"""Minimal LoRA (Low-Rank Adaptation) for NinjaMind.

Instead of updating a full weight matrix W (d_out x d_in), LoRA freezes W and
learns a low-rank correction B @ A (rank r << d), so the effective weight is
W + (alpha / r) * B @ A. Only A and B train — a tiny fraction of the params.
"""
from __future__ import annotations

from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.scale = alpha / rank
        # A random, B zero => the correction starts at exactly 0 and the
        # adapted model is identical to the base model at step 0.
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(x)) * self.scale


def apply_lora(model, rank=8, alpha=16, targets=DEFAULT_TARGETS):
    """Attach a LoRA branch to every target nn.Linear (base weights untouched)."""
    targets = tuple(targets)
    attached = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            if hasattr(module, "lora"):
                continue
            lora = LoRA(module.in_features, module.out_features, rank, alpha)
            lora = lora.to(device=module.weight.device, dtype=module.weight.dtype)
            module.lora = lora  # registers as a submodule -> shows in state_dict
            original_forward = module.forward
            module.forward = (
                lambda x, _orig=original_forward, _lora=lora: _orig(x) + _lora(x)
            )
            attached += 1
    if attached == 0:
        raise ValueError(f"no linear modules matched LoRA targets: {targets}")
    model._lora_config = {"rank": rank, "alpha": alpha, "targets": list(targets)}
    return model


def freeze_non_lora(model):
    """Freeze everything except LoRA params; returns the trainable params."""
    trainable = []
    for param_name, param in model.named_parameters():
        param.requires_grad = ".lora." in param_name
        if param.requires_grad:
            trainable.append(param)
    return trainable


def lora_state_dict(model):
    """Return only adapter tensors, detached on CPU for a compact checkpoint."""
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if ".lora." in key
    }


def save_lora(model, path, metadata=None):
    """Save adapters and enough metadata to inspect/reload the checkpoint."""
    first_lora = next((module for module in model.modules() if isinstance(module, LoRA)), None)
    if first_lora is None:
        raise ValueError("no LoRA adapters are attached to the model")
    adapter_config = getattr(model, "_lora_config", None)
    if adapter_config is None:
        adapter_config = {
            "rank": first_lora.A.out_features,
            "alpha": first_lora.scale * first_lora.A.out_features,
            "targets": list(DEFAULT_TARGETS),
        }
    payload = {
        "format_version": 1,
        "adapter_config": adapter_config,
        "metadata": metadata or {},
        "state_dict": lora_state_dict(model),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_lora(model, path):
    """Load a structured adapter checkpoint (or the legacy raw state dict)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    if not state or any(".lora." not in key for key in state):
        raise ValueError("adapter checkpoint did not contain only LoRA parameters")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"unexpected LoRA keys: {unexpected[:5]}")
    return missing, unexpected


@torch.no_grad()
def merge_lora(model):
    """Fold every adapter into its base ``nn.Linear`` and remove the branch.

    After merging, inference no longer pays for the two extra low-rank matrix
    multiplications and the resulting model can be saved as a normal base
    checkpoint.
    """
    merged = 0
    for module in model.modules():
        lora = getattr(module, "lora", None)
        if not isinstance(module, nn.Linear) or not isinstance(lora, LoRA):
            continue
        delta = (lora.B.weight @ lora.A.weight) * lora.scale
        module.weight.add_(delta.to(device=module.weight.device, dtype=module.weight.dtype))
        del module.lora
        module.forward = MethodType(nn.Linear.forward, module)
        merged += 1
    if merged == 0:
        raise ValueError("no LoRA adapters are attached to the model")
    return model
