"""Minimal LoRA (Low-Rank Adaptation) for NinjaMind.

Instead of updating a full weight matrix W (d_out x d_in), LoRA freezes W and
learns a low-rank correction B @ A (rank r << d), so the effective weight is
W + (alpha / r) * B @ A. Only A and B train — a tiny fraction of the params.
"""
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
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.split(".")[-1] in targets:
            lora = LoRA(module.in_features, module.out_features, rank, alpha)
            lora = lora.to(module.weight.device)
            module.lora = lora  # registers as a submodule -> shows in state_dict
            original_forward = module.forward
            module.forward = (
                lambda x, _orig=original_forward, _lora=lora: _orig(x) + _lora(x)
            )
    return model


def freeze_non_lora(model):
    """Freeze everything except LoRA params; returns the trainable params."""
    trainable = []
    for param_name, param in model.named_parameters():
        param.requires_grad = ".lora." in param_name
        if param.requires_grad:
            trainable.append(param)
    return trainable


def save_lora(model, path):
    state = {k: v for k, v in model.state_dict().items() if ".lora." in k}
    torch.save(state, path)


def load_lora(model, path):
    state = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"unexpected LoRA keys: {unexpected[:5]}"
