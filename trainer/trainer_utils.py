"""Shared training, checkpoint, distributed, and RL utilities.

The helpers in this module deliberately keep single-process CPU/MPS training
as the default.  When launched with ``torchrun`` (``WORLD_SIZE > 1``), the
same entry points initialize DDP, shard data with ``DistributedSampler``, and
restrict logs/checkpoints to rank zero.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import time
from collections import Counter
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset


# --------------------------------------------------------------------------- #
# Distributed environment / model plumbing                                    #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DistributedContext:
    """Runtime information shared by all trainers.

    ``initialized_here`` is tracked so callers can safely use
    :func:`cleanup_distributed` without tearing down a process group owned by
    an embedding application or a test harness.
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device = torch.device("cpu")
    backend: str | None = None
    initialized_here: bool = False

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _env_int(environ: Mapping[str, str], name: str, default: int) -> int:
    value = environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def parse_distributed_env(
    environ: Mapping[str, str] | None = None,
    local_rank: int | None = None,
) -> tuple[int, int, int]:
    """Parse the variables populated by ``torchrun``.

    Returns ``(rank, local_rank, world_size)`` and provides a deterministic
    ``(0, 0, 1)`` fallback when the script is launched normally.  Keeping
    parsing side-effect free makes the fallback independently testable.
    """

    environ = os.environ if environ is None else environ
    world_size = _env_int(environ, "WORLD_SIZE", 1)
    rank = _env_int(environ, "RANK", 0)
    parsed_local_rank = _env_int(environ, "LOCAL_RANK", 0)
    if local_rank is not None:
        parsed_local_rank = local_rank
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"RANK must be in [0, WORLD_SIZE), got {rank}/{world_size}")
    if parsed_local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be >= 0, got {parsed_local_rank}")
    return rank, parsed_local_rank, world_size


def get_device(preference: str = "auto", local_rank: int = 0) -> torch.device:
    """Resolve the requested device, selecting the local CUDA device for DDP."""

    if preference != "auto":
        device = torch.device(preference)
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", local_rank)
        return device
    if torch.cuda.is_available():
        return torch.device("cuda", local_rank)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def add_distributed_args(parser):
    """Add optional torchrun/DDP arguments to an ``ArgumentParser``."""

    parser.add_argument(
        "--local-rank", "--local_rank", dest="local_rank", type=int, default=None,
        help="local worker rank (normally supplied through LOCAL_RANK by torchrun)",
    )
    parser.add_argument(
        "--dist-backend", "--dist_backend", dest="dist_backend",
        choices=("auto", "gloo", "nccl"), default="auto",
        help="DDP process-group backend",
    )


def setup_distributed(args=None) -> DistributedContext:
    """Initialize an optional process group and return its runtime context."""

    local_rank_arg = getattr(args, "local_rank", None)
    rank, local_rank, world_size = parse_distributed_env(local_rank=local_rank_arg)
    device = get_device(getattr(args, "device", "auto"), local_rank=local_rank)
    requested_backend = getattr(args, "dist_backend", "auto")
    backend = None
    initialized_here = False

    if dist.is_available() and dist.is_initialized():
        # An embedding application (or test harness) may own the group even
        # when torchrun variables are not present in this process.
        rank, world_size = dist.get_rank(), dist.get_world_size()
        backend = str(dist.get_backend())

    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable in this PyTorch build")
        if device.type == "mps":
            raise RuntimeError("multi-process MPS training is unsupported; use CPU/gloo or CUDA")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        backend = (
            "nccl" if requested_backend == "auto" and device.type == "cuda"
            else "gloo" if requested_backend == "auto"
            else requested_backend
        )
        if backend == "nccl" and device.type != "cuda":
            raise ValueError("the nccl backend requires a CUDA device")
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
            initialized_here = True
        else:
            # Trust the existing group rather than attempting a second init.
            rank, world_size = dist.get_rank(), dist.get_world_size()
            backend = str(dist.get_backend())

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        backend=backend,
        initialized_here=initialized_here,
    )


# Backwards-friendly alias: both verbs are common in training entry points.
init_distributed = setup_distributed


def cleanup_distributed(context: DistributedContext | None = None) -> None:
    """Destroy a process group initialized by :func:`setup_distributed`."""

    should_destroy = context is None or context.initialized_here
    if should_destroy and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(context: DistributedContext | None = None) -> bool:
    if context is not None:
        return context.is_main
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


def rank0_print(*values, context: DistributedContext | None = None, **kwargs) -> None:
    if is_main_process(context):
        print(*values, **kwargs)


def distributed_mean_metrics(
    metrics: Mapping[str, float],
    context: DistributedContext | None = None,
) -> dict[str, float]:
    """Average scalar logging metrics across workers in one collective.

    Every rank must call this helper at the same training step. Keeping the
    collective separate from ``rank0_print`` prevents rank zero from reporting
    only its local data shard while the other workers skip synchronization.
    """

    if not metrics:
        return {}
    context = context or DistributedContext()
    names = tuple(metrics)
    values = torch.tensor(
        [float(metrics[name]) for name in names],
        dtype=torch.float32,
        device=context.device,
    )
    if context.distributed:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("distributed metric reduction requires an initialized process group")
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values /= context.world_size
    return dict(zip(names, values.cpu().tolist(), strict=True))


def unwrap_model(model):
    """Return the underlying module for either a plain model or DDP wrapper."""

    return model.module if isinstance(model, DistributedDataParallel) else model


def wrap_ddp(model, context: DistributedContext):
    if not context.distributed:
        return model
    kwargs = {"device_ids": [context.local_rank], "output_device": context.local_rank}
    if context.device.type != "cuda":
        kwargs = {}
    return DistributedDataParallel(model, **kwargs)


def build_dataloader(
    dataset,
    args,
    context: DistributedContext | None = None,
    *,
    shuffle: bool = True,
    drop_last: bool = False,
    collate_fn=None,
    batch_size: int | None = None,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Create a sharded loader under DDP and a normal loader otherwise."""

    context = context or DistributedContext(device=get_device(getattr(args, "device", "auto")))
    sampler = None
    if context.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=int(getattr(args, "seed", 42)),
        )
    # RandomSampler rejects an empty dataset.  Disabling shuffle gives empty
    # datasets a clean zero-step behavior, useful for validation and tests.
    loader_shuffle = shuffle and sampler is None and len(dataset) > 0
    return DataLoader(
        dataset,
        batch_size=batch_size or args.batch_size,
        shuffle=loader_shuffle,
        sampler=sampler,
        drop_last=drop_last,
        num_workers=getattr(args, "num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=context.device.type == "cuda",
        generator=generator,
    )


def set_dataloader_epoch(loader: DataLoader, epoch: int) -> None:
    """Reseed a DistributedSampler once per epoch (a no-op otherwise)."""

    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)


def autocast_ctx(device: torch.device):
    """Mixed precision on CUDA; full precision elsewhere (MPS/CPU)."""

    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def make_grad_scaler(device: torch.device):
    """Enable loss scaling only for CUDA fp16 (bf16 does not need it)."""

    enabled = device.type == "cuda" and not torch.cuda.is_bf16_supported()
    return torch.amp.GradScaler("cuda", enabled=enabled)


def add_model_args(parser):
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--use_moe", action="store_true")


def add_train_args(parser, default_lr):
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=default_lr)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_steps", type=int, default=0,
        help="stop after N optimizer updates (0 = full run)",
    )
    add_distributed_args(parser)


def add_supervised_eval_args(parser):
    """Add validation, artifact, and exact supervised-resume options."""

    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.1,
        help="fraction of unique, grouped samples reserved for validation (0 disables)",
    )
    parser.add_argument(
        "--split_strategy",
        choices=("exact", "full"),
        default="exact",
        help=(
            "exact: tensor-level dedup + deterministic validation split; "
            "full: trust a prevalidated source and train every record without a hold-out"
        ),
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="stable train/validation split seed (defaults to --seed)",
    )
    parser.add_argument(
        "--metrics_dir",
        type=str,
        default="",
        help="JSON/CSV/SVG metric output directory (defaults to OUT_DIR/metrics)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="strict v2 training checkpoint to resume (distinct from --init_from)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=0,
        help="save resumable latest checkpoint every N optimizer steps (0 = epoch boundary)",
    )


def build_model(args, vocab_size, device, context: DistributedContext | None = None):
    from model.model import NinjaMindConfig, NinjaMindForCausalLM

    if args.num_attention_heads < 1 or args.num_key_value_heads < 1:
        raise ValueError("attention head counts must be >= 1")
    if args.hidden_size % args.num_attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if args.num_attention_heads % args.num_key_value_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    config = NinjaMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        use_moe=args.use_moe,
        vocab_size=vocab_size,
    )
    model = NinjaMindForCausalLM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    rank0_print(
        f"model: {n_params / 1e6:.2f}M params | vocab {vocab_size} | device {device}",
        context=context,
    )
    return model, config


# --------------------------------------------------------------------------- #
# Versioned checkpoints with legacy raw-state compatibility                    #
# --------------------------------------------------------------------------- #
def _plain_value(value: Any) -> Any:
    """Convert metadata to objects accepted by weights-only ``torch.load``."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(v) for v in value]
    return str(value)


def _tokenizer_metadata(tokenizer) -> dict[str, Any] | None:
    if tokenizer is None:
        return None
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "special_tokens_map": _plain_value(getattr(tokenizer, "special_tokens_map", {})),
    }


def tokenizer_fingerprint(tokenizer) -> str:
    """Hash the tokenizer state that changes supervised tokenization.

    A vocabulary-size check alone cannot distinguish two tokenizers that map
    the same number of tokens to different IDs.  Strict resume identities use
    this digest to reject that silent preprocessing change.
    """

    if tokenizer is None or not hasattr(tokenizer, "get_vocab"):
        raise TypeError("tokenizer must provide get_vocab()")
    payload = {
        "class": tokenizer.__class__.__name__,
        "vocab": tokenizer.get_vocab(),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "special_tokens_map": _plain_value(
            getattr(tokenizer, "special_tokens_map", {})
        ),
        "special_token_ids": {
            name: getattr(tokenizer, name, None)
            for name in (
                "bos_token_id",
                "eos_token_id",
                "pad_token_id",
                "unk_token_id",
            )
        },
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def checkpoint_model_state(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    """Extract model weights from a versioned or legacy checkpoint object."""

    for key in ("model_state_dict", "state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise ValueError("checkpoint does not contain a model state dict")


def load_checkpoint_state(path: str) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"unsupported checkpoint type: {type(checkpoint).__name__}")
    state_dict = dict(checkpoint_model_state(checkpoint))
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_weights(model, path, device, context: DistributedContext | None = None):
    """Load either new metadata checkpoints or historical raw state dicts."""

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"unsupported checkpoint type: {type(checkpoint).__name__}")
    state_dict = dict(checkpoint_model_state(checkpoint))
    # Accept checkpoints saved directly from an older DDP wrapper.
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    missing, unexpected = unwrap_model(model).load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        rank0_print(
            f"load_weights: missing={len(missing)} unexpected={len(unexpected)}",
            context=context,
        )
    model.to(device)
    rank0_print(f"loaded weights from {path}", context=context)
    if "model_state_dict" not in checkpoint:
        return {"format_version": 0, "legacy": True}
    return {key: value for key, value in checkpoint.items() if key != "model_state_dict"}


def save_checkpoint(
    path: str,
    model,
    *,
    config=None,
    tokenizer=None,
    args=None,
    optimizer=None,
    scaler=None,
    step: int | None = None,
    stage: str | None = None,
    training_state: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    context: DistributedContext | None = None,
) -> bool:
    """Save model weights plus reproducibility metadata on rank zero.

    Returns ``True`` on the rank that wrote the file and ``False`` elsewhere.
    Every rank participates when ``training_state`` is supplied under DDP so
    rank-local RNG and in-epoch metric state are preserved.
    ``load_weights`` remains compatible with historical raw state dicts.
    """

    resumable_v2 = training_state is not None
    if resumable_v2 and (optimizer is None or scaler is None):
        raise ValueError("resumable checkpoints require optimizer and scaler state")
    rank_resume_states = None
    if resumable_v2:
        from trainer.checkpointing import gather_rank_resume_states

        rank_resume_states = gather_rank_resume_states(training_state)
    if not is_main_process(context):
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    base_model = unwrap_model(model)
    config = config or getattr(base_model, "config", None)
    config_dict = config.to_dict() if hasattr(config, "to_dict") else config
    train_args = vars(args) if args is not None and hasattr(args, "__dict__") else args
    format_version = 2 if resumable_v2 else 1
    payload: dict[str, Any] = {
        "format_version": format_version,
        "model_state_dict": base_model.state_dict(),
        "config": _plain_value(config_dict),
        "tokenizer": _tokenizer_metadata(tokenizer),
        "training_args": _plain_value(train_args),
        "stage": stage,
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if resumable_v2:
        assert rank_resume_states is not None
        training_states = [
            _plain_value(state["training_state"]) for state in rank_resume_states
        ]
        rng_states = [state["rng_state"] for state in rank_resume_states]
        identities = [state.get("run_identity", {}) for state in training_states]
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError("all DDP ranks must share the same run identity")
        payload["training_state"] = training_states[0]
        payload["run_identity"] = identities[0]
        payload["world_size"] = len(rank_resume_states)
        if len(rank_resume_states) == 1:
            payload["rng_state"] = rng_states[0]
        else:
            payload["rng_state_by_rank"] = rng_states
            payload["training_state_by_rank"] = training_states
    elif training_state is not None:
        payload["training_state"] = _plain_value(training_state)
    if extra:
        payload["extra"] = _plain_value(extra)
    from trainer.checkpointing import atomic_torch_save

    atomic_torch_save(payload, path)
    rank0_print(f"checkpoint saved: {path}", context=context)
    return True


def ckpt_name(out_dir, stage, config):
    moe = "_moe" if config.use_moe else ""
    return os.path.join(out_dir, f"{stage}_{config.hidden_size}{moe}.pth")


# --------------------------------------------------------------------------- #
# Loss / schedule                                                              #
# --------------------------------------------------------------------------- #
def cosine_lr(step, total_steps, max_lr, warmup_ratio=0.02, min_ratio=0.1):
    """Linear warmup then cosine decay to ``min_ratio * max_lr``."""

    warmup = max(1, int(total_steps * warmup_ratio))
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return max_lr * (min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress)))


def masked_ce_sum_and_count(logits, targets, mask):
    """Return summed token NLL and valid-token count for a masked batch."""

    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="none"
    )
    flat_mask = mask.reshape(-1).to(dtype=token_loss.dtype)
    return (token_loss * flat_mask).sum(), flat_mask.sum()


def masked_ce(logits, targets, mask):
    """Cross entropy averaged over positions where ``mask`` is 1."""

    loss_sum, token_count = masked_ce_sum_and_count(logits, targets, mask)
    return loss_sum / token_count.clamp(min=1)


def _dataset_attr(dataset, name, default=None):
    """Read an attribute through nested dataset index views/subsets."""

    current = dataset
    while current is not None:
        if hasattr(current, name):
            return getattr(current, name)
        current = getattr(current, "dataset", None)
    return default


@torch.no_grad()
def evaluate_supervised(
    model,
    dataset,
    args,
    device,
    context: DistributedContext | None = None,
):
    """Compute exact token-weighted validation CE/perplexity.

    Validation indices are strided across ranks without padding, so no sample
    is duplicated when the dataset size is not divisible by ``world_size``.
    The unwrapped module avoids DDP forward collectives when ranks receive a
    different number of batches; all ranks only synchronize the final NLL sum
    and token count. ``no_grad`` is intentional: inference mode would mark
    lazily-created RoPE/cache buffers as inference tensors that DDP cannot
    subsequently synchronize during training.
    """

    context = context or DistributedContext(device=device)
    local_indices = range(context.rank, len(dataset), context.world_size)
    local_dataset = Subset(dataset, tuple(local_indices))
    local_context = DistributedContext(device=device)
    loader = build_dataloader(
        local_dataset,
        args,
        local_context,
        shuffle=False,
        drop_last=False,
    )
    base_model = unwrap_model(model)
    was_training = base_model.training
    base_model.eval()
    pad_id = _dataset_attr(dataset, "pad_id")
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    try:
        for X, Y, loss_mask in loader:
            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)
            model_inputs = {"input_ids": X}
            if pad_id is not None:
                model_inputs["attention_mask"] = X.ne(pad_id)
            with autocast_ctx(device):
                output = base_model(**model_inputs)
            loss_sum, token_count = masked_ce_sum_and_count(
                output.logits,
                Y,
                loss_mask,
            )
            totals[0] += loss_sum.detach().double()
            totals[1] += token_count.detach().double()
    finally:
        base_model.train(was_training)

    if context.distributed:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("distributed validation requires an initialized process group")
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)

    token_count = int(totals[1].item())
    if token_count == 0:
        raise ValueError("validation split contains no supervised target tokens")
    validation_ce = totals[0].item() / token_count
    try:
        perplexity = math.exp(validation_ce)
    except OverflowError:
        perplexity = None
    if perplexity is not None and not math.isfinite(perplexity):
        perplexity = None
    return {
        "validation_ce": validation_ce,
        "validation_perplexity": perplexity,
        "validation_tokens": token_count,
        "perplexity_overflow": perplexity is None,
    }


def normalize_token_weighted_gradients(
    params,
    local_token_count: int,
    context: DistributedContext,
) -> int:
    """Normalize accumulated NLL-sum gradients by the global token count.

    DDP averages synchronized gradients across ranks.  Multiplying by
    ``world_size / global_tokens`` therefore turns the averaged NLL-sum
    gradient back into the gradient of one global token-weighted mean.
    """

    token_count = torch.tensor(
        float(local_token_count),
        dtype=torch.float64,
        device=context.device,
    )
    if context.distributed:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "distributed gradient normalization requires an initialized process group"
            )
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
    global_token_count = int(token_count.item())
    if global_token_count <= 0:
        raise ValueError("an optimizer window contains no supervised target tokens")
    scale = context.world_size / global_token_count
    for parameter in params:
        if parameter.grad is not None:
            parameter.grad.mul_(scale)
    return global_token_count


def masked_mean(x, mask, dim=None):
    mask = mask.to(dtype=x.dtype)
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp(min=1)
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1)


def whiten(x, mask):
    """Normalize masked entries to zero mean / unit variance."""

    mask = mask.to(dtype=x.dtype)
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    normalized = (x - mean) * torch.rsqrt(var + 1e-8)
    return normalized * mask


# --------------------------------------------------------------------------- #
# Supervised training loop (pretrain / SFT / LoRA)                             #
# --------------------------------------------------------------------------- #
def train_supervised(
    model,
    dataset,
    args,
    device,
    save_fn,
    params=None,
    context: DistributedContext | None = None,
    *,
    validation_dataset=None,
    epoch_callback=None,
    stage: str | None = None,
    run_identity: Mapping[str, Any] | None = None,
):
    """Train supervised stages with exact validation and resumable cursors.

    The final partial batch and the final partial accumulation window both
    produce an optimizer update.  ``max_steps`` counts optimizer updates, not
    micro-batches. Checkpoints are only emitted at optimizer boundaries.
    """

    if args.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    if getattr(args, "save_interval", 0) < 0:
        raise ValueError("save_interval must be >= 0")
    context = context or DistributedContext(device=device)
    loader_generator = torch.Generator()
    loader = build_dataloader(
        dataset,
        args,
        context,
        shuffle=True,
        drop_last=False,
        generator=loader_generator,
    )
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = list(params)
    if not params:
        raise ValueError("model has no trainable parameters")

    model = wrap_ddp(model, context)
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scaler = make_grad_scaler(device)
    updates_per_epoch = math.ceil(len(loader) / args.accumulation_steps) if len(loader) else 0
    total_steps = updates_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    seed = int(getattr(args, "seed", 42))

    identity = {
        **dict(run_identity or {}),
        "training_contract": "supervised-v2",
        "stage": stage,
        "seed": seed,
        "learning_rate": float(args.lr),
        "grad_clip": float(args.grad_clip),
        "epochs": int(args.epochs),
        "max_steps": int(args.max_steps),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "accumulation_steps": int(args.accumulation_steps),
        "batches_per_epoch": len(loader),
        "dataset_max_length": _dataset_attr(dataset, "max_length"),
        "planned_total_steps": total_steps,
        "world_size": context.world_size,
    }

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    start_epoch = 0
    start_batch = 0
    history: list[dict[str, Any]] = []
    best_validation_ce: float | None = None
    partial_nll_sum = 0.0
    partial_token_count = 0
    partial_duration_seconds = 0.0
    checkpoint_finalized = False

    resume_path = getattr(args, "resume", "")
    if resume_path:
        from trainer.checkpointing import load_training_state

        restored = load_training_state(
            resume_path,
            unwrap_model(model),
            optimizer,
            scaler,
            device=device,
            expected_stage=stage,
            expected_run_identity=identity,
        )
        start_epoch = int(restored["epoch"])
        start_batch = int(restored["batch_in_epoch"])
        optimizer_step = int(restored["optimizer_step"])
        micro_step = int(restored.get("micro_steps_completed", 0))
        history = list(restored.get("history", []))
        best_validation_ce = restored.get("best_validation_ce")
        partial_nll_sum = float(restored.get("partial_train_nll_sum", 0.0))
        partial_token_count = int(restored.get("partial_train_tokens", 0))
        partial_duration_seconds = float(restored.get("partial_duration_seconds", 0.0))
        checkpoint_finalized = bool(restored.get("checkpoint_finalized", False))
        rank0_print(
            f"resumed {stage or 'supervised'} at epoch {start_epoch + 1} "
            f"batch {start_batch} step {optimizer_step}",
            context=context,
        )

    def training_state(
        *,
        epoch_cursor,
        batch_cursor,
        epoch_nll_sum,
        epoch_token_count,
        epoch_duration_seconds,
        finalized,
    ):
        return {
            "epoch": int(epoch_cursor),
            "batch_in_epoch": int(batch_cursor),
            "optimizer_step": int(optimizer_step),
            # Checkpoints are emitted only after optimizer.step(), so no
            # accumulation window is pending at a save boundary.
            "micro_step": 0,
            "micro_steps_completed": int(micro_step),
            "planned_total_steps": int(total_steps),
            "history": history,
            "best_validation_ce": best_validation_ce,
            "partial_train_nll_sum": float(epoch_nll_sum),
            "partial_train_tokens": int(epoch_token_count),
            "partial_duration_seconds": float(epoch_duration_seconds),
            "checkpoint_finalized": bool(finalized),
            "run_identity": identity,
        }

    def save(kind, state):
        callback_kwargs = {
            "optimizer": optimizer,
            "scaler": scaler,
            "training_state": state,
            "kind": kind,
        }
        try:
            inspect.signature(save_fn).bind(unwrap_model(model), **callback_kwargs)
        except (TypeError, ValueError):
            # Preserve the historical one-argument callback for small embedded
            # callers. Such callbacks save weights only and cannot be resumed.
            if is_main_process(context):
                save_fn(unwrap_model(model))
        else:
            # Full callbacks run on every rank because resumable DDP saves
            # collect rank-local RNG/progress before rank zero writes.
            save_fn(unwrap_model(model), **callback_kwargs)

    if validation_dataset is not None and not history and optimizer_step == 0:
        baseline = {
            "epoch": 0,
            "epoch_complete": True,
            "optimizer_step": 0,
            "learning_rate": None,
            "train_ce": None,
            "train_tokens": 0,
            "duration_seconds": 0.0,
            **evaluate_supervised(model, validation_dataset, args, device, context),
        }
        history.append(baseline)
        best_validation_ce = float(baseline["validation_ce"])
        baseline_state = training_state(
            epoch_cursor=0,
            batch_cursor=0,
            epoch_nll_sum=0.0,
            epoch_token_count=0,
            epoch_duration_seconds=0.0,
            finalized=False,
        )
        save("best", baseline_state)
        if epoch_callback is not None and is_main_process(context):
            epoch_callback(history, baseline_state)

    run_finished = checkpoint_finalized and (
        start_epoch >= args.epochs
        or (args.max_steps and optimizer_step >= args.max_steps)
    )
    if run_finished:
        restored_state = training_state(
            epoch_cursor=start_epoch,
            batch_cursor=start_batch,
            epoch_nll_sum=partial_nll_sum,
            epoch_token_count=partial_token_count,
            epoch_duration_seconds=partial_duration_seconds,
            finalized=True,
        )
        # Re-emitting final outputs makes a resume repair a process that died
        # after the latest checkpoint but before best/artifact writes.
        save("latest", restored_state)
        if history and best_validation_ce is not None:
            last_validation_ce = history[-1].get("validation_ce")
            if (
                last_validation_ce is not None
                and float(last_validation_ce) == float(best_validation_ce)
            ):
                save("best", restored_state)
        if epoch_callback is not None and is_main_process(context):
            epoch_callback(history, restored_state)
        return {
            "optimizer_steps": optimizer_step,
            "micro_steps": micro_step,
            "total_steps": total_steps,
            "history": history,
            "best_validation_ce": best_validation_ce,
        }

    stopped = False

    for epoch in range(start_epoch, args.epochs):
        if stopped:
            break
        loader_generator.manual_seed(seed + epoch)
        set_dataloader_epoch(loader, epoch)
        window_objective_sum = 0.0
        window_token_count = 0
        epoch_nll_sum = partial_nll_sum if epoch == start_epoch else 0.0
        epoch_token_count = partial_token_count if epoch == start_epoch else 0
        elapsed_before = partial_duration_seconds if epoch == start_epoch else 0.0
        epoch_started = time.perf_counter()
        next_batch_cursor = start_batch if epoch == start_epoch else 0
        for batch_index, (X, Y, loss_mask) in enumerate(loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            if args.max_steps and optimizer_step >= args.max_steps:
                stopped = True
                break

            should_update = (
                (batch_index + 1) % args.accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if batch_index % args.accumulation_steps == 0:
                lr = cosine_lr(optimizer_step, max(total_steps, 1), args.lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr

            X = X.to(device, non_blocking=True)
            Y = Y.to(device, non_blocking=True)
            loss_mask = loss_mask.to(device, non_blocking=True)
            pad_id = getattr(dataset, "pad_id", None)
            attention_mask = X.ne(pad_id) if pad_id is not None else None
            model_inputs = {"input_ids": X}
            if attention_mask is not None:
                model_inputs["attention_mask"] = attention_mask
            sync_ctx = model.no_sync() if context.distributed and not should_update else nullcontext()
            with sync_ctx:
                with autocast_ctx(device):
                    out = model(**model_inputs)
                    batch_nll_sum, batch_token_count = masked_ce_sum_and_count(
                        out.logits,
                        Y,
                        loss_mask,
                    )
                    objective_sum = batch_nll_sum
                    aux = getattr(out, "aux_loss", None)
                    if torch.is_tensor(aux):
                        # Treat router aux as a per-target-token penalty so the
                        # same global normalization applies without mixing it
                        # into the reported pure CE metric.
                        objective_sum = objective_sum + aux * batch_token_count
                    loss = objective_sum
                scaler.scale(loss).backward()

            window_objective_sum += objective_sum.detach().float().item()
            window_token_count += int(batch_token_count.detach().item())
            epoch_nll_sum += batch_nll_sum.detach().double().item()
            epoch_token_count += int(batch_token_count.detach().item())
            micro_step += 1
            next_batch_cursor = batch_index + 1
            if not should_update:
                continue

            # AMP order is important: unscale -> clip -> optimizer step.
            scaler.unscale_(optimizer)
            normalize_token_weighted_gradients(
                params,
                window_token_count,
                context,
            )
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            if (optimizer_step - 1) % args.log_interval == 0:
                rank0_print(
                    f"epoch {epoch + 1}/{args.epochs} step {optimizer_step}/{total_steps} "
                    f"loss {window_objective_sum / window_token_count:.4f} lr {lr:.2e}",
                    context=context,
                )
            window_objective_sum = 0.0
            window_token_count = 0

            save_interval = int(getattr(args, "save_interval", 0))
            if (
                save_interval
                and optimizer_step % save_interval == 0
                and optimizer_step < total_steps
                and next_batch_cursor < len(loader)
            ):
                cursor_epoch = epoch
                cursor_batch = next_batch_cursor
                save(
                    "latest",
                    training_state(
                        epoch_cursor=cursor_epoch,
                        batch_cursor=cursor_batch,
                        epoch_nll_sum=0.0 if cursor_epoch > epoch else epoch_nll_sum,
                        epoch_token_count=0 if cursor_epoch > epoch else epoch_token_count,
                        epoch_duration_seconds=(
                            0.0
                            if cursor_epoch > epoch
                            else elapsed_before + time.perf_counter() - epoch_started
                        ),
                        finalized=False,
                    ),
                )

        epoch_duration = elapsed_before + time.perf_counter() - epoch_started
        epoch_complete = next_batch_cursor >= len(loader)
        aggregate = torch.tensor(
            [epoch_nll_sum, float(epoch_token_count)],
            dtype=torch.float64,
            device=device,
        )
        if context.distributed:
            dist.all_reduce(aggregate, op=dist.ReduceOp.SUM)
        global_tokens = int(aggregate[1].item())
        record = {
            "epoch": epoch + 1,
            "epoch_complete": epoch_complete,
            "optimizer_step": optimizer_step,
            "learning_rate": (
                float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else None
            ),
            "train_ce": aggregate[0].item() / global_tokens if global_tokens else None,
            "train_tokens": global_tokens,
            "duration_seconds": epoch_duration,
        }
        if validation_dataset is not None:
            record.update(evaluate_supervised(model, validation_dataset, args, device, context))
        else:
            record.update({
                "validation_ce": None,
                "validation_perplexity": None,
                "validation_tokens": 0,
                "perplexity_overflow": False,
            })
        history.append(record)
        rank0_print(
            f"epoch {epoch + 1} metrics train_ce={record['train_ce']} "
            f"validation_ce={record['validation_ce']} "
            f"perplexity={record['validation_perplexity']}",
            context=context,
        )

        improved = (
            record["validation_ce"] is not None
            and (
                best_validation_ce is None
                or float(record["validation_ce"]) < best_validation_ce
            )
        )
        if improved:
            best_validation_ce = float(record["validation_ce"])

        cursor_epoch = epoch + 1 if epoch_complete else epoch
        cursor_batch = 0 if epoch_complete else next_batch_cursor
        state = training_state(
            epoch_cursor=cursor_epoch,
            batch_cursor=cursor_batch,
            epoch_nll_sum=0.0 if epoch_complete else epoch_nll_sum,
            epoch_token_count=0 if epoch_complete else epoch_token_count,
            epoch_duration_seconds=0.0 if epoch_complete else epoch_duration,
            finalized=True,
        )
        save("latest", state)
        if improved:
            save("best", state)
        if epoch_callback is not None and is_main_process(context):
            epoch_callback(history, state)

        start_batch = 0
        partial_nll_sum = 0.0
        partial_token_count = 0
        partial_duration_seconds = 0.0

    return {
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "total_steps": total_steps,
        "history": history,
        "best_validation_ce": best_validation_ce,
    }


# --------------------------------------------------------------------------- #
# RL helpers (PPO / GRPO)                                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_generate(
    model,
    input_ids,
    attention_mask,
    max_new_tokens,
    eos_id,
    pad_id,
    temperature=1.0,
    top_k=30,
):
    """Batched sampling with a KV cache from left-padded prompts.

    Returns ``(seq, gen_mask, attn_mask)`` where ``seq`` is prompt+completion,
    ``gen_mask`` marks real generated tokens (up to and including eos) and
    ``attn_mask`` covers the full sequence (prompt pads and post-eos pads = 0).
    """

    if max_new_tokens <= 0:
        empty = input_ids.new_zeros((input_ids.size(0), 0))
        return input_ids, empty, attention_mask
    was_training = model.training
    model.eval()
    device = input_ids.device
    batch_size = input_ids.size(0)

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1]
    done = torch.zeros(batch_size, dtype=torch.bool, device=device)
    tokens, gen_mask_cols = [], []

    # Keep the forward-call count identical on every DDP rank. Ranks can reach
    # EOS at different times; a local early break would otherwise let one rank
    # leave while another is still entering DDP buffer broadcasts.
    for token_index in range(max_new_tokens):
        logits = next_logits.float() / max(temperature, 1e-6)
        if top_k:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(
                logits < top_values[:, [-1]],
                torch.full_like(logits, float("-inf")),
                logits,
            )
        next_token = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
        next_token = torch.where(done, torch.full_like(next_token, pad_id), next_token)

        gen_mask_cols.append((~done).long())
        tokens.append(next_token)
        done = done | (next_token == eos_id)
        attention_mask = torch.cat([attention_mask, gen_mask_cols[-1].unsqueeze(1)], dim=1)
        if token_index + 1 < max_new_tokens:
            out = model(
                input_ids=next_token.unsqueeze(1),
                attention_mask=attention_mask,
                use_cache=True,
                past_key_values=past,
            )
            past = out.past_key_values
            next_logits = out.logits[:, -1]

    if was_training:
        model.train()
    seq = torch.cat([input_ids, torch.stack(tokens, dim=1)], dim=1)
    gen_mask = torch.stack(gen_mask_cols, dim=1)
    return seq, gen_mask, attention_mask


def token_logprobs(model, seq, attn_mask, prompt_len):
    """Log-probs of each generated token ``seq[:, prompt_len:]`` — shape (B, G)."""

    out = model(input_ids=seq, attention_mask=attn_mask)
    logp = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
    gathered = logp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
    return gathered[:, prompt_len - 1:]


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Generalized Advantage Estimation over generated-token positions."""

    if rewards.shape != values.shape or rewards.shape != mask.shape:
        raise ValueError("rewards, values, and mask must have identical shapes")
    batch_size, generated_length = rewards.shape
    mask = mask.to(dtype=rewards.dtype)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(generated_length)):
        next_value = (
            values[:, t + 1] if t < generated_length - 1
            else torch.zeros_like(last_gae)
        )
        next_nonterminal = (
            mask[:, t + 1] if t < generated_length - 1
            else torch.zeros_like(last_gae)
        )
        delta = rewards[:, t] + gamma * next_value * next_nonterminal - values[:, t]
        last_gae = (delta + gamma * lam * next_nonterminal * last_gae) * mask[:, t]
        advantages[:, t] = last_gae
    returns = (advantages + values) * mask
    return advantages, returns


def containment_reward(completion, answer):
    """Toy rule-based reward: 1 if the reference answer appears verbatim."""

    answer = (answer or "").strip()
    return 1.0 if answer and answer in completion else 0.0


def reference_overlap_reward(completion, answer):
    """Continuous character n-gram F1 against a preferred reference answer.

    This lightweight reward is intended for small, reproducible PPO/GRPO
    experiments where loading a separate billion-parameter reward model would
    dominate the policy itself.  It mixes unigram and bigram overlap so weak
    policies still receive a graded signal instead of almost-always-zero exact
    match rewards.  It is not a substitute for a learned quality judge.
    """

    def normalized_characters(text):
        return [character.lower() for character in str(text) if character.isalnum()]

    def ngram_f1(candidate, reference, width):
        candidate_ngrams = [tuple(candidate[i : i + width]) for i in range(len(candidate) - width + 1)]
        reference_ngrams = [tuple(reference[i : i + width]) for i in range(len(reference) - width + 1)]
        if not candidate_ngrams or not reference_ngrams:
            return 0.0
        overlap = sum((Counter(candidate_ngrams) & Counter(reference_ngrams)).values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(candidate_ngrams)
        recall = overlap / len(reference_ngrams)
        return 2 * precision * recall / (precision + recall)

    candidate = normalized_characters(completion)
    reference = normalized_characters(answer)
    if not candidate or not reference:
        return 0.0
    unigram = ngram_f1(candidate, reference, 1)
    bigram = ngram_f1(candidate, reference, 2)
    containment_bonus = 0.1 if "".join(reference) in "".join(candidate) else 0.0
    return min(1.0, 0.6 * unigram + 0.3 * bigram + containment_bonus)
