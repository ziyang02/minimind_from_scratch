"""Shared training, checkpoint, distributed, and RL utilities.

The helpers in this module deliberately keep single-process CPU/MPS training
as the default.  When launched with ``torchrun`` (``WORLD_SIZE > 1``), the
same entry points initialize DDP, shard data with ``DistributedSampler``, and
restrict logs/checkpoints to rank zero.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler


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
    step: int | None = None,
    stage: str | None = None,
    extra: Mapping[str, Any] | None = None,
    context: DistributedContext | None = None,
) -> bool:
    """Save model weights plus reproducibility metadata on rank zero.

    Returns ``True`` on the rank that wrote the file and ``False`` elsewhere.
    ``load_weights`` remains compatible with the old raw ``state_dict`` files.
    """

    if not is_main_process(context):
        return False
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    base_model = unwrap_model(model)
    config = config or getattr(base_model, "config", None)
    config_dict = config.to_dict() if hasattr(config, "to_dict") else config
    train_args = vars(args) if args is not None and hasattr(args, "__dict__") else args
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_state_dict": base_model.state_dict(),
        "config": _plain_value(config_dict),
        "tokenizer": _tokenizer_metadata(tokenizer),
        "training_args": _plain_value(train_args),
        "stage": stage,
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload["extra"] = _plain_value(extra)
    torch.save(payload, path)
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


def masked_ce(logits, targets, mask):
    """Cross entropy averaged over positions where ``mask`` is 1."""

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="none"
    )
    mask = mask.reshape(-1).float()
    return (loss * mask).sum() / mask.sum().clamp(min=1)


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
):
    """Train over ``(X, Y, loss_mask)`` triples with AMP/DDP accumulation.

    The final partial batch and the final partial accumulation window both
    produce an optimizer update.  ``max_steps`` counts optimizer updates, not
    micro-batches.
    """

    if args.accumulation_steps < 1:
        raise ValueError("accumulation_steps must be >= 1")
    context = context or DistributedContext(device=device)
    loader = build_dataloader(dataset, args, context, shuffle=True, drop_last=False)
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

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    stopped = False

    for epoch in range(args.epochs):
        if stopped:
            break
        set_dataloader_epoch(loader, epoch)
        window_loss = 0.0
        for batch_index, (X, Y, loss_mask) in enumerate(loader):
            if args.max_steps and optimizer_step >= args.max_steps:
                stopped = True
                break

            group_start = (batch_index // args.accumulation_steps) * args.accumulation_steps
            window_size = min(args.accumulation_steps, len(loader) - group_start)
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
                    raw_loss = masked_ce(out.logits, Y, loss_mask)
                    aux = getattr(out, "aux_loss", None)
                    if torch.is_tensor(aux):
                        raw_loss = raw_loss + aux
                    loss = raw_loss / window_size
                scaler.scale(loss).backward()

            window_loss += raw_loss.detach().float().item() / window_size
            micro_step += 1
            if not should_update:
                continue

            # AMP order is important: unscale -> clip -> optimizer step.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            if (optimizer_step - 1) % args.log_interval == 0:
                rank0_print(
                    f"epoch {epoch + 1}/{args.epochs} step {optimizer_step}/{total_steps} "
                    f"loss {window_loss:.4f} lr {lr:.2e}",
                    context=context,
                )
            window_loss = 0.0

        if is_main_process(context):
            save_fn(unwrap_model(model))

    # An empty dataset still produces a valid initialized checkpoint.
    if not len(loader) and is_main_process(context):
        save_fn(unwrap_model(model))
    return {
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "total_steps": total_steps,
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
