"""Shared utilities for every training stage.

Supervised stages (pretrain / SFT / LoRA) use :func:`train_supervised`.
RL stages (PPO / GRPO) use :func:`sample_generate` for rollouts plus the
GAE / masking helpers below.
"""
import math
import os
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# --------------------------------------------------------------------------- #
# Environment / model plumbing                                                 #
# --------------------------------------------------------------------------- #
def get_device(preference="auto"):
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device):
    """Mixed precision on CUDA; full precision elsewhere (MPS/CPU)."""
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def add_model_args(parser):
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
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
    parser.add_argument("--max_steps", type=int, default=0, help="stop after N steps (0 = full run)")


def build_model(args, vocab_size, device):
    from model.model import NinjaMindConfig, NinjaMindForCausalLM

    config = NinjaMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe,
        vocab_size=vocab_size,
    )
    model = NinjaMindForCausalLM(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M params | vocab {vocab_size} | device {device}")
    return model, config


def load_weights(model, path, device):
    state_dict = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"load_weights: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    print(f"loaded weights from {path}")


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
    return max_lr * (min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress)))


def masked_ce(logits, targets, mask):
    """Cross entropy averaged over positions where ``mask`` is 1."""
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="none"
    )
    mask = mask.reshape(-1).float()
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def masked_mean(x, mask, dim=None):
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp(min=1)
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1)


def whiten(x, mask):
    """Normalize to zero mean / unit variance over the masked entries."""
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return (x - mean) * torch.rsqrt(var + 1e-8)


# --------------------------------------------------------------------------- #
# Supervised training loop (pretrain / SFT / LoRA)                             #
# --------------------------------------------------------------------------- #
def train_supervised(model, dataset, args, device, save_fn, params=None):
    """Generic loop over a dataset yielding ``(X, Y, loss_mask)`` triples."""
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=args.num_workers,
    )
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    total_steps = len(loader) * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    ctx = autocast_ctx(device)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for X, Y, loss_mask in loader:
            if args.max_steps and step >= args.max_steps:
                save_fn(model)
                return
            X, Y, loss_mask = X.to(device), Y.to(device), loss_mask.to(device)
            lr = cosine_lr(step, total_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with ctx:
                out = model(input_ids=X)
                loss = masked_ce(out.logits, Y, loss_mask)
                aux = getattr(out, "aux_loss", None)
                if torch.is_tensor(aux):
                    loss = loss + aux
                loss = loss / args.accumulation_steps
            loss.backward()

            if (step + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % args.log_interval == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} step {step}/{total_steps} "
                    f"loss {loss.item() * args.accumulation_steps:.4f} lr {lr:.2e}"
                )
            step += 1
        save_fn(model)


# --------------------------------------------------------------------------- #
# RL helpers (PPO / GRPO)                                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_generate(model, input_ids, attention_mask, max_new_tokens,
                    eos_id, pad_id, temperature=1.0, top_k=30):
    """Batched sampling with a KV cache from left-padded prompts.

    Returns ``(seq, gen_mask, attn_mask)`` where ``seq`` is prompt+completion,
    ``gen_mask`` marks real generated tokens (up to and including eos) and
    ``attn_mask`` covers the full sequence (prompt pads and post-eos pads = 0).
    """
    was_training = model.training
    model.eval()
    device = input_ids.device
    bsz = input_ids.size(0)

    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1]
    done = torch.zeros(bsz, dtype=torch.bool, device=device)
    tokens, gen_mask_cols = [], []

    for _ in range(max_new_tokens):
        logits = next_logits.float() / max(temperature, 1e-6)
        if top_k:
            topv, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits = torch.where(logits < topv[:, [-1]], torch.full_like(logits, float("-inf")), logits)
        next_tok = torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze(-1)
        next_tok = torch.where(done, torch.full_like(next_tok, pad_id), next_tok)

        gen_mask_cols.append((~done).long())
        tokens.append(next_tok)
        done = done | (next_tok == eos_id)
        attention_mask = torch.cat([attention_mask, gen_mask_cols[-1].unsqueeze(1)], dim=1)
        if done.all():
            break
        out = model(input_ids=next_tok.unsqueeze(1), attention_mask=attention_mask,
                    use_cache=True, past_key_values=past)
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
    lp = logp.gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
    return lp[:, prompt_len - 1:]


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Generalized Advantage Estimation over generated-token positions."""
    bsz, glen = rewards.shape
    adv = torch.zeros_like(rewards)
    lastgaelam = torch.zeros(bsz, device=rewards.device)
    for t in reversed(range(glen)):
        next_value = values[:, t + 1] if t < glen - 1 else torch.zeros(bsz, device=rewards.device)
        next_nonterminal = mask[:, t + 1] if t < glen - 1 else torch.zeros(bsz, device=rewards.device)
        delta = rewards[:, t] + gamma * next_value * next_nonterminal - values[:, t]
        lastgaelam = delta + gamma * lam * next_nonterminal * lastgaelam
        adv[:, t] = lastgaelam
    returns = adv + values
    return adv, returns


def containment_reward(completion, answer):
    """Toy rule-based reward: 1 if the reference answer appears verbatim."""
    answer = (answer or "").strip()
    return 1.0 if answer and answer in completion else 0.0
