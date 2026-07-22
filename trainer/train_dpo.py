"""Stage 3 — Direct Preference Optimization (DPO).

DPO trains a policy from preferred/rejected response pairs without fitting a
separate reward model.  For policy ``pi`` and frozen reference ``pi_ref`` the
per-example objective is::

    -log sigmoid(beta * ((log pi(y_w|x) - log pi(y_l|x))
                         - (log pi_ref(y_w|x) - log pi_ref(y_l|x))))

Only assistant-response tokens contribute to each sequence log-probability.

Usage:
    uv run python trainer/train_dpo.py --data_path dataset/demo/dpo_demo.jsonl \
        --init_from out/full_sft_512.pth
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import (
    add_model_args,
    add_train_args,
    autocast_ctx,
    build_dataloader,
    build_model,
    ckpt_name,
    cleanup_distributed,
    cosine_lr,
    distributed_mean_metrics,
    load_weights,
    make_grad_scaler,
    rank0_print,
    save_checkpoint,
    set_dataloader_epoch,
    setup_distributed,
    unwrap_model,
    wrap_ddp,
)


def sequence_log_probs(logits, labels, loss_mask, average_log_prob=False):
    """Return masked response log-probability for each sequence.

    ``logits`` has shape ``(batch, sequence, vocab)`` while ``labels`` and
    ``loss_mask`` have shape ``(batch, sequence)``.  DPO conventionally uses
    a sum over response tokens; ``average_log_prob`` is exposed for diagnostics
    and length-normalized experiments.
    """

    if logits.shape[:-1] != labels.shape or labels.shape != loss_mask.shape:
        raise ValueError(
            "expected logits (B,T,V), labels (B,T), and loss_mask (B,T); "
            f"got {tuple(logits.shape)}, {tuple(labels.shape)}, {tuple(loss_mask.shape)}"
        )
    token_log_probs = F.log_softmax(logits.float(), dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    mask = loss_mask.to(dtype=token_log_probs.dtype)
    sequence_sums = (token_log_probs * mask).sum(dim=-1)
    if average_log_prob:
        return sequence_sums / mask.sum(dim=-1).clamp(min=1)
    return sequence_sums


def get_batch_logps(logits, labels, loss_mask, average_log_prob=False):
    """Compatibility alias using the name common in DPO implementations."""

    return sequence_log_probs(logits, labels, loss_mask, average_log_prob)


def dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta=0.1,
):
    """Compute standard reference-anchored DPO losses and implicit rewards.

    Returns per-example ``(losses, chosen_rewards, rejected_rewards)``.  The
    rewards are detached because they are logging metrics, not optimization
    targets.
    """

    if beta <= 0:
        raise ValueError("beta must be > 0")
    policy_log_ratio = policy_chosen_logps - policy_rejected_logps
    reference_log_ratio = reference_chosen_logps - reference_rejected_logps
    preference_logits = policy_log_ratio - reference_log_ratio
    losses = -F.logsigmoid(beta * preference_logits)
    chosen_rewards = (beta * (policy_chosen_logps - reference_chosen_logps)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - reference_rejected_logps)).detach()
    return losses, chosen_rewards, rejected_rewards


def dpo_metrics(losses, chosen_rewards, rejected_rewards):
    """Aggregate the metrics reported by the trainer as scalar tensors."""

    margin = chosen_rewards - rejected_rewards
    return {
        "loss": losses.mean(),
        "preference_accuracy": (margin > 0).float().mean(),
        "chosen_reward": chosen_rewards.mean(),
        "rejected_reward": rejected_rewards.mean(),
        "reward_margin": margin.mean(),
    }


def freeze_reference_model(model):
    """Put a reference policy in eval mode and permanently disable gradients."""

    model.eval()
    model.requires_grad_(False)
    return model


def preference_forward(model, batch, pad_id):
    """Run both preference sides together and return log-probs plus policy aux loss."""

    chosen_batch_size = batch["x_chosen"].size(0)
    input_ids = torch.cat([batch["x_chosen"], batch["x_rejected"]], dim=0)
    labels = torch.cat([batch["y_chosen"], batch["y_rejected"]], dim=0)
    masks = torch.cat([batch["mask_chosen"], batch["mask_rejected"]], dim=0)
    attention_mask = input_ids.ne(pad_id)
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = sequence_log_probs(output.logits, labels, masks)
    return (
        log_probs[:chosen_batch_size],
        log_probs[chosen_batch_size:],
        output.aux_loss,
    )


def _move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def main():
    parser = argparse.ArgumentParser(description="DPO fine-tune NinjaMind")
    parser.add_argument("--data_path", type=str, default="dataset/demo/dpo_demo.jsonl")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    parser.add_argument("--init_from", type=str, default="out/full_sft_512.pth")
    parser.add_argument(
        "--reference_from",
        type=str,
        default="",
        help="optional reference checkpoint (defaults to --init_from)",
    )
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.1)
    add_model_args(parser)
    add_train_args(parser, default_lr=5e-7)
    args = parser.parse_args()
    if args.accumulation_steps < 1:
        parser.error("--accumulation_steps must be >= 1")
    if args.beta <= 0:
        parser.error("--beta must be > 0")

    context = setup_distributed(args)
    try:
        torch.manual_seed(args.seed + context.rank)
        device = context.device
        os.makedirs(args.out_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        policy, config = build_model(args, len(tokenizer), device, context)
        load_weights(policy, args.init_from, device, context)

        if args.reference_from and args.reference_from != args.init_from:
            reference, _ = build_model(args, len(tokenizer), device, context)
            load_weights(reference, args.reference_from, device, context)
        else:
            reference = copy.deepcopy(policy)
        freeze_reference_model(reference)

        dataset = DPODataset(args.data_path, tokenizer, max_length=args.max_length)
        loader = build_dataloader(dataset, args, context, shuffle=True, drop_last=False)
        rank0_print(f"dataset: {len(dataset)} preference pairs", context=context)

        policy = wrap_ddp(policy, context)
        trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)
        scaler = make_grad_scaler(device)
        optimizer.zero_grad(set_to_none=True)

        updates_per_epoch = math.ceil(len(loader) / args.accumulation_steps) if len(loader) else 0
        total_steps = updates_per_epoch * args.epochs
        if args.max_steps:
            total_steps = min(total_steps, args.max_steps)
        optimizer_step = 0
        stopped = False
        policy.train()

        for epoch in range(args.epochs):
            if stopped:
                break
            set_dataloader_epoch(loader, epoch)
            metric_window = {
                "loss": 0.0,
                "preference_accuracy": 0.0,
                "chosen_reward": 0.0,
                "rejected_reward": 0.0,
                "reward_margin": 0.0,
            }
            for batch_index, batch in enumerate(loader):
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

                batch = _move_batch(batch, device)
                with torch.no_grad(), autocast_ctx(device):
                    ref_chosen_logps, ref_rejected_logps, _ = preference_forward(
                        reference, batch, tokenizer.pad_token_id
                    )

                sync_ctx = (
                    policy.no_sync()
                    if context.distributed and not should_update
                    else nullcontext()
                )
                with sync_ctx:
                    with autocast_ctx(device):
                        (
                            policy_chosen_logps,
                            policy_rejected_logps,
                            policy_aux_loss,
                        ) = preference_forward(policy, batch, tokenizer.pad_token_id)
                        losses, chosen_rewards, rejected_rewards = dpo_loss(
                            policy_chosen_logps,
                            policy_rejected_logps,
                            ref_chosen_logps,
                            ref_rejected_logps,
                            beta=args.beta,
                        )
                        loss = losses.mean() + policy_aux_loss
                    scaler.scale(loss / window_size).backward()

                metrics = dpo_metrics(losses, chosen_rewards, rejected_rewards)
                metrics["loss"] = loss.detach()
                for name, value in metrics.items():
                    metric_window[name] += value.detach().float().item() / window_size
                if not should_update:
                    continue

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if (optimizer_step - 1) % args.log_interval == 0:
                    global_metrics = distributed_mean_metrics(metric_window, context)
                    rank0_print(
                        f"epoch {epoch + 1}/{args.epochs} step {optimizer_step}/{total_steps} "
                        f"loss {global_metrics['loss']:.4f} "
                        f"preference_accuracy {global_metrics['preference_accuracy']:.3f} "
                        f"chosen_reward {global_metrics['chosen_reward']:.4f} "
                        f"rejected_reward {global_metrics['rejected_reward']:.4f} "
                        f"margin {global_metrics['reward_margin']:.4f} lr {lr:.2e}",
                        context=context,
                    )
                for name in metric_window:
                    metric_window[name] = 0.0

        save_path = ckpt_name(args.out_dir, "dpo", config)
        save_checkpoint(
            save_path,
            unwrap_model(policy),
            config=config,
            tokenizer=tokenizer,
            args=args,
            optimizer=optimizer,
            step=optimizer_step,
            stage="dpo",
            extra={"reference_checkpoint": args.reference_from or args.init_from},
            context=context,
        )
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
