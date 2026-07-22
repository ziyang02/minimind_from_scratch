"""Stage 3b — critic-free Group Relative Policy Optimization (GRPO).

Each prompt is repeated ``G`` times.  Its completion rewards are normalized
within the group and broadcast over generated tokens, while a frozen reference
policy supplies a stable KL penalty.

Usage:
    uv run python trainer/train_grpo.py --data_path dataset/demo/rl_demo.jsonl \
        --init_from out/full_sft_512.pth
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from contextlib import nullcontext
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from dataset.lm_dataset import RLAIFDataset, decode_completion, rl_collate
from trainer.trainer_utils import (
    add_model_args,
    add_train_args,
    autocast_ctx,
    build_dataloader,
    build_model,
    ckpt_name,
    cleanup_distributed,
    containment_reward,
    cosine_lr,
    load_weights,
    make_grad_scaler,
    masked_mean,
    rank0_print,
    sample_generate,
    save_checkpoint,
    set_dataloader_epoch,
    setup_distributed,
    token_logprobs,
    unwrap_model,
    wrap_ddp,
)


def group_advantages(rewards, group_size, eps=1e-4):
    """Normalize one scalar reward per completion within prompt groups."""

    if group_size < 2:
        raise ValueError("GRPO needs at least two completions per group")
    if rewards.ndim != 1 or rewards.numel() % group_size:
        raise ValueError("rewards must be flat and divisible by group_size")
    grouped = rewards.view(-1, group_size)
    normalized = (grouped - grouped.mean(dim=1, keepdim=True)) / (
        grouped.std(dim=1, keepdim=True, unbiased=False) + eps
    )
    return normalized.reshape(-1, 1)


def _train(args, context):
    torch.manual_seed(args.seed + context.rank)
    device = context.device
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id

    policy, config = build_model(args, len(tokenizer), device, context)
    load_weights(policy, args.init_from, device, context)
    reference = copy.deepcopy(policy).eval().requires_grad_(False)
    policy = wrap_ddp(policy, context)
    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = make_grad_scaler(device)
    optimizer.zero_grad(set_to_none=True)

    dataset = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_prompt_len)
    loader = build_dataloader(
        dataset,
        args,
        context,
        shuffle=True,
        drop_last=False,
        collate_fn=partial(
            rl_collate,
            tokenizer=tokenizer,
            max_prompt_len=args.max_prompt_len,
        ),
    )
    rank0_print(
        f"dataset: {len(dataset)} prompts | G={args.group_size}", context=context
    )

    updates_per_rollout = math.ceil(args.update_epochs / args.accumulation_steps)
    total_steps = len(loader) * args.epochs * updates_per_rollout
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    optimizer_step = 0
    stopped = False
    group_size = args.group_size

    for epoch in range(args.epochs):
        if stopped:
            break
        set_dataloader_epoch(loader, epoch)
        for batch in loader:
            if args.max_steps and optimizer_step >= args.max_steps:
                stopped = True
                break
            # Duplicate each prompt G times, producing one contiguous group.
            input_ids = batch["input_ids"].repeat_interleave(group_size, dim=0).to(device)
            attention_mask = batch["attention_mask"].repeat_interleave(
                group_size, dim=0
            ).to(device)
            answers = [answer for answer in batch["answer"] for _ in range(group_size)]
            prompt_len = input_ids.size(1)

            seq, generated_mask, full_attention_mask = sample_generate(
                policy,
                input_ids,
                attention_mask,
                args.max_new_tokens,
                eos_id=eos_id,
                pad_id=pad_id,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            generated_mask_f = generated_mask.float()

            completions = [
                decode_completion(tokenizer, seq[index].tolist(), prompt_len)
                for index in range(seq.size(0))
            ]
            rewards = torch.tensor(
                [
                    containment_reward(completion, answer)
                    for completion, answer in zip(completions, answers, strict=True)
                ],
                device=device,
            )
            advantages = group_advantages(rewards, group_size)

            with torch.no_grad():
                old_logps = token_logprobs(policy, seq, full_attention_mask, prompt_len)
                reference_logps = token_logprobs(
                    reference, seq, full_attention_mask, prompt_len
                )

            policy.train()
            for update_index in range(args.update_epochs):
                if args.max_steps and optimizer_step >= args.max_steps:
                    stopped = True
                    break
                group_start = (
                    update_index // args.accumulation_steps
                ) * args.accumulation_steps
                window_size = min(
                    args.accumulation_steps, args.update_epochs - group_start
                )
                should_update = (
                    (update_index + 1) % args.accumulation_steps == 0
                    or update_index + 1 == args.update_epochs
                )
                if update_index % args.accumulation_steps == 0:
                    lr = cosine_lr(optimizer_step, max(total_steps, 1), args.lr)
                    for group in optimizer.param_groups:
                        group["lr"] = lr

                sync_ctx = (
                    policy.no_sync()
                    if context.distributed and not should_update
                    else nullcontext()
                )
                with sync_ctx:
                    with autocast_ctx(device):
                        output = policy(input_ids=seq, attention_mask=full_attention_mask)
                        logits = output.logits[:, prompt_len - 1:-1].float()
                        all_logps = F.log_softmax(logits, dim=-1)
                        new_logps = all_logps.gather(
                            -1, seq[:, prompt_len:].unsqueeze(-1)
                        ).squeeze(-1)

                        ratio = (new_logps - old_logps).exp()
                        policy_objective = torch.max(
                            -advantages * ratio,
                            -advantages
                            * ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps),
                        )
                        # k3 estimator: exp(ref-new) - (ref-new) - 1 >= 0.
                        reference_log_ratio = reference_logps - new_logps
                        kl = (
                            reference_log_ratio.exp() - reference_log_ratio - 1
                        )
                        loss = masked_mean(
                            policy_objective + args.kl_coef * kl,
                            generated_mask_f,
                        ) + output.aux_loss
                    scaler.scale(loss / window_size).backward()

                if not should_update:
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if (optimizer_step - 1) % args.log_interval == 0:
                    rank0_print(
                        f"epoch {epoch + 1}/{args.epochs} step {optimizer_step}/{total_steps} "
                        f"loss {loss.detach().float().item():.4f} "
                        f"reward {rewards.mean().item():.3f} "
                        f"kl {masked_mean(kl, generated_mask_f).item():.4f}",
                        context=context,
                    )

    save_path = ckpt_name(args.out_dir, "grpo", config)
    save_checkpoint(
        save_path,
        unwrap_model(policy),
        config=config,
        tokenizer=tokenizer,
        args=args,
        optimizer=optimizer,
        step=optimizer_step,
        stage="grpo",
        extra={"reference_checkpoint": args.init_from},
        context=context,
    )


def main():
    parser = argparse.ArgumentParser(description="GRPO NinjaMind")
    parser.add_argument("--data_path", type=str, default="dataset/demo/rl_demo.jsonl")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    parser.add_argument("--init_from", type=str, default="out/full_sft_512.pth")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_prompt_len", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--kl_coef", type=float, default=0.04)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--update_epochs", type=int, default=1)
    add_model_args(parser)
    add_train_args(parser, default_lr=1e-6)
    args = parser.parse_args()
    if args.group_size < 2:
        parser.error("--group_size must be >= 2")
    if args.update_epochs < 1:
        parser.error("--update_epochs must be >= 1")
    if args.accumulation_steps < 1:
        parser.error("--accumulation_steps must be >= 1")

    context = setup_distributed(args)
    try:
        _train(args, context)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
