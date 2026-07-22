"""Stage 3a — PPO with a frozen KL reference, critic, and masked GAE.

The built-in reward is intentionally a small deterministic containment rule;
replace it with a learned reward model for real preference optimization.

Usage:
    uv run python trainer/train_ppo.py --data_path dataset/demo/rl_demo.jsonl \
        --init_from out/full_sft_512.pth
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from contextlib import ExitStack
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from dataset.lm_dataset import RLAIFDataset, decode_completion, rl_collate
from model.model import NinjaMindModel
from trainer.trainer_utils import (
    add_model_args,
    add_train_args,
    autocast_ctx,
    build_dataloader,
    build_model,
    ckpt_name,
    cleanup_distributed,
    compute_gae,
    containment_reward,
    cosine_lr,
    load_checkpoint_state,
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
    whiten,
    wrap_ddp,
)


class Critic(nn.Module):
    """Value network: same backbone architecture + a scalar head per position."""

    def __init__(self, config):
        super().__init__()
        self.model = NinjaMindModel(config)
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden_states, _, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.value_head(hidden_states).squeeze(-1), aux_loss


def _train(args, context):
    torch.manual_seed(args.seed + context.rank)
    device = context.device
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id

    # Actor (trained), frozen reference (KL anchor), critic (value estimates).
    actor, config = build_model(args, len(tokenizer), device, context)
    load_weights(actor, args.init_from, device, context)
    reference = copy.deepcopy(actor).eval().requires_grad_(False)
    critic = Critic(config).to(device)
    actor_state = load_checkpoint_state(args.init_from)
    backbone_state = {
        key[len("model."):]: value
        for key, value in actor_state.items()
        if key.startswith("model.")
    }
    critic.model.load_state_dict(backbone_state, strict=False)

    actor = wrap_ddp(actor, context)
    critic = wrap_ddp(critic, context)
    actor_params = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    critic_params = [parameter for parameter in critic.parameters() if parameter.requires_grad]
    actor_optimizer = torch.optim.AdamW(actor_params, lr=args.lr)
    critic_optimizer = torch.optim.AdamW(critic_params, lr=args.critic_lr)
    scaler = make_grad_scaler(device)

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
    rank0_print(f"dataset: {len(dataset)} prompts", context=context)

    updates_per_rollout = math.ceil(args.ppo_epochs / args.accumulation_steps)
    total_steps = len(loader) * args.epochs * updates_per_rollout
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    optimizer_step = 0
    stopped = False
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        if stopped:
            break
        set_dataloader_epoch(loader, epoch)
        for batch in loader:
            if args.max_steps and optimizer_step >= args.max_steps:
                stopped = True
                break
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            prompt_len = input_ids.size(1)

            # 1) Roll out completions from the current actor.
            seq, generated_mask, full_attention_mask = sample_generate(
                actor,
                input_ids,
                attention_mask,
                args.max_new_tokens,
                eos_id=eos_id,
                pad_id=pad_id,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            generated_mask_f = generated_mask.float()

            # 2) Environment reward plus a per-token KL penalty.
            with torch.no_grad():
                old_logps = token_logprobs(actor, seq, full_attention_mask, prompt_len)
                reference_logps = token_logprobs(
                    reference, seq, full_attention_mask, prompt_len
                )
                old_values, _ = critic(seq, full_attention_mask)
                old_values = old_values[:, prompt_len - 1:-1]

            completions = [
                decode_completion(tokenizer, seq[index].tolist(), prompt_len)
                for index in range(seq.size(0))
            ]
            environment_reward = torch.tensor(
                [
                    containment_reward(completion, answer)
                    for completion, answer in zip(
                        completions, batch["answer"], strict=True
                    )
                ],
                device=device,
            )
            kl = old_logps - reference_logps
            token_rewards = -args.kl_coef * kl * generated_mask_f
            last_index = generated_mask.sum(dim=1).long().clamp(min=1) - 1
            token_rewards[
                torch.arange(seq.size(0), device=device), last_index
            ] += environment_reward

            # 3) Masked generalized advantage estimation.
            advantages, returns = compute_gae(
                token_rewards,
                old_values,
                generated_mask_f,
                args.gamma,
                args.lam,
            )
            advantages = whiten(advantages, generated_mask_f)

            # 4) PPO clipped actor/value updates. Accumulation windows are
            # local to this rollout, so their tail is always flushed.
            actor.train()
            critic.train()
            for update_index in range(args.ppo_epochs):
                if args.max_steps and optimizer_step >= args.max_steps:
                    stopped = True
                    break
                group_start = (
                    update_index // args.accumulation_steps
                ) * args.accumulation_steps
                window_size = min(
                    args.accumulation_steps, args.ppo_epochs - group_start
                )
                should_update = (
                    (update_index + 1) % args.accumulation_steps == 0
                    or update_index + 1 == args.ppo_epochs
                )
                if update_index % args.accumulation_steps == 0:
                    lr = cosine_lr(optimizer_step, max(total_steps, 1), args.lr)
                    for group in actor_optimizer.param_groups:
                        group["lr"] = lr

                with ExitStack() as sync_stack:
                    if context.distributed and not should_update:
                        sync_stack.enter_context(actor.no_sync())
                        sync_stack.enter_context(critic.no_sync())
                    with autocast_ctx(device):
                        output = actor(input_ids=seq, attention_mask=full_attention_mask)
                        logits = output.logits[:, prompt_len - 1:-1].float()
                        all_logps = F.log_softmax(logits, dim=-1)
                        new_logps = all_logps.gather(
                            -1, seq[:, prompt_len:].unsqueeze(-1)
                        ).squeeze(-1)
                        entropy = -(all_logps.exp() * all_logps).sum(-1)

                        ratio = (new_logps - old_logps).exp()
                        policy_objective = torch.max(
                            -advantages * ratio,
                            -advantages
                            * ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps),
                        )
                        policy_loss = masked_mean(policy_objective, generated_mask_f)
                        entropy_bonus = masked_mean(entropy, generated_mask_f)

                        value_prediction, critic_aux_loss = critic(
                            seq,
                            full_attention_mask,
                        )
                        value_prediction = value_prediction[:, prompt_len - 1:-1]
                        clipped_value = old_values + (value_prediction - old_values).clamp(
                            -args.clip_eps, args.clip_eps
                        )
                        value_loss = 0.5 * masked_mean(
                            torch.max(
                                (value_prediction - returns) ** 2,
                                (clipped_value - returns) ** 2,
                            ),
                            generated_mask_f,
                        )
                        loss = (
                            policy_loss
                            + args.vf_coef * value_loss
                            - args.ent_coef * entropy_bonus
                            + output.aux_loss
                            + critic_aux_loss
                        )
                    scaler.scale(loss / window_size).backward()

                if not should_update:
                    continue
                scaler.unscale_(actor_optimizer)
                scaler.unscale_(critic_optimizer)
                torch.nn.utils.clip_grad_norm_(actor_params, args.grad_clip)
                torch.nn.utils.clip_grad_norm_(critic_params, args.grad_clip)
                scaler.step(actor_optimizer)
                scaler.step(critic_optimizer)
                scaler.update()
                actor_optimizer.zero_grad(set_to_none=True)
                critic_optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if (optimizer_step - 1) % args.log_interval == 0:
                    rank0_print(
                        f"epoch {epoch + 1}/{args.epochs} step {optimizer_step}/{total_steps} "
                        f"loss {loss.detach().float().item():.4f} "
                        f"reward {environment_reward.mean().item():.3f} "
                        f"kl {masked_mean(kl, generated_mask_f).item():.4f} "
                        f"pg {policy_loss.item():.4f} vf {value_loss.item():.4f}",
                        context=context,
                    )

    save_path = ckpt_name(args.out_dir, "ppo", config)
    save_checkpoint(
        save_path,
        unwrap_model(actor),
        config=config,
        tokenizer=tokenizer,
        args=args,
        optimizer=actor_optimizer,
        step=optimizer_step,
        stage="ppo",
        extra={"critic_trained": True, "reference_checkpoint": args.init_from},
        context=context,
    )


def main():
    parser = argparse.ArgumentParser(description="PPO NinjaMind")
    parser.add_argument("--data_path", type=str, default="dataset/demo/rl_demo.jsonl")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    parser.add_argument("--init_from", type=str, default="out/full_sft_512.pth")
    parser.add_argument("--max_prompt_len", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--critic_lr", type=float, default=1e-5)
    add_model_args(parser)
    add_train_args(parser, default_lr=1e-6)
    args = parser.parse_args()
    if args.accumulation_steps < 1:
        parser.error("--accumulation_steps must be >= 1")
    if args.ppo_epochs < 1:
        parser.error("--ppo_epochs must be >= 1")

    context = setup_distributed(args)
    try:
        _train(args, context)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
