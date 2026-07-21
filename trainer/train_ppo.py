"""Stage 3a — PPO (Proximal Policy Optimization) with a critic and GAE.

The loop per batch:
  1. rollout      — sample completions from the actor for a batch of prompts
  2. score        — env reward (rule-based here) + per-token KL penalty vs ref
  3. estimate     — critic values -> GAE advantages / returns
  4. update       — clipped policy loss for the actor, clipped value loss
                    for the critic, for a few inner epochs

The reward here is a toy containment check (reference answer appears in the
completion) — swap `containment_reward` for a real reward model when you have
one.

Usage:
    uv run python trainer/train_ppo.py --data_path dataset/rl_demo.jsonl \
        --init_from out/full_sft_512.pth
"""
import argparse
import copy
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.lm_dataset import RLAIFDataset, decode_completion, rl_collate
from model.model import NinjaMindModel
from trainer.trainer_utils import (
    add_model_args, add_train_args, build_model, ckpt_name, compute_gae,
    containment_reward, get_device, load_weights, masked_mean, sample_generate,
    token_logprobs, whiten,
)


class Critic(nn.Module):
    """Value network: same backbone architecture + a scalar head per position."""

    def __init__(self, config):
        super().__init__()
        self.model = NinjaMindModel(config)
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, input_ids, attention_mask=None):
        hidden_states, _, _ = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self.value_head(hidden_states).squeeze(-1)  # (B, T)


def main():
    parser = argparse.ArgumentParser(description="PPO NinjaMind")
    parser.add_argument("--data_path", type=str, default="dataset/rl_demo.jsonl")
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

    torch.manual_seed(42)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id

    # Actor (trained), frozen reference (KL anchor), critic (value estimates).
    actor, config = build_model(args, len(tokenizer), device)
    load_weights(actor, args.init_from, device)
    ref = copy.deepcopy(actor).eval().requires_grad_(False)
    critic = Critic(config).to(device)
    state_dict = torch.load(args.init_from, map_location="cpu")
    critic.model.load_state_dict(
        {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}
    )

    actor_opt = torch.optim.AdamW(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)

    dataset = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_prompt_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=partial(rl_collate, tokenizer=tokenizer, max_prompt_len=args.max_prompt_len),
    )
    print(f"dataset: {len(dataset)} prompts")

    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            if args.max_steps and step >= args.max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            prompt_len = input_ids.size(1)

            # ---- 1. rollout -------------------------------------------------
            seq, gen_mask, full_attn = sample_generate(
                actor, input_ids, attn, args.max_new_tokens,
                eos_id=eos_id, pad_id=pad_id,
                temperature=args.temperature, top_k=args.top_k,
            )
            gen_mask_f = gen_mask.float()

            # ---- 2. score: env reward + per-token KL penalty ---------------
            with torch.no_grad():
                old_logp = token_logprobs(actor, seq, full_attn, prompt_len)
                ref_logp = token_logprobs(ref, seq, full_attn, prompt_len)
                values = critic(seq, full_attn)[:, prompt_len - 1:-1]

            completions = [
                decode_completion(tokenizer, seq[b].tolist(), prompt_len)
                for b in range(seq.size(0))
            ]
            env_reward = torch.tensor(
                [containment_reward(c, a) for c, a in zip(completions, batch["answer"])],
                device=device,
            )
            kl = old_logp - ref_logp
            rewards = -args.kl_coef * kl * gen_mask_f
            last_idx = gen_mask.sum(dim=1).long().clamp(min=1) - 1
            rewards[torch.arange(seq.size(0), device=device), last_idx] += env_reward

            # ---- 3. GAE ------------------------------------------------------
            adv, returns = compute_gae(rewards, values, gen_mask_f, args.gamma, args.lam)
            adv = whiten(adv, gen_mask_f)

            # ---- 4. clipped PPO updates -------------------------------------
            actor.train()
            for _ in range(args.ppo_epochs):
                out = actor(input_ids=seq, attention_mask=full_attn)
                logits = out.logits[:, prompt_len - 1:-1].float()
                logp_all = F.log_softmax(logits, dim=-1)
                new_logp = logp_all.gather(-1, seq[:, prompt_len:].unsqueeze(-1)).squeeze(-1)
                entropy = -(logp_all.exp() * logp_all).sum(-1)

                ratio = (new_logp - old_logp).exp()
                pg = torch.max(
                    -adv * ratio,
                    -adv * ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps),
                )
                pg_loss = masked_mean(pg, gen_mask_f)
                ent_loss = masked_mean(entropy, gen_mask_f)

                value_pred = critic(seq, full_attn)[:, prompt_len - 1:-1]
                value_clip = values + (value_pred - values).clamp(-args.clip_eps, args.clip_eps)
                vf_loss = 0.5 * masked_mean(
                    torch.max((value_pred - returns) ** 2, (value_clip - returns) ** 2),
                    gen_mask_f,
                )

                loss = pg_loss + args.vf_coef * vf_loss - args.ent_coef * ent_loss
                actor_opt.zero_grad(set_to_none=True)
                critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip)
                actor_opt.step()
                critic_opt.step()

            if step % args.log_interval == 0:
                print(
                    f"epoch {epoch + 1} step {step} "
                    f"reward {env_reward.mean().item():.3f} "
                    f"kl {masked_mean(kl, gen_mask_f).item():.4f} "
                    f"pg {pg_loss.item():.4f} vf {vf_loss.item():.4f}"
                )
            step += 1

    save_path = ckpt_name(args.out_dir, "ppo", config)
    torch.save(actor.state_dict(), save_path)
    print(f"checkpoint saved: {save_path}")


if __name__ == "__main__":
    main()
