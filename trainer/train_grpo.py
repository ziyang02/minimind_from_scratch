"""Stage 3b — GRPO (Group Relative Policy Optimization): critic-free RL.

Instead of a learned value function (PPO's critic), GRPO samples a GROUP of
G completions per prompt and uses the group-normalized reward as the
advantage for every token of each completion:

    A_i = (r_i - mean(r_group)) / (std(r_group) + eps)

plus a token-level KL penalty (k3 estimator) against a frozen reference
model. Half the memory of PPO (no critic) and simpler — the approach used by
DeepSeek-R1.

Usage:
    uv run python trainer/train_grpo.py --data_path dataset/rl_demo.jsonl \
        --init_from out/full_sft_512.pth
"""
import argparse
import copy
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.lm_dataset import RLAIFDataset, decode_completion, rl_collate
from trainer.trainer_utils import (
    add_model_args, add_train_args, build_model, ckpt_name, containment_reward,
    get_device, load_weights, masked_mean, sample_generate, token_logprobs,
)


def main():
    parser = argparse.ArgumentParser(description="GRPO NinjaMind")
    parser.add_argument("--data_path", type=str, default="dataset/rl_demo.jsonl")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    parser.add_argument("--init_from", type=str, default="out/full_sft_512.pth")
    parser.add_argument("--group_size", type=int, default=4, help="completions per prompt (G)")
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
    assert args.group_size >= 2, "GRPO needs at least 2 samples per group"

    torch.manual_seed(42)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    eos_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id

    policy, config = build_model(args, len(tokenizer), device)
    load_weights(policy, args.init_from, device)
    ref = copy.deepcopy(policy).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    dataset = RLAIFDataset(args.data_path, tokenizer, max_length=args.max_prompt_len)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        collate_fn=partial(rl_collate, tokenizer=tokenizer, max_prompt_len=args.max_prompt_len),
    )
    print(f"dataset: {len(dataset)} prompts | G={args.group_size}")

    G = args.group_size
    step = 0
    for epoch in range(args.epochs):
        for batch in loader:
            if args.max_steps and step >= args.max_steps:
                break
            # Duplicate each prompt G times -> one group per prompt.
            input_ids = batch["input_ids"].repeat_interleave(G, dim=0).to(device)
            attn = batch["attention_mask"].repeat_interleave(G, dim=0).to(device)
            answers = [a for a in batch["answer"] for _ in range(G)]
            prompt_len = input_ids.size(1)
            n_prompts = input_ids.size(0) // G

            # ---- rollout the whole group ------------------------------------
            seq, gen_mask, full_attn = sample_generate(
                policy, input_ids, attn, args.max_new_tokens,
                eos_id=eos_id, pad_id=pad_id,
                temperature=args.temperature, top_k=args.top_k,
            )
            gen_mask_f = gen_mask.float()

            # ---- rewards -> group-relative advantages -----------------------
            completions = [
                decode_completion(tokenizer, seq[b].tolist(), prompt_len)
                for b in range(seq.size(0))
            ]
            rewards = torch.tensor(
                [containment_reward(c, a) for c, a in zip(completions, answers)],
                device=device,
            )
            grouped = rewards.view(n_prompts, G)
            adv = (grouped - grouped.mean(dim=1, keepdim=True)) / (
                grouped.std(dim=1, keepdim=True) + 1e-4
            )
            adv = adv.view(-1, 1)  # broadcast one advantage over all tokens

            # ---- old / reference log-probs ----------------------------------
            with torch.no_grad():
                old_logp = token_logprobs(policy, seq, full_attn, prompt_len)
                ref_logp = token_logprobs(ref, seq, full_attn, prompt_len)

            # ---- policy update ----------------------------------------------
            policy.train()
            for _ in range(args.update_epochs):
                out = policy(input_ids=seq, attention_mask=full_attn)
                logits = out.logits[:, prompt_len - 1:-1].float()
                logp_all = F.log_softmax(logits, dim=-1)
                new_logp = logp_all.gather(-1, seq[:, prompt_len:].unsqueeze(-1)).squeeze(-1)

                ratio = (new_logp - old_logp).exp()
                pg = torch.max(
                    -adv * ratio,
                    -adv * ratio.clamp(1 - args.clip_eps, 1 + args.clip_eps),
                )
                # k3 KL estimator: exp(ref - new) - (ref - new) - 1  (>= 0)
                log_ratio_ref = ref_logp - new_logp
                kl = log_ratio_ref.exp() - log_ratio_ref - 1

                loss = masked_mean(pg + args.kl_coef * kl, gen_mask_f)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
                optimizer.step()

            if step % args.log_interval == 0:
                print(
                    f"epoch {epoch + 1} step {step} "
                    f"reward {rewards.mean().item():.3f} "
                    f"kl {masked_mean(kl, gen_mask_f).item():.4f} "
                    f"loss {loss.item():.4f}"
                )
            step += 1

    save_path = ckpt_name(args.out_dir, "grpo", config)
    torch.save(policy.state_dict(), save_path)
    print(f"checkpoint saved: {save_path}")


if __name__ == "__main__":
    main()
