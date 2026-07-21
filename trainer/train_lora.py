"""Stage 2b — LoRA fine-tuning: freeze the base model, train low-rank adapters.

Usage:
    uv run python trainer/train_lora.py --data_path dataset/lora_identity.jsonl \
        --init_from out/full_sft_512.pth
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from dataset.lm_dataset import SFTDataset
from model.model_lora import apply_lora, freeze_non_lora, save_lora
from trainer.trainer_utils import (
    add_model_args, add_train_args, build_model, get_device, load_weights, train_supervised,
)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune NinjaMind")
    parser.add_argument("--data_path", type=str, default="dataset/lora_identity.jsonl")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    parser.add_argument("--init_from", type=str, default="out/full_sft_512.pth")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_name", type=str, default="lora")
    add_model_args(parser)
    add_train_args(parser, default_lr=1e-4)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model, config = build_model(args, len(tokenizer), device)
    load_weights(model, args.init_from, device)

    apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    model.to(device)
    trainable = freeze_non_lora(model)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {n_trainable / 1e3:.1f}K trainable / {n_total / 1e6:.2f}M total "
          f"({100 * n_trainable / n_total:.2f}%)")

    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_length)
    print(f"dataset: {len(dataset)} samples")

    save_path = os.path.join(args.out_dir, f"{args.lora_name}_{config.hidden_size}.pth")

    def save_fn(m):
        save_lora(m, save_path)
        print(f"LoRA weights saved: {save_path}")

    train_supervised(model, dataset, args, device, save_fn, params=trainable)


if __name__ == "__main__":
    main()
