"""Stage 2 — SFT: supervised fine-tuning on chat data (loss on assistant turns only).

Usage:
    uv run python trainer/train_sft.py --data_path dataset/sft_mini.jsonl \
        --init_from out/pretrain_512.pth
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import (
    add_model_args, add_train_args, build_model, ckpt_name, get_device,
    load_weights, train_supervised,
)


def main():
    parser = argparse.ArgumentParser(description="SFT NinjaMind")
    parser.add_argument("--data_path", type=str, default="dataset/sft_mini.jsonl")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    parser.add_argument("--init_from", type=str, default="out/pretrain_512.pth")
    parser.add_argument("--max_length", type=int, default=1024)
    add_model_args(parser)
    add_train_args(parser, default_lr=5e-6)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    model, config = build_model(args, len(tokenizer), device)
    load_weights(model, args.init_from, device)
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_length)
    print(f"dataset: {len(dataset)} samples")

    save_path = ckpt_name(args.out_dir, "full_sft", config)

    def save_fn(m):
        torch.save(m.state_dict(), save_path)
        print(f"checkpoint saved: {save_path}")

    train_supervised(model, dataset, args, device, save_fn)


if __name__ == "__main__":
    main()
