"""Stage 2b — LoRA fine-tuning: freeze the base model, train low-rank adapters.

Usage:
    uv run python trainer/train_lora.py --data_path dataset/demo/sft_demo.jsonl \
        --init_from out/full_sft_512.pth
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from dataset.lm_dataset import SFTDataset, split_supervised_dataset
from model.model_lora import apply_lora, freeze_non_lora, save_lora
from trainer.artifacts import write_training_artifacts
from trainer.trainer_utils import (
    add_model_args,
    add_supervised_eval_args,
    add_train_args,
    build_model,
    cleanup_distributed,
    load_weights,
    rank0_print,
    save_checkpoint,
    setup_distributed,
    tokenizer_fingerprint,
    train_supervised,
)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune NinjaMind")
    parser.add_argument("--data_path", type=str, default="dataset/demo/sft_demo.jsonl")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    parser.add_argument("--init_from", type=str, default="out/full_sft_512.pth")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_name", type=str, default="lora")
    add_model_args(parser)
    add_train_args(parser, default_lr=1e-4)
    add_supervised_eval_args(parser)
    args = parser.parse_args()

    context = setup_distributed(args)
    try:
        torch.manual_seed(args.seed + context.rank)
        device = context.device
        os.makedirs(args.out_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        model, config = build_model(args, len(tokenizer), device, context)
        if not args.resume:
            load_weights(model, args.init_from, device, context)

        apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
        model.to(device)
        trainable = freeze_non_lora(model)
        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in model.parameters())
        rank0_print(
            f"LoRA: {n_trainable / 1e3:.1f}K trainable / {n_total / 1e6:.2f}M total "
            f"({100 * n_trainable / n_total:.2f}%)",
            context=context,
        )

        full_dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_length)
        split_seed = args.seed if args.split_seed is None else args.split_seed
        dataset, validation_dataset, split_metadata = split_supervised_dataset(
            full_dataset,
            validation_fraction=args.validation_fraction,
            seed=split_seed,
        )
        rank0_print(
            f"dataset: {split_metadata['raw_samples']} raw -> "
            f"{split_metadata['train_samples']} train / "
            f"{split_metadata['validation_samples']} validation "
            f"({split_metadata['duplicates_removed']} duplicates removed)",
            context=context,
        )

        save_path = os.path.join(args.out_dir, f"{args.lora_name}_{config.hidden_size}.pth")
        best_adapter_path = save_path.removesuffix(".pth") + "_best.pth"
        training_path = save_path.removesuffix(".pth") + "_train.pth"
        metrics_dir = args.metrics_dir or os.path.join(args.out_dir, "metrics")

        def save_fn(
            trained_model,
            *,
            optimizer,
            scaler,
            training_state,
            kind,
        ):
            adapter_path = best_adapter_path if kind == "best" else save_path
            if kind != "best":
                # Preserve the resumable state first. If adapter serialization
                # then fails, --resume can deterministically regenerate it.
                save_checkpoint(
                    training_path,
                    trained_model,
                    config=config,
                    tokenizer=tokenizer,
                    args=args,
                    optimizer=optimizer,
                    scaler=scaler,
                    step=training_state["optimizer_step"],
                    stage="lora",
                    training_state=training_state,
                    extra={
                        "split": split_metadata,
                        "checkpoint_kind": kind,
                        "adapter_checkpoint": adapter_path,
                        "base_checkpoint": args.init_from,
                    },
                    context=context,
                )
            save_lora(
                trained_model,
                adapter_path,
                metadata={
                    "stage": "lora",
                    "base_checkpoint": args.init_from,
                    "config": config.to_dict(),
                    "training_args": vars(args),
                    "tokenizer": args.tokenizer_dir,
                    "optimizer_step": training_state["optimizer_step"],
                    "split": split_metadata,
                },
            )
            rank0_print(f"LoRA weights saved: {adapter_path}", context=context)

        def write_metrics(history, _training_state):
            paths = write_training_artifacts(
                metrics_dir,
                "lora",
                history,
                split_metadata,
                args,
                config,
            )
            if paths:
                rank0_print(f"metrics updated: {paths['json']}", context=context)

        train_supervised(
            model,
            dataset,
            args,
            device,
            save_fn,
            params=trainable,
            context=context,
            validation_dataset=validation_dataset,
            epoch_callback=write_metrics,
            stage="lora",
            run_identity={
                "split_fingerprint": split_metadata["split_fingerprint"],
                "train_samples": split_metadata["train_samples"],
                "validation_samples": split_metadata["validation_samples"],
                "tokenizer_vocab_size": len(tokenizer),
                "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
                "model_config": config.to_dict(),
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_targets": model._lora_config["targets"],
            },
        )
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
