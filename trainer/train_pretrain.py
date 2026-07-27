"""Stage 1 — Pretraining: next-token prediction over raw text.

Usage:
    uv run python trainer/train_pretrain.py --data_path dataset/demo/pretrain_demo.jsonl
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset, split_supervised_dataset
from trainer.artifacts import write_training_artifacts
from trainer.trainer_utils import (
    add_model_args,
    add_supervised_eval_args,
    add_train_args,
    build_model,
    ckpt_name,
    cleanup_distributed,
    rank0_print,
    save_checkpoint,
    setup_distributed,
    tokenizer_fingerprint,
    train_supervised,
)


def main():
    parser = argparse.ArgumentParser(description="Pretrain NinjaMind")
    parser.add_argument("--data_path", type=str, default="dataset/demo/pretrain_demo.jsonl")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer")
    parser.add_argument("--max_length", type=int, default=512)
    add_model_args(parser)
    add_train_args(parser, default_lr=5e-4)
    add_supervised_eval_args(parser)
    args = parser.parse_args()

    context = setup_distributed(args)
    try:
        torch.manual_seed(args.seed + context.rank)
        device = context.device
        os.makedirs(args.out_dir, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
        model, config = build_model(args, len(tokenizer), device, context)
        full_dataset = PretrainDataset(
            args.data_path,
            tokenizer,
            max_length=args.max_length,
        )
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

        save_path = ckpt_name(args.out_dir, "pretrain", config)
        best_path = save_path.removesuffix(".pth") + "_best.pth"
        metrics_dir = args.metrics_dir or os.path.join(args.out_dir, "metrics")

        def save_fn(
            trained_model,
            *,
            optimizer,
            scaler,
            training_state,
            kind,
        ):
            checkpoint_kwargs = {
                "config": config,
                "tokenizer": tokenizer,
                "args": args,
                "step": training_state["optimizer_step"],
                "stage": "pretrain",
                "extra": {"split": split_metadata, "checkpoint_kind": kind},
                "context": context,
            }
            if kind == "best":
                # Best is a compact inference/warm-start checkpoint. Exact
                # resume always uses the latest checkpoint with optimizer/RNG.
                save_checkpoint(best_path, trained_model, **checkpoint_kwargs)
            else:
                save_checkpoint(
                    save_path,
                    trained_model,
                    optimizer=optimizer,
                    scaler=scaler,
                    training_state=training_state,
                    **checkpoint_kwargs,
                )

        def write_metrics(history, _training_state):
            paths = write_training_artifacts(
                metrics_dir,
                "pretrain",
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
            context=context,
            validation_dataset=validation_dataset,
            epoch_callback=write_metrics,
            stage="pretrain",
            run_identity={
                "split_fingerprint": split_metadata["split_fingerprint"],
                "train_samples": split_metadata["train_samples"],
                "validation_samples": split_metadata["validation_samples"],
                "tokenizer_vocab_size": len(tokenizer),
                "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
                "model_config": config.to_dict(),
            },
        )
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
