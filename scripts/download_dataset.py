"""Download current MiniMind-3 training data and its matching tokenizer.

Repo: https://huggingface.co/datasets/jingyaogong/minimind_dataset

Files in the repo (name -> stage -> approx size):
    pretrain_t2t.jsonl        pretrain (full)     about 10 GB
    pretrain_t2t_mini.jsonl   pretrain (mini)     1.2 GB
    sft_t2t.jsonl             SFT (full)          about 14 GB
    sft_t2t_mini.jsonl        SFT (mini)          1.7 GB
    dpo.jsonl                 DPO preference      50 MB
    rlaif.jsonl               RLAIF (PPO/GRPO)    20 MB
    agent_rl.jsonl            Agent RL            80 MB
    agent_rl_math.jsonl       Agent RL (math)     20 MB
    lora_identity.jsonl       LoRA demo           tiny
    lora_medical.jsonl        LoRA domain         30 MB
    lora_exam.jsonl           LoRA domain         20 MB

Usage:
    uv run python scripts/download_dataset.py                       # mini pretrain/SFT + tokenizer
    uv run python scripts/download_dataset.py --full                # full pretrain/SFT + tokenizer
    uv run python scripts/download_dataset.py --files dpo.jsonl
    uv run python scripts/download_dataset.py --mirror              # via hf-mirror.com (faster in CN)
"""
import argparse
import os

REPO_ID = "jingyaogong/minimind_dataset"
TOKENIZER_REPO_ID = "jingyaogong/minimind-3"
DEFAULT_FILES = [
    "pretrain_t2t_mini.jsonl",
    "sft_t2t_mini.jsonl",
]
FULL_FILES = ["pretrain_t2t.jsonl", "sft_t2t.jsonl"]
TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]


def main():
    parser = argparse.ArgumentParser(description="Download MiniMind datasets")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                        help="filenames inside the dataset repo")
    parser.add_argument("--full", action="store_true",
                        help="download full pretrain/SFT files instead of the mini pair")
    parser.add_argument("--out_dir", type=str, default="dataset")
    parser.add_argument("--tokenizer_dir", type=str, default="tokenizer_minimind3")
    parser.add_argument("--skip_tokenizer", action="store_true")
    parser.add_argument("--mirror", action="store_true",
                        help="download via https://hf-mirror.com (mainland China)")
    args = parser.parse_args()

    if args.mirror:
        # Must be set before importing huggingface_hub.
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from huggingface_hub import hf_hub_download

    files = FULL_FILES if args.full else args.files
    os.makedirs(args.out_dir, exist_ok=True)
    for filename in files:
        print(f"downloading {filename} ...")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=args.out_dir,  # interrupted downloads resume automatically
        )
        size_mb = os.path.getsize(path) / 1e6
        print(f"  -> {path} ({size_mb:.1f} MB)")

    if not args.skip_tokenizer:
        os.makedirs(args.tokenizer_dir, exist_ok=True)
        print(f"downloading MiniMind-3 tokenizer to {args.tokenizer_dir} ...")
        for filename in TOKENIZER_FILES:
            hf_hub_download(
                repo_id=TOKENIZER_REPO_ID,
                filename=filename,
                repo_type="model",
                local_dir=args.tokenizer_dir,
            )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
        if len(tokenizer) != 6400:
            raise RuntimeError(
                f"expected the MiniMind-3 6400-token vocabulary, got {len(tokenizer)}"
            )
        print(f"  -> tokenizer verified ({len(tokenizer)} tokens)")
    print("done")


if __name__ == "__main__":
    main()
