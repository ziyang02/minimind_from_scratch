"""Download MiniMind training datasets from HuggingFace.

Repo: https://huggingface.co/datasets/jingyaogong/minimind_dataset

Files in the repo (name -> stage -> approx size):
    pretrain_t2t.jsonl        pretrain (full)     8.3 GB
    pretrain_t2t_mini.jsonl   pretrain (mini)     1.2 GB
    sft_t2t.jsonl             SFT (full)          14.1 GB
    sft_t2t_mini.jsonl        SFT (mini)          1.7 GB
    dpo.jsonl                 DPO preference      50 MB
    rlaif.jsonl               RLAIF (PPO/GRPO)    20 MB
    agent_rl.jsonl            Agent RL            80 MB
    agent_rl_math.jsonl       Agent RL (math)     20 MB
    lora_identity.jsonl       LoRA demo           tiny
    lora_medical.jsonl        LoRA domain         30 MB
    lora_exam.jsonl           LoRA domain         20 MB

Usage:
    uv run python scripts/download_dataset.py                       # default: mini pretrain/SFT + RL files
    uv run python scripts/download_dataset.py --files dpo.jsonl
    uv run python scripts/download_dataset.py --mirror              # via hf-mirror.com (faster in CN)
"""
import argparse
import os

REPO_ID = "jingyaogong/minimind_dataset"
DEFAULT_FILES = [
    "pretrain_t2t_mini.jsonl",
    "sft_t2t_mini.jsonl",
    "rlaif.jsonl",
    "dpo.jsonl",
    "agent_rl_math.jsonl",
]


def main():
    parser = argparse.ArgumentParser(description="Download MiniMind datasets")
    parser.add_argument("--files", nargs="+", default=DEFAULT_FILES,
                        help="filenames inside the dataset repo")
    parser.add_argument("--out_dir", type=str, default="dataset")
    parser.add_argument("--mirror", action="store_true",
                        help="download via https://hf-mirror.com (mainland China)")
    args = parser.parse_args()

    if args.mirror:
        # Must be set before importing huggingface_hub.
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from huggingface_hub import hf_hub_download

    os.makedirs(args.out_dir, exist_ok=True)
    for filename in args.files:
        print(f"downloading {filename} ...")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=args.out_dir,  # interrupted downloads resume automatically
        )
        size_mb = os.path.getsize(path) / 1e6
        print(f"  -> {path} ({size_mb:.1f} MB)")
    print("done")


if __name__ == "__main__":
    main()
