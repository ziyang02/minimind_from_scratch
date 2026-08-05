#!/usr/bin/env bash
set -euo pipefail

# Run three independent post-training branches from the same SFT checkpoint.
# These defaults are bounded experimental runs for one 32 GB GPU; override any
# value through the environment when a longer comparison is desired.

BASE_CHECKPOINT="${BASE_CHECKPOINT:-out/minimind3/full_sft_768.pth}"
TOKENIZER_DIR="${TOKENIZER_DIR:-tokenizer_minimind3}"
DPO_DATA="${DPO_DATA:-dataset/dpo.jsonl}"
RL_DATA="${RL_DATA:-dataset/rl_from_dpo.jsonl}"
OUT_ROOT="${OUT_ROOT:-out/post_training}"
RL_SAMPLES="${RL_SAMPLES:-2000}"
DPO_STEPS="${DPO_STEPS:-1000}"
PPO_STEPS="${PPO_STEPS:-300}"
GRPO_STEPS="${GRPO_STEPS:-300}"
NUM_WORKERS="${NUM_WORKERS:-4}"
USE_SYSTEM_PYTHON="${USE_SYSTEM_PYTHON:-0}"

if [[ "$USE_SYSTEM_PYTHON" == "1" ]]; then
  PYTHON_RUNNER=(python)
else
  PYTHON_RUNNER=(uv run python)
fi

for required in "$BASE_CHECKPOINT" "$TOKENIZER_DIR/tokenizer.json" "$DPO_DATA"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 2
  fi
done

mkdir -p "$OUT_ROOT/dpo" "$OUT_ROOT/ppo" "$OUT_ROOT/grpo"

"${PYTHON_RUNNER[@]}" scripts/prepare_rl_from_dpo.py \
  --input "$DPO_DATA" \
  --output "$RL_DATA" \
  --limit "$RL_SAMPLES" \
  --seed 42

MODEL_ARGS=(
  --tokenizer_dir "$TOKENIZER_DIR"
  --hidden_size 768
  --num_hidden_layers 8
  --num_attention_heads 8
  --num_key_value_heads 4
  --init_from "$BASE_CHECKPOINT"
  --epochs 1
  --num_workers "$NUM_WORKERS"
  --device cuda
  --seed 42
)

echo "branch 1/3: DPO from the SFT baseline"
"${PYTHON_RUNNER[@]}" trainer/train_dpo.py \
  "${MODEL_ARGS[@]}" \
  --data_path "$DPO_DATA" \
  --out_dir "$OUT_ROOT/dpo" \
  --max_length 768 \
  --batch_size 4 \
  --accumulation_steps 2 \
  --max_steps "$DPO_STEPS" \
  --beta 0.1 \
  --lr 5e-7

echo "branch 2/3: PPO from the same SFT baseline"
"${PYTHON_RUNNER[@]}" trainer/train_ppo.py \
  "${MODEL_ARGS[@]}" \
  --data_path "$RL_DATA" \
  --out_dir "$OUT_ROOT/ppo" \
  --max_prompt_len 256 \
  --max_new_tokens 64 \
  --batch_size 4 \
  --accumulation_steps 1 \
  --max_steps "$PPO_STEPS" \
  --ppo_epochs 2 \
  --reward_mode overlap \
  --lr 1e-6 \
  --critic_lr 1e-5

echo "branch 3/3: GRPO from the same SFT baseline"
"${PYTHON_RUNNER[@]}" trainer/train_grpo.py \
  "${MODEL_ARGS[@]}" \
  --data_path "$RL_DATA" \
  --out_dir "$OUT_ROOT/grpo" \
  --max_prompt_len 256 \
  --max_new_tokens 64 \
  --batch_size 2 \
  --accumulation_steps 1 \
  --max_steps "$GRPO_STEPS" \
  --group_size 4 \
  --update_epochs 1 \
  --reward_mode overlap \
  --lr 1e-6

echo "post-training comparison complete"
echo "DPO:  $OUT_ROOT/dpo/dpo_768.pth"
echo "PPO:  $OUT_ROOT/ppo/ppo_768.pth"
echo "GRPO: $OUT_ROOT/grpo/grpo_768.pth"
