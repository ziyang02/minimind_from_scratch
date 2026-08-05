#!/usr/bin/env bash
set -euo pipefail

# Reproduce a practical MiniMind-3 Zero model with this repository.
# Override any setting through the environment, for example:
#   PRETRAIN_BATCH=8 SFT_BATCH=4 PRETRAIN_ACCUMULATION=16 bash scripts/train_minimind3.sh

DATA_DIR="${DATA_DIR:-dataset}"
TOKENIZER_DIR="${TOKENIZER_DIR:-tokenizer_minimind3}"
OUT_DIR="${OUT_DIR:-out/minimind3}"
DATA_VARIANT="${DATA_VARIANT:-mini}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-1}"
PRETRAIN_BATCH="${PRETRAIN_BATCH:-32}"
SFT_BATCH="${SFT_BATCH:-16}"
PRETRAIN_ACCUMULATION="${PRETRAIN_ACCUMULATION:-8}"
SFT_ACCUMULATION="${SFT_ACCUMULATION:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
USE_SYSTEM_PYTHON="${USE_SYSTEM_PYTHON:-0}"

if [[ "$USE_SYSTEM_PYTHON" == "1" ]]; then
  PYTHON_RUNNER=(python)
else
  PYTHON_RUNNER=(uv run python)
fi

if [[ "$DATA_VARIANT" == "mini" ]]; then
  PRETRAIN_DATA="$DATA_DIR/pretrain_t2t_mini.jsonl"
  SFT_DATA="$DATA_DIR/sft_t2t_mini.jsonl"
  PRETRAIN_LENGTH="${PRETRAIN_LENGTH:-340}"
elif [[ "$DATA_VARIANT" == "full" ]]; then
  PRETRAIN_DATA="$DATA_DIR/pretrain_t2t.jsonl"
  SFT_DATA="$DATA_DIR/sft_t2t.jsonl"
  PRETRAIN_LENGTH="${PRETRAIN_LENGTH:-380}"
else
  echo "DATA_VARIANT must be mini or full" >&2
  exit 2
fi
SFT_LENGTH="${SFT_LENGTH:-768}"

for required in "$PRETRAIN_DATA" "$SFT_DATA" "$TOKENIZER_DIR/tokenizer.json"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    if [[ "$DATA_VARIANT" == "full" ]]; then
      echo "run: uv run python scripts/download_dataset.py --full" >&2
    else
      echo "run: uv run python scripts/download_dataset.py" >&2
    fi
    exit 2
  fi
done

"${PYTHON_RUNNER[@]}" - "$TOKENIZER_DIR" <<'PY'
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
if len(tokenizer) != 6400:
    raise SystemExit(f"MiniMind-3 requires the official 6400-token tokenizer; got {len(tokenizer)}")
print(f"tokenizer OK: {len(tokenizer)} tokens")
PY

COMMON_ARGS=(
  --tokenizer_dir "$TOKENIZER_DIR"
  --out_dir "$OUT_DIR"
  --hidden_size 768
  --num_hidden_layers 8
  --num_attention_heads 8
  --num_key_value_heads 4
  --epochs "$EPOCHS"
  --num_workers "$NUM_WORKERS"
  --save_interval "$SAVE_INTERVAL"
  --split_strategy full
  --validation_fraction 0
  --device "$DEVICE"
  --seed 42
)

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  if [[ "$USE_SYSTEM_PYTHON" == "1" ]]; then
    TRAIN_LAUNCHER=(torchrun --standalone --nproc_per_node "$NPROC_PER_NODE")
  else
    TRAIN_LAUNCHER=(uv run torchrun --standalone --nproc_per_node "$NPROC_PER_NODE")
  fi
else
  TRAIN_LAUNCHER=("${PYTHON_RUNNER[@]}")
fi

echo "stage 1/2: pretraining MiniMind-3"
"${TRAIN_LAUNCHER[@]}" trainer/train_pretrain.py \
  "${COMMON_ARGS[@]}" \
  --data_path "$PRETRAIN_DATA" \
  --max_length "$PRETRAIN_LENGTH" \
  --batch_size "$PRETRAIN_BATCH" \
  --accumulation_steps "$PRETRAIN_ACCUMULATION" \
  --lr 5e-4

echo "stage 2/2: supervised fine-tuning"
"${TRAIN_LAUNCHER[@]}" trainer/train_sft.py \
  "${COMMON_ARGS[@]}" \
  --data_path "$SFT_DATA" \
  --init_from "$OUT_DIR/pretrain_768.pth" \
  --max_length "$SFT_LENGTH" \
  --batch_size "$SFT_BATCH" \
  --accumulation_steps "$SFT_ACCUMULATION" \
  --lr 1e-5

echo "training complete: $OUT_DIR/full_sft_768.pth"
echo "chat with: uv run ninjamind --checkpoint $OUT_DIR/full_sft_768.pth --tokenizer-dir $TOKENIZER_DIR"
