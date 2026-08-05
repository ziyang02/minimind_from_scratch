"""Create reference-reward rollout prompts from MiniMind DPO preferences.

The official RLAIF file intentionally contains blank assistant placeholders and
expects a separate learned reward model.  This project also supports a smaller
single-GPU experiment: sample preferred DPO responses as references and score
online PPO/GRPO completions with continuous character n-gram overlap.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def preferred_rollout_record(sample: object, line_number: int) -> dict | None:
    if not isinstance(sample, dict):
        raise ValueError(f"line {line_number}: expected a JSON object")
    chosen = sample.get("chosen")
    if not isinstance(chosen, list) or len(chosen) < 2:
        return None
    final = chosen[-1]
    if not isinstance(final, dict) or final.get("role") != "assistant":
        return None
    answer = final.get("content")
    if not isinstance(answer, str) or not answer.strip():
        return None
    prompt_messages = chosen[:-1]
    if not prompt_messages or not all(isinstance(message, dict) for message in prompt_messages):
        return None
    return {"conversations": prompt_messages, "answer": answer}


def build_dataset(source: Path, destination: Path, *, limit: int, seed: int) -> int:
    rng = random.Random(seed)
    reservoir: list[dict] = []
    valid_seen = 0
    with source.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            if not raw_line.strip():
                continue
            try:
                sample = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON ({exc})") from exc
            record = preferred_rollout_record(sample, line_number)
            if record is None:
                continue
            valid_seen += 1
            if len(reservoir) < limit:
                reservoir.append(record)
            else:
                replacement = rng.randrange(valid_seen)
                if replacement < limit:
                    reservoir[replacement] = record

    if not reservoir:
        raise ValueError(f"no usable preferred answers found in {source}")
    rng.shuffle(reservoir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in reservoir:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(reservoir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PPO/GRPO prompts from DPO chosen answers")
    parser.add_argument("--input", default="dataset/dpo.jsonl", type=Path)
    parser.add_argument("--output", default="dataset/rl_from_dpo.jsonl", type=Path)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be >= 1")
    count = build_dataset(args.input, args.output, limit=args.limit, seed=args.seed)
    print(f"wrote {count} rollout prompts to {args.output}")


if __name__ == "__main__":
    main()
