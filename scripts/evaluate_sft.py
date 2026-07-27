"""Evaluate greedy generation on the deterministic held-out SFT split.

The evaluator uses the same exact-deduplicated, prompt-grouped split as SFT
training.  It loads the model once, generates one completion per selected
validation record, and atomically writes a strict machine-readable JSON
artifact.  This measures exact reproduction only; it intentionally contains
no arithmetic- or task-specific answer rules.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import SFTDataset, split_supervised_dataset  # noqa: E402
from inference import SamplingConfig, load_model_and_tokenizer, stream_text  # noqa: E402

SCHEMA_NAME = "ninjamind.sft_generation_evaluation"
SCHEMA_VERSION = 1


def extract_last_assistant_target(
    sample: Mapping[str, Any],
) -> tuple[list[dict[str, str]], str]:
    """Return messages before the last assistant turn and its exact content."""

    conversations = sample.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("SFT sample must contain a non-empty 'conversations' list")

    normalised_messages: list[dict[str, str]] = []
    last_assistant_index: int | None = None
    for index, message in enumerate(conversations):
        if not isinstance(message, Mapping):
            raise ValueError(f"SFT message {index} must be a JSON object")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(
                f"SFT message {index} must contain string role/content fields"
            )
        normalised_messages.append({"role": role, "content": content})
        if role == "assistant":
            last_assistant_index = index

    if last_assistant_index is None:
        raise ValueError("SFT sample has no assistant target")
    if last_assistant_index == 0:
        raise ValueError("SFT assistant target must have preceding prompt/history")

    history = normalised_messages[:last_assistant_index]
    expected = normalised_messages[last_assistant_index]["content"]
    return history, expected


def conservative_normalize(text: str) -> str:
    """Normalize canonical Unicode, newline encoding, and outer whitespace.

    Case, punctuation, internal spacing, wording, and numeric representation
    remain untouched.  The metric is consequently still an exact-match metric,
    not a task-aware correctness heuristic.
    """

    if not isinstance(text, str):
        raise TypeError("normalization input must be a string")
    canonical = unicodedata.normalize("NFC", text)
    return canonical.replace("\r\n", "\n").replace("\r", "\n").strip()


def aggregate_matches(samples: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    """Aggregate strict and conservative-normalized exact-match booleans."""

    evaluated = len(samples)
    strict_matches = sum(bool(sample["strict_exact_match"]) for sample in samples)
    normalized_matches = sum(
        bool(sample["normalized_exact_match"]) for sample in samples
    )
    denominator = evaluated or 1
    return {
        "evaluated_samples": evaluated,
        "strict_exact_matches": strict_matches,
        "strict_exact_match_rate": strict_matches / denominator if evaluated else 0.0,
        "normalized_exact_matches": normalized_matches,
        "normalized_exact_match_rate": (
            normalized_matches / denominator if evaluated else 0.0
        ),
    }


def _strict_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"


def write_evaluation_json(path: str | os.PathLike[str], payload: Any) -> Path:
    """Atomically write strict JSON, leaving an existing artifact intact on failure."""

    output_path = Path(path)
    # Render first: a NaN/Infinity error must not create a directory or touch a
    # previously valid artifact.
    content = _strict_json(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


def render_generation_prompt(tokenizer: Any, history: Sequence[Mapping[str, str]]) -> str:
    """Render the exact held-out history with an assistant generation marker."""

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("SFT generation evaluation requires a tokenizer chat template")
    return tokenizer.apply_chat_template(
        list(history),
        tokenize=False,
        add_generation_prompt=True,
    )


def generate_greedy_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: SamplingConfig,
) -> str:
    """Consume the existing streaming API and return its final cumulative text."""

    completion = ""
    for cumulative in stream_text(
        model,
        tokenizer,
        prompt,
        chat=False,
        config=config,
    ):
        completion = cumulative
    return completion


def evaluate_sft(
    *,
    checkpoint: str | os.PathLike[str],
    tokenizer_dir: str | os.PathLike[str],
    data_path: str | os.PathLike[str],
    max_length: int,
    validation_fraction: float,
    split_seed: int,
    device: str,
    max_new_tokens: int,
    limit: int,
    output: str | os.PathLike[str],
    model_loader: Callable[..., Any] | None = None,
    completion_generator: Callable[[Any, Any, str, SamplingConfig], str] | None = None,
) -> dict[str, Any]:
    """Run held-out greedy generation and write the schema-v1 evaluation artifact."""

    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be finite and in (0, 1)")
    if isinstance(split_seed, bool) or not isinstance(split_seed, int):
        raise TypeError("split_seed must be an integer")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if max_length < 2:
        raise ValueError("max_length must be >= 2")
    if limit < 0:
        raise ValueError("limit must be >= 0 (0 evaluates the full validation split)")

    loader = model_loader or load_model_and_tokenizer
    generator = completion_generator or generate_greedy_completion
    # This is deliberately outside the sample loop.  Loading once matters much
    # more than micro-optimising evaluation of this small held-out set.
    loaded = loader(
        tokenizer_dir=tokenizer_dir,
        checkpoint=checkpoint,
        device=device,
    )

    full_dataset = SFTDataset(data_path, loaded.tokenizer, max_length=max_length)
    _, validation_dataset, split_metadata = split_supervised_dataset(
        full_dataset,
        validation_fraction=validation_fraction,
        seed=split_seed,
    )
    if validation_dataset is None or not len(validation_dataset):
        raise ValueError("held-out split produced no validation samples")

    selected_indices = validation_dataset.indices
    if limit:
        selected_indices = selected_indices[:limit]
    sampling = SamplingConfig(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_k=0,
        use_cache=True,
        seed=None,
    )

    sample_results: list[dict[str, Any]] = []
    for validation_position, source_index in enumerate(selected_indices):
        try:
            history, expected = extract_last_assistant_target(
                full_dataset.samples[source_index]
            )
        except ValueError as exc:
            raise ValueError(
                f"{full_dataset.data_path}: validation sample {source_index}: {exc}"
            ) from exc
        prompt = render_generation_prompt(loaded.tokenizer, history)
        completion = generator(loaded.model, loaded.tokenizer, prompt, sampling)
        if not isinstance(completion, str):
            raise TypeError("completion_generator must return a string")
        strict_match = completion == expected
        normalized_match = conservative_normalize(completion) == conservative_normalize(
            expected
        )
        sample_results.append(
            {
                "validation_position": validation_position,
                "source_index": source_index,
                "history": history,
                "prompt": prompt,
                "expected": expected,
                "completion": completion,
                "strict_exact_match": strict_match,
                "normalized_exact_match": normalized_match,
            }
        )

    payload = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "split_metadata": split_metadata,
        "checkpoint": {
            "path": str(Path(checkpoint).expanduser().resolve()),
            "source": str(loaded.source),
        },
        "decoding": {
            "strategy": "greedy",
            "temperature": 0.0,
            "top_k": 0,
            "use_cache": True,
            "max_new_tokens": max_new_tokens,
            "limit": limit,
        },
        "samples": sample_results,
        "aggregate": aggregate_matches(sample_results),
    }
    write_evaluation_json(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer-dir", "--tokenizer_dir", default="tokenizer")
    parser.add_argument("--data", "--data-path", "--data_path", dest="data_path", required=True)
    parser.add_argument(
        "--validation-fraction",
        "--validation_fraction",
        dest="validation_fraction",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--split-seed", "--split_seed", dest="split_seed", type=int, default=42
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-length", "--max_length", dest="max_length", type=int, default=1024
    )
    parser.add_argument(
        "--max-new-tokens",
        "--max_new_tokens",
        dest="max_new_tokens",
        type=int,
        default=64,
    )
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all held-out samples")
    parser.add_argument("--output", default="artifacts/sft_generation_evaluation.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = evaluate_sft(
        checkpoint=args.checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        data_path=args.data_path,
        max_length=args.max_length,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        limit=args.limit,
        output=args.output,
    )
    aggregate = payload["aggregate"]
    print(
        f"evaluated {aggregate['evaluated_samples']} held-out samples | "
        f"strict EM {aggregate['strict_exact_match_rate']:.3f} | "
        f"normalized EM {aggregate['normalized_exact_match_rate']:.3f}"
    )
    print(f"artifact: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
