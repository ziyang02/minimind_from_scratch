"""Strip optimizer/resume state from a training checkpoint for deployment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

INFERENCE_KEYS = {
    "format_version",
    "model_state_dict",
    "config",
    "tokenizer",
    "training_args",
    "stage",
    "step",
    "extra",
}


def export_checkpoint(source: Path, destination: Path) -> None:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{source} is not a structured full-model checkpoint")
    compact = {key: value for key, value in checkpoint.items() if key in INFERENCE_KEYS}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(compact, temporary)
    temporary.replace(destination)
    print(
        f"exported {checkpoint.get('stage')} step={checkpoint.get('step')} "
        f"from {source} to {destination}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    export_checkpoint(args.source, args.destination)


if __name__ == "__main__":
    main()
