"""Run the complete supervised/preference pipeline on the committed demo data.

The script launches the real trainer entry points rather than maintaining a
second toy training loop.  It writes machine-readable losses, environment
metadata, and three deterministic generation samples; with matplotlib
installed it also creates a compact loss-curve image.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LOSS_PATTERN = re.compile(r"\bloss(?:\s+|=)(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def run_command(command: list[str]) -> tuple[str, float]:
    """Run one stage, echo its log, and fail with the original output."""

    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
    )
    duration = time.perf_counter() - started
    print(result.stdout, end="")
    if result.returncode:
        rendered = " ".join(command)
        raise RuntimeError(f"smoke command failed ({result.returncode}): {rendered}")
    return result.stdout, duration


def git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def stage_commands(output_dir: Path, steps: int) -> list[tuple[str, list[str]]]:
    common = [
        "--tokenizer_dir",
        "tokenizer",
        "--out_dir",
        str(output_dir),
        "--hidden_size",
        "32",
        "--num_hidden_layers",
        "1",
        "--num_attention_heads",
        "4",
        "--num_key_value_heads",
        "2",
        "--max_length",
        "96",
        "--batch_size",
        "2",
        "--max_steps",
        str(steps),
        "--log_interval",
        "1",
        "--device",
        "cpu",
        "--seed",
        "42",
    ]
    python = sys.executable
    sft_checkpoint = output_dir / "full_sft_32.pth"
    return [
        (
            "pretrain",
            [
                python,
                "trainer/train_pretrain.py",
                "--data_path",
                "dataset/demo/pretrain_demo.jsonl",
                *common,
            ],
        ),
        (
            "sft",
            [
                python,
                "trainer/train_sft.py",
                "--data_path",
                "dataset/demo/sft_demo.jsonl",
                "--init_from",
                str(output_dir / "pretrain_32.pth"),
                *common,
            ],
        ),
        (
            "lora",
            [
                python,
                "trainer/train_lora.py",
                "--data_path",
                "dataset/demo/sft_demo.jsonl",
                "--init_from",
                str(sft_checkpoint),
                "--lora_rank",
                "2",
                "--lora_alpha",
                "4",
                *common,
            ],
        ),
        (
            "dpo",
            [
                python,
                "trainer/train_dpo.py",
                "--data_path",
                "dataset/demo/dpo_demo.jsonl",
                "--init_from",
                str(sft_checkpoint),
                *common,
            ],
        ),
    ]


def generate_samples(checkpoint: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    from inference import SamplingConfig, load_model_and_tokenizer, stream_text

    loaded = load_model_and_tokenizer(
        tokenizer_dir=ROOT / "tokenizer",
        checkpoint=checkpoint,
        device="cpu",
        num_attention_heads=4,
    )
    config = SamplingConfig(
        max_new_tokens=12,
        temperature=0,
        top_k=0,
        use_cache=True,
        seed=42,
    )
    samples = []
    for prompt in ("What is 2 plus 2?", "What is 3 plus 4?", "What is 8 plus 1?"):
        chunks = list(stream_text(loaded.model, loaded.tokenizer, prompt, config=config))
        samples.append({"prompt": prompt, "completion": chunks[-1] if chunks else ""})
    model_metadata = {
        "parameters": sum(parameter.numel() for parameter in loaded.model.parameters()),
        "hidden_size": loaded.model.config.hidden_size,
        "num_hidden_layers": loaded.model.config.num_hidden_layers,
        "num_attention_heads": loaded.model.config.num_attention_heads,
        "num_key_value_heads": loaded.model.config.num_key_value_heads,
        "vocab_size": loaded.model.config.vocab_size,
    }
    return samples, model_metadata


def write_loss_plot(stages: list[dict], path: Path) -> bool:
    """Write a dependency-free SVG with one loss panel per training stage."""

    width, height = 900, 500
    panel_width, panel_height = 380, 155
    origins = ((70, 80), (500, 80), (70, 300), (500, 300))
    colors = ("#377eb8", "#4daf4a", "#984ea3", "#ff7f00")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="36" text-anchor="middle" font-family="sans-serif" '
        'font-size="24" font-weight="600">CPU demo smoke losses</text>',
    ]
    for stage, (origin_x, origin_y), color in zip(stages, origins, colors, strict=True):
        losses = stage["losses"]
        low, high = min(losses), max(losses)
        padding = max((high - low) * 0.15, max(abs(high), 1.0) * 0.002)
        low, high = low - padding, high + padding
        plot_left, plot_right = origin_x + 45, origin_x + panel_width
        plot_top, plot_bottom = origin_y + 25, origin_y + panel_height
        elements.extend(
            [
                f'<text x="{origin_x}" y="{origin_y + 12}" font-family="sans-serif" '
                f'font-size="17" font-weight="600" fill="{color}">{stage["stage"]}</text>',
                f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" '
                f'y2="{plot_bottom}" stroke="#444"/>',
                f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" '
                f'y2="{plot_bottom}" stroke="#444"/>',
                f'<text x="{plot_left - 7}" y="{plot_top + 5}" text-anchor="end" '
                f'font-family="monospace" font-size="11">{high:.4f}</text>',
                f'<text x="{plot_left - 7}" y="{plot_bottom + 4}" text-anchor="end" '
                f'font-family="monospace" font-size="11">{low:.4f}</text>',
            ]
        )
        denominator = max(len(losses) - 1, 1)
        points = []
        for index, loss in enumerate(losses):
            x_pos = plot_left + (plot_right - plot_left) * index / denominator
            y_pos = plot_bottom - (plot_bottom - plot_top) * (loss - low) / (high - low)
            points.append((x_pos, y_pos))
        point_text = " ".join(f"{x_pos:.1f},{y_pos:.1f}" for x_pos, y_pos in points)
        elements.append(
            f'<polyline points="{point_text}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round"/>'
        )
        for index, ((x_pos, y_pos), loss) in enumerate(zip(points, losses, strict=True), 1):
            elements.extend(
                [
                    f'<circle cx="{x_pos:.1f}" cy="{y_pos:.1f}" r="4" fill="{color}"/>',
                    f'<text x="{x_pos:.1f}" y="{plot_bottom + 18}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="11">{index}</text>',
                    f'<text x="{x_pos:.1f}" y="{y_pos - 8:.1f}" text-anchor="middle" '
                    f'font-family="monospace" font-size="10">{loss:.4f}</text>',
                ]
            )
    elements.append(
        '<text x="450" y="490" text-anchor="middle" font-family="sans-serif" '
        'font-size="12" fill="#555">optimizer step (each panel uses its own y-axis)</text>'
    )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("out/smoke"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/smoke_train.json"))
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be at least 1")

    output_dir = args.output_dir.resolve()
    artifact_path = args.artifact.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = []
    for stage, command in stage_commands(output_dir, args.steps):
        print(f"\n[{stage}] {' '.join(command)}")
        log, duration = run_command(command)
        losses = [float(value) for value in LOSS_PATTERN.findall(log)]
        if not losses:
            raise RuntimeError(f"{stage} completed without logging a parseable loss")
        stages.append(
            {
                "stage": stage,
                "command": command,
                "duration_seconds": round(duration, 4),
                "losses": losses,
                "log_lines": [line for line in log.splitlines() if "loss" in line.lower()],
            }
        )

    sample_path = artifact_path.with_name("generation_samples.json")
    samples, model_metadata = generate_samples(output_dir / "full_sft_32.pth")
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_text(
        json.dumps(
            {
                "checkpoint": str(output_dir / "full_sft_32.pth"),
                "decoding": {"temperature": 0, "max_new_tokens": 12, "seed": 42},
                "samples": samples,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    plot_path = artifact_path.with_name("smoke_loss.svg")
    plot_written = write_loss_plot(stages, plot_path)
    dirty = git_value("status", "--porcelain")
    artifact = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": bool(dirty),
        "seed": 42,
        "model": model_metadata,
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": "cpu",
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        },
        "stages": stages,
        "generation_samples": str(sample_path),
        "loss_plot": str(plot_path) if plot_written else None,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nsmoke artifact: {artifact_path}")
    print(f"generation samples: {sample_path}")
    if plot_written:
        print(f"loss plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
