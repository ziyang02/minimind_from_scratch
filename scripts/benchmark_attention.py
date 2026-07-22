"""Reproducible MHA/GQA/MQA and KV-cache inference benchmark.

The default workload is intentionally small enough for a laptop CPU.  It
writes machine-readable JSON/CSV plus a PNG comparison chart to ``artifacts``.
All reported timings come from real executions; no synthetic performance
numbers are filled in when a metric is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from inference import resolve_device  # noqa: E402
from model.model import NinjaMindConfig, NinjaMindForCausalLM  # noqa: E402


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _cache_nbytes(past_key_values: Any) -> int | None:
    """Return actual tensor payload bytes, independent of allocator overhead."""

    if not isinstance(past_key_values, (tuple, list)):
        return None
    tensors: list[torch.Tensor] = []
    for layer in past_key_values:
        if not isinstance(layer, (tuple, list)):
            return None
        tensors.extend(value for value in layer if torch.is_tensor(value))
    return sum(value.numel() * value.element_size() for value in tensors)


def _decode_cached(
    model: NinjaMindForCausalLM,
    prompt: torch.Tensor,
    decode_tokens: int,
    device: torch.device,
) -> tuple[float, Any]:
    output = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
    past = output.past_key_values
    _synchronise(device)
    start = time.perf_counter()
    for _ in range(decode_tokens):
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        output = model(
            input_ids=next_token,
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        past = output.past_key_values
    _synchronise(device)
    return time.perf_counter() - start, past


def _decode_without_cache(
    model: NinjaMindForCausalLM,
    prompt: torch.Tensor,
    decode_tokens: int,
    device: torch.device,
) -> float:
    sequence = prompt
    _synchronise(device)
    start = time.perf_counter()
    for _ in range(decode_tokens):
        output = model(input_ids=sequence, use_cache=False, logits_to_keep=1)
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        sequence = torch.cat([sequence, next_token], dim=1)
    _synchronise(device)
    return time.perf_counter() - start


def _architecture_name(q_heads: int, kv_heads: int) -> str:
    if kv_heads == q_heads:
        return "MHA"
    if kv_heads == 1:
        return "MQA"
    return "GQA"


def _probe_prefill_sdpa(model: NinjaMindForCausalLM, prompt: torch.Tensor) -> bool:
    """Detect a real SDPA call outside the timed benchmark region."""

    original_sdpa = F.scaled_dot_product_attention
    calls = 0

    def tracked_sdpa(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_sdpa(*args, **kwargs)

    F.scaled_dot_product_attention = tracked_sdpa
    try:
        with torch.inference_mode():
            model(input_ids=prompt, use_cache=False, logits_to_keep=1)
    finally:
        F.scaled_dot_product_attention = original_sdpa
    return calls > 0


def benchmark_variant(
    args: argparse.Namespace,
    *,
    kv_heads: int,
    prompt_cpu: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    config = NinjaMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        vocab_size=args.vocab_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=kv_heads,
        max_position_embeddings=args.seq_len + args.decode_tokens + 8,
        flash_attn=not args.no_sdpa,
        dropout=0.0,
        use_moe=False,
    )
    model = NinjaMindForCausalLM(config).to(device).eval()
    prompt = prompt_cpu.to(device)
    prefill_sdpa_executed = _probe_prefill_sdpa(model, prompt)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    dtype_bytes = next(model.parameters()).element_size()
    peak_allocated: int | None = None

    with torch.inference_mode():
        for _ in range(args.warmup):
            model(input_ids=prompt, use_cache=True, logits_to_keep=1)
            _decode_cached(model, prompt, args.decode_tokens, device)
            _decode_without_cache(model, prompt, args.decode_tokens, device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        prefill_seconds: list[float] = []
        for _ in range(args.repeats):
            _synchronise(device)
            start = time.perf_counter()
            for _ in range(args.prefill_iterations):
                prefill_output = model(
                    input_ids=prompt,
                    use_cache=True,
                    logits_to_keep=1,
                )
            _synchronise(device)
            prefill_seconds.append(
                (time.perf_counter() - start) / args.prefill_iterations
            )

        cached_seconds: list[float] = []
        cache_payload_bytes: int | None = None
        for _ in range(args.repeats):
            elapsed, final_cache = _decode_cached(
                model, prompt, args.decode_tokens, device
            )
            cached_seconds.append(elapsed)
            cache_payload_bytes = _cache_nbytes(final_cache)

        uncached_seconds = [
            _decode_without_cache(model, prompt, args.decode_tokens, device)
            for _ in range(args.repeats)
        ]
        if device.type == "cuda":
            peak_allocated = int(torch.cuda.max_memory_allocated(device))

    prefill_ms = [value * 1000 for value in prefill_seconds]
    decoded_token_count = args.batch_size * args.decode_tokens
    cached_tps = [decoded_token_count / value for value in cached_seconds]
    uncached_tps = [decoded_token_count / value for value in uncached_seconds]
    prefill_mean, prefill_std = _mean_std(prefill_ms)
    cached_mean, cached_std = _mean_std(cached_tps)
    uncached_mean, uncached_std = _mean_std(uncached_tps)
    final_sequence_length = args.seq_len + args.decode_tokens
    theoretical_cache_bytes = (
        2
        * args.num_hidden_layers
        * args.batch_size
        * final_sequence_length
        * kv_heads
        * (args.hidden_size // args.num_attention_heads)
        * dtype_bytes
    )
    prefill_cache_bytes = _cache_nbytes(prefill_output.past_key_values)

    return {
        "attention": _architecture_name(args.num_attention_heads, kv_heads),
        "query_heads": args.num_attention_heads,
        "key_value_heads": kv_heads,
        "group_size": args.num_attention_heads // kv_heads,
        "parameter_count": parameter_count,
        "prefill_sdpa_executed": prefill_sdpa_executed,
        "prefill_latency_ms_mean": prefill_mean,
        "prefill_latency_ms_std": prefill_std,
        "prefill_latency_ms_samples": prefill_ms,
        "decode_cached_tokens_per_second_mean": cached_mean,
        "decode_cached_tokens_per_second_std": cached_std,
        "decode_cached_tokens_per_second_samples": cached_tps,
        "decode_no_cache_tokens_per_second_mean": uncached_mean,
        "decode_no_cache_tokens_per_second_std": uncached_std,
        "decode_no_cache_tokens_per_second_samples": uncached_tps,
        "cache_decode_speedup": cached_mean / uncached_mean,
        "kv_cache_sequence_length": final_sequence_length,
        "kv_cache_theoretical_bytes": theoretical_cache_bytes,
        "kv_cache_tensor_payload_bytes": cache_payload_bytes,
        "kv_cache_prefill_tensor_payload_bytes": prefill_cache_bytes,
        "cuda_peak_allocated_bytes": peak_allocated,
    }


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    try:
        output = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return bool(output.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or platform.machine()


def _write_csv(path: Path, results: list[dict[str, Any]]) -> None:
    scalar_keys = [
        key
        for key, value in results[0].items()
        if not isinstance(value, (list, dict))
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_keys)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in scalar_keys})


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    payload = kind + data
    return (
        struct.pack(">I", len(data))
        + payload
        + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    )


def _fallback_plot(path: Path, results: list[dict[str, Any]]) -> None:
    """Write a dependency-free two-panel bar chart if matplotlib is absent."""

    width, height = 900, 480
    pixels = bytearray([255] * width * height * 3)

    def rectangle(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(max(y0, 0), min(y1, height)):
            for x in range(max(x0, 0), min(x1, width)):
                offset = (y * width + x) * 3
                pixels[offset : offset + 3] = bytes(color)

    rectangle(60, 30, 64, 420, (30, 30, 30))
    rectangle(60, 416, 420, 420, (30, 30, 30))
    rectangle(500, 30, 504, 420, (30, 30, 30))
    rectangle(500, 416, 860, 420, (30, 30, 30))
    prefill_max = max(item["prefill_latency_ms_mean"] for item in results)
    decode_max = max(
        item["decode_cached_tokens_per_second_mean"] for item in results
    )
    colors = [(55, 126, 184), (77, 175, 74), (152, 78, 163)]
    for index, (item, color) in enumerate(zip(results, colors, strict=True)):
        x = 100 + index * 100
        bar_height = int(350 * item["prefill_latency_ms_mean"] / prefill_max)
        rectangle(x, 416 - bar_height, x + 55, 416, color)
        x = 535 + index * 105
        cache_height = int(
            350 * item["decode_cached_tokens_per_second_mean"] / decode_max
        )
        no_cache_height = int(
            350 * item["decode_no_cache_tokens_per_second_mean"] / decode_max
        )
        rectangle(x, 416 - cache_height, x + 32, 416, color)
        rectangle(x + 34, 416 - no_cache_height, x + 66, 416, (180, 180, 180))

    scanlines = b"".join(
        b"\x00" + bytes(pixels[row * width * 3 : (row + 1) * width * 3])
        for row in range(height)
    )
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _write_plot(path: Path, results: list[dict[str, Any]]) -> str:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/minimind-matplotlib")
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _fallback_plot(path, results)
        return "stdlib fallback"

    labels = [item["attention"] for item in results]
    colors = ["#377eb8", "#4daf4a", "#984ea3"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    axes[0].bar(
        labels,
        [item["prefill_latency_ms_mean"] for item in results],
        yerr=[item["prefill_latency_ms_std"] for item in results],
        color=colors,
        capsize=3,
    )
    axes[0].set_title("Prefill latency (lower is better)")
    axes[0].set_ylabel("milliseconds")
    x_positions = list(range(len(results)))
    axes[1].bar(
        [value - 0.18 for value in x_positions],
        [item["decode_cached_tokens_per_second_mean"] for item in results],
        width=0.36,
        label="KV cache",
        color=colors,
    )
    axes[1].bar(
        [value + 0.18 for value in x_positions],
        [item["decode_no_cache_tokens_per_second_mean"] for item in results],
        width=0.36,
        label="No cache",
        color="#bdbdbd",
    )
    axes[1].set_xticks(x_positions, labels)
    axes[1].set_title("Decode throughput (higher is better)")
    axes[1].set_ylabel("tokens / second")
    axes[1].legend()
    figure.suptitle("NinjaMind attention benchmark (mean of repeated runs)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return f"matplotlib {matplotlib.__version__}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prefill-iterations", type=int, default=20)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=1)
    parser.add_argument("--num-attention-heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--no-sdpa", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    if args.num_attention_heads < 4 or args.num_attention_heads % 2:
        parser.error("--num-attention-heads must be an even integer >= 4")
    if args.hidden_size % args.num_attention_heads:
        parser.error("--hidden-size must be divisible by --num-attention-heads")
    if min(
        args.batch_size,
        args.seq_len,
        args.decode_tokens,
        args.num_hidden_layers,
        args.prefill_iterations,
    ) <= 0:
        parser.error("batch size, lengths, and layer count must be positive")

    device = resolve_device(args.device)
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    prompt = torch.randint(
        0,
        args.vocab_size,
        (args.batch_size, args.seq_len),
        generator=generator,
    )
    kv_variants = [args.num_attention_heads, args.num_attention_heads // 2, 1]
    results = [
        benchmark_variant(args, kv_heads=kv_heads, prompt_cpu=prompt, device=device)
        for kv_heads in kv_variants
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "benchmark.json"
    csv_path = output_dir / "benchmark.csv"
    plot_path = output_dir / "benchmark.png"
    plot_backend = _write_plot(plot_path, results)
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "device_type": device.type,
            "device_name": _device_name(device),
            "torch_num_threads": torch.get_num_threads(),
            "dtype": str(torch.get_default_dtype()).removeprefix("torch."),
            "git_commit": _git_commit(),
            "git_worktree_dirty": _git_dirty(),
            "plot_backend": plot_backend,
        },
        "config": {
            "seed": args.seed,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "prefill_iterations_per_sample": args.prefill_iterations,
            "batch_size": args.batch_size,
            "sequence_length": args.seq_len,
            "decode_tokens": args.decode_tokens,
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "vocab_size": args.vocab_size,
            "logits_to_keep": 1,
            "sdpa_config_enabled": not args.no_sdpa,
            "sdpa_measurement": (
                "prefill_sdpa_executed is measured with a non-timed call probe; "
                "timed calls do not include probe overhead"
            ),
            "kv_cache_formula": (
                "2 * layers * batch * cache_sequence_length * kv_heads * "
                "head_dim * dtype_bytes"
            ),
            "cache_measurement": (
                "sum(numel * element_size) across returned K/V tensors; "
                "allocator overhead excluded"
            ),
        },
        "results": results,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(csv_path, results)
    print(f"benchmark complete on {device}: {args.repeats} measured repeats per variant")
    for result in results:
        print(
            f"{result['attention']}: params={result['parameter_count']:,} "
            f"prefill={result['prefill_latency_ms_mean']:.3f} ms "
            f"decode(cache/no-cache)="
            f"{result['decode_cached_tokens_per_second_mean']:.1f}/"
            f"{result['decode_no_cache_tokens_per_second_mean']:.1f} tok/s "
            f"KV={result['kv_cache_tensor_payload_bytes']:,} bytes"
        )
    print(f"wrote {json_path}, {csv_path}, and {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
