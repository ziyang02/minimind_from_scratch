"""Dependency-free, versioned training metric artifacts.

The writer intentionally uses only the Python standard library so training can
emit machine-readable metrics and a small loss curve without installing the
benchmark/plotting extras.  Each output file is written to a temporary file in
its destination directory and atomically moved into place.
"""

from __future__ import annotations

import csv
import dataclasses
import html
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

SCHEMA_NAME = "ninjamind.training_artifacts"
SCHEMA_VERSION = 1

_PREFERRED_COLUMNS = (
    "epoch",
    "optimizer_step",
    "train_ce",
    "validation_ce",
    "validation_perplexity",
)
_TRAIN_CE_KEYS = ("train_ce", "train_loss", "training_loss")
_VALIDATION_CE_KEYS = ("validation_ce", "val_ce", "validation_loss", "val_loss")


def _object_mapping(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
        if isinstance(converted, Mapping):
            return converted
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        converted = dataclasses.asdict(value)
        if isinstance(converted, Mapping):
            return converted
    try:
        return vars(value)
    except TypeError as exc:
        raise TypeError(f"expected a mapping-like object, got {type(value).__name__}") from exc


def _json_value(value: Any) -> Any:
    """Convert common metadata values to strict JSON-compatible objects."""

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError("training artifacts cannot contain NaN or infinity")
        return converted
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, (set, frozenset)):
        return [_json_value(item) for item in sorted(value, key=repr)]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    return str(value)


def _normalise_history(history: Any) -> list[dict[str, Any]]:
    if history is None:
        return []
    records = []
    for index, record in enumerate(history):
        try:
            mapping = _object_mapping(record)
        except TypeError as exc:
            raise TypeError(f"history record {index} must be mapping-like") from exc
        records.append({str(key): _json_value(value) for key, value in mapping.items()})
    return records


def _rank_from_args(args: Any) -> int:
    values = _object_mapping(args)
    rank = values.get("rank", values.get("global_rank", 0))
    if rank is None:
        return 0
    try:
        return int(rank)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"rank must be an integer, got {rank!r}") from exc


def _safe_stage(stage: Any) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(stage or "training"))
    return slug.strip("._") or "training"


def _strict_json(payload: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=indent,
        separators=None if indent is not None else (",", ":"),
    )


def _csv_text(history: list[dict[str, Any]]) -> str:
    present = {key for record in history for key in record}
    columns = [key for key in _PREFERRED_COLUMNS if key in present]
    columns.extend(sorted(present.difference(columns)))
    if not columns:
        columns = list(_PREFERRED_COLUMNS)

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for record in history:
        row = {}
        for column in columns:
            value = record.get(column)
            if value is None:
                row[column] = ""
            elif isinstance(value, (dict, list)):
                row[column] = _strict_json(value)
            else:
                row[column] = value
        writer.writerow(row)
    return stream.getvalue()


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric(record: Mapping[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in record:
            return _finite_number(record[key])
    return None


def _svg_text(stage: str, history: list[dict[str, Any]]) -> str:
    width, height = 820, 460
    left, right, top, bottom = 72, 28, 48, 62
    plot_width = width - left - right
    plot_height = height - top - bottom

    train_points = []
    validation_points = []
    x_values = []
    for index, record in enumerate(history):
        x_value = _finite_number(record.get("epoch"))
        if x_value is None:
            x_value = float(index)
        x_values.append(x_value)
        train_ce = _metric(record, _TRAIN_CE_KEYS)
        validation_ce = _metric(record, _VALIDATION_CE_KEYS)
        if train_ce is not None:
            train_points.append((x_value, train_ce))
        if validation_ce is not None:
            validation_points.append((x_value, validation_ce))

    all_points = train_points + validation_points
    if not x_values:
        x_values = [0.0]
    x_min, x_max = min(x_values), max(x_values)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5

    if all_points:
        y_min = min(point[1] for point in all_points)
        y_max = max(point[1] for point in all_points)
        padding = max((y_max - y_min) * 0.1, max(abs(y_min), abs(y_max), 1.0) * 0.03)
        y_min -= padding
        y_max += padding
    else:
        y_min, y_max = 0.0, 1.0

    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        px = left + (x - x_min) / (x_max - x_min) * plot_width
        py = top + (y_max - y) / (y_max - y_min) * plot_height
        return px, py

    def series(points: list[tuple[float, float]], colour: str, label: str) -> str:
        if not points:
            return ""
        projected = [project(point) for point in points]
        coords = " ".join(f"{x:.2f},{y:.2f}" for x, y in projected)
        lines = [
            f'<polyline points="{coords}" fill="none" stroke="{colour}" '
            'stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>',
        ]
        for x, y in projected:
            lines.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{colour}">'
                f"<title>{html.escape(label)}</title></circle>"
            )
        return "\n".join(lines)

    title = html.escape(f"{stage} train / validation cross-entropy")
    x_start = f"{x_min:g}"
    x_end = f"{x_max:g}"
    y_start = f"{y_min:.4g}"
    y_end = f"{y_max:.4g}"
    empty_note = ""
    if not all_points:
        empty_note = (
            f'<text x="{left + plot_width / 2:.1f}" y="{top + plot_height / 2:.1f}" '
            'text-anchor="middle" fill="#6b7280" font-size="14">No finite CE values</text>'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="sans-serif" font-size="17" font-weight="600">{title}</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#374151"/>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#374151"/>
<line x1="{left}" y1="{top}" x2="{left + plot_width}" y2="{top}" stroke="#e5e7eb"/>
<line x1="{left}" y1="{top + plot_height / 2:.1f}" x2="{left + plot_width}" y2="{top + plot_height / 2:.1f}" stroke="#e5e7eb"/>
<text x="{left}" y="{height - 35}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_start}</text>
<text x="{left + plot_width}" y="{height - 35}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_end}</text>
<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="sans-serif" font-size="12">{y_end}</text>
<text x="{left - 8}" y="{top + plot_height + 4}" text-anchor="end" font-family="sans-serif" font-size="12">{y_start}</text>
<text x="{left + plot_width / 2:.1f}" y="{height - 12}" text-anchor="middle" font-family="sans-serif" font-size="13">epoch</text>
<text x="18" y="{top + plot_height / 2:.1f}" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 18 {top + plot_height / 2:.1f})">cross-entropy</text>
{series(train_points, "#2563eb", "train CE")}
{series(validation_points, "#dc2626", "validation CE")}
{empty_note}
<line x1="{width - 218}" y1="24" x2="{width - 194}" y2="24" stroke="#2563eb" stroke-width="2.5"/>
<text x="{width - 188}" y="28" font-family="sans-serif" font-size="12">train CE</text>
<line x1="{width - 112}" y1="24" x2="{width - 88}" y2="24" stroke="#dc2626" stroke-width="2.5"/>
<text x="{width - 82}" y="28" font-family="sans-serif" font-size="12">validation CE</text>
</svg>
"""


def _atomic_write_text(path: Path, content: str) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_training_artifacts(
    metrics_dir: str | os.PathLike[str],
    stage: str,
    history: Any,
    split_metadata: Any,
    args: Any,
    config: Any,
) -> dict[str, Path]:
    """Write JSON, CSV, and SVG training metrics using schema version 1.

    ``history`` is an iterable of mapping-like epoch records.  JSON and CSV
    retain every supplied field; the SVG reads ``train_ce`` and
    ``validation_ce`` (with common ``*_loss`` aliases).  ``None`` validation
    values and epoch zero are valid, and a single point remains visible.

    Set ``args.rank`` (or ``args.global_rank``) to a non-zero value to skip all
    filesystem writes on worker ranks.  The return value is empty when skipped
    and otherwise maps ``json``, ``csv``, and ``svg`` to their final paths.
    """

    if _rank_from_args(args) != 0:
        return {}

    normalised_history = _normalise_history(history)
    payload = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "stage": str(stage),
        "history": normalised_history,
        "split_metadata": _json_value(_object_mapping(split_metadata)),
        "training_args": _json_value(_object_mapping(args)),
        "model_config": _json_value(_object_mapping(config)),
    }

    # Render all content before touching existing files.  In particular,
    # allow_nan=False makes a non-finite metric fail without a partial update.
    json_content = _strict_json(payload, indent=2) + "\n"
    csv_content = _csv_text(normalised_history)
    svg_content = _svg_text(str(stage), normalised_history)

    directory = Path(metrics_dir)
    directory.mkdir(parents=True, exist_ok=True)
    prefix = _safe_stage(stage)
    paths = {
        "json": directory / f"{prefix}_metrics.json",
        "csv": directory / f"{prefix}_metrics.csv",
        "svg": directory / f"{prefix}_ce.svg",
    }
    _atomic_write_text(paths["json"], json_content)
    _atomic_write_text(paths["csv"], csv_content)
    _atomic_write_text(paths["svg"], svg_content)
    return paths
