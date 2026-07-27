import csv
import json
from argparse import Namespace
from pathlib import Path

import pytest

import trainer.artifacts as artifacts
from trainer.artifacts import write_training_artifacts


class _Config:
    def to_dict(self):
        return {"hidden_size": 32, "use_moe": False}


def test_training_artifacts_json_csv_and_svg_are_consistent(tmp_path: Path):
    history = [
        {
            "epoch": 0,
            "optimizer_step": 2,
            "train_ce": 1.25,
            "validation_ce": None,
            "validation_perplexity": None,
        },
        {
            "epoch": 1,
            "optimizer_step": 4,
            "train_ce": 1.0,
            "validation_ce": 1.125,
            "validation_perplexity": 3.080216848918031,
        },
    ]
    paths = write_training_artifacts(
        tmp_path,
        "sft",
        history,
        {"train_samples": 8, "validation_samples": 2},
        Namespace(rank=0, lr=1e-4, data_path=Path("dataset/demo/sft_demo.jsonl")),
        _Config(),
    )

    assert set(paths) == {"json", "csv", "svg"}
    assert all(path.exists() for path in paths.values())

    raw_json = paths["json"].read_text(encoding="utf-8")
    assert "NaN" not in raw_json
    assert "Infinity" not in raw_json
    payload = json.loads(raw_json)
    assert payload["schema"] == "ninjamind.training_artifacts"
    assert payload["schema_version"] == 1
    assert payload["stage"] == "sft"
    assert payload["history"] == history
    assert payload["split_metadata"] == {"train_samples": 8, "validation_samples": 2}
    assert payload["training_args"]["data_path"] == "dataset/demo/sft_demo.jsonl"
    assert payload["model_config"] == {"hidden_size": 32, "use_moe": False}

    with paths["csv"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(payload["history"])
    for csv_row, json_row in zip(rows, payload["history"], strict=True):
        assert int(csv_row["epoch"]) == json_row["epoch"]
        assert int(csv_row["optimizer_step"]) == json_row["optimizer_step"]
        assert float(csv_row["train_ce"]) == json_row["train_ce"]
        if json_row["validation_ce"] is None:
            assert csv_row["validation_ce"] == ""
            assert csv_row["validation_perplexity"] == ""
        else:
            assert float(csv_row["validation_ce"]) == json_row["validation_ce"]
            assert float(csv_row["validation_perplexity"]) == json_row["validation_perplexity"]

    svg = paths["svg"].read_text(encoding="utf-8")
    assert svg.startswith("<svg")
    assert "train / validation cross-entropy" in svg
    assert "train CE" in svg
    assert "validation CE" in svg
    assert "<circle" in svg
    assert "nan" not in svg.lower()
    assert "inf" not in svg.lower()
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    "history",
    [
        [{"epoch": 0, "train_ce": 2.0, "validation_ce": None}],
        [{"epoch": None, "train_ce": None, "validation_ce": None}],
    ],
)
def test_training_artifact_svg_handles_one_point_epoch_zero_and_none(tmp_path: Path, history):
    paths = write_training_artifacts(
        tmp_path,
        "pretrain",
        history,
        {},
        Namespace(rank=0),
        {},
    )

    svg = paths["svg"].read_text(encoding="utf-8")
    assert svg.endswith("</svg>\n")
    assert "nan" not in svg.lower()
    assert "inf" not in svg.lower()
    if history[0]["train_ce"] is None:
        assert "No finite CE values" in svg
    else:
        assert "<circle" in svg
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_non_main_rank_skips_all_artifact_writes(tmp_path: Path):
    metrics_dir = tmp_path / "metrics"
    paths = write_training_artifacts(
        metrics_dir,
        "pretrain",
        [{"epoch": 0, "train_ce": 1.0}],
        {},
        Namespace(rank=1),
        {},
    )

    assert paths == {}
    assert not metrics_dir.exists()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_metrics_fail_before_writing(tmp_path: Path, invalid: float):
    metrics_dir = tmp_path / "metrics"
    with pytest.raises(ValueError, match="NaN or infinity"):
        write_training_artifacts(
            metrics_dir,
            "pretrain",
            [{"epoch": 0, "train_ce": invalid}],
            {},
            Namespace(rank=0),
            {},
        )

    assert not metrics_dir.exists()


def test_atomic_overwrite_leaves_no_temporary_files(tmp_path: Path):
    arguments = Namespace(rank=0)
    first = write_training_artifacts(
        tmp_path,
        "pretrain",
        [{"epoch": 0, "train_ce": 3.0}],
        {},
        arguments,
        {},
    )
    second = write_training_artifacts(
        tmp_path,
        "pretrain",
        [{"epoch": 0, "train_ce": 2.0, "validation_ce": 2.5}],
        {},
        arguments,
        {},
    )

    assert first == second
    payload = json.loads(second["json"].read_text(encoding="utf-8"))
    assert payload["history"][0]["train_ce"] == 2.0
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_atomic_replace_cleans_temporary_file(tmp_path: Path, monkeypatch):
    def fail_replace(source, destination):
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        write_training_artifacts(
            tmp_path,
            "pretrain",
            [{"epoch": 0, "train_ce": 2.0}],
            {},
            Namespace(rank=0),
            {},
        )

    assert not list(tmp_path.iterdir())
