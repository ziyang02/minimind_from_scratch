"""Verify exact two-rank DDP recovery against uninterrupted training.

The fixture intentionally interrupts both ranks after optimizer step two,
resumes from the shared checkpoint, and compares the final model, optimizer,
scaler, and every rank-local RNG stream with a continuous four-step run.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import random
import sys
import tempfile
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from trainer.trainer_utils import (  # noqa: E402
    DistributedContext,
    save_checkpoint,
    train_supervised,
)

SCHEMA_NAME = "ninjamind.ddp_resume_verification"
SCHEMA_VERSION = 1


class _InjectedInterruption(RuntimeError):
    """Internal control flow used to stop a run after a valid checkpoint."""


class _ResumeDataset(Dataset):
    """Deterministic samples that shard differently across two ranks."""

    def __init__(self) -> None:
        self.examples = []
        for index in range(8):
            inputs = torch.tensor(
                [(index + offset) % 17 for offset in range(4)],
                dtype=torch.long,
            )
            targets = torch.tensor(
                [(index + offset + 1) % 17 for offset in range(4)],
                dtype=torch.long,
            )
            mask = torch.tensor([1, 1, 1, 1], dtype=torch.long)
            self.examples.append((inputs, targets, mask))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]


class _DropoutLM(nn.Module):
    """Small stochastic LM whose RNG stream must be restored per rank."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(17, 12)
        self.dropout = nn.Dropout(0.35)
        self.head = nn.Linear(12, 17)

    def forward(self, input_ids, attention_mask=None):
        del attention_mask
        hidden = self.dropout(self.embedding(input_ids))
        return SimpleNamespace(logits=self.head(hidden))


def _args(*, resume: str = "") -> Namespace:
    return Namespace(
        batch_size=1,
        num_workers=0,
        device="cpu",
        lr=1e-2,
        epochs=1,
        accumulation_steps=1,
        grad_clip=1.0,
        max_steps=4,
        log_interval=100,
        save_interval=2,
        seed=123,
        resume=resume,
    )


def _seeded_model(rank: int, seed: int) -> _DropoutLM:
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    return _DropoutLM()


def _save_callback(
    *,
    context: DistributedContext,
    intermediate_path: Path,
    final_path: Path,
    interrupt: bool,
):
    def save(model, *, optimizer, scaler, training_state, kind):
        step = int(training_state["optimizer_step"])
        finalized = bool(training_state["checkpoint_finalized"])
        target = intermediate_path if step == 2 and not finalized else final_path
        save_checkpoint(
            str(target),
            model,
            optimizer=optimizer,
            scaler=scaler,
            step=step,
            stage="ddp-resume-verification",
            training_state=training_state,
            extra={"checkpoint_kind": kind, "fault_injected": interrupt},
            context=context,
        )
        if interrupt and step == 2 and not finalized:
            raise _InjectedInterruption("simulated failure after optimizer step 2")

    return save


def _worker(
    rank: int,
    world_size: int,
    store_path: str,
    intermediate_path: str,
    continuous_path: str,
    resumed_path: str,
) -> None:
    os.environ.setdefault(
        "GLOO_SOCKET_IFNAME",
        "lo0" if platform.system() == "Darwin" else "lo",
    )
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    context = DistributedContext(
        rank=rank,
        local_rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
        backend="gloo",
    )
    dataset = _ResumeDataset()
    identity = {"fixture": "dropout-lm-v1"}
    intermediate = Path(intermediate_path)
    continuous = Path(continuous_path)
    resumed = Path(resumed_path)
    try:
        train_supervised(
            _seeded_model(rank, 700),
            dataset,
            _args(),
            torch.device("cpu"),
            _save_callback(
                context=context,
                intermediate_path=intermediate,
                final_path=continuous,
                interrupt=False,
            ),
            context=context,
            stage="ddp-resume-verification",
            run_identity=identity,
        )
        dist.barrier()

        try:
            train_supervised(
                _seeded_model(rank, 700),
                dataset,
                _args(),
                torch.device("cpu"),
                _save_callback(
                    context=context,
                    intermediate_path=intermediate,
                    final_path=resumed,
                    interrupt=True,
                ),
                context=context,
                stage="ddp-resume-verification",
                run_identity=identity,
            )
        except _InjectedInterruption:
            pass
        else:
            raise AssertionError("fault-injected DDP run did not stop")
        dist.barrier()

        train_supervised(
            _seeded_model(rank, 900),
            dataset,
            _args(resume=str(intermediate)),
            torch.device("cpu"),
            _save_callback(
                context=context,
                intermediate_path=intermediate,
                final_path=resumed,
                interrupt=False,
            ),
            context=context,
            stage="ddp-resume-verification",
            run_identity=identity,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _nested_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _serialized_sha256(value: Any) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
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


def run_verification(output_dir: Path, artifact_path: Path) -> dict[str, Any]:
    """Run the fault-injection experiment and return its artifact payload."""

    output_dir.mkdir(parents=True, exist_ok=True)
    store_path = output_dir / "process_group.store"
    intermediate_path = output_dir / "interrupted_step_2.pth"
    continuous_path = output_dir / "continuous_step_4.pth"
    resumed_path = output_dir / "resumed_step_4.pth"
    store_path.unlink(missing_ok=True)

    mp.spawn(
        _worker,
        args=(
            2,
            str(store_path),
            str(intermediate_path),
            str(continuous_path),
            str(resumed_path),
        ),
        nprocs=2,
        join=True,
    )

    continuous = torch.load(continuous_path, map_location="cpu", weights_only=True)
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=True)
    checks = {
        "model_state_exact": _nested_equal(
            continuous["model_state_dict"], resumed["model_state_dict"]
        ),
        "optimizer_state_exact": _nested_equal(
            continuous["optimizer_state_dict"], resumed["optimizer_state_dict"]
        ),
        "scaler_state_exact": _nested_equal(
            continuous["scaler_state_dict"], resumed["scaler_state_dict"]
        ),
        "rank_rng_states_exact": _nested_equal(
            continuous["rng_state_by_rank"], resumed["rng_state_by_rank"]
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError("DDP resume mismatch: " + ", ".join(failed))

    rank_rng_states = continuous["rng_state_by_rank"]
    rank_rng_streams_distinct = not torch.equal(
        rank_rng_states[0]["torch_cpu"],
        rank_rng_states[1]["torch_cpu"],
    )
    if not rank_rng_streams_distinct:
        raise AssertionError("fixture did not create distinct per-rank RNG streams")

    payload = {
        "schema": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "backend": "gloo",
            "world_size": 2,
        },
        "experiment": {
            "planned_optimizer_steps": 4,
            "fault_after_optimizer_step": 2,
            "resume_checkpoint": intermediate_path.name,
            "continuous_checkpoint": continuous_path.name,
            "resumed_checkpoint": resumed_path.name,
        },
        "checkpoint": {
            "format_version": continuous["format_version"],
            "world_size": continuous["world_size"],
            "rank_rng_states": len(rank_rng_states),
            "rank_training_states": len(continuous["training_state_by_rank"]),
            "rank_rng_streams_distinct": rank_rng_streams_distinct,
        },
        "checks": checks,
        "hashes": {
            "model_state_dict": _serialized_sha256(continuous["model_state_dict"]),
            "optimizer_state_dict": _serialized_sha256(
                continuous["optimizer_state_dict"]
            ),
            "scaler_state_dict": _serialized_sha256(
                continuous["scaler_state_dict"]
            ),
            "rng_state_by_rank": _serialized_sha256(
                continuous["rng_state_by_rank"]
            ),
        },
        "limitations": [
            "This fixture validates two-process CPU/Gloo training.",
            "CUDA/NCCL and multi-node recovery require separate hardware validation.",
            "Exact resume requires the same world size and run identity.",
        ],
    }
    _atomic_json(artifact_path, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/ninjamind-ddp-resume"),
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=Path("artifacts/ddp_resume.json"),
    )
    args = parser.parse_args()
    payload = run_verification(args.output_dir.resolve(), args.artifact.resolve())
    print(
        "DDP exact resume verified | "
        f"world_size={payload['environment']['world_size']} | "
        f"fault_step={payload['experiment']['fault_after_optimizer_step']} | "
        f"final_step={payload['experiment']['planned_optimizer_steps']}"
    )
    print(f"artifact: {args.artifact.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
