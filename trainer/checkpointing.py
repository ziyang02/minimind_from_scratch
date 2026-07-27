"""Strict v2 checkpoints for exact single-process and DDP training resumption.

The existing project checkpoints intentionally remain useful as model-weight
initializers.  This module defines a separate contract for ``--resume``:
checkpoints must contain model, optimizer, scaler, progress, run identity, and
RNG state.  Raw state dicts and version-1 checkpoints are rejected with an
explicit instruction to use ``--init_from`` instead.

Distributed checkpoints keep one RNG state and one in-epoch progress state per
rank. Model, optimizer, scaler, shared history, and run identity remain common
and are written once by rank zero.
"""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

FORMAT_VERSION = 2

_REQUIRED_PAYLOAD_KEYS = {
    "stage",
    "run_identity",
    "world_size",
    "model_state_dict",
    "optimizer_state_dict",
    "scaler_state_dict",
    "training_state",
}
_REQUIRED_TRAINING_STATE_KEYS = {
    "epoch",
    "batch_in_epoch",
    "optimizer_step",
    "micro_step",
    "planned_total_steps",
}


class ResumeCheckpointError(RuntimeError):
    """Base error for checkpoints that cannot provide a strict resume."""


class LegacyCheckpointError(ResumeCheckpointError):
    """Raised when a raw/v1 checkpoint is supplied to the resume path."""


class IncompleteCheckpointError(ResumeCheckpointError):
    """Raised when a v2 checkpoint omits required state."""


class CheckpointCompatibilityError(ResumeCheckpointError):
    """Raised when a checkpoint belongs to a different training run."""


def _world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise ResumeCheckpointError("WORLD_SIZE must be an integer") from exc


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    try:
        rank = int(os.environ.get("RANK", "0"))
    except ValueError as exc:
        raise ResumeCheckpointError("RANK must be an integer") from exc
    world_size = _world_size()
    if not 0 <= rank < world_size:
        raise ResumeCheckpointError(
            f"RANK must be in [0, WORLD_SIZE), got {rank}/{world_size}"
        )
    return rank


def _require_single_process(operation: str) -> None:
    world_size = _world_size()
    if world_size != 1:
        raise ResumeCheckpointError(
            f"{operation} is a single-process convenience; distributed trainers "
            "must gather rank-local state before rank zero writes"
        )


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, torch CPU, MPS, and every visible CUDA generator.

    The returned object contains only primitives and tensors, so it is accepted
    by ``torch.load(..., weights_only=True)``.
    """

    cuda_states: list[torch.Tensor] = []
    if torch.cuda.is_available():
        cuda_states = [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
    mps_state = None
    if torch.backends.mps.is_available():
        mps_state = torch.mps.get_rng_state().cpu().clone()
    return {
        "format_version": 2,
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state().cpu().clone(),
        "torch_cuda": cuda_states,
        "torch_mps": mps_state,
    }


def _validate_rng_state(
    state: Mapping[str, Any],
    device: torch.device,
) -> tuple[object, torch.Tensor, list[torch.Tensor], torch.Tensor | None]:
    missing = {"python", "torch_cpu", "torch_cuda", "torch_mps"} - set(state)
    if missing:
        raise IncompleteCheckpointError(
            f"checkpoint RNG state is incomplete; missing: {', '.join(sorted(missing))}"
        )

    python_state = state["python"]
    cpu_state = state["torch_cpu"]
    cuda_states = state["torch_cuda"]
    mps_state = state["torch_mps"]
    if not torch.is_tensor(cpu_state):
        raise IncompleteCheckpointError("checkpoint torch CPU RNG state must be a tensor")
    if not isinstance(cuda_states, (list, tuple)) or not all(
        torch.is_tensor(cuda_state) for cuda_state in cuda_states
    ):
        raise IncompleteCheckpointError(
            "checkpoint torch CUDA RNG state must be a list of tensors"
        )
    if mps_state is not None and not torch.is_tensor(mps_state):
        raise IncompleteCheckpointError(
            "checkpoint torch MPS RNG state must be a tensor or null"
        )
    if device.type not in {"cpu", "cuda", "mps"}:
        raise CheckpointCompatibilityError(
            "strict RNG resume currently supports cpu/cuda/mps devices, "
            f"got {device.type!r}"
        )

    cuda_states = [cuda_state.cpu() for cuda_state in cuda_states]
    if cuda_states:
        if not torch.cuda.is_available():
            raise CheckpointCompatibilityError(
                "checkpoint contains CUDA RNG state, but CUDA is unavailable"
            )
        current_count = torch.cuda.device_count()
        if len(cuda_states) != current_count:
            raise CheckpointCompatibilityError(
                "CUDA device count changed since checkpoint creation: "
                f"checkpoint={len(cuda_states)} current={current_count}"
            )
    elif device.type == "cuda":
        raise CheckpointCompatibilityError(
            "checkpoint has no CUDA RNG state but resume device is CUDA"
        )

    if mps_state is not None:
        if not torch.backends.mps.is_available():
            raise CheckpointCompatibilityError(
                "checkpoint contains MPS RNG state, but MPS is unavailable"
            )
        mps_state = mps_state.cpu()
    elif device.type == "mps":
        raise CheckpointCompatibilityError(
            "checkpoint has no MPS RNG state but resume device is MPS"
        )
    return python_state, cpu_state.cpu(), cuda_states, mps_state


def restore_rng_state(
    state: Mapping[str, Any],
    device: torch.device | str = "cpu",
) -> None:
    """Restore a state produced by :func:`capture_rng_state`.

    Validation happens before any generator is mutated.  CUDA checkpoints are
    strict about device count because dropping or inventing generator states
    would make a purported exact resume nondeterministic.
    """

    target_device = torch.device(device)
    python_state, cpu_state, cuda_states, mps_state = _validate_rng_state(
        state,
        target_device,
    )
    try:
        random.setstate(python_state)
    except (TypeError, ValueError) as exc:
        raise IncompleteCheckpointError("checkpoint Python RNG state is invalid") from exc
    torch.set_rng_state(cpu_state)
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)
    if mps_state is not None:
        torch.mps.set_rng_state(mps_state)


def atomic_torch_save(payload: Any, path: str | os.PathLike[str]) -> None:
    """Atomically replace ``path`` with a torch-serialized payload.

    The temporary file lives beside the destination so ``os.replace`` remains
    atomic.  It is removed on both serialization and replacement failures.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def _plain_identity(value: Any, *, location: str = "run_identity") -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {
            str(key): _plain_identity(item, location=f"{location}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _plain_identity(item, location=f"{location}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{location} must contain only paths, primitives, lists, and mappings; "
        f"got {type(value).__name__}"
    )


def _validated_training_state(state: Mapping[str, Any]) -> dict[str, Any]:
    missing = _REQUIRED_TRAINING_STATE_KEYS - set(state)
    if missing:
        raise IncompleteCheckpointError(
            "checkpoint training_state is incomplete; missing: "
            + ", ".join(sorted(missing))
        )
    validated = dict(state)
    for key in _REQUIRED_TRAINING_STATE_KEYS:
        value = validated[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise IncompleteCheckpointError(
                f"checkpoint training_state.{key} must be a non-negative integer"
            )
    if validated["micro_step"] != 0:
        raise IncompleteCheckpointError(
            "strict checkpoints may only be saved at an optimizer boundary "
            "(training_state.micro_step must be 0)"
        )
    if validated["optimizer_step"] > validated["planned_total_steps"]:
        raise IncompleteCheckpointError(
            "training_state.optimizer_step exceeds planned_total_steps"
        )
    return validated


def gather_rank_resume_states(
    training_state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Collect rank-local RNG and progress at a shared optimizer boundary.

    All ranks must call this function. The complete list is returned on every
    worker so rank zero can write one payload while failures remain visible to
    the whole process group.
    """

    local_state = {
        "rng_state": capture_rng_state(),
        "training_state": _validated_training_state(training_state),
    }
    world_size = _world_size()
    if world_size == 1:
        return [local_state]
    if not dist.is_available() or not dist.is_initialized():
        raise ResumeCheckpointError(
            "distributed checkpoint capture requires an initialized process group"
        )

    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, local_state)
    validated = []
    for rank, state in enumerate(gathered):
        if not isinstance(state, Mapping):
            raise IncompleteCheckpointError(
                f"rank {rank} resume state must be a mapping"
            )
        rng_state = state.get("rng_state")
        progress = state.get("training_state")
        if not isinstance(rng_state, Mapping):
            raise IncompleteCheckpointError(
                f"rank {rank} RNG state must be a mapping"
            )
        if not isinstance(progress, Mapping):
            raise IncompleteCheckpointError(
                f"rank {rank} training state must be a mapping"
            )
        validated.append(
            {
                "rng_state": dict(rng_state),
                "training_state": _validated_training_state(progress),
            }
        )
    return validated


def build_training_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    stage: str,
    run_identity: Mapping[str, Any],
    training_state: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete v2 payload without writing it.

    Callers must pass an unwrapped model. This low-level builder remains a
    single-process convenience; distributed trainers first collect rank-local
    state with :func:`gather_rank_resume_states`.
    """

    _require_single_process("v2 checkpoint construction")
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a non-empty string")
    if not isinstance(run_identity, Mapping):
        raise TypeError("run_identity must be a mapping")
    if not isinstance(training_state, Mapping):
        raise TypeError("training_state must be a mapping")
    if scaler is None or not hasattr(scaler, "state_dict"):
        raise TypeError("scaler must provide state_dict/load_state_dict")

    payload: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "stage": stage,
        "run_identity": _plain_identity(run_identity),
        "world_size": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "training_state": _validated_training_state(training_state),
    }
    if extra is not None:
        payload["extra"] = dict(extra)
    return payload


def save_training_checkpoint(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    stage: str,
    run_identity: Mapping[str, Any],
    training_state: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Build and atomically save a complete v2 training checkpoint."""

    payload = build_training_checkpoint(
        model,
        optimizer,
        scaler,
        stage=stage,
        run_identity=run_identity,
        training_state=training_state,
        extra=extra,
    )
    atomic_torch_save(payload, path)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def optimizer_to(optimizer: torch.optim.Optimizer, device: torch.device | str) -> None:
    """Move every tensor nested in optimizer state to ``device`` in place."""

    target_device = torch.device(device)
    for parameter, state in list(optimizer.state.items()):
        optimizer.state[parameter] = _move_to_device(state, target_device)


def _load_payload(path: str | os.PathLike[str]) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise ResumeCheckpointError(f"could not load resume checkpoint {path!s}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LegacyCheckpointError(
            "checkpoint is not a resumable v2 payload; use --init_from for legacy weights"
        )
    version = payload.get("format_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < FORMAT_VERSION:
        raise LegacyCheckpointError(
            "raw and format_version 0/1 checkpoints are weight initializers, not complete "
            "resume checkpoints; use --init_from instead"
        )
    missing = _REQUIRED_PAYLOAD_KEYS - set(payload)
    if missing:
        raise IncompleteCheckpointError(
            "v2 checkpoint is incomplete; missing: " + ", ".join(sorted(missing))
        )
    world_size = payload.get("world_size")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise IncompleteCheckpointError(
            "checkpoint world_size must be a positive integer"
        )
    if world_size == 1:
        if not isinstance(payload.get("rng_state"), Mapping):
            raise IncompleteCheckpointError(
                "single-process checkpoint rng_state must be a mapping"
            )
    else:
        rng_states = payload.get("rng_state_by_rank")
        training_states = payload.get("training_state_by_rank")
        if not isinstance(rng_states, (list, tuple)) or len(rng_states) != world_size:
            raise IncompleteCheckpointError(
                "distributed checkpoint rng_state_by_rank must contain one state per rank"
            )
        if (
            not isinstance(training_states, (list, tuple))
            or len(training_states) != world_size
        ):
            raise IncompleteCheckpointError(
                "distributed checkpoint training_state_by_rank must contain "
                "one state per rank"
            )
    return payload


def _rank_resume_state(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    checkpoint_world_size = int(payload["world_size"])
    runtime_world_size = _world_size()
    runtime_rank = _rank()
    if checkpoint_world_size != runtime_world_size:
        raise CheckpointCompatibilityError(
            "checkpoint world size mismatch: "
            f"checkpoint={checkpoint_world_size} current={runtime_world_size}"
        )
    if checkpoint_world_size == 1:
        return payload["rng_state"], payload["training_state"]

    rng_state = payload["rng_state_by_rank"][runtime_rank]
    training_state = payload["training_state_by_rank"][runtime_rank]
    if not isinstance(rng_state, Mapping):
        raise IncompleteCheckpointError(
            f"checkpoint RNG state for rank {runtime_rank} must be a mapping"
        )
    if not isinstance(training_state, Mapping):
        raise IncompleteCheckpointError(
            f"checkpoint training state for rank {runtime_rank} must be a mapping"
        )
    return rng_state, training_state


def load_training_state(
    path: str | os.PathLike[str],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    device: torch.device | str,
    expected_stage: str,
    expected_run_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly restore a complete v2 checkpoint and return its progress state.

    Compatibility checks are performed before model, optimizer, scaler, or RNG
    mutation.  The model state uses ``strict=True`` and optimizer tensors are
    moved from the CPU load location to the requested runtime device.
    """

    target_device = torch.device(device)
    payload = _load_payload(path)

    actual_stage = payload["stage"]
    if actual_stage != expected_stage:
        raise CheckpointCompatibilityError(
            f"checkpoint stage mismatch: expected {expected_stage!r}, got {actual_stage!r}"
        )
    if not isinstance(expected_run_identity, Mapping):
        raise TypeError("expected_run_identity must be a mapping")
    actual_identity = payload["run_identity"]
    if not isinstance(actual_identity, Mapping):
        raise IncompleteCheckpointError("checkpoint run_identity must be a mapping")
    normalized_expected = _plain_identity(expected_run_identity)
    normalized_actual = _plain_identity(actual_identity)
    if normalized_actual != normalized_expected:
        raise CheckpointCompatibilityError(
            "checkpoint run identity mismatch: "
            f"expected {normalized_expected!r}, got {normalized_actual!r}"
        )
    model_state = payload["model_state_dict"]
    optimizer_state = payload["optimizer_state_dict"]
    scaler_state = payload["scaler_state_dict"]
    rng_state, training_state = _rank_resume_state(payload)
    if not isinstance(model_state, Mapping):
        raise IncompleteCheckpointError("checkpoint model_state_dict must be a mapping")
    if not isinstance(optimizer_state, Mapping):
        raise IncompleteCheckpointError("checkpoint optimizer_state_dict must be a mapping")
    if not isinstance(scaler_state, Mapping):
        raise IncompleteCheckpointError("checkpoint scaler_state_dict must be a mapping")
    if not isinstance(rng_state, Mapping):
        raise IncompleteCheckpointError("checkpoint rng_state must be a mapping")
    if not isinstance(training_state, Mapping):
        raise IncompleteCheckpointError("checkpoint training_state must be a mapping")
    validated_state = _validated_training_state(training_state)
    local_identity = validated_state.get("run_identity")
    if local_identity is not None and _plain_identity(local_identity) != normalized_expected:
        raise CheckpointCompatibilityError(
            f"checkpoint rank {_rank()} training state has a mismatched run identity"
        )
    _validate_rng_state(rng_state, target_device)
    if scaler is None or not hasattr(scaler, "load_state_dict"):
        raise TypeError("scaler must provide state_dict/load_state_dict")

    model.to(target_device)
    try:
        model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint model state is incompatible with the runtime model: {exc}"
        ) from exc
    try:
        optimizer.load_state_dict(optimizer_state)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint optimizer state is incompatible: {exc}"
        ) from exc
    optimizer_to(optimizer, target_device)
    try:
        scaler.load_state_dict(dict(scaler_state))
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(
            f"checkpoint scaler state is incompatible: {exc}"
        ) from exc
    restore_rng_state(rng_state, target_device)
    return validated_state
