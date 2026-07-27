import random

import pytest
import torch
from torch import nn

from trainer.checkpointing import (
    CheckpointCompatibilityError,
    IncompleteCheckpointError,
    LegacyCheckpointError,
    atomic_torch_save,
    capture_rng_state,
    load_training_state,
    restore_rng_state,
    save_training_checkpoint,
)


class _TestScaler:
    def __init__(self, scale=128.0):
        self.scale = float(scale)

    def state_dict(self):
        return {"scale": self.scale}

    def load_state_dict(self, state):
        self.scale = float(state["scale"])


def _progress(**overrides):
    state = {
        "epoch": 1,
        "batch_in_epoch": 2,
        "optimizer_step": 3,
        "micro_step": 0,
        "planned_total_steps": 10,
    }
    state.update(overrides)
    return state


def _model_and_optimizer():
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.randn(5, 3)).square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def test_v2_roundtrip_restores_model_optimizer_scaler_progress_and_rng(tmp_path):
    random.seed(1234)
    torch.manual_seed(5678)
    model, optimizer = _model_and_optimizer()
    scaler = _TestScaler(256.0)
    identity = {"dataset": "demo.jsonl", "config": {"hidden_size": 4}}
    path = tmp_path / "resume.pth"

    expected_model = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    expected_exp_avg = next(iter(optimizer.state.values()))["exp_avg"].detach().clone()
    save_training_checkpoint(
        path,
        model,
        optimizer,
        scaler,
        stage="sft",
        run_identity=identity,
        training_state=_progress(),
    )
    expected_python = random.random()
    expected_torch = torch.rand(4)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    for state in optimizer.state.values():
        state["exp_avg"].zero_()
    scaler.scale = 1.0
    random.seed(9)
    torch.manual_seed(9)

    restored = load_training_state(
        path,
        model,
        optimizer,
        scaler,
        torch.device("cpu"),
        "sft",
        identity,
    )

    assert restored == _progress()
    assert scaler.scale == pytest.approx(256.0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected_model[name])
    restored_exp_avg = next(iter(optimizer.state.values()))["exp_avg"]
    torch.testing.assert_close(restored_exp_avg, expected_exp_avg)
    assert all(
        value.device.type == "cpu"
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    assert random.random() == pytest.approx(expected_python)
    torch.testing.assert_close(torch.rand(4), expected_torch)
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_capture_and_restore_rng_replays_python_and_torch_streams():
    random.seed(17)
    torch.manual_seed(23)
    state = capture_rng_state()
    assert isinstance(state["torch_cuda"], list)
    assert state["torch_mps"] is None or torch.is_tensor(state["torch_mps"])

    expected_python = [random.random() for _ in range(3)]
    expected_torch = torch.rand(3)
    random.seed(999)
    torch.manual_seed(999)

    restore_rng_state(state)
    assert [random.random() for _ in range(3)] == pytest.approx(expected_python)
    torch.testing.assert_close(torch.rand(3), expected_torch)


def test_capture_and_restore_rng_supports_mps_state(monkeypatch):
    mps_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    restored = []
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda: mps_state.clone())
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda state: restored.append(state))

    state = capture_rng_state()
    torch.testing.assert_close(state["torch_mps"], mps_state)
    restore_rng_state(state, torch.device("mps"))

    assert len(restored) == 1
    torch.testing.assert_close(restored[0], mps_state)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_capture_and_restore_rng_replays_cuda_stream():
    torch.cuda.manual_seed_all(31)
    state = capture_rng_state()
    expected = torch.rand(4, device="cuda")
    torch.cuda.manual_seed_all(99)

    restore_rng_state(state, torch.device("cuda"))
    torch.testing.assert_close(torch.rand(4, device="cuda"), expected)


@pytest.mark.parametrize(
    "payload",
    [
        {"weight": torch.ones(1)},
        {
            "format_version": 1,
            "model_state_dict": {"weight": torch.ones(1)},
            "optimizer_state_dict": {},
        },
    ],
    ids=("raw-state-dict", "v1"),
)
def test_resume_rejects_legacy_checkpoints_with_init_from_guidance(tmp_path, payload):
    path = tmp_path / "legacy.pth"
    torch.save(payload, path)
    model = nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())

    with pytest.raises(LegacyCheckpointError, match="--init_from"):
        load_training_state(
            path,
            model,
            optimizer,
            _TestScaler(),
            "cpu",
            "pretrain",
            {"dataset": "demo"},
        )


@pytest.mark.parametrize(
    ("expected_stage", "expected_identity", "message"),
    [
        ("dpo", {"dataset": "demo"}, "stage mismatch"),
        ("sft", {"dataset": "other"}, "identity mismatch"),
    ],
)
def test_resume_rejects_stage_and_identity_mismatch_before_mutation(
    tmp_path,
    expected_stage,
    expected_identity,
    message,
):
    model, optimizer = _model_and_optimizer()
    path = tmp_path / "resume.pth"
    save_training_checkpoint(
        path,
        model,
        optimizer,
        _TestScaler(),
        stage="sft",
        run_identity={"dataset": "demo"},
        training_state=_progress(),
    )
    with torch.no_grad():
        model[0].weight.fill_(123)
    before = model[0].weight.detach().clone()

    with pytest.raises(CheckpointCompatibilityError, match=message):
        load_training_state(
            path,
            model,
            optimizer,
            _TestScaler(),
            "cpu",
            expected_stage,
            expected_identity,
        )
    torch.testing.assert_close(model[0].weight, before)


def test_resume_rejects_incomplete_v2_training_state(tmp_path):
    model, optimizer = _model_and_optimizer()
    path = tmp_path / "incomplete.pth"
    save_training_checkpoint(
        path,
        model,
        optimizer,
        _TestScaler(),
        stage="sft",
        run_identity={"dataset": "demo"},
        training_state=_progress(),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    del payload["training_state"]["batch_in_epoch"]
    torch.save(payload, path)

    with pytest.raises(IncompleteCheckpointError, match="batch_in_epoch"):
        load_training_state(
            path,
            model,
            optimizer,
            _TestScaler(),
            "cpu",
            "sft",
            {"dataset": "demo"},
        )


def test_atomic_save_cleans_temporary_file_after_serialization_failure(tmp_path, monkeypatch):
    path = tmp_path / "failed.pth"

    def fail_save(*_args, **_kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="serialization failed"):
        atomic_torch_save({"value": 1}, path)

    assert not path.exists()
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
