import torch

from scripts.verify_ddp_resume import SCHEMA_NAME, run_verification


def test_two_rank_ddp_resume_matches_uninterrupted_training(tmp_path):
    artifact = run_verification(
        tmp_path / "run",
        tmp_path / "ddp_resume.json",
    )

    assert artifact["schema"] == SCHEMA_NAME
    assert artifact["environment"]["world_size"] == 2
    assert artifact["checkpoint"] == {
        "format_version": 2,
        "world_size": 2,
        "rank_rng_states": 2,
        "rank_training_states": 2,
        "rank_rng_streams_distinct": True,
    }
    assert all(artifact["checks"].values())

    checkpoint = torch.load(
        tmp_path / "run" / "interrupted_step_2.pth",
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["training_state"]["optimizer_step"] == 2
    assert len(checkpoint["rng_state_by_rank"]) == 2
    assert len(checkpoint["training_state_by_rank"]) == 2
