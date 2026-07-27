import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer

import scripts.evaluate_sft as evaluation
from dataset.lm_dataset import SFTDataset, split_supervised_dataset
from scripts.evaluate_sft import (
    SCHEMA_NAME,
    aggregate_matches,
    conservative_normalize,
    evaluate_sft,
    extract_last_assistant_target,
    write_evaluation_json,
)


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_extracts_history_before_last_assistant_and_ignores_later_suffix():
    history, expected = extract_last_assistant_target(
        {
            "conversations": [
                {"role": "system", "content": "Be concise.", "ignored": 1},
                {"role": "user", "content": "First question"},
                {"role": "assistant", "content": "First answer"},
                {"role": "user", "content": "Second question"},
                {"role": "assistant", "content": "Final target"},
                {"role": "user", "content": "Unused suffix"},
            ]
        }
    )

    assert expected == "Final target"
    assert history == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]
    with pytest.raises(ValueError, match="no assistant target"):
        extract_last_assistant_target(
            {"conversations": [{"role": "user", "content": "No answer"}]}
        )
    with pytest.raises(ValueError, match="preceding prompt/history"):
        extract_last_assistant_target(
            {"conversations": [{"role": "assistant", "content": "No prompt"}]}
        )


def test_conservative_normalization_does_not_add_task_specific_equivalence():
    assert conservative_normalize("\r\n Cafe\u0301 \r\n") == "Café"
    assert conservative_normalize(" answer \n") == "answer"
    assert conservative_normalize("two  spaces") != conservative_normalize("two spaces")
    assert conservative_normalize("Four.") != conservative_normalize("four")
    assert conservative_normalize("4") != conservative_normalize("four")
    with pytest.raises(TypeError, match="string"):
        conservative_normalize(4)


def test_match_aggregate_reports_counts_and_rates():
    samples = [
        {"strict_exact_match": True, "normalized_exact_match": True},
        {"strict_exact_match": False, "normalized_exact_match": True},
        {"strict_exact_match": False, "normalized_exact_match": False},
    ]
    assert aggregate_matches(samples) == {
        "evaluated_samples": 3,
        "strict_exact_matches": 1,
        "strict_exact_match_rate": pytest.approx(1 / 3),
        "normalized_exact_matches": 2,
        "normalized_exact_match_rate": pytest.approx(2 / 3),
    }
    assert aggregate_matches([]) == {
        "evaluated_samples": 0,
        "strict_exact_matches": 0,
        "strict_exact_match_rate": 0.0,
        "normalized_exact_matches": 0,
        "normalized_exact_match_rate": 0.0,
    }


def test_evaluator_reuses_validation_split_loads_once_and_writes_strict_json(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("tokenizer", local_files_only=True)
    rows = [
        {
            "conversations": [
                {"role": "user", "content": f"held-out question {index}"},
                {"role": "assistant", "content": f"expected answer {index}"},
            ]
        }
        for index in range(8)
    ]
    data_path = tmp_path / "sft.jsonl"
    write_jsonl(data_path, rows)
    checkpoint = tmp_path / "fake_checkpoint.pth"
    output = tmp_path / "evaluation.json"

    load_calls = []

    def fake_loader(**kwargs):
        load_calls.append(kwargs)
        return SimpleNamespace(
            model=object(),
            tokenizer=tokenizer,
            device=torch.device("cpu"),
            source="fake checkpoint for unit test",
        )

    generation_calls = []

    def fake_generator(model, used_tokenizer, prompt, config):
        assert used_tokenizer is tokenizer
        assert config.temperature == 0
        assert config.top_k == 0
        assert config.use_cache is True
        assert config.max_new_tokens == 12
        matching_index = next(
            index for index in range(8) if f"held-out question {index}" in prompt
        )
        expected = f"expected answer {matching_index}"
        generation_calls.append((model, prompt, config))
        return f" \r\n{expected}\r\n " if len(generation_calls) == 1 else expected

    payload = evaluate_sft(
        checkpoint=checkpoint,
        tokenizer_dir="tokenizer",
        data_path=data_path,
        max_length=64,
        validation_fraction=0.5,
        split_seed=23,
        device="cpu",
        max_new_tokens=12,
        limit=3,
        output=output,
        model_loader=fake_loader,
        completion_generator=fake_generator,
    )

    assert len(load_calls) == 1
    assert load_calls[0] == {
        "tokenizer_dir": "tokenizer",
        "checkpoint": checkpoint,
        "device": "cpu",
    }
    assert len(generation_calls) == 3
    expected_dataset = SFTDataset(data_path, tokenizer, max_length=64)
    _, expected_validation, expected_split = split_supervised_dataset(
        expected_dataset,
        validation_fraction=0.5,
        seed=23,
    )
    assert payload["schema"] == SCHEMA_NAME
    assert payload["schema_version"] == 1
    assert payload["split_metadata"] == expected_split
    assert [sample["source_index"] for sample in payload["samples"]] == list(
        expected_validation.indices[:3]
    )
    assert all(sample["history"][-1]["role"] == "user" for sample in payload["samples"])
    assert all("<|im_start|>assistant\n" in sample["prompt"] for sample in payload["samples"])
    assert payload["aggregate"] == {
        "evaluated_samples": 3,
        "strict_exact_matches": 2,
        "strict_exact_match_rate": pytest.approx(2 / 3),
        "normalized_exact_matches": 3,
        "normalized_exact_match_rate": 1.0,
    }
    assert payload["checkpoint"] == {
        "path": str(checkpoint.resolve()),
        "source": "fake checkpoint for unit test",
    }
    assert payload["decoding"] == {
        "strategy": "greedy",
        "temperature": 0.0,
        "top_k": 0,
        "use_cache": True,
        "max_new_tokens": 12,
        "limit": 3,
    }
    raw = output.read_text(encoding="utf-8")
    assert json.loads(raw) == payload
    assert "NaN" not in raw
    assert "Infinity" not in raw
    assert not list(tmp_path.glob(".*.tmp"))


def test_atomic_json_overwrite_and_non_finite_failure_preserve_valid_artifact(tmp_path):
    output = tmp_path / "evaluation.json"
    write_evaluation_json(output, {"schema_version": 1, "value": 1.0})
    write_evaluation_json(output, {"schema_version": 1, "value": 2.0})
    valid_content = output.read_text(encoding="utf-8")
    assert json.loads(valid_content)["value"] == 2.0

    with pytest.raises(ValueError, match="Out of range float values"):
        write_evaluation_json(output, {"value": float("nan")})

    assert output.read_text(encoding="utf-8") == valid_content
    assert not list(tmp_path.glob(".*.tmp"))


def test_failed_atomic_replace_cleans_temporary_file(tmp_path, monkeypatch):
    output = tmp_path / "evaluation.json"

    def fail_replace(source, destination):
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(evaluation.os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        write_evaluation_json(output, {"schema_version": 1})

    assert not list(tmp_path.iterdir())
