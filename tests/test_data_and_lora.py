import json

import pytest
import torch
from transformers import AutoTokenizer

from dataset.lm_dataset import (
    DPODataset,
    IndexedDataset,
    PretrainDataset,
    SFTDataset,
    rl_collate,
    split_supervised_dataset,
)
from model import model_lora
from model.model import NinjaMindConfig, NinjaMindForCausalLM
from model.model_lora import (
    apply_lora,
    freeze_non_lora,
    load_lora,
    merge_lora,
    save_lora,
)


def tiny_model():
    config = NinjaMindConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=64,
        flash_attn=False,
    )
    return NinjaMindForCausalLM(config).eval()


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_tokenizer_special_ids_and_chat_template():
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    assert tokenizer.pad_token_id == tokenizer.unk_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2

    rendered = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "4"},
        ],
        tokenize=False,
    )
    assert "<|im_start|>user\n2+2?<|im_end|>" in rendered
    assert "<|im_start|>assistant\n4<|im_end|>" in rendered


def test_dataset_shapes_and_sft_masks_only_assistant_response():
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    pretrain = PretrainDataset("dataset/demo/pretrain_demo.jsonl", tokenizer, max_length=96)
    sft = SFTDataset("dataset/demo/sft_demo.jsonl", tokenizer, max_length=96)
    dpo = DPODataset("dataset/demo/dpo_demo.jsonl", tokenizer, max_length=96)

    assert all(tensor.shape == (95,) for tensor in pretrain[0])
    _, targets, mask = sft[0]
    assert mask.sum() > 0
    supervised = tokenizer.decode(targets[mask.bool()].tolist(), skip_special_tokens=False)
    assert supervised == "0 plus 0 equals 0.<|im_end|>"
    assert "What is" not in supervised

    preference = dpo[0]
    assert set(preference) == {
        "x_chosen",
        "y_chosen",
        "mask_chosen",
        "x_rejected",
        "y_rejected",
        "mask_rejected",
    }
    assert preference["mask_chosen"].sum() > 0
    assert preference["mask_rejected"].sum() > 0


def test_rl_collate_left_truncates_and_restores_tokenizer_state():
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    prompt = "old context " * 40 + "<|im_start|>assistant\n"
    full_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    assert len(full_ids) > 16

    batch = rl_collate(
        [{"prompt": prompt, "answer": "demo"}],
        tokenizer,
        max_prompt_len=16,
    )

    assert batch["input_ids"][0].tolist() == full_ids[-16:]
    assert batch["attention_mask"][0].tolist() == [1] * 16
    assert batch["answer"] == ["demo"]
    assert tokenizer.padding_side == "right"
    assert tokenizer.truncation_side == "right"


def test_sft_left_truncation_preserves_latest_assistant_response(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    path = tmp_path / "long_sft.jsonl"
    sample = {
        "conversations": [
            {"role": "user", "content": "discard this old context " * 40},
            {"role": "assistant", "content": "The answer is four."},
        ]
    }
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    _, targets, mask = SFTDataset(path, tokenizer, max_length=32)[0]

    supervised = tokenizer.decode(targets[mask.bool()].tolist(), skip_special_tokens=False)
    assert supervised == "The answer is four.<|im_end|>"
    assert "old context" not in supervised


def test_pretrain_split_exact_dedup_is_stable_and_preserves_dataset_metadata(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    rows = [
        {"text": "alpha", "source": "first"},
        {"text": "beta"},
        {"text": "gamma"},
        {"text": "alpha", "source": "ignored duplicate metadata"},
        {"text": "delta"},
        {"text": "epsilon"},
    ]
    path = tmp_path / "pretrain.jsonl"
    write_jsonl(path, rows)
    dataset = PretrainDataset(path, tokenizer, max_length=24)

    train, validation, metadata = split_supervised_dataset(
        dataset,
        validation_fraction=0.4,
        seed=17,
    )
    train_again, validation_again, metadata_again = split_supervised_dataset(
        dataset,
        validation_fraction=0.4,
        seed=17,
    )

    assert isinstance(train, IndexedDataset)
    assert isinstance(validation, IndexedDataset)
    assert train.pad_id == validation.pad_id == dataset.pad_id
    assert train.indices == train_again.indices
    assert validation.indices == validation_again.indices
    assert metadata == metadata_again
    assert metadata == {
        "schema_version": 1,
        "algorithm": "sha256-group-prefix-v1",
        "deduplication": "encoded-supervised-tensors-exact-sha256-v2",
        "stage": "pretrain",
        "source_path": str(path),
        "max_length": 24,
        "seed": 17,
        "requested_validation_fraction": 0.4,
        "actual_validation_fraction": 0.4,
        "raw_samples": 6,
        "unique_samples": 5,
        "duplicates_removed": 1,
        "group_count": 5,
        "train_samples": 3,
        "validation_samples": 2,
        "train_groups": 3,
        "validation_groups": 2,
        "split_fingerprint": metadata["split_fingerprint"],
    }
    assert metadata["split_fingerprint"].startswith("sha256:")
    assert 3 not in train.indices + validation.indices

    train_keys = {dataset.dedup_key(dataset.samples[index]) for index in train.indices}
    validation_keys = {
        dataset.dedup_key(dataset.samples[index]) for index in validation.indices
    }
    assert train_keys.isdisjoint(validation_keys)
    for tensor, expected in zip(train[0], dataset[train.indices[0]], strict=True):
        assert torch.equal(tensor, expected)

    # Hash-based assignment and fingerprint depend on content rather than row
    # order or ignored metadata.
    reordered_path = tmp_path / "pretrain_reordered.jsonl"
    write_jsonl(reordered_path, list(reversed(rows)))
    reordered = PretrainDataset(reordered_path, tokenizer, max_length=24)
    reordered_train, reordered_validation, reordered_metadata = split_supervised_dataset(
        reordered,
        validation_fraction=0.4,
        seed=17,
    )
    assert reordered_metadata["split_fingerprint"] == metadata["split_fingerprint"]
    assert [
        dataset.dedup_key(dataset.samples[index]) for index in train.indices
    ] == [
        reordered.dedup_key(reordered.samples[index])
        for index in reordered_train.indices
    ]
    assert [
        dataset.dedup_key(dataset.samples[index]) for index in validation.indices
    ] == [
        reordered.dedup_key(reordered.samples[index])
        for index in reordered_validation.indices
    ]


def test_sft_split_groups_alternative_answers_and_ignores_unused_metadata(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    same_user = {"role": "user", "content": "What is two plus two?"}
    answer_four = {"role": "assistant", "content": "Four."}
    rows = [
        {"conversations": [same_user, answer_four], "record_id": 1},
        {
            "conversations": [
                same_user,
                {"role": "assistant", "content": "The answer is 4."},
            ]
        },
        {
            "conversations": [
                {"content": "What is two plus two?", "role": "user", "trace": "ignored"},
                {"content": "Four.", "role": "assistant", "score": 1},
            ],
            "record_id": 999,
        },
        {
            "conversations": [
                {"role": "user", "content": "Name a primary colour."},
                {"role": "assistant", "content": "Red."},
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "Name a shape."},
                {"role": "assistant", "content": "Circle."},
            ]
        },
    ]
    path = tmp_path / "sft.jsonl"
    write_jsonl(path, rows)
    dataset = SFTDataset(path, tokenizer, max_length=64)

    train, validation, metadata = split_supervised_dataset(
        dataset,
        validation_fraction=0.5,
        seed=42,
    )
    train_indices = set(train.indices)
    validation_indices = set(validation.indices)

    assert dataset.dedup_key(dataset.samples[0]) == dataset.dedup_key(dataset.samples[2])
    assert dataset.split_group_key(dataset.samples[0]) == dataset.split_group_key(
        dataset.samples[1]
    )
    assert (0 in train_indices) == (1 in train_indices)
    assert (0 in validation_indices) == (1 in validation_indices)
    assert 2 not in train_indices | validation_indices
    train_group_keys = {
        dataset.split_group_key(dataset.samples[index]) for index in train.indices
    }
    validation_group_keys = {
        dataset.split_group_key(dataset.samples[index]) for index in validation.indices
    }
    assert train_group_keys.isdisjoint(validation_group_keys)
    assert metadata["raw_samples"] == 5
    assert metadata["unique_samples"] == 4
    assert metadata["duplicates_removed"] == 1
    assert metadata["group_count"] == 3
    assert metadata["train_groups"] + metadata["validation_groups"] == 3
    assert metadata["train_samples"] and metadata["validation_samples"]


def test_pretrain_split_deduplicates_text_that_differs_only_after_truncation(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    visible_prefix = "same visible prefix " * 20
    rows = [
        {"text": visible_prefix + "discarded tail A"},
        {"text": "another training example"},
        {"text": visible_prefix + "discarded tail B"},
        {"text": "a third training example"},
    ]
    path = tmp_path / "truncated_pretrain.jsonl"
    write_jsonl(path, rows)
    dataset = PretrainDataset(path, tokenizer, max_length=24)

    train, validation, metadata = split_supervised_dataset(
        dataset,
        validation_fraction=0.5,
        seed=9,
    )

    assert dataset.dedup_key(rows[0]) == dataset.dedup_key(rows[2])
    assert all(
        torch.equal(left, right)
        for left, right in zip(dataset[0], dataset[2], strict=True)
    )
    assert metadata["raw_samples"] == 4
    assert metadata["unique_samples"] == 3
    assert metadata["duplicates_removed"] == 1
    assert not ({0, 2} <= set(train.indices) | set(validation.indices))


def test_sft_split_uses_visible_truncated_tensors_and_prompt_skeleton(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    common_prompt = {"role": "user", "content": "What is the final answer?"}
    rows = [
        {
            "conversations": [
                {"role": "user", "content": "old alpha context " * 80},
                common_prompt,
                {"role": "assistant", "content": "First answer."},
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "old beta context " * 80},
                common_prompt,
                {"role": "assistant", "content": "First answer."},
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "old gamma context " * 80},
                common_prompt,
                {"role": "assistant", "content": "A different answer."},
            ]
        },
        {
            "conversations": [
                {"role": "user", "content": "independent prompt"},
                {"role": "assistant", "content": "Independent answer."},
            ]
        },
    ]
    path = tmp_path / "truncated_sft.jsonl"
    write_jsonl(path, rows)
    dataset = SFTDataset(path, tokenizer, max_length=48)

    train, validation, metadata = split_supervised_dataset(
        dataset,
        validation_fraction=0.5,
        seed=5,
    )

    # Rows 0/1 become the exact same (X, Y, mask) after left truncation, so
    # only one representative can occur in either split.
    assert dataset.dedup_key(rows[0]) == dataset.dedup_key(rows[1])
    assert all(
        torch.equal(left, right)
        for left, right in zip(dataset[0], dataset[1], strict=True)
    )
    retained = set(train.indices) | set(validation.indices)
    assert not ({0, 1} <= retained)
    assert metadata["duplicates_removed"] == 1

    # The old context is outside the encoded skeleton and assistant content is
    # represented structurally, so the alternative answer remains in one group.
    group_keys = [dataset.split_group_key(row) for row in rows[:3]]
    assert len(set(group_keys)) == 1
    retained_prompt_variants = {0, 1, 2} & retained
    assert retained_prompt_variants <= set(train.indices) or retained_prompt_variants <= set(
        validation.indices
    )

    reordered_path = tmp_path / "truncated_sft_reordered.jsonl"
    write_jsonl(reordered_path, list(reversed(rows)))
    reordered = SFTDataset(reordered_path, tokenizer, max_length=48)
    reordered_train, reordered_validation, reordered_metadata = split_supervised_dataset(
        reordered,
        validation_fraction=0.5,
        seed=5,
    )
    assert reordered_metadata["split_fingerprint"] == metadata["split_fingerprint"]
    assert [dataset.dedup_key(dataset.samples[index]) for index in train.indices] == [
        reordered.dedup_key(reordered.samples[index]) for index in reordered_train.indices
    ]
    assert [
        dataset.dedup_key(dataset.samples[index]) for index in validation.indices
    ] == [
        reordered.dedup_key(reordered.samples[index])
        for index in reordered_validation.indices
    ]


def test_supervised_split_validates_fraction_groups_and_schema(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    pretrain_path = tmp_path / "two_groups.jsonl"
    write_jsonl(pretrain_path, [{"text": "left"}, {"text": "right"}, {"text": "left"}])
    pretrain = PretrainDataset(pretrain_path, tokenizer, max_length=16)

    train, validation, metadata = split_supervised_dataset(
        pretrain,
        validation_fraction=0,
        seed=1,
    )
    assert validation is None
    assert len(train) == 2
    assert metadata["duplicates_removed"] == 1

    train, validation, _ = split_supervised_dataset(
        pretrain,
        validation_fraction=0.99,
        seed=1,
    )
    assert len(train) == len(validation) == 1

    for invalid_fraction in (-0.1, 1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="validation_fraction"):
            split_supervised_dataset(
                pretrain,
                validation_fraction=invalid_fraction,
                seed=1,
            )
    with pytest.raises(TypeError, match="validation_fraction"):
        split_supervised_dataset(pretrain, validation_fraction=True, seed=1)
    with pytest.raises(TypeError, match="seed"):
        split_supervised_dataset(pretrain, validation_fraction=0.5, seed=True)

    one_group_path = tmp_path / "one_sft_group.jsonl"
    write_jsonl(
        one_group_path,
        [
            {
                "conversations": [
                    {"role": "user", "content": "same prompt"},
                    {"role": "assistant", "content": answer},
                ]
            }
            for answer in ("answer one", "answer two")
        ],
    )
    one_group = SFTDataset(one_group_path, tokenizer, max_length=32)
    with pytest.raises(ValueError, match="at least two unique split groups"):
        split_supervised_dataset(one_group, validation_fraction=0.5, seed=1)

    malformed_path = tmp_path / "malformed_sft.jsonl"
    write_jsonl(
        malformed_path,
        [{"conversations": [{"role": "user", "content": 123}]}],
    )
    malformed = SFTDataset(malformed_path, tokenizer, max_length=32)
    with pytest.raises(ValueError, match=r"sample 0.*string role/content"):
        split_supervised_dataset(malformed, validation_fraction=0, seed=1)


def test_dpo_left_truncation_preserves_both_response_masks(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    path = tmp_path / "long_dpo.jsonl"
    old_user = {"role": "user", "content": "discard this old context " * 40}
    sample = {
        "chosen": [old_user, {"role": "assistant", "content": "four"}],
        "rejected": [old_user, {"role": "assistant", "content": "five"}],
    }
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    preference = DPODataset(path, tokenizer, max_length=24)[0]

    chosen = tokenizer.decode(
        preference["y_chosen"][preference["mask_chosen"].bool()].tolist(),
        skip_special_tokens=False,
    )
    rejected = tokenizer.decode(
        preference["y_rejected"][preference["mask_rejected"].bool()].tolist(),
        skip_special_tokens=False,
    )
    assert chosen == "four<|im_end|>\n"
    assert rejected == "five<|im_end|>\n"


def test_dpo_uses_one_shared_truncated_prompt_for_unequal_responses(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    path = tmp_path / "unequal_dpo.jsonl"
    old_user = {"role": "user", "content": "shared context " * 35}
    sample = {
        "chosen": [old_user, {"role": "assistant", "content": "good"}],
        "rejected": [old_user, {"role": "assistant", "content": "bad " * 12}],
    }
    path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

    preference = DPODataset(path, tokenizer, max_length=96)[0]
    chosen_start = int(preference["mask_chosen"].nonzero()[0])
    rejected_start = int(preference["mask_rejected"].nonzero()[0])

    assert chosen_start == rejected_start
    assert torch.equal(
        preference["x_chosen"][:chosen_start],
        preference["x_rejected"][:rejected_start],
    )


def test_lora_initial_output_trainable_params_roundtrip_and_merge(tmp_path):
    torch.manual_seed(7)
    model = tiny_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    with torch.no_grad():
        base_logits = model(input_ids).logits

    apply_lora(model, rank=2, alpha=4)
    trainable = freeze_non_lora(model)
    with torch.no_grad():
        initial_logits = model(input_ids).logits
    torch.testing.assert_close(initial_logits, base_logits, atol=0, rtol=0)
    assert trainable
    assert all(".lora." in name for name, p in model.named_parameters() if p.requires_grad)

    # Make the adapter observable, then verify structured save/load.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("lora.B.weight"):
                parameter.fill_(0.01)
    adapted_logits = model(input_ids).logits.detach()
    path = tmp_path / "adapter.pth"
    save_lora(model, path, metadata={"stage": "test"})
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []

    reloaded = tiny_model()
    reloaded.load_state_dict(
        {k: v for k, v in model.state_dict().items() if ".lora." not in k}, strict=False
    )
    apply_lora(reloaded, rank=2, alpha=4)
    load_lora(reloaded, path)
    torch.testing.assert_close(reloaded(input_ids).logits, adapted_logits)

    merge_lora(reloaded)
    assert not any(".lora." in name for name, _ in reloaded.named_parameters())
    torch.testing.assert_close(reloaded(input_ids).logits, adapted_logits, atol=1e-5, rtol=1e-5)


def test_save_lora_serialization_failure_preserves_existing_adapter(tmp_path, monkeypatch):
    model = tiny_model()
    apply_lora(model, rank=2, alpha=4)
    path = tmp_path / "adapter.pth"
    original = b"existing adapter contents"
    path.write_bytes(original)

    def fail_save(_payload, handle):
        handle.write(b"incomplete replacement")
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="serialization failed"):
        save_lora(model, path)

    assert path.read_bytes() == original
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []


def test_save_lora_replace_failure_preserves_existing_adapter(tmp_path, monkeypatch):
    model = tiny_model()
    apply_lora(model, rank=2, alpha=4)
    path = tmp_path / "adapter.pth"
    original = b"existing adapter contents"
    path.write_bytes(original)

    def fail_replace(source, destination):
        assert source.parent == destination.parent == tmp_path
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(model_lora.os, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        save_lora(model, path)

    assert path.read_bytes() == original
    assert list(tmp_path.glob(f".{path.name}.*.tmp")) == []
