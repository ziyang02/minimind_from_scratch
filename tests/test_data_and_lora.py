import json

import torch
from transformers import AutoTokenizer

from dataset.lm_dataset import DPODataset, PretrainDataset, SFTDataset, rl_collate
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
