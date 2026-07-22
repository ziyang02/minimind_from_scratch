from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from inference import (
    SamplingConfig,
    decode_token_stream,
    load_model_and_tokenizer,
    sample_next_token,
    stream_token_ids,
)
from model.model import NinjaMindConfig, NinjaMindForCausalLM
from model.model_lora import LoRA, apply_lora, save_lora
from webui import _history_pairs

ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = ROOT / "tokenizer"


def tiny_config(vocab_size: int = 64) -> NinjaMindConfig:
    return NinjaMindConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=vocab_size,
        max_position_embeddings=64,
        flash_attn=False,
        dropout=0.0,
    )


def test_sample_next_token_greedy_and_top_k() -> None:
    logits = torch.tensor([[0.0, 1.0, 4.0, 3.0]])
    greedy = sample_next_token(logits, temperature=0.0, top_k=0)
    sampled = sample_next_token(
        logits,
        temperature=1.0,
        top_k=1,
        generator=torch.Generator().manual_seed(123),
    )
    assert greedy.item() == 2
    assert sampled.item() == 2
    with pytest.raises(ValueError, match="temperature"):
        sample_next_token(logits, temperature=-1.0, top_k=0)


def test_byte_level_streaming_waits_for_complete_unicode_characters() -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    token_ids = tokenizer("你好", add_special_tokens=False).input_ids

    pieces = list(decode_token_stream(tokenizer, iter(token_ids)))

    assert pieces == ["你", "好"]
    assert "".join(pieces) == tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    assert all("\ufffd" not in piece for piece in pieces)


def test_web_history_accepts_gradio_messages_and_legacy_pairs() -> None:
    messages = [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "incomplete turn"},
    ]
    assert _history_pairs(messages) == [("one", "first")]
    assert _history_pairs([["two", "second"]]) == [("two", "second")]


def test_stream_generation_cache_matches_full_recompute() -> None:
    torch.manual_seed(7)
    model = NinjaMindForCausalLM(tiny_config()).eval()
    prompt = torch.tensor([[1, 5, 9, 2]])
    common = {"max_new_tokens": 5, "temperature": 0.0, "top_k": 0}
    cached = list(
        stream_token_ids(
            model,
            prompt,
            config=SamplingConfig(use_cache=True, **common),
        )
    )
    uncached = list(
        stream_token_ids(
            model,
            prompt,
            config=SamplingConfig(use_cache=False, **common),
        )
    )
    assert cached == uncached
    assert len(cached) == 5


def test_structured_checkpoint_uses_saved_attention_config(tmp_path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    config = tiny_config(vocab_size=len(tokenizer))
    config.num_key_value_heads = 1
    model = NinjaMindForCausalLM(config)
    checkpoint = tmp_path / "structured.pth"
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "stage": "sft",
        },
        checkpoint,
    )

    loaded = load_model_and_tokenizer(
        tokenizer_dir=TOKENIZER_DIR,
        checkpoint=checkpoint,
        device="cpu",
        # Deliberately wrong CLI values: embedded config must take precedence.
        num_attention_heads=8,
        num_key_value_heads=4,
    )
    assert loaded.model.config.num_attention_heads == 4
    assert loaded.model.config.num_key_value_heads == 1
    assert "sft" in loaded.source


def test_structured_lora_uses_saved_nondefault_alpha(tmp_path: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    config = tiny_config(vocab_size=len(tokenizer))
    torch.manual_seed(11)
    base = NinjaMindForCausalLM(config).eval()
    base_checkpoint = tmp_path / "base.pth"
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": base.state_dict(),
            "config": config.to_dict(),
            "stage": "pretrain",
        },
        base_checkpoint,
    )

    adapted = NinjaMindForCausalLM(config).eval()
    adapted.load_state_dict(base.state_dict())
    apply_lora(adapted, rank=2, alpha=7, targets=("q_proj",))
    with torch.no_grad():
        for module in adapted.modules():
            if isinstance(module, LoRA):
                module.A.weight.fill_(0.1)
                module.B.weight.fill_(0.2)
    adapter_checkpoint = tmp_path / "adapter.pth"
    save_lora(adapted, adapter_checkpoint)

    loaded = load_model_and_tokenizer(
        tokenizer_dir=TOKENIZER_DIR,
        checkpoint=base_checkpoint,
        lora_checkpoint=adapter_checkpoint,
        device="cpu",
        lora_alpha=16,
    )
    loaded_adapters = [module for module in loaded.model.modules() if isinstance(module, LoRA)]
    assert loaded_adapters
    assert all(module.scale == pytest.approx(3.5) for module in loaded_adapters)

    input_ids = torch.tensor([[1, 4, 8]])
    with torch.inference_mode():
        expected = adapted(input_ids=input_ids).logits
        actual = loaded.model(input_ids=input_ids).logits
    torch.testing.assert_close(actual, expected)
