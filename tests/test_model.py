import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache, StaticCache

from model.model import (
    Attention,
    MoEFeedForward,
    NinjaMindConfig,
    NinjaMindForCausalLM,
    RMSNorm,
    precompute_freqs_cis,
    repeat_kv,
)


def tiny_config(**overrides):
    values = {
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 32,
        "intermediate_size": 32,
        "max_position_embeddings": 32,
        "dropout": 0.0,
        "flash_attn": False,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    values.update(overrides)
    return NinjaMindConfig(**values)


def test_rmsnorm_matches_reference_and_is_stable():
    norm = RMSNorm(4, eps=1e-6)
    x = torch.tensor(
        [[[0.0, 0.0, 0.0, 0.0], [1.0, -2.0, 3.0, -4.0]]],
        requires_grad=True,
    )

    output = norm(x)
    expected = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected)
    output.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    "override, message",
    [
        ({"hidden_size": 15}, "divisible"),
        ({"num_key_value_heads": 3}, "num_key_value_heads"),
        ({"head_dim": 8}, "head_dim"),
        ({"num_experts": 2, "num_experts_per_tok": 3}, "cannot exceed"),
    ],
)
def test_config_rejects_inconsistent_dimensions(override, message):
    with pytest.raises(ValueError, match=message):
        tiny_config(**override)


def test_causal_mask_blocks_future_tokens_and_left_padding_is_equivalent():
    torch.manual_seed(0)
    model = NinjaMindForCausalLM(tiny_config()).eval()

    original = torch.tensor([[3, 4, 5, 6]])
    changed_future = torch.tensor([[3, 4, 20, 21]])
    with torch.no_grad():
        original_logits = model(original).logits
        changed_logits = model(changed_future).logits
    torch.testing.assert_close(original_logits[:, :2], changed_logits[:, :2])

    unpadded = torch.tensor([[3, 4, 5]])
    padded = torch.tensor([[0, 0, 3, 4, 5]])
    with torch.no_grad():
        unpadded_logits = model(unpadded, attention_mask=torch.ones_like(unpadded)).logits
        padded_logits = model(
            padded, attention_mask=torch.tensor([[0, 0, 1, 1, 1]])
        ).logits
    torch.testing.assert_close(padded_logits[:, -3:], unpadded_logits, atol=2e-6, rtol=2e-5)


def test_gqa_repeat_order_attention_shape_and_cache_shape():
    # KV head 0 must feed query heads 0/1 and KV head 1 heads 2/3.
    values = torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]])
    repeated = repeat_kv(values, 2)
    expected = torch.tensor(
        [[[[1.0], [1.0], [2.0], [2.0]], [[3.0], [3.0], [4.0], [4.0]]]]
    )
    torch.testing.assert_close(repeated, expected)

    config = tiny_config(num_hidden_layers=1)
    attention = Attention(config).eval()
    x = torch.randn(2, 3, config.hidden_size)
    position_embeddings = precompute_freqs_cis(config.head_dim, end=3)
    with torch.no_grad():
        output, cache = attention(x, position_embeddings, use_cache=True)

    assert output.shape == x.shape
    assert cache[0].shape == (2, config.num_key_value_heads, 3, config.head_dim)
    assert cache[1].shape == cache[0].shape


def test_incremental_cache_matches_full_forward_for_left_padded_batch():
    torch.manual_seed(1)
    config = tiny_config()
    model = NinjaMindForCausalLM(config).eval()
    prompt = torch.tensor([[0, 0, 3, 4], [0, 5, 6, 7]])
    prompt_mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    next_token = torch.tensor([[8], [9]])
    full_ids = torch.cat((prompt, next_token), dim=1)
    full_mask = torch.cat((prompt_mask, torch.ones(2, 1, dtype=torch.long)), dim=1)

    with torch.no_grad():
        expected = model(full_ids, attention_mask=full_mask).logits[:, -1]

        prefill = model(prompt, attention_mask=prompt_mask, use_cache=True)
        legacy_step = model(
            next_token,
            attention_mask=full_mask,
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        dynamic_cache = DynamicCache(config=config)
        dynamic_prefill = model(
            prompt,
            attention_mask=prompt_mask,
            past_key_values=dynamic_cache,
            use_cache=True,
        )
        dynamic_step = model(
            next_token,
            attention_mask=full_mask,
            past_key_values=dynamic_prefill.past_key_values,
            use_cache=True,
        )

    assert isinstance(prefill.past_key_values, tuple)
    assert prefill.past_key_values[0][0].shape[2] == prompt.shape[1]
    assert legacy_step.past_key_values[0][0].shape[2] == full_ids.shape[1]
    assert dynamic_prefill.past_key_values is dynamic_cache
    assert dynamic_step.past_key_values is dynamic_cache
    assert dynamic_cache.get_seq_length() == full_ids.shape[1]
    torch.testing.assert_close(legacy_step.logits[:, -1], expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(dynamic_step.logits[:, -1], expected, atol=2e-6, rtol=2e-5)


def test_static_cache_and_generation_mixin_are_supported():
    torch.manual_seed(2)
    config = tiny_config(num_hidden_layers=1)
    model = NinjaMindForCausalLM(config).eval()
    prompt = torch.tensor([[0, 3, 4], [5, 6, 7]])
    prompt_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    next_token = torch.tensor([[8], [9]])
    full_mask = torch.cat((prompt_mask, torch.ones(2, 1, dtype=torch.long)), dim=1)
    cache = StaticCache(config=config, max_cache_len=8)

    with torch.no_grad():
        full = model(
            torch.cat((prompt, next_token), dim=1), attention_mask=full_mask
        ).logits[:, -1]
        model(
            prompt,
            attention_mask=prompt_mask,
            past_key_values=cache,
            cache_position=torch.arange(prompt.shape[1]),
            use_cache=True,
        )
        step = model(
            next_token,
            attention_mask=full_mask,
            past_key_values=cache,
            cache_position=torch.tensor([prompt.shape[1]]),
            use_cache=True,
        )
        generated = model.generate(
            prompt,
            attention_mask=prompt_mask,
            max_new_tokens=2,
            do_sample=False,
            cache_implementation="static",
        )

    torch.testing.assert_close(step.logits[:, -1], full, atol=2e-6, rtol=2e-5)
    assert generated.shape == (2, prompt.shape[1] + 2)


def test_hf_directory_roundtrip_rebuilds_rope_and_preserves_cached_logits(tmp_path):
    torch.manual_seed(23)
    config = tiny_config(num_hidden_layers=1)
    model = NinjaMindForCausalLM(config).eval()
    prompt = torch.tensor([[0, 3, 4], [5, 6, 7]])
    prompt_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    next_token = torch.tensor([[8], [9]])
    full_ids = torch.cat((prompt, next_token), dim=1)
    full_mask = torch.cat((prompt_mask, torch.ones(2, 1, dtype=torch.long)), dim=1)

    with torch.no_grad():
        expected_full = model(full_ids, attention_mask=full_mask).logits
        expected_prefill = model(prompt, attention_mask=prompt_mask, use_cache=True)
        expected_step = model(
            next_token,
            attention_mask=full_mask,
            past_key_values=expected_prefill.past_key_values,
            use_cache=True,
        )

    # RoPE tables are derived from config and must not inflate the checkpoint.
    assert model.model.freqs_cos.shape == (
        config.max_position_embeddings,
        config.head_dim,
    )
    assert "model.freqs_cos" not in model.state_dict()
    assert "model.freqs_sin" not in model.state_dict()
    model.save_pretrained(tmp_path)

    loaded = NinjaMindForCausalLM.from_pretrained(
        tmp_path, local_files_only=True
    ).eval()
    # Transformers 5 preserves the empty sentinel when materializing a model
    # created on the meta device; the first forward must rebuild real tables.
    assert loaded.model.freqs_cos.shape == (0, config.head_dim)
    assert loaded.model.freqs_sin.shape == (0, config.head_dim)

    with torch.no_grad():
        actual_full = loaded(full_ids, attention_mask=full_mask).logits
        actual_prefill = loaded(prompt, attention_mask=prompt_mask, use_cache=True)
        actual_step = loaded(
            next_token,
            attention_mask=full_mask,
            past_key_values=actual_prefill.past_key_values,
            use_cache=True,
        )

    assert loaded.model.freqs_cos.shape == (
        config.max_position_embeddings,
        config.head_dim,
    )
    assert torch.count_nonzero(loaded.model.freqs_cos).item() > 0
    torch.testing.assert_close(actual_full, expected_full, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        actual_prefill.logits, expected_prefill.logits, atol=2e-6, rtol=2e-5
    )
    torch.testing.assert_close(
        actual_step.logits, expected_step.logits, atol=2e-6, rtol=2e-5
    )
    for expected_layer, actual_layer in zip(
        expected_step.past_key_values, actual_step.past_key_values, strict=True
    ):
        torch.testing.assert_close(actual_layer[0], expected_layer[0])
        torch.testing.assert_close(actual_layer[1], expected_layer[1])


class _ScaleExpert(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


def test_moe_topk_weights_stay_paired_with_token_indices():
    config = tiny_config(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=8,
        num_experts=3,
        num_experts_per_tok=2,
        use_moe=True,
    )
    moe = MoEFeedForward(config).eval()
    moe.experts = nn.ModuleList([_ScaleExpert(1.0), _ScaleExpert(2.0), _ScaleExpert(4.0)])
    with torch.no_grad():
        moe.gate.weight.copy_(
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, -1.0, 0.0, 0.0]]
            )
        )
    x = torch.tensor([[[2.0, 1.0, 0.0, 0.0], [-1.0, 3.0, 0.0, 0.0]]])

    router_probs = F.softmax(moe.gate(x.reshape(-1, 4)), dim=-1)
    weights, indices = torch.topk(router_probs, k=2, dim=-1, sorted=False)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    scales = torch.tensor([1.0, 2.0, 4.0])
    mixed_scale = (weights * scales[indices]).sum(dim=-1).view(1, 2, 1)
    expected = x * mixed_scale

    torch.testing.assert_close(moe(x), expected)


def test_moe_aux_loss_formula_and_router_gradient_for_topk_two():
    torch.manual_seed(3)
    config = tiny_config(
        num_hidden_layers=1,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.1,
    )
    moe = MoEFeedForward(config).train()
    x = torch.randn(2, 3, config.hidden_size)
    moe(x)

    router_probs = F.softmax(moe.gate(x.reshape(-1, config.hidden_size)).float(), dim=-1)
    indices = torch.topk(router_probs, k=2, dim=-1, sorted=False).indices
    load = F.one_hot(indices, num_classes=config.num_experts).float().mean(dim=(0, 1))
    expected = (
        config.num_experts
        * (load * router_probs.mean(dim=0)).sum()
        * config.router_aux_loss_coef
    )
    torch.testing.assert_close(moe.aux_loss, expected)

    moe.aux_loss.backward()
    assert moe.gate.weight.grad is not None
    assert moe.gate.weight.grad.abs().sum() > 0


def test_moe_aux_loss_ignores_padding_tokens_through_model_mask():
    torch.manual_seed(31)
    config = tiny_config(
        num_hidden_layers=1,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.1,
    )
    model = NinjaMindForCausalLM(config).train()
    unpadded = torch.tensor([[1, 3, 4]])
    padded = torch.tensor([[1, 3, 4, 0, 0]])

    base_aux = model(unpadded, attention_mask=torch.ones_like(unpadded)).aux_loss
    padded_aux = model(
        padded,
        attention_mask=torch.tensor([[1, 1, 1, 0, 0]]),
    ).aux_loss

    torch.testing.assert_close(padded_aux, base_aux, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("use_moe", [False, True])
def test_dense_and_moe_language_model_forward_backward(use_moe):
    torch.manual_seed(4)
    config = tiny_config(
        num_hidden_layers=1,
        use_moe=use_moe,
        num_experts=4,
        num_experts_per_tok=2,
    )
    model = NinjaMindForCausalLM(config).train()
    input_ids = torch.tensor([[1, 3, 4, 2], [1, 5, 6, 2]])
    output = model(input_ids, labels=input_ids)

    assert output.logits.shape == (2, 4, config.vocab_size)
    assert output.loss.ndim == 0 and torch.isfinite(output.loss)
    assert output.aux_loss.ndim == 0 and torch.isfinite(output.aux_loss)
    (output.loss + output.aux_loss).backward()
    assert model.model.embed_tokens.weight.grad is not None
    if use_moe:
        assert output.aux_loss > 0
        assert model.model.layers[0].mlp.gate.weight.grad is not None
    else:
        assert output.aux_loss.item() == 0.0
