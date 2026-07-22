"""Core NinjaMind decoder-only Transformer implementation.

The module intentionally stays small, but its cache and masking conventions are
compatible with Hugging Face generation:

* legacy caches use ``(batch, kv_heads, sequence, head_dim)`` tensors;
* ``transformers.cache_utils.Cache`` objects are updated in place;
* RoPE positions are derived from the 2-D attention mask, so left-padded
  batches use the same logical positions as their unpadded counterparts.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeCausalLMOutputWithPast


class NinjaMindConfig(PretrainedConfig):
    model_type = "ninjamind"

    def __init__(
        self,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        use_moe: bool = False,
        **kwargs,
    ):
        dropout = kwargs.pop("dropout", 0.0)
        vocab_size = kwargs.pop("vocab_size", 6400)
        bos_token_id = kwargs.pop("bos_token_id", 1)
        eos_token_id = kwargs.pop("eos_token_id", 2)
        pad_token_id = kwargs.pop("pad_token_id", None)
        flash_attn = kwargs.pop("flash_attn", True)
        num_attention_heads = kwargs.pop("num_attention_heads", 8)
        num_key_value_heads = kwargs.pop("num_key_value_heads", 4)
        requested_head_dim = kwargs.pop("head_dim", None)
        hidden_act = kwargs.pop("hidden_act", "silu")
        intermediate_size = kwargs.pop(
            "intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64
        )
        max_position_embeddings = kwargs.pop("max_position_embeddings", 32768)
        rms_norm_eps = kwargs.pop("rms_norm_eps", 1e-6)
        rope_theta = kwargs.pop("rope_theta", 1e6)
        tie_word_embeddings = kwargs.pop("tie_word_embeddings", True)
        inference_rope_scaling = kwargs.pop("inference_rope_scaling", False)
        use_cache = kwargs.pop("use_cache", True)

        num_experts = kwargs.pop("num_experts", 4)
        num_experts_per_tok = kwargs.pop("num_experts_per_tok", 1)
        moe_intermediate_size = kwargs.pop("moe_intermediate_size", intermediate_size)
        norm_topk_prob = kwargs.pop("norm_topk_prob", True)
        router_aux_loss_coef = kwargs.pop("router_aux_loss_coef", 5e-4)

        # A saved config may contain these base-class fields. NinjaMind is
        # always a decoder-only model, so avoid passing conflicting duplicates.
        kwargs.pop("is_decoder", None)
        kwargs.pop("is_encoder_decoder", None)
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_decoder=True,
            is_encoder_decoder=False,
            **kwargs,
        )

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.flash_attn = flash_attn
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        inferred_head_dim = (
            hidden_size // num_attention_heads
            if isinstance(num_attention_heads, int) and num_attention_heads > 0
            else 0
        )
        self.head_dim = inferred_head_dim if requested_head_dim is None else requested_head_dim
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.use_cache = use_cache
        self.rope_scaling = (
            {
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 16,
                "original_max_position_embeddings": 2048,
                "attention_factor": 1.0,
                "type": "yarn",
            }
            if inference_rope_scaling
            else None
        )

        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
        self._validate()

    def _validate(self) -> None:
        positive_ints = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "vocab_size": self.vocab_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
        }
        for name, value in positive_ints.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads "
                f"({self.hidden_size} vs {self.num_attention_heads})"
            )
        expected_head_dim = self.hidden_size // self.num_attention_heads
        if self.head_dim != expected_head_dim:
            raise ValueError(
                f"head_dim must equal hidden_size // num_attention_heads "
                f"({expected_head_dim}), got {self.head_dim}"
            )
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, got {self.head_dim}")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads "
                f"({self.num_attention_heads} vs {self.num_key_value_heads})"
            )
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError(
                "num_experts_per_tok cannot exceed num_experts "
                f"({self.num_experts_per_tok} vs {self.num_experts})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")
        if self.router_aux_loss_coef < 0:
            raise ValueError(
                "router_aux_loss_coef must be non-negative, got "
                f"{self.router_aux_loss_coef}"
            )
        if self.hidden_act not in ACT2FN:
            raise ValueError(f"unknown hidden_act: {self.hidden_act!r}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        return work * torch.rsqrt(work.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    # Retain the original public spelling for callers that used it directly.
    def _Norm(self, x: torch.Tensor) -> torch.Tensor:  # noqa: N802
        return torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self._norm(x).to(dtype=x.dtype)
        return normalized * self.weight.to(dtype=x.dtype)


def precompute_freqs_cis(
    dim: int,
    end: int = 32 * 1024,
    rope_base: float = 1e6,
    rope_scaling: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dim <= 0 or dim % 2:
        raise ValueError(f"RoPE dimension must be a positive even integer, got {dim}")
    freqs = 1.0 / rope_base ** (torch.arange(0, dim, 2).float() / dim)
    attention_factor = 1.0

    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow, attention_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32),
            rope_scaling.get("beta_slow", 1),
            rope_scaling.get("attention_factor", 1.0),
        )
        if end > orig_max:
            inv_dim = lambda b: (  # noqa: E731
                dim * math.log(orig_max / (2 * b * math.pi))
            ) / (2 * math.log(rope_base))
            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(dim // 2, device=freqs.device).float() - low)
                / max(high - low, 0.01),
                0,
                1,
            )
            # YaRN interpolates between extrapolated (1/factor) and original
            # frequencies. attention_factor scales cos/sin amplitudes only.
            freqs = freqs * ((1 - ramp) / factor + ramp)

    angles = torch.outer(torch.arange(end, device=freqs.device), freqs).float()
    freqs_cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1) * attention_factor
    freqs_sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1) * attention_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = -2,
) -> tuple[torch.Tensor, torch.Tensor]:
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        midpoint = x.shape[-1] // 2
        return torch.cat((-x[..., midpoint:], x[..., :midpoint]), dim=-1)

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = q * cos.to(q.dtype) + rotate_half(q) * sin.to(q.dtype)
    k_embed = k * cos.to(k.dtype) + rotate_half(k) * sin.to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat sequence-first KV heads without copying where possible.

    Args:
        x: ``(batch, sequence, kv_heads, head_dim)`` tensor.
        n_rep: Number of query heads represented by each KV head.
    """

    if x.ndim != 4:
        raise ValueError(f"repeat_kv expects a rank-4 tensor, got shape {tuple(x.shape)}")
    if n_rep <= 0:
        raise ValueError(f"n_rep must be positive, got {n_rep}")
    if n_rep == 1:
        return x
    batch_size, seq_len, num_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, num_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, num_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, config: NinjaMindConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_heads = config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, self.head_dim * config.num_attention_heads, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention") and config.flash_attn

    def _legacy_cache_to_sequence_first(
        self, past_key_value: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        past_key, past_value = past_key_value
        if past_key.ndim != 4 or past_value.shape != past_key.shape:
            raise ValueError("each legacy cache entry must contain two equal rank-4 tensors")
        if past_key.shape[1] == self.num_key_value_heads:
            # Hugging Face convention: (B, Hkv, S, D).
            return past_key.transpose(1, 2), past_value.transpose(1, 2)
        if past_key.shape[2] == self.num_key_value_heads:
            # Backward compatibility with this project's old (B, S, Hkv, D).
            return past_key, past_value
        raise ValueError(
            "legacy cache has no KV-head dimension matching "
            f"num_key_value_heads={self.num_key_value_heads}: {tuple(past_key.shape)}"
        )

    @staticmethod
    def _query_cache_positions(
        cache_position: torch.Tensor | None,
        batch_size: int,
        query_length: int,
        key_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        if cache_position is None:
            positions = torch.arange(
                key_length - query_length, key_length, device=device, dtype=torch.long
            )
            return positions.unsqueeze(0)
        positions = cache_position.to(device=device, dtype=torch.long)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0)
        if positions.ndim != 2 or positions.shape[-1] != query_length:
            raise ValueError(
                "cache_position must have shape (query_length,) or "
                f"(batch, query_length), got {tuple(positions.shape)}"
            )
        if positions.shape[0] not in (1, batch_size):
            raise ValueError(
                f"cache_position batch size must be 1 or {batch_size}, "
                f"got {positions.shape[0]}"
            )
        return positions

    @staticmethod
    def _pad_2d_key_mask(
        mask: torch.Tensor, key_length: int, filled_length: int
    ) -> torch.Tensor:
        mask_length = mask.shape[-1]
        if mask_length == key_length:
            return mask
        if mask_length > key_length:
            return mask[:, :key_length]
        missing = key_length - mask_length
        if key_length > filled_length:
            # A static cache exposes its full capacity; unfilled slots live on
            # the right and must remain masked.
            return F.pad(mask, (0, missing), value=0)
        # Friendly legacy-cache fallback when only the current-token mask was
        # supplied. The caller cannot describe past padding in this form.
        return F.pad(mask, (missing, 0), value=1)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | Cache | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | Cache | None]:
        batch_size, seq_len, _ = x.shape
        cos, sin = position_embeddings

        xq = self.q_proj(x).view(batch_size, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        xv = self.v_proj(x).view(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if isinstance(past_key_value, Cache):
            cache_kwargs = (
                {"cache_position": cache_position.to(device=x.device)}
                if cache_position is not None
                else None
            )
            cached_key, cached_value = past_key_value.update(
                xk.transpose(1, 2),
                xv.transpose(1, 2),
                self.layer_idx,
                cache_kwargs,
            )
            xk = cached_key.transpose(1, 2)
            xv = cached_value.transpose(1, 2)
            present_key_value = past_key_value if use_cache else None
        else:
            if past_key_value is not None:
                past_key, past_value = self._legacy_cache_to_sequence_first(past_key_value)
                xk = torch.cat((past_key, xk), dim=1)
                xv = torch.cat((past_value, xv), dim=1)
            present_key_value = (
                (xk.transpose(1, 2), xv.transpose(1, 2)) if use_cache else None
            )

        key_len = xk.shape[1]
        query_positions = self._query_cache_positions(
            cache_position, batch_size, seq_len, key_len, x.device
        )
        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # SDPA's native causal alignment is correct only when query and key
        # lengths match. Cached decoding uses an explicit bottom-right mask.
        use_fast_causal = (
            self.flash
            and key_len == seq_len
            and attention_mask is None
        )
        if use_fast_causal:
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            key_positions = torch.arange(key_len, device=x.device).view(1, 1, 1, key_len)
            causal_mask = key_positions > query_positions[:, None, :, None]
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(causal_mask, mask_value)

            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    if attention_mask.shape[0] != batch_size:
                        raise ValueError(
                            "attention_mask batch size does not match input: "
                            f"{attention_mask.shape[0]} vs {batch_size}"
                        )
                    filled_length = min(int(query_positions.max().item()) + 1, key_len)
                    key_mask = self._pad_2d_key_mask(
                        attention_mask.to(device=x.device), key_len, filled_length
                    )
                    scores = scores.masked_fill(key_mask[:, None, None, :] == 0, mask_value)
                elif attention_mask.ndim == 4:
                    mask = attention_mask.to(device=x.device)
                    if mask.shape[-2] != seq_len:
                        mask = mask[..., -seq_len:, :]
                    if mask.shape[-1] != key_len:
                        if mask.shape[-1] > key_len:
                            mask = mask[..., :key_len]
                        else:
                            mask = F.pad(mask, (0, key_len - mask.shape[-1]), value=False)
                    if mask.dtype == torch.bool:
                        scores = scores.masked_fill(~mask, mask_value)
                    else:
                        scores = scores + mask.to(dtype=scores.dtype)
                else:
                    raise ValueError(
                        "attention_mask must be rank 2 or 4, got "
                        f"shape {tuple(attention_mask.shape)}"
                    )

            probabilities = F.softmax(scores.float(), dim=-1).to(dtype=xq.dtype)
            probabilities = torch.nan_to_num(probabilities, nan=0.0)
            output = self.attn_dropout(probabilities) @ xv

        output = output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, present_key_value


class FeedForwrd(nn.Module):
    """SwiGLU feed-forward network (name retained for API compatibility)."""

    def __init__(self, config: NinjaMindConfig, intermediate_size: int | None = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoEFeedForward(nn.Module):
    def __init__(self, config: NinjaMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                FeedForwrd(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        self.aux_loss = torch.tensor(0.0)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.reshape(-1, hidden_size)
        if token_mask is not None:
            if token_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    f"token_mask must have shape {(batch_size, seq_len)}, "
                    f"got {tuple(token_mask.shape)}"
                )
            valid_tokens = token_mask.to(device=x.device, dtype=torch.bool).reshape(-1)
        else:
            valid_tokens = None
        router_probs = F.softmax(self.gate(x_flat).float(), dim=-1)
        topk_weight, topk_idx = torch.topk(
            router_probs,
            k=self.config.num_experts_per_tok,
            dim=-1,
            sorted=False,
        )
        if self.config.norm_topk_prob:
            topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True).clamp_min(1e-20)

        output = torch.zeros_like(x_flat)
        for expert_idx, expert in enumerate(self.experts):
            # Keep token and top-k slot paired. Flattening the boolean mask and
            # independently deduplicating token IDs breaks this for top-k > 1.
            token_idx, slot_idx = (topk_idx == expert_idx).nonzero(as_tuple=True)
            if token_idx.numel() > 0:
                expert_weight = topk_weight[token_idx, slot_idx].to(dtype=x.dtype).unsqueeze(-1)
                expert_output = expert(x_flat.index_select(0, token_idx)) * expert_weight
                output.index_add_(0, token_idx, expert_output.to(dtype=output.dtype))
            elif self.training:
                # Preserve a zero-gradient connection for DDP runs in which an
                # expert receives no tokens on this rank.
                output = output + sum(parameter.sum() * 0.0 for parameter in expert.parameters())

        if self.training and self.config.router_aux_loss_coef > 0:
            # Fraction of assignments received by each expert across both token
            # and top-k dimensions. The old mean(0) left a top-k dimension and
            # scaled the loss incorrectly whenever k > 1.
            aux_topk_idx = topk_idx if valid_tokens is None else topk_idx[valid_tokens]
            aux_router_probs = (
                router_probs if valid_tokens is None else router_probs[valid_tokens]
            )
            if aux_router_probs.shape[0] == 0:
                self.aux_loss = router_probs.sum() * 0.0
            else:
                expert_load = F.one_hot(
                    aux_topk_idx, num_classes=self.config.num_experts
                ).float().mean(dim=(0, 1))
                mean_router_prob = aux_router_probs.mean(dim=0)
                self.aux_loss = (
                    self.config.num_experts
                    * torch.sum(expert_load * mean_router_prob)
                    * self.config.router_aux_loss_coef
                )
        else:
            self.aux_loss = router_probs.new_zeros(())
        return output.view(batch_size, seq_len, hidden_size)


class NinjaMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: NinjaMindConfig):
        super().__init__()
        self.self_attn = Attention(config, layer_idx=layer_id)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForwrd(config) if not config.use_moe else MoEFeedForward(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_value: tuple[torch.Tensor, torch.Tensor] | Cache | None = None,
        use_cache: bool = False,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | Cache | None]:
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
            attention_mask=attention_mask,
            cache_position=cache_position,
        )
        hidden_states = hidden_states + residual
        mlp_input = self.post_attention_layernorm(hidden_states)
        if isinstance(self.mlp, MoEFeedForward):
            mlp_output = self.mlp(mlp_input, token_mask=token_mask)
        else:
            mlp_output = self.mlp(mlp_input)
        hidden_states = hidden_states + mlp_output
        return hidden_states, present_key_value


class NinjaMindModel(nn.Module):
    def __init__(self, config: NinjaMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [NinjaMindBlock(layer_id, config) for layer_id in range(self.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Build RoPE tables lazily.  Keeping these derived tensors
        # non-persistent avoids adding two large, reproducible tables to every
        # Hugging Face checkpoint.  The zero-length shape is also an explicit
        # initialization sentinel: Transformers' low-memory ``from_pretrained``
        # path materializes non-persistent meta buffers without restoring their
        # values, but it preserves their shape.  A precomputed full-size table
        # would therefore become silently zero-filled after loading, whereas an
        # empty table is rebuilt on the first real forward pass below.
        self.register_buffer(
            "freqs_cos", torch.empty(0, config.head_dim), persistent=False
        )
        self.register_buffer(
            "freqs_sin", torch.empty(0, config.head_dim), persistent=False
        )

    @staticmethod
    def _legacy_past_length(
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor] | None, ...] | list,
        num_key_value_heads: int,
    ) -> int:
        first = next((entry for entry in past_key_values if entry is not None), None)
        if first is None:
            return 0
        key = first[0]
        return key.shape[2] if key.shape[1] == num_key_value_heads else key.shape[1]

    def _rope_for_positions(
        self, position_ids: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.numel() == 0:
            shape = (*position_ids.shape, self.config.head_dim)
            empty = torch.empty(shape, device=device, dtype=self.freqs_cos.dtype)
            return empty, empty
        if torch.any(position_ids < 0):
            raise ValueError("position_ids cannot contain negative values")
        required = int(position_ids.max().item()) + 1
        needs_rebuild = self.freqs_cos.device.type == "meta" or required > self.freqs_cos.shape[0]
        if needs_rebuild:
            capacity = max(required, self.config.max_position_embeddings)
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=capacity,
                rope_base=self.config.rope_theta,
                rope_scaling=self.config.rope_scaling,
            )
            cache_dtype = self.freqs_cos.dtype
            self.freqs_cos = freqs_cos.to(device=device, dtype=cache_dtype)
            self.freqs_sin = freqs_sin.to(device=device, dtype=cache_dtype)
        flat_positions = position_ids.to(device=self.freqs_cos.device).reshape(-1)
        cos = self.freqs_cos.index_select(0, flat_positions).view(
            *position_ids.shape, self.config.head_dim
        )
        sin = self.freqs_sin.index_select(0, flat_positions).view(
            *position_ids.shape, self.config.head_dim
        )
        return cos.to(device=device), sin.to(device=device)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | tuple | list | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, Cache | tuple | None, torch.Tensor]:
        del kwargs
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        batch_size, seq_length = inputs_embeds.shape[:2]

        cache_is_object = isinstance(past_key_values, Cache)
        if cache_is_object:
            start_pos = past_key_values.get_seq_length()
            layer_past = [past_key_values] * len(self.layers)
        else:
            if past_key_values is None:
                layer_past = [None] * len(self.layers)
            else:
                if len(past_key_values) != len(self.layers):
                    raise ValueError(
                        f"past_key_values must have {len(self.layers)} layers, "
                        f"got {len(past_key_values)}"
                    )
                layer_past = list(past_key_values)
            start_pos = self._legacy_past_length(
                layer_past, self.config.num_key_value_heads
            )

        if cache_position is None:
            cache_position = torch.arange(
                start_pos,
                start_pos + seq_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )
        if position_ids is None:
            if attention_mask is not None and attention_mask.ndim == 2:
                logical_positions = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
                logical_positions.masked_fill_(attention_mask == 0, 0)
                position_ids = logical_positions[:, -seq_length:]
            else:
                position_ids = cache_position
        position_ids = position_ids.to(device=inputs_embeds.device, dtype=torch.long)
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        if position_ids.shape != (batch_size, seq_length):
            raise ValueError(
                f"position_ids must have shape {(batch_size, seq_length)}, "
                f"got {tuple(position_ids.shape)}"
            )

        hidden_states = self.dropout(inputs_embeds)
        position_embeddings = self._rope_for_positions(position_ids, hidden_states.device)
        token_mask = None
        if attention_mask is not None and attention_mask.ndim == 2:
            token_mask = attention_mask[:, -seq_length:]
        presents = []
        for layer, past_key_value in zip(self.layers, layer_past, strict=True):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
                cache_position=cache_position,
                token_mask=token_mask,
            )
            if not cache_is_object:
                presents.append(present)
        hidden_states = self.norm(hidden_states)

        aux_loss = hidden_states.new_zeros(())
        for layer in self.layers:
            if isinstance(layer.mlp, MoEFeedForward):
                aux_loss = aux_loss + layer.mlp.aux_loss.to(hidden_states.device)

        if not use_cache:
            present_key_values = None
        elif cache_is_object:
            present_key_values = past_key_values
        else:
            present_key_values = tuple(presents)
        return hidden_states, present_key_values, aux_loss


class NinjaMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NinjaMindConfig
    base_model_prefix = "model"
    # Transformers 5 expects an output->input mapping here.  A mapping also
    # remains iterable as output keys on Transformers 4, while a v4-style list
    # crashes v5's expanded tied-weight discovery during ``post_init``.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: NinjaMindConfig | None = None):
        config = config or NinjaMindConfig()
        super().__init__(config)
        self.model = NinjaMindModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        if config.tie_word_embeddings:
            self.tie_weights()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, value: nn.Linear) -> None:
        self.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | tuple | list | None = None,
        use_cache: bool = False,
        logits_to_keep: int | torch.Tensor = 0,
        labels: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> MoeCausalLMOutputWithPast:
        hidden_states, present_key_values, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        # Labels require all time steps for the shifted language-model loss.
        full_logits = self.lm_head(hidden_states) if labels is not None else None
        if labels is not None:
            shift_logits = full_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            loss = None

        if isinstance(logits_to_keep, int):
            if logits_to_keep < 0:
                raise ValueError("logits_to_keep must be non-negative")
            hidden_slice = hidden_states[:, -logits_to_keep:, :] if logits_to_keep else hidden_states
        else:
            hidden_slice = hidden_states[:, logits_to_keep, :]
        return_full_logits = (
            labels is not None
            and isinstance(logits_to_keep, int)
            and logits_to_keep == 0
        )
        logits = full_logits if return_full_logits else self.lm_head(hidden_slice)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=present_key_values,
            hidden_states=hidden_states,
        )
