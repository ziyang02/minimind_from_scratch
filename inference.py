"""Reusable local inference utilities for NinjaMind.

The training scripts save versioned local checkpoints.  This module also
supports their legacy plain ``state_dict`` files, Hugging Face-style local
model directories, and LoRA adapters, without downloading weights or a
tokenizer from the network.
"""

from __future__ import annotations

import argparse
import math
import re
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from model.model import NinjaMindConfig, NinjaMindForCausalLM
from model.model_lora import apply_lora, load_lora


@dataclass(frozen=True)
class SamplingConfig:
    """Controls autoregressive decoding.

    ``temperature=0`` selects greedy decoding. ``top_k=0`` keeps the full
    distribution.  A local generator is used when ``seed`` is set, so callers
    do not have to mutate PyTorch's process-wide RNG state.
    """

    max_new_tokens: int = 64
    temperature: float = 0.8
    top_k: int = 40
    use_cache: bool = True
    seed: int | None = None

    def validate(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")


@dataclass
class LoadedModel:
    """A model/tokenizer pair plus a short description of loaded weights."""

    model: NinjaMindForCausalLM
    tokenizer: PreTrainedTokenizerBase
    device: torch.device
    source: str


def resolve_device(preference: str = "auto") -> torch.device:
    """Resolve ``auto`` to CUDA, MPS, or CPU (in that order)."""

    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _safe_torch_load(path: str | Path) -> dict[str, Any]:
    """Load tensor-only checkpoints while remaining compatible with old torch."""

    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0 has no weights_only argument
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, dict):
        raise ValueError(f"checkpoint must contain a mapping, got {type(value).__name__}")
    return value


def _extract_state(payload: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = payload
    for wrapper_key in ("state_dict", "model_state_dict", "model"):
        wrapped = payload.get(wrapper_key)
        if isinstance(wrapped, dict):
            state = wrapped
            break
    if not state or not all(torch.is_tensor(value) for value in state.values()):
        raise ValueError("checkpoint does not contain a tensor state dict")
    return {_normalise_key(str(key)): tensor for key, tensor in state.items()}


def _normalise_key(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _is_lora_state(state: dict[str, Any]) -> bool:
    tensor_keys = [key for key, value in state.items() if torch.is_tensor(value)]
    return bool(tensor_keys) and all(".lora." in key for key in tensor_keys)


def _validate_attention_config(config: NinjaMindConfig) -> None:
    q_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    hidden_size = int(config.hidden_size)
    if q_heads <= 0 or kv_heads <= 0:
        raise ValueError("attention head counts must be positive")
    if hidden_size % q_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if q_heads % kv_heads:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")


def _infer_config(
    state: dict[str, Any],
    *,
    num_attention_heads: int,
    max_position_embeddings: int,
) -> NinjaMindConfig:
    """Infer the shape-bearing config fields from a plain model state dict."""

    embed = state.get("model.embed_tokens.weight")
    if not torch.is_tensor(embed) or embed.ndim != 2:
        raise ValueError("cannot infer config: model.embed_tokens.weight is missing")
    vocab_size, hidden_size = embed.shape
    layer_ids = {
        int(match.group(1))
        for key in state
        if (match := re.match(r"model\.layers\.(\d+)\.", key)) is not None
    }
    if not layer_ids:
        raise ValueError("cannot infer config: no transformer layer weights found")
    num_hidden_layers = max(layer_ids) + 1
    if num_attention_heads <= 0 or hidden_size % num_attention_heads:
        raise ValueError(
            f"hidden_size={hidden_size} is not divisible by "
            f"num_attention_heads={num_attention_heads}"
        )
    head_dim = hidden_size // num_attention_heads
    k_proj = state.get("model.layers.0.self_attn.k_proj.weight")
    if not torch.is_tensor(k_proj) or k_proj.shape[0] % head_dim:
        raise ValueError("cannot infer num_key_value_heads from k_proj.weight")
    num_key_value_heads = k_proj.shape[0] // head_dim
    gate = state.get("model.layers.0.mlp.gate_proj.weight")
    use_moe = not torch.is_tensor(gate)
    kwargs: dict[str, Any] = {}
    if torch.is_tensor(gate):
        kwargs["intermediate_size"] = int(gate.shape[0])
    else:
        expert_gate = state.get("model.layers.0.mlp.experts.0.gate_proj.weight")
        router = state.get("model.layers.0.mlp.gate.weight")
        if torch.is_tensor(expert_gate):
            kwargs["moe_intermediate_size"] = int(expert_gate.shape[0])
        if torch.is_tensor(router):
            kwargs["num_experts"] = int(router.shape[0])
    return NinjaMindConfig(
        hidden_size=int(hidden_size),
        num_hidden_layers=num_hidden_layers,
        vocab_size=int(vocab_size),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(num_key_value_heads),
        max_position_embeddings=max_position_embeddings,
        use_moe=use_moe,
        **kwargs,
    )


def _load_full_checkpoint(
    checkpoint: Path,
    *,
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    max_position_embeddings: int,
    use_moe: bool,
) -> tuple[NinjaMindForCausalLM, str]:
    if checkpoint.is_dir():
        model = NinjaMindForCausalLM.from_pretrained(checkpoint, local_files_only=True)
        _validate_attention_config(model.config)
        return model, str(checkpoint)

    payload = _safe_torch_load(checkpoint)
    state = _extract_state(payload)
    if _is_lora_state(state):
        raise ValueError(
            f"{checkpoint} contains only LoRA weights; pass it with "
            "--lora-checkpoint and provide the base model via --checkpoint"
        )
    saved_config = payload.get("config")
    if isinstance(saved_config, dict):
        config = NinjaMindConfig(**saved_config)
    else:
        try:
            config = _infer_config(
                state,
                num_attention_heads=num_attention_heads,
                max_position_embeddings=max_position_embeddings,
            )
        except ValueError:
            config = NinjaMindConfig(
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                vocab_size=vocab_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                max_position_embeddings=max_position_embeddings,
                use_moe=use_moe,
            )
    _validate_attention_config(config)
    model = NinjaMindForCausalLM(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint does not match the model config: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    stage = payload.get("stage")
    description = f"{checkpoint} ({stage})" if stage else str(checkpoint)
    return model, description


def load_model_and_tokenizer(
    *,
    tokenizer_dir: str | Path = "tokenizer",
    checkpoint: str | Path | None = None,
    lora_checkpoint: str | Path | None = None,
    device: str = "auto",
    hidden_size: int = 128,
    num_hidden_layers: int = 2,
    num_attention_heads: int = 8,
    num_key_value_heads: int = 4,
    max_position_embeddings: int = 2048,
    use_moe: bool = False,
    lora_rank: int = 8,
    lora_alpha: float = 16,
) -> LoadedModel:
    """Load a base/SFT model and, optionally, a LoRA adapter from local files.

    Base and SFT training checkpoints share the same full-state format.  A LoRA
    adapter is applied on top of ``checkpoint`` and never accepted without a
    base checkpoint, since a random base would produce misleading output.
    """

    tokenizer_path = Path(tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, use_fast=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token

    if lora_checkpoint is not None and checkpoint is None:
        raise ValueError("a LoRA adapter requires --checkpoint with base/SFT weights")

    source = "randomly initialized smoke model"
    if checkpoint is None:
        config = NinjaMindConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            vocab_size=len(tokenizer),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            use_moe=use_moe,
        )
        _validate_attention_config(config)
        model = NinjaMindForCausalLM(config)
    else:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
        model, source = _load_full_checkpoint(
            checkpoint_path,
            vocab_size=len(tokenizer),
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            use_moe=use_moe,
        )

    model_vocab_size = int(model.config.vocab_size)
    if model_vocab_size != len(tokenizer):
        raise ValueError(
            f"tokenizer/model vocabulary mismatch: {len(tokenizer)} != {model_vocab_size}"
        )

    if lora_checkpoint is not None:
        lora_path = Path(lora_checkpoint)
        if not lora_path.is_file():
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")
        adapter_payload = _safe_torch_load(lora_path)
        adapter_state = _extract_state(adapter_payload)
        if not _is_lora_state(adapter_state):
            raise ValueError(f"not a LoRA-only checkpoint: {lora_path}")
        adapter_config = adapter_payload.get("adapter_config", {})
        if isinstance(adapter_config, dict):
            lora_rank = int(adapter_config.get("rank", lora_rank))
            lora_alpha = float(adapter_config.get("alpha", lora_alpha))
            raw_targets = adapter_config.get("targets")
            targets = tuple(raw_targets) if isinstance(raw_targets, (list, tuple)) else None
        else:
            targets = None
        if not isinstance(adapter_config, dict) or "rank" not in adapter_config:
            ranks = {
                int(value.shape[0])
                for key, value in adapter_state.items()
                if key.endswith(".lora.A.weight") and torch.is_tensor(value)
            }
            if len(ranks) == 1:
                lora_rank = ranks.pop()
        if lora_rank <= 0 or not math.isfinite(lora_alpha):
            raise ValueError("LoRA rank must be positive and alpha must be finite")
        apply_kwargs: dict[str, Any] = {"rank": lora_rank, "alpha": lora_alpha}
        if targets:
            apply_kwargs["targets"] = targets
        apply_lora(model, **apply_kwargs)
        load_lora(model, lora_path)
        expected_lora = {key for key in model.state_dict() if ".lora." in key}
        missing_lora = expected_lora.difference(adapter_state)
        if missing_lora:
            raise ValueError(f"LoRA checkpoint is missing keys: {sorted(missing_lora)[:5]}")
        source = f"{source} + LoRA {lora_path}"

    target_device = resolve_device(device)
    model.to(target_device).eval()
    return LoadedModel(model=model, tokenizer=tokenizer, device=target_device, source=source)


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample one token per row from ``(batch, vocab)`` logits."""

    if logits.ndim != 2:
        raise ValueError(f"expected 2-D logits, got shape {tuple(logits.shape)}")
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and non-negative")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    scores = logits.float() / max(temperature, 1e-6)
    if top_k:
        k = min(top_k, scores.shape[-1])
        threshold = torch.topk(scores, k, dim=-1).values[:, [-1]]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    probabilities = F.softmax(scores, dim=-1)
    if generator is not None and probabilities.device.type != generator.device.type:
        # PyTorch does not currently expose an MPS Generator. Sampling this
        # tiny vector on CPU preserves a local deterministic RNG in that case.
        return torch.multinomial(probabilities.cpu(), 1, generator=generator).to(
            probabilities.device
        )
    return torch.multinomial(probabilities, 1, generator=generator)


def stream_token_ids(
    model: NinjaMindForCausalLM,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    eos_token_id: int | Sequence[int] | None = None,
    config: SamplingConfig | None = None,
) -> Iterator[int]:
    """Yield generated token IDs one by one for a single prompt.

    When ``use_cache`` is true, the prompt is evaluated once and subsequent
    calls receive only the newest token plus each layer's K/V tensors.
    """

    config = config or SamplingConfig()
    config.validate()
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("stream_token_ids currently supports one prompt at a time")
    if input_ids.shape[1] == 0:
        raise ValueError("input_ids must contain at least one token")

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    # A single tokenizer-produced prompt is unpadded in the common case.  Not
    # forwarding its redundant all-ones mask lets Attention use SDPA prefill.
    has_padding = not bool(torch.all(attention_mask == 1).item())

    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and input_ids.shape[1] + config.max_new_tokens > max_positions:
        raise ValueError(
            f"prompt + generation length exceeds max_position_embeddings={max_positions}"
        )
    if eos_token_id is None:
        eos_ids: set[int] = set()
    elif isinstance(eos_token_id, int):
        eos_ids = {eos_token_id}
    else:
        eos_ids = {int(value) for value in eos_token_id}

    generator = None
    if config.seed is not None:
        generator_device = device if device.type == "cuda" else torch.device("cpu")
        generator = torch.Generator(device=generator_device).manual_seed(config.seed)

    was_training = model.training
    model.eval()
    inference_context = torch.inference_mode if hasattr(torch, "inference_mode") else nullcontext
    try:
        with inference_context():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask if has_padding else None,
                use_cache=config.use_cache,
                logits_to_keep=1,
            )
            past_key_values = output.past_key_values if config.use_cache else None
            sequence = input_ids
            for _ in range(config.max_new_tokens):
                next_token = sample_next_token(
                    output.logits[:, -1],
                    temperature=config.temperature,
                    top_k=config.top_k,
                    generator=generator,
                )
                token_id = int(next_token.item())
                yield token_id
                if token_id in eos_ids:
                    break

                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((1, 1))], dim=1
                )
                if config.use_cache:
                    output = model(
                        input_ids=next_token,
                        attention_mask=attention_mask if has_padding else None,
                        past_key_values=past_key_values,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    past_key_values = output.past_key_values
                else:
                    sequence = torch.cat([sequence, next_token], dim=1)
                    output = model(
                        input_ids=sequence,
                        attention_mask=attention_mask if has_padding else None,
                        use_cache=False,
                        logits_to_keep=1,
                    )
    finally:
        if was_training:
            model.train()


def render_chat_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    history: Sequence[tuple[str, str]] | None = None,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """Render one chat turn using the tokenizer's local chat template."""

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for user_text, assistant_text in history or ():
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    messages.append({"role": "user", "content": prompt})
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"


def decode_token_stream(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Iterator[int],
) -> Iterator[str]:
    """Decode generated IDs into stable text pieces.

    A byte-level tokenizer can split one UTF-8 character across several
    tokens. Decoding every cumulative prefix then slicing strings exposes
    temporary replacement characters (``\ufffd``) and can duplicate text when
    the completed character rewrites that prefix. Buffering only the pending
    byte tokens keeps both terminal streaming and the WebUI monotonic.
    """

    pending: list[int] = []
    for token_id in token_ids:
        pending.append(token_id)
        piece = tokenizer.decode(
            pending,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if "\ufffd" in piece:
            continue
        pending.clear()
        if piece:
            yield piece

    if pending:
        piece = tokenizer.decode(
            pending,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if piece:
            yield piece


def stream_text(
    model: NinjaMindForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    history: Sequence[tuple[str, str]] | None = None,
    system_prompt: str = "You are a helpful assistant.",
    chat: bool = True,
    config: SamplingConfig | None = None,
) -> Iterator[str]:
    """Yield the cumulative decoded completion after every generated token."""

    text = (
        render_chat_prompt(tokenizer, prompt, history=history, system_prompt=system_prompt)
        if chat
        else prompt
    )
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    cumulative = ""
    token_ids = stream_token_ids(
        model,
        encoded["input_ids"],
        attention_mask=encoded.get("attention_mask"),
        eos_token_id=tokenizer.eos_token_id,
        config=config,
    )
    for piece in decode_token_stream(tokenizer, token_ids):
        cumulative += piece
        yield cumulative


def build_parser(*, smoke_defaults: bool = False) -> argparse.ArgumentParser:
    """Create the shared CLI parser used by ``main.py`` and run_model.py."""

    parser = argparse.ArgumentParser(description="Stream text from a local NinjaMind model")
    default_prompt = "MiniMind" if smoke_defaults else "你好，请简单介绍一下你自己。"
    parser.add_argument("--prompt", default=default_prompt)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--raw-prompt", action="store_true", help="do not apply the chat template")
    parser.add_argument("--tokenizer-dir", default="tokenizer")
    parser.add_argument("--checkpoint", help="base/SFT .pth file or local HF model directory")
    parser.add_argument("--lora-checkpoint", help="LoRA-only .pth applied on top of --checkpoint")
    parser.add_argument("--device", default="cpu" if smoke_defaults else "auto")
    parser.add_argument("--max-new-tokens", type=int, default=8 if smoke_defaults else 64)
    parser.add_argument("--temperature", type=float, default=0.0 if smoke_defaults else 0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=64 if smoke_defaults else 128)
    parser.add_argument("--num-hidden-layers", type=int, default=1 if smoke_defaults else 2)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--num-key-value-heads", type=int, default=4)
    parser.add_argument("--max-position-embeddings", type=int, default=512 if smoke_defaults else 2048)
    parser.add_argument("--use-moe", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16)
    return parser


def cli_main(argv: Sequence[str] | None = None, *, smoke_defaults: bool = False) -> int:
    """Load a model and print its response incrementally to stdout."""

    args = build_parser(smoke_defaults=smoke_defaults).parse_args(argv)
    torch.manual_seed(args.seed)
    loaded = load_model_and_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        checkpoint=args.checkpoint,
        lora_checkpoint=args.lora_checkpoint,
        device=args.device,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        max_position_embeddings=args.max_position_embeddings,
        use_moe=args.use_moe,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    params = sum(parameter.numel() for parameter in loaded.model.parameters())
    print(f"model: {params / 1e6:.2f}M params | device: {loaded.device} | source: {loaded.source}")
    print("assistant: ", end="", flush=True)
    previous = ""
    sampling = SamplingConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        use_cache=not args.no_cache,
        seed=args.seed,
    )
    use_chat_template = not args.raw_prompt and not (
        smoke_defaults and args.checkpoint is None
    )
    for cumulative in stream_text(
        loaded.model,
        loaded.tokenizer,
        args.prompt,
        system_prompt=args.system_prompt,
        chat=use_chat_template,
        config=sampling,
    ):
        piece = cumulative[len(previous) :] if cumulative.startswith(previous) else cumulative
        print(piece, end="", flush=True)
        previous = cumulative
    print()
    if smoke_defaults:
        print("smoke inference OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
