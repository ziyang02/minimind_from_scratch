"""Minimal Gradio chat UI backed by :mod:`inference`.

Gradio is imported lazily so the core project and CLI remain usable without
the optional web dependency.
"""

from __future__ import annotations

import argparse
import inspect
import threading
from collections.abc import Sequence
from typing import Any

from inference import (
    LoadedModel,
    SamplingConfig,
    build_parser,
    load_model_and_tokenizer,
    stream_text,
)


def _history_pairs(history: Sequence[Any] | None) -> list[tuple[str, str]]:
    """Normalise both legacy tuple history and Gradio message dictionaries."""

    if not history:
        return []
    first = history[0]
    if isinstance(first, (tuple, list)):
        return [(str(item[0]), str(item[1])) for item in history if len(item) >= 2]

    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for item in history:
        if not isinstance(item, dict):
            continue
        role, content = item.get("role"), item.get("content", "")
        if not isinstance(content, str):
            continue
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, content))
            pending_user = None
    return pairs


def create_demo(
    loaded: LoadedModel,
    *,
    system_prompt: str = "You are a helpful assistant.",
    sampling: SamplingConfig | None = None,
):
    """Build a streaming ``gr.ChatInterface`` for a loaded local model."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "Gradio is optional. Install the project's web dependencies before "
            "running webui.py."
        ) from exc

    sampling = sampling or SamplingConfig()
    model_lock = threading.Lock()

    def respond(message: str, history: Sequence[Any]):
        if not message.strip():
            yield ""
            return
        pairs = _history_pairs(history)
        with model_lock:
            yield from stream_text(
                loaded.model,
                loaded.tokenizer,
                message,
                history=pairs,
                system_prompt=system_prompt,
                chat=True,
                config=sampling,
            )

    interface_kwargs = {
        "fn": respond,
        "title": "NinjaMind Local Chat",
        "description": (
            f"Local streaming inference on {loaded.device}; weights: {loaded.source}. "
            "No prompt or response is sent to a remote model service."
        ),
    }
    # Gradio 5 accepts ``type='messages'``; Gradio 6 removed the argument and
    # uses message dictionaries unconditionally. Keep the optional web extra
    # compatible with both supported API generations.
    if "type" in inspect.signature(gr.ChatInterface).parameters:
        interface_kwargs["type"] = "messages"
    return gr.ChatInterface(
        **interface_kwargs,
    )


def build_web_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Launch the local NinjaMind Gradio chat UI"
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_web_parser().parse_args(argv)
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
    demo = create_demo(
        loaded,
        system_prompt=args.system_prompt,
        sampling=SamplingConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            use_cache=not args.no_cache,
            seed=args.seed,
        ),
    )
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
