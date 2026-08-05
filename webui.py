"""Minimal Gradio chat UI backed by :mod:`inference`.

Gradio is imported lazily so the core project and CLI remain usable without
the optional web dependency.
"""

from __future__ import annotations

import argparse
import inspect
import threading
from collections.abc import Mapping, Sequence
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
    loaded: LoadedModel | Mapping[str, LoadedModel],
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
    loaded_models = dict(loaded) if isinstance(loaded, Mapping) else {"Model": loaded}
    if not loaded_models:
        raise ValueError("at least one loaded model is required")
    model_locks = {name: threading.Lock() for name in loaded_models}

    def stream_response(message: str, history: Sequence[Any], model_name: str):
        if not message.strip():
            yield ""
            return
        pairs = _history_pairs(history)
        selected = loaded_models[model_name]
        with model_locks[model_name]:
            yield from stream_text(
                selected.model,
                selected.tokenizer,
                message,
                history=pairs,
                system_prompt=system_prompt,
                chat=True,
                config=sampling,
            )

    if len(loaded_models) == 1:
        only_name = next(iter(loaded_models))

        def respond(message: str, history: Sequence[Any]):
            yield from stream_response(message, history, only_name)

        interface_kwargs = {
            "fn": respond,
            "title": "NinjaMind Local Chat",
            "description": (
                f"Local streaming inference on {loaded_models[only_name].device}; "
                f"weights: {loaded_models[only_name].source}. "
                "No prompt or response is sent to a remote model service."
            ),
        }
    else:
        default_name = next(iter(loaded_models))
        selector = gr.Dropdown(
            choices=list(loaded_models),
            value=default_name,
            label="模型",
            info="切换模型后请清空旧对话，以便公平比较。",
        )

        def respond_comparison(
            message: str,
            history: Sequence[Any],
            model_name: str,
        ):
            yield from stream_response(message, history, model_name)

        details = "; ".join(
            f"{name}: {model.source}" for name, model in loaded_models.items()
        )
        interface_kwargs = {
            "fn": respond_comparison,
            "additional_inputs": [selector],
            "additional_inputs_accordion": "模型选择",
            "title": "NinjaMind SFT / DPO / PPO / GRPO 对比",
            "description": (
                f"All models run locally on {loaded_models[default_name].device}. "
                f"{details}. No prompt or response is sent to a remote service."
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
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="load a named model for comparison; repeat for multiple checkpoints",
    )
    return parser


def _parse_model_specs(specs: Sequence[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid --model {spec!r}; expected NAME=CHECKPOINT")
        name, checkpoint = (part.strip() for part in spec.split("=", 1))
        if not name or not checkpoint:
            raise ValueError(f"invalid --model {spec!r}; expected NAME=CHECKPOINT")
        if name in parsed:
            raise ValueError(f"duplicate model name: {name}")
        parsed[name] = checkpoint
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args = build_web_parser().parse_args(argv)
    model_specs = _parse_model_specs(args.model)
    if model_specs and (args.checkpoint or args.lora_checkpoint):
        raise ValueError("use either repeated --model or --checkpoint/--lora-checkpoint")

    def load(checkpoint, lora_checkpoint=None):
        return load_model_and_tokenizer(
            tokenizer_dir=args.tokenizer_dir,
            checkpoint=checkpoint,
            lora_checkpoint=lora_checkpoint,
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

    loaded = (
        {name: load(checkpoint) for name, checkpoint in model_specs.items()}
        if model_specs
        else load(args.checkpoint, args.lora_checkpoint)
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
