"""Quick smoke test: build the model and run forward passes.

Run with:  uv run python scripts/run_model.py
"""
import sys
from pathlib import Path

# Make the repo root importable so `model` is found regardless of how this
# script is launched (as a file or with -m).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from model.model import NinjaMindConfig, NinjaMindForCausalLM


def main():
    torch.manual_seed(0)

    # A small config so it builds instantly on CPU.
    cfg = NinjaMindConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        vocab_size=256,
        use_moe=False,   # flip to True to exercise the MoE path
    )
    model = NinjaMindForCausalLM(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model built: {n_params/1e6:.2f}M params")

    # Fake a batch of token ids: (batch=2, seq_len=16).
    input_ids = torch.randint(0, cfg.vocab_size, (2, 16))

    # Training-style forward: pass labels to get a loss.
    out = model(input_ids=input_ids, labels=input_ids)
    print(f"logits {tuple(out.logits.shape)} | loss {out.loss.item():.4f}")

    # Inference-style forward with KV cache (incremental decoding).
    model.eval()
    with torch.no_grad():
        prefill = model(input_ids=input_ids, use_cache=True)
        next_tok = prefill.logits[:, -1:].argmax(-1)
        step = model(input_ids=next_tok, past_key_values=prefill.past_key_values, use_cache=True)
    print(f"decode step logits {tuple(step.logits.shape)}")
    print("OK")


if __name__ == "__main__":
    main()
