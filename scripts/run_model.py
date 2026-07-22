"""Fast, offline CPU inference smoke test and checkpoint-capable CLI.

With no arguments this builds a tiny random model, uses the checked-in local
tokenizer, streams eight tokens, and exits.  The same flags as ``main.py`` can
load base/SFT weights and an optional LoRA adapter::

    python scripts/run_model.py --checkpoint out/full_sft_512.pth \
        --hidden-size 512 --prompt "你好"
"""
import sys
from pathlib import Path

# Make the repo root importable so `model` is found regardless of how this
# script is launched (as a file or with -m).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import cli_main


def main() -> int:
    return cli_main(smoke_defaults=True)


if __name__ == "__main__":
    raise SystemExit(main())
