"""Train the NinjaMind BPE tokenizer (MiniMind-style, default 6400 vocab).

Reads a pretrain-format jsonl ({"text": ...} per line), trains a byte-level
BPE tokenizer with the three special tokens the model config expects:

    <unk> = 0 (pad)   <|im_start|> = 1 (bos)   <|im_end|> = 2 (eos)

and saves an HF-compatible tokenizer (tokenizer.json + tokenizer_config.json
with the chat template) that `AutoTokenizer.from_pretrained` can load.

Usage:
    uv run python scripts/train_tokenizer.py --data_path dataset/pretrain_hq.jsonl
"""
import argparse
import json
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<unk>", "<|im_start|>", "<|im_end|>"]

# ChatML-style template. Every user turn is followed by the assistant header,
# so a conversation ending in a user turn is already "generation ready".
CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' }}"
    "{% else %}"
    "{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}"
    "{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n<|im_start|>assistant\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + '<|im_end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
)


def iter_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)["text"]


def main():
    parser = argparse.ArgumentParser(description="Train the BPE tokenizer")
    parser.add_argument("--data_path", type=str, required=True, help="pretrain jsonl with a 'text' field")
    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--out_dir", type=str, default="tokenizer")
    args = parser.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        # Seed with all 256 bytes so any input is representable (no real <unk>).
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(iter_texts(args.data_path), trainer=trainer)

    # The model config hardcodes bos=1 / eos=2; make sure ids line up.
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<|im_start|>") == 1
    assert tokenizer.token_to_id("<|im_end|>") == 2

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "pad_token": "<unk>",
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "model_max_length": 32768,
        "chat_template": CHAT_TEMPLATE,
    }
    with open(os.path.join(args.out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, ensure_ascii=False, indent=2)
    print(f"tokenizer saved to {args.out_dir}/ (vocab_size={tokenizer.get_vocab_size()})")

    # Self-check: reload through transformers and render a conversation.
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.out_dir)
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    rendered = tok.apply_chat_template(messages, tokenize=False)
    print("chat render check:", rendered.replace("\n", "\\n"))
    ids = tok("hello world").input_ids
    print("roundtrip check:", repr(tok.decode(ids)))


if __name__ == "__main__":
    main()
