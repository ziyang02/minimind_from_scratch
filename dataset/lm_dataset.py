"""Datasets for every training stage of the NinjaMind pipeline.

All datasets read newline-delimited JSON (``.jsonl``), one sample per line.
The expected schema per stage:

    Pretrain   {"text": "raw document text ..."}
    SFT        {"conversations": [{"role": "user", "content": "..."},
                                   {"role": "assistant", "content": "..."}, ...]}
    DPO        {"chosen":   [ ...chat messages ending in the preferred reply... ],
                "rejected": [ ...chat messages ending in the dispreferred reply... ]}
    RLAIF      {"conversations": [...user turn(s)...], "answer": "reference answer"}
    AgentRL    {"conversations": [...], "tools": [ ...JSON tool schemas... ],
                "answer": "gold final answer" (optional),
                "gold_tool_calls": [ ... ] (optional)}

Return conventions
------------------
Supervised stages (Pretrain / SFT) return ``(X, Y, loss_mask)`` where
``Y`` is ``X`` shifted left by one (next-token targets) and ``loss_mask``
marks which target positions contribute to the loss.

DPO returns a dict with a ``chosen`` and a ``rejected`` ``(X, Y, mask)`` triple.

RL stages (RLAIF / AgentRL) return a dict holding the *prompt* only (the
completion is generated online during the rollout) plus reference fields
used by the reward function.  Use :func:`rl_collate` to batch and
left-pad prompts for generation.

The chat-based stages assume the tokenizer defines a chat template using
``<|im_start|>`` / ``<|im_end|>`` markers (the MiniMind tokenizer).
"""

import hashlib
import json
import math
import os

import torch
from torch.utils.data import Dataset

# HF tokenizers fork a thread pool; silence the warning when used with
# a DataLoader that already uses multiprocessing workers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__all__ = [
    "load_jsonl",
    "build_chat_prompt",
    "pad_to_length",
    "make_supervised_tensors",
    "generate_response_loss_mask",
    "decode_completion",
    "rl_collate",
    "IndexedDataset",
    "split_supervised_dataset",
    "PretrainDataset",
    "SFTDataset",
    "DPODataset",
    "RLAIFDataset",
    "AgentRLDataset",
]


# --------------------------------------------------------------------------- #
# Pre-processing helpers                                                       #
# --------------------------------------------------------------------------- #
def load_jsonl(path):
    """Load a ``.jsonl`` file into a list of dicts (one per line)."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({e})") from e
    return samples


def build_chat_prompt(tokenizer, messages, add_generation_prompt=False):
    """Render a list of ``{"role", "content"}`` messages into a single string.

    Delegates to the tokenizer's chat template so the special markers
    (``<|im_start|>`` / ``<|im_end|>``) match what the model was trained on.
    Set ``add_generation_prompt=True`` when you want the string to end with
    the assistant header, ready for the model to continue (used at inference
    and for RL rollouts).
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def pad_to_length(
    ids,
    max_length,
    pad_id,
    padding_side="right",
    truncation_side="right",
):
    """Truncate or pad a list of token ids to exactly ``max_length``."""
    if truncation_side not in {"left", "right"}:
        raise ValueError("truncation_side must be 'left' or 'right'")
    ids = list(ids[-max_length:] if truncation_side == "left" else ids[:max_length])
    pad = [pad_id] * (max_length - len(ids))
    return (ids + pad) if padding_side == "right" else (pad + ids)


def generate_response_loss_mask(input_ids, start_ids, end_ids, max_length):
    """Mark (with 1) only the assistant-response tokens in ``input_ids``.

    Scans for each ``start_ids`` marker (the tokens of ``<|im_start|>assistant``)
    and turns the mask on from just after it up to and including the following
    ``end_ids`` marker (``<|im_end|>``).  Everything else — system prompt, user
    turns, padding — stays 0, so the loss is computed on the assistant's words
    only.  This is what separates SFT from plain language-model pretraining.
    """
    mask = [0] * len(input_ids)
    n, i = len(input_ids), 0
    while i < n:
        # Does an assistant-start marker begin at position i?
        if input_ids[i : i + len(start_ids)] == start_ids:
            start = i + len(start_ids)
            end = start
            while end < n and input_ids[end : end + len(end_ids)] != end_ids:
                end += 1
            # Supervise the response tokens plus its closing <|im_end|>.
            for j in range(start, min(end + len(end_ids), max_length)):
                mask[j] = 1
            i = end + len(end_ids)
        else:
            i += 1
    return mask


# --------------------------------------------------------------------------- #
# Post-processing helpers                                                      #
# --------------------------------------------------------------------------- #
def make_supervised_tensors(input_ids, loss_mask):
    """Turn a padded id list + mask into shifted ``(X, Y, loss_mask)`` tensors.

    ``Y[t]`` is the token the model should predict at step ``t`` (i.e. the next
    token), and ``loss_mask[t]`` says whether that prediction is scored.
    """
    x = torch.tensor(input_ids[:-1], dtype=torch.long)
    y = torch.tensor(input_ids[1:], dtype=torch.long)
    mask = torch.tensor(loss_mask[1:], dtype=torch.long)
    return x, y, mask


def decode_completion(tokenizer, generated_ids, prompt_len):
    """Decode only the newly generated tokens (everything after the prompt)."""
    new_ids = generated_ids[prompt_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def rl_collate(batch, tokenizer, max_prompt_len):
    """Collate RL samples: left-pad tokenized prompts for batched generation.

    Left padding keeps every prompt's final token flush against the position
    where generation starts, which is what causal-LM ``generate`` expects.
    Long prompts are truncated from the left for the same reason: the most
    recent user turn and assistant generation header must remain present.
    Reference fields (``answer``, ``tools``, ...) are returned as plain lists.
    """
    prompts = [b["prompt"] for b in batch]
    original_padding_side = tokenizer.padding_side
    original_truncation_side = tokenizer.truncation_side
    try:
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        enc = tokenizer(
            prompts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_prompt_len,
            padding="max_length",
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = original_padding_side
        tokenizer.truncation_side = original_truncation_side
    out = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "prompt": prompts,
    }
    # Pass through any reference fields the reward function needs.
    for key in ("answer", "tools", "gold_tool_calls"):
        if key in batch[0]:
            out[key] = [b.get(key) for b in batch]
    return out


# --------------------------------------------------------------------------- #
# Base class                                                                   #
# --------------------------------------------------------------------------- #
class _JSONLDataset(Dataset):
    """Shared plumbing: load a jsonl file and cache assistant-span markers."""

    def __init__(self, data_path, tokenizer, max_length):
        super().__init__()
        self.data_path = os.fspath(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_id = tokenizer.pad_token_id
        self.samples = load_jsonl(data_path)

    def __len__(self):
        return len(self.samples)

    def _response_markers(self):
        """Token ids that bracket an assistant response in a rendered prompt."""
        # Include the header newline: it is prompt formatting, not an answer token.
        start = self.tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        end = self.tokenizer("<|im_end|>", add_special_tokens=False).input_ids
        return start, end

    def dedup_payload(self, sample):
        """Return the fields that affect one encoded training example."""
        raise TypeError(f"{type(self).__name__} does not support supervised splitting")

    def split_group_payload(self, sample):
        """Return fields whose examples must stay on the same side of a split."""
        return self.dedup_payload(sample)

    def dedup_key(self, sample):
        return _stable_payload_hash(self.dedup_payload(sample))

    def split_group_key(self, sample):
        return _stable_payload_hash(self.split_group_payload(sample))


class IndexedDataset(Dataset):
    """A deterministic dataset view that preserves supervised metadata.

    ``torch.utils.data.Subset`` hides attributes such as ``pad_id`` that the
    shared trainer needs to construct an attention mask.  This small view keeps
    those attributes while storing only integer indices into the source
    dataset, so a split does not duplicate tokenized samples or JSON records.
    """

    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = tuple(indices)
        for index in self.indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(f"dataset indices must be integers, got {index!r}")
            if not 0 <= index < len(dataset):
                raise IndexError(f"dataset index out of range: {index}")
        self.pad_id = getattr(dataset, "pad_id", None)
        self.tokenizer = getattr(dataset, "tokenizer", None)
        self.max_length = getattr(dataset, "max_length", None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]


def _canonical_payload(payload):
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_payload_hash(payload):
    canonical = _canonical_payload(payload).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _supervised_tensor_payload(tensors):
    """Return a JSON-stable representation of the exact training tensors."""
    input_ids, target_ids, loss_mask = tensors
    return {
        "input_ids": input_ids.tolist(),
        "target_ids": target_ids.tolist(),
        "loss_mask": loss_mask.tolist(),
    }


# --------------------------------------------------------------------------- #
# 1. Pretraining — plain next-token prediction over raw text                   #
# --------------------------------------------------------------------------- #
class PretrainDataset(_JSONLDataset):
    """Every non-pad token is a target; the model learns to predict text."""

    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__(data_path, tokenizer, max_length)

    def _encode_sample(self, sample):
        if not isinstance(sample, dict) or "text" not in sample:
            raise ValueError("pretrain samples must contain a 'text' field")
        # Wrap in bos/eos so the model learns document boundaries.
        text = str(sample["text"])
        text = f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        ids = pad_to_length(ids, self.max_length, self.pad_id)
        # Supervise every real (non-padding) token.
        loss_mask = [1 if tok != self.pad_id else 0 for tok in ids]
        return make_supervised_tensors(ids, loss_mask)

    def dedup_payload(self, sample):
        # Hash what training consumes, not the raw text.  Text that differs only
        # after right truncation is therefore one exact training example.
        return _supervised_tensor_payload(self._encode_sample(sample))

    def __getitem__(self, index):
        return self._encode_sample(self.samples[index])


# --------------------------------------------------------------------------- #
# 2. SFT — supervised fine-tuning; loss only on assistant replies              #
# --------------------------------------------------------------------------- #
class SFTDataset(_JSONLDataset):
    """Teaches instruction following: score only the assistant turns."""

    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__(data_path, tokenizer, max_length)
        self.start_ids, self.end_ids = self._response_markers()

    @staticmethod
    def _conversation_payload(sample, *, hide_assistant_content=False):
        if not isinstance(sample, dict):
            raise ValueError("SFT samples must be JSON objects")
        conversations = sample.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError("SFT samples must contain a non-empty 'conversations' list")

        payload = []
        for message_index, message in enumerate(conversations):
            if not isinstance(message, dict):
                raise ValueError(f"SFT message {message_index} must be a JSON object")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError(
                    f"SFT message {message_index} must contain string role/content fields"
                )
            payload.append(
                {
                    "role": role,
                    "content": None
                    if hide_assistant_content and role == "assistant"
                    else content,
                }
            )
        return {"conversations": payload}

    def _encode_conversation(self, conversations):
        prompt = build_chat_prompt(self.tokenizer, conversations, add_generation_prompt=False)
        ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        # Preserve the newest assistant response when a conversation is too
        # long. Right truncation can remove the response marker entirely and
        # silently turn the sample into a zero-gradient batch item.
        return pad_to_length(
            ids,
            self.max_length,
            self.pad_id,
            truncation_side="left",
        )

    def _encode_sample(self, sample, *, index=None):
        # Normalising to role/content mirrors the fields consumed by this chat
        # template and deliberately ignores record/message metadata.
        conversations = self._conversation_payload(sample)["conversations"]
        ids = self._encode_conversation(conversations)
        loss_mask = generate_response_loss_mask(ids, self.start_ids, self.end_ids, self.max_length)
        if not any(loss_mask):
            location = f" {index}" if index is not None else ""
            raise ValueError(
                f"SFT sample{location} has no complete assistant response within "
                f"max_length={self.max_length}; increase --max_length"
            )
        return make_supervised_tensors(ids, loss_mask)

    def dedup_payload(self, sample):
        # This is the exact post-template, post-truncation, shifted example used
        # by the loss.  Differences that the model cannot see are duplicates.
        return _supervised_tensor_payload(self._encode_sample(sample))

    def split_group_payload(self, sample):
        # Alternative assistant answers share one prompt group.  Encode the
        # answer-hidden skeleton through the real chat template and left-
        # truncation path so differences outside the visible window cannot put
        # equivalent prompts on opposite sides of the split.
        conversations = self._conversation_payload(
            sample,
            hide_assistant_content=True,
        )["conversations"]
        for message in conversations:
            if message["role"] == "assistant":
                message["content"] = ""
        ids = self._encode_conversation(conversations)
        assistant_slots = generate_response_loss_mask(
            ids,
            self.start_ids,
            self.end_ids,
            self.max_length,
        )
        # Token ids are non-negative.  -1 is therefore an unambiguous structural
        # answer slot, unlike a string sentinel that real content could match.
        visible_skeleton = [
            -1 if is_assistant_slot else token_id
            for token_id, is_assistant_slot in zip(ids, assistant_slots, strict=True)
        ]
        return {"visible_prompt_skeleton": visible_skeleton}

    def __getitem__(self, index):
        return self._encode_sample(self.samples[index], index=index)


def _split_fingerprint(stage, seed, validation_fraction, train_keys, validation_keys):
    payload = {
        "algorithm": "sha256-group-prefix-v1",
        "stage": stage,
        "seed": seed,
        "requested_validation_fraction": validation_fraction,
        "train_keys": sorted(train_keys),
        "validation_keys": sorted(validation_keys),
    }
    return f"sha256:{_stable_payload_hash(payload)}"


def split_supervised_dataset(dataset, *, validation_fraction, seed):
    """Exact-deduplicate and deterministically split Pretrain/SFT datasets.

    Records that encode to identical supervised examples are retained once.
    Unique records are then assigned as complete stage-specific groups, so SFT
    examples with the same conversation skeleton but alternative assistant
    answers cannot straddle the split.  Group order comes from SHA-256 rather
    than process-global RNG state, making the result identical on every DDP
    rank and stable when the source file is reordered.
    """

    if not isinstance(dataset, (PretrainDataset, SFTDataset)):
        raise TypeError("split_supervised_dataset requires PretrainDataset or SFTDataset")
    if isinstance(validation_fraction, bool) or not isinstance(
        validation_fraction, (int, float)
    ):
        raise TypeError("validation_fraction must be a number in [0, 1)")
    validation_fraction = float(validation_fraction)
    if not math.isfinite(validation_fraction) or not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be finite and in [0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    stage = "pretrain" if isinstance(dataset, PretrainDataset) else "sft"
    unique_entries = []
    seen_dedup_keys = set()
    groups = {}

    for index, sample in enumerate(dataset.samples):
        try:
            dedup_key = dataset.dedup_key(sample)
            group_key = dataset.split_group_key(sample)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{dataset.data_path}: sample {index}: {exc}") from exc
        if dedup_key in seen_dedup_keys:
            continue
        seen_dedup_keys.add(dedup_key)
        entry = (index, dedup_key, group_key)
        unique_entries.append(entry)
        groups.setdefault(group_key, []).append(entry)

    raw_count = len(dataset)
    unique_count = len(unique_entries)
    duplicate_count = raw_count - unique_count

    if validation_fraction == 0:
        validation_group_keys = set()
    else:
        if len(groups) < 2:
            raise ValueError(
                "a positive validation_fraction requires at least two unique split groups"
            )
        target_validation_count = min(
            unique_count - 1,
            max(1, round(unique_count * validation_fraction)),
        )
        ordered_group_keys = sorted(
            groups,
            key=lambda key: (
                hashlib.sha256(f"{seed}\0{key}".encode()).hexdigest(),
                key,
            ),
        )

        # A prefix of a seed-hashed group order gives deterministic assignment.
        # Select the closest prefix while reserving at least one group for train.
        prefix_count = 0
        candidates = []
        for prefix_length, group_key in enumerate(ordered_group_keys[:-1], start=1):
            prefix_count += len(groups[group_key])
            candidates.append(
                (
                    abs(prefix_count - target_validation_count),
                    prefix_count > target_validation_count,
                    prefix_length,
                )
            )
        best_prefix_length = min(candidates)[2]
        validation_group_keys = set(ordered_group_keys[:best_prefix_length])

    # Canonical key order makes the dataset view itself stable when a source
    # JSONL is reordered; a seeded sampler then produces the same sample order.
    train_entries = sorted(
        (entry for entry in unique_entries if entry[2] not in validation_group_keys),
        key=lambda entry: entry[1],
    )
    validation_entries = sorted(
        (entry for entry in unique_entries if entry[2] in validation_group_keys),
        key=lambda entry: entry[1],
    )
    train_indices = [entry[0] for entry in train_entries]
    validation_indices = [entry[0] for entry in validation_entries]
    train_keys = [entry[1] for entry in train_entries]
    validation_keys = [entry[1] for entry in validation_entries]

    train_dataset = IndexedDataset(dataset, train_indices)
    validation_dataset = (
        IndexedDataset(dataset, validation_indices) if validation_indices else None
    )
    train_group_count = len(groups) - len(validation_group_keys)
    validation_group_count = len(validation_group_keys)
    metadata = {
        "schema_version": 1,
        "algorithm": "sha256-group-prefix-v1",
        "deduplication": "encoded-supervised-tensors-exact-sha256-v2",
        "stage": stage,
        "source_path": dataset.data_path,
        "max_length": dataset.max_length,
        "seed": seed,
        "requested_validation_fraction": validation_fraction,
        "actual_validation_fraction": (
            len(validation_indices) / unique_count if unique_count else 0.0
        ),
        "raw_samples": raw_count,
        "unique_samples": unique_count,
        "duplicates_removed": duplicate_count,
        "group_count": len(groups),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "train_groups": train_group_count,
        "validation_groups": validation_group_count,
        "split_fingerprint": _split_fingerprint(
            stage,
            seed,
            validation_fraction,
            train_keys,
            validation_keys,
        ),
    }
    return train_dataset, validation_dataset, metadata


# --------------------------------------------------------------------------- #
# 3. DPO — Direct Preference Optimization; chosen vs. rejected pairs           #
# --------------------------------------------------------------------------- #
class DPODataset(_JSONLDataset):
    """Returns a preferred and a dispreferred completion for the same prompt.

    DPO pushes the policy to raise the likelihood of ``chosen`` relative to
    ``rejected`` (compared against a frozen reference model in the trainer).
    Both are masked so only the assistant tokens count.
    """

    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__(data_path, tokenizer, max_length)
        self.start_ids, self.end_ids = self._response_markers()

    def _encode_pair(self, chosen, rejected, index):
        if not chosen or not rejected:
            raise ValueError(f"DPO sample {index} must contain chosen and rejected messages")
        if chosen[:-1] != rejected[:-1]:
            raise ValueError(f"DPO sample {index} chosen/rejected prompts must be identical")
        if chosen[-1].get("role") != "assistant" or rejected[-1].get("role") != "assistant":
            raise ValueError(f"DPO sample {index} must end both sides with an assistant response")

        prompt_messages = chosen[:-1]
        prompt_text = build_chat_prompt(
            self.tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False).input_ids

        def response_suffix(messages):
            full_text = build_chat_prompt(
                self.tokenizer,
                messages,
                add_generation_prompt=False,
            )
            full_ids = self.tokenizer(full_text, add_special_tokens=False).input_ids
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError(
                    f"DPO sample {index} chat template does not preserve the common prompt"
                )
            return full_ids[len(prompt_ids) :]

        chosen_response = response_suffix(chosen)
        rejected_response = response_suffix(rejected)
        response_budget = max(len(chosen_response), len(rejected_response))
        prompt_budget = self.max_length - response_budget
        if prompt_budget < len(self.start_ids):
            raise ValueError(
                f"DPO sample {index} response does not fit within max_length={self.max_length}; "
                "increase --max_length"
            )
        common_prompt = prompt_ids[-prompt_budget:]
        if common_prompt[-len(self.start_ids) :] != self.start_ids:
            raise ValueError(
                f"DPO sample {index} cannot retain the assistant marker within "
                f"max_length={self.max_length}; increase --max_length"
            )

        def tensors(response_ids):
            ids = common_prompt + response_ids
            loss_mask = [0] * len(common_prompt) + [1] * len(response_ids)
            ids = pad_to_length(ids, self.max_length, self.pad_id)
            loss_mask = pad_to_length(loss_mask, self.max_length, 0)
            return make_supervised_tensors(ids, loss_mask)

        return tensors(chosen_response), tensors(rejected_response)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen, rejected = self._encode_pair(
            sample["chosen"],
            sample["rejected"],
            index,
        )
        x_c, y_c, m_c = chosen
        x_r, y_r, m_r = rejected
        return {
            "x_chosen": x_c, "y_chosen": y_c, "mask_chosen": m_c,
            "x_rejected": x_r, "y_rejected": y_r, "mask_rejected": m_r,
        }


# --------------------------------------------------------------------------- #
# 4. RLAIF — RL from AI Feedback; prompts for online generation                #
# --------------------------------------------------------------------------- #
class RLAIFDataset(_JSONLDataset):
    """Yields prompts to roll out; a reward model/AI judge scores completions.

    Unlike the supervised stages there are no targets here: the trainer
    samples completions from the current policy, scores them, and updates the
    policy (e.g. GRPO/PPO).  We therefore return the *generation-ready* prompt
    text plus a reference ``answer`` the reward function may use.
    """

    def __init__(self, data_path, tokenizer, max_length=1024):
        super().__init__(data_path, tokenizer, max_length)

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = build_chat_prompt(
            self.tokenizer, sample["conversations"], add_generation_prompt=True
        )
        return {"prompt": prompt, "answer": sample.get("answer", "")}


# --------------------------------------------------------------------------- #
# 5. AgentRL — RL for tool-using agents                                        #
# --------------------------------------------------------------------------- #
class AgentRLDataset(_JSONLDataset):
    """Prompts for agentic rollouts where the policy may call tools.

    Extends RLAIF with the tool schemas available for the episode (injected
    into the chat template if the tokenizer supports a ``tools`` argument) and
    optional gold references (final answer and/or expected tool calls) that
    the environment/reward uses to score a trajectory.
    """

    def __init__(self, data_path, tokenizer, max_length=2048):
        super().__init__(data_path, tokenizer, max_length)

    def __getitem__(self, index):
        sample = self.samples[index]
        tools = sample.get("tools")
        # Pass tools to the template when provided so the system prompt lists
        # the callable functions; fall back gracefully if unsupported.
        try:
            prompt = self.tokenizer.apply_chat_template(
                sample["conversations"],
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
        except TypeError:
            prompt = build_chat_prompt(
                self.tokenizer, sample["conversations"], add_generation_prompt=True
            )
        return {
            "prompt": prompt,
            "tools": tools,
            "answer": sample.get("answer", ""),
            "gold_tool_calls": sample.get("gold_tool_calls", []),
        }
