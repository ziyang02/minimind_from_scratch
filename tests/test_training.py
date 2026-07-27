from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, TensorDataset
from transformers import AutoTokenizer

from model.model import NinjaMindConfig, NinjaMindForCausalLM
from trainer.train_dpo import (
    dpo_loss,
    dpo_metrics,
    freeze_reference_model,
    preference_forward,
    sequence_log_probs,
)
from trainer.train_grpo import group_advantages
from trainer.trainer_utils import (
    DistributedContext,
    add_distributed_args,
    build_dataloader,
    compute_gae,
    containment_reward,
    distributed_mean_metrics,
    evaluate_supervised,
    load_weights,
    masked_mean,
    normalize_token_weighted_gradients,
    parse_distributed_env,
    sample_generate,
    save_checkpoint,
    setup_distributed,
    tokenizer_fingerprint,
    train_supervised,
    whiten,
)


def test_sequence_log_probs_scores_only_masked_response_tokens():
    logits = torch.tensor(
        [
            [[2.0, 0.0, -1.0], [0.0, 3.0, 0.0], [1.0, 0.0, 2.0]],
            [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
        ]
    )
    labels = torch.tensor([[0, 1, 2], [1, 0, 2]])
    mask = torch.tensor([[0, 1, 1], [1, 0, 1]])

    token_logps = F.log_softmax(logits, dim=-1).gather(
        -1, labels.unsqueeze(-1)
    ).squeeze(-1)
    expected = (token_logps * mask).sum(-1)
    assert torch.allclose(sequence_log_probs(logits, labels, mask), expected)
    assert torch.allclose(
        sequence_log_probs(logits, labels, mask, average_log_prob=True),
        expected / mask.sum(-1),
    )


def test_sequence_log_probs_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="expected logits"):
        sequence_log_probs(torch.zeros(2, 3, 4), torch.zeros(2, 2, dtype=torch.long), torch.ones(2, 2))


def test_dpo_loss_matches_standard_reference_anchored_formula():
    policy_chosen = torch.tensor([3.0, 1.0], requires_grad=True)
    policy_rejected = torch.tensor([1.0, 2.0], requires_grad=True)
    ref_chosen = torch.tensor([1.5, 0.5])
    ref_rejected = torch.tensor([1.0, 0.5])
    beta = 0.2

    losses, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta
    )
    expected_logits = beta * (
        (policy_chosen - policy_rejected) - (ref_chosen - ref_rejected)
    )
    assert torch.allclose(losses, -F.logsigmoid(expected_logits))
    assert torch.allclose(chosen_rewards, beta * (policy_chosen - ref_chosen))
    assert torch.allclose(rejected_rewards, beta * (policy_rejected - ref_rejected))
    assert not chosen_rewards.requires_grad
    assert not rejected_rewards.requires_grad

    metrics = dpo_metrics(losses, chosen_rewards, rejected_rewards)
    assert set(metrics) == {
        "loss",
        "preference_accuracy",
        "chosen_reward",
        "rejected_reward",
        "reward_margin",
    }
    assert metrics["preference_accuracy"].item() == pytest.approx(0.5)
    assert metrics["reward_margin"].item() == pytest.approx(
        (chosen_rewards - rejected_rewards).mean().item()
    )
    losses.mean().backward()
    assert policy_chosen.grad is not None


def test_dpo_rejects_non_positive_beta_and_freezes_reference():
    values = torch.zeros(1)
    with pytest.raises(ValueError, match="beta"):
        dpo_loss(values, values, values, values, beta=0)

    reference = nn.Sequential(nn.Linear(2, 2), nn.Dropout(0.5)).train()
    assert freeze_reference_model(reference) is reference
    assert not reference.training
    assert all(not parameter.requires_grad for parameter in reference.parameters())


def test_dpo_policy_forward_surfaces_moe_router_aux_loss():
    config = NinjaMindConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=32,
        max_position_embeddings=16,
        use_moe=True,
        num_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.1,
        dropout=0.0,
        flash_attn=False,
    )
    model = NinjaMindForCausalLM(config).train()
    batch = {
        "x_chosen": torch.tensor([[1, 3, 4]]),
        "y_chosen": torch.tensor([[3, 4, 2]]),
        "mask_chosen": torch.tensor([[0, 1, 1]]),
        "x_rejected": torch.tensor([[1, 5, 6]]),
        "y_rejected": torch.tensor([[5, 6, 2]]),
        "mask_rejected": torch.tensor([[0, 1, 1]]),
    }

    _, _, aux_loss = preference_forward(model, batch, pad_id=0)

    assert aux_loss > 0
    aux_loss.backward()
    assert model.model.layers[0].mlp.gate.weight.grad is not None


def test_compute_gae_respects_terminal_and_padding_mask():
    rewards = torch.tensor([[0.0, 1.0, 99.0], [2.0, 99.0, 99.0]])
    values = torch.zeros_like(rewards)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    advantages, returns = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert torch.equal(advantages, torch.tensor([[1.0, 1.0, 0.0], [2.0, 0.0, 0.0]]))
    assert torch.equal(returns, advantages)

    with pytest.raises(ValueError, match="identical shapes"):
        compute_gae(rewards, values[:, :2], mask)


def test_masked_statistics_and_rule_reward():
    values = torch.tensor([[1.0, 2.0, 100.0], [3.0, 4.0, 100.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    assert masked_mean(values, mask).item() == pytest.approx(2.5)
    normalized = whiten(values, mask)
    assert masked_mean(normalized, mask).item() == pytest.approx(0.0, abs=1e-6)
    assert masked_mean(normalized.square(), mask).item() == pytest.approx(1.0, abs=1e-5)
    assert torch.equal(normalized[:, 2], torch.zeros(2))
    assert containment_reward("The answer is Sydney.", "Sydney") == 1.0
    assert containment_reward("The answer is Sydney.", "Melbourne") == 0.0
    assert containment_reward("anything", "   ") == 0.0


def test_grpo_group_advantages_are_normalized_per_prompt():
    rewards = torch.tensor([0.0, 2.0, 10.0, 10.0])
    advantages = group_advantages(rewards, group_size=2)
    assert advantages.shape == (4, 1)
    assert torch.allclose(advantages[:2].mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages[2:], torch.zeros(2, 1))
    with pytest.raises(ValueError, match="at least two"):
        group_advantages(rewards, group_size=1)


class _ScriptedPolicy(nn.Module):
    """Emit EOS for row 0 immediately and for row 1 one token later."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(
        self, input_ids, attention_mask=None, use_cache=False, past_key_values=None
    ):
        del attention_mask, use_cache, past_key_values
        logits = torch.full((*input_ids.shape, 4), -1000.0)
        if self.calls == 0:
            logits[0, -1, 2] = 1000.0  # EOS now.
            logits[1, -1, 1] = 1000.0  # Continue once.
        else:
            logits[:, -1, 2] = 1000.0
        self.calls += 1
        return SimpleNamespace(logits=logits, past_key_values=())


def test_rollout_mask_includes_eos_and_excludes_post_eos_padding():
    policy = _ScriptedPolicy().train()
    prompts = torch.tensor([[1, 1], [1, 1]])
    prompt_mask = torch.ones_like(prompts)
    sequence, generated_mask, full_mask = sample_generate(
        policy,
        prompts,
        prompt_mask,
        max_new_tokens=3,
        eos_id=2,
        pad_id=0,
        top_k=1,
    )
    assert torch.equal(generated_mask, torch.tensor([[1, 0, 0], [1, 1, 0]]))
    assert torch.equal(sequence[:, 2:], torch.tensor([[2, 0, 0], [1, 2, 0]]))
    assert torch.equal(full_mask[:, 2:], generated_mask)
    assert policy.calls == 3  # fixed count, even though every row hit EOS at step 2
    assert policy.training  # sample_generate restores the caller's mode.


def test_distributed_environment_parsing_and_cli_aliases():
    assert parse_distributed_env({}) == (0, 0, 1)
    env = {"RANK": "2", "LOCAL_RANK": "1", "WORLD_SIZE": "4"}
    assert parse_distributed_env(env) == (2, 1, 4)
    assert parse_distributed_env(env, local_rank=3) == (2, 3, 4)
    with pytest.raises(ValueError, match="RANK"):
        parse_distributed_env({"RANK": "2", "WORLD_SIZE": "2"})
    with pytest.raises(ValueError, match="WORLD_SIZE"):
        parse_distributed_env({"WORLD_SIZE": "many"})

    parser = ArgumentParser()
    add_distributed_args(parser)
    args = parser.parse_args(["--local_rank", "3", "--dist-backend", "gloo"])
    assert args.local_rank == 3
    assert args.dist_backend == "gloo"


def test_distributed_metric_mean_reduces_every_rank(monkeypatch):
    context = DistributedContext(rank=0, world_size=2, device=torch.device("cpu"))

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(values, op):
        assert op == torch.distributed.ReduceOp.SUM
        values.add_(torch.tensor([3.0, 0.5]))

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    averaged = distributed_mean_metrics(
        {"loss": 1.0, "preference_accuracy": 0.5},
        context,
    )
    assert averaged == {"loss": 2.0, "preference_accuracy": 0.5}


def test_setup_distributed_has_side_effect_free_single_process_fallback(monkeypatch):
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(name, raising=False)
    args = Namespace(device="cpu", local_rank=None, dist_backend="auto")
    context = setup_distributed(args)
    assert context == DistributedContext(device=torch.device("cpu"))
    assert not context.distributed
    assert context.is_main


def test_dataloader_keeps_dataset_smaller_than_batch_and_shards_when_requested():
    dataset = TensorDataset(torch.arange(2))
    args = Namespace(batch_size=8, num_workers=0, device="cpu")
    loader = build_dataloader(dataset, args, drop_last=False)
    assert len(loader) == 1
    assert next(iter(loader))[0].numel() == 2

    distributed = DistributedContext(rank=1, local_rank=1, world_size=2)
    sharded = build_dataloader(dataset, args, distributed, shuffle=False)
    assert sharded.sampler.num_replicas == 2
    assert sharded.sampler.rank == 1


class _TripleDataset(Dataset):
    def __init__(self, size=3):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x = torch.tensor([index % 3, (index + 1) % 3])
        y = torch.tensor([(index + 1) % 3, (index + 2) % 3])
        return x, y, torch.ones(2)


class _UnevenTokenDataset(Dataset):
    """Two conflicting examples with one versus three supervised tokens."""

    examples = (
        (
            torch.tensor([1, 1, 1]),
            torch.tensor([0, 0, 0]),
            torch.tensor([1, 0, 0]),
        ),
        (
            torch.tensor([1, 1, 1]),
            torch.tensor([1, 1, 1]),
            torch.tensor([1, 1, 1]),
        ),
    )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 4)
        self.head = nn.Linear(4, 3)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def _train_args(**overrides):
    values = {
        "batch_size": 1,
        "num_workers": 0,
        "device": "cpu",
        "lr": 1e-2,
        "epochs": 1,
        "accumulation_steps": 2,
        "grad_clip": 1.0,
        "max_steps": 0,
        "log_interval": 100,
    }
    values.update(overrides)
    return Namespace(**values)


def test_supervised_loop_flushes_partial_accumulation_window():
    model = _TinyLM()
    saves = []
    stats = train_supervised(
        model,
        _TripleDataset(size=3),
        _train_args(),
        torch.device("cpu"),
        lambda saved_model: saves.append(saved_model),
    )
    assert {
        key: stats[key] for key in ("optimizer_steps", "micro_steps", "total_steps")
    } == {"optimizer_steps": 2, "micro_steps": 3, "total_steps": 2}
    assert len(stats["history"]) == 1
    assert stats["history"][0]["epoch_complete"] is True
    assert saves == [model]


def test_accumulation_matches_one_token_weighted_full_batch_update():
    torch.manual_seed(31)
    accumulated = _TinyLM()
    full_batch = _TinyLM()
    full_batch.load_state_dict(accumulated.state_dict())
    dataset = _UnevenTokenDataset()

    train_supervised(
        accumulated,
        dataset,
        _train_args(batch_size=1, accumulation_steps=2),
        torch.device("cpu"),
        lambda _model: None,
    )
    train_supervised(
        full_batch,
        dataset,
        _train_args(batch_size=2, accumulation_steps=1),
        torch.device("cpu"),
        lambda _model: None,
    )

    for accumulated_parameter, batched_parameter in zip(
        accumulated.parameters(),
        full_batch.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(
            accumulated_parameter,
            batched_parameter,
            rtol=1e-6,
            atol=1e-7,
        )


def test_ddp_gradient_normalization_uses_global_token_count(monkeypatch):
    parameter = nn.Parameter(torch.tensor([0.0]))
    # DDP has already averaged local NLL-sum gradients 3 and 400.
    parameter.grad = torch.tensor([(3.0 + 400.0) / 2])
    context = DistributedContext(world_size=2, device=torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def fake_all_reduce(token_count, op):
        assert op == torch.distributed.ReduceOp.SUM
        token_count.add_(100)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    global_tokens = normalize_token_weighted_gradients([parameter], 1, context)

    assert global_tokens == 101
    assert parameter.grad.item() == pytest.approx((3.0 + 400.0) / 101)


def test_final_planned_step_only_emits_a_finalized_checkpoint():
    events = []

    def record_save(_model, *, optimizer, scaler, training_state, kind):
        del optimizer, scaler
        events.append(
            (
                kind,
                training_state["optimizer_step"],
                training_state["checkpoint_finalized"],
            )
        )

    train_supervised(
        _TinyLM(),
        _TripleDataset(size=4),
        _train_args(
            accumulation_steps=1,
            max_steps=2,
            save_interval=2,
        ),
        torch.device("cpu"),
        record_save,
    )

    assert events == [("latest", 2, True)]


def test_resuming_finalized_checkpoint_reemits_recoverable_outputs(tmp_path):
    checkpoint = tmp_path / "finalized.pth"
    args = _train_args(
        accumulation_steps=1,
        max_steps=1,
        save_interval=1,
        seed=71,
        resume="",
    )

    def persist(model, *, optimizer, scaler, training_state, kind):
        assert kind == "latest"
        save_checkpoint(
            str(checkpoint),
            model,
            optimizer=optimizer,
            scaler=scaler,
            stage="finalization-test",
            training_state=training_state,
        )

    train_supervised(
        _TinyLM(),
        _TripleDataset(size=3),
        args,
        torch.device("cpu"),
        persist,
        stage="finalization-test",
    )

    reemitted = []
    artifact_updates = []

    def recover(model, *, optimizer, scaler, training_state, kind):
        del model, optimizer, scaler
        reemitted.append((kind, training_state["checkpoint_finalized"]))

    train_supervised(
        _TinyLM(),
        _TripleDataset(size=3),
        Namespace(**{**vars(args), "resume": str(checkpoint)}),
        torch.device("cpu"),
        recover,
        epoch_callback=lambda history, state: artifact_updates.append(
            (len(history), state["checkpoint_finalized"])
        ),
        stage="finalization-test",
    )

    assert reemitted == [("latest", True)]
    assert artifact_updates == [(1, True)]


def test_checkpoint_loader_supports_structured_and_legacy_state_dicts(tmp_path):
    source = nn.Linear(2, 2)
    legacy_path = tmp_path / "legacy.pth"
    torch.save(source.state_dict(), legacy_path)

    legacy_target = nn.Linear(2, 2)
    metadata = load_weights(legacy_target, str(legacy_path), torch.device("cpu"))
    assert metadata["legacy"] is True
    assert torch.equal(legacy_target.weight, source.weight)

    structured_path = tmp_path / "structured.pth"
    assert save_checkpoint(
        str(structured_path),
        source,
        args=Namespace(lr=1e-3),
        stage="test",
        step=7,
    )
    payload = torch.load(structured_path, map_location="cpu")
    assert payload["format_version"] == 1
    assert payload["stage"] == "test"
    assert payload["step"] == 7
    assert payload["training_args"]["lr"] == pytest.approx(1e-3)

    structured_target = nn.Linear(2, 2)
    loaded_metadata = load_weights(
        structured_target, str(structured_path), torch.device("cpu")
    )
    assert loaded_metadata["format_version"] == 1
    assert torch.equal(structured_target.bias, source.bias)


def test_tokenizer_fingerprint_covers_vocabulary_and_chat_template():
    tokenizer = AutoTokenizer.from_pretrained("tokenizer", local_files_only=True)
    original_template = tokenizer.chat_template
    original_fingerprint = tokenizer_fingerprint(tokenizer)

    assert tokenizer_fingerprint(tokenizer) == original_fingerprint
    tokenizer.chat_template = f"{original_template}\n"
    assert tokenizer_fingerprint(tokenizer) != original_fingerprint


class _MaskedValidationDataset(Dataset):
    pad_id = 0

    def __init__(self, zero_mask=False):
        masks = [
            torch.zeros(3, dtype=torch.long)
            if zero_mask
            else torch.tensor([1, 0, 0]),
            torch.zeros(3, dtype=torch.long)
            if zero_mask
            else torch.tensor([1, 1, 0]),
        ]
        self.examples = [
            (torch.tensor([1, 0, 0]), torch.tensor([0, 2, 1]), masks[0]),
            (torch.tensor([2, 2, 0]), torch.tensor([1, 0, 2]), masks[1]),
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class _FixedValidationLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "logit_table",
            torch.tensor(
                [
                    [100.0, -100.0, -100.0],
                    [2.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
        )
        self.forward_modes = []
        self.attention_masks = []

    def forward(self, input_ids, attention_mask=None):
        self.forward_modes.append((self.training, torch.is_grad_enabled()))
        self.attention_masks.append(attention_mask.detach().clone())
        return SimpleNamespace(logits=self.logit_table[input_ids])


class _LazyValidationBufferLM(_FixedValidationLM):
    def forward(self, input_ids, attention_mask=None):
        if not hasattr(self, "lazy_buffer"):
            self.register_buffer("lazy_buffer", torch.ones(1))
        return super().forward(input_ids, attention_mask=attention_mask)


def test_validation_is_token_weighted_padding_aware_and_restores_mode():
    model = _FixedValidationLM().train()
    dataset = _MaskedValidationDataset()
    args = Namespace(batch_size=1, num_workers=0, device="cpu")

    metrics = evaluate_supervised(model, dataset, args, torch.device("cpu"))

    first = F.cross_entropy(
        model.logit_table[1].unsqueeze(0),
        torch.tensor([0]),
        reduction="sum",
    )
    second = F.cross_entropy(
        model.logit_table[torch.tensor([2, 2])],
        torch.tensor([1, 0]),
        reduction="sum",
    )
    expected_ce = ((first + second) / 3).item()
    assert metrics["validation_ce"] == pytest.approx(expected_ce)
    assert metrics["validation_perplexity"] == pytest.approx(torch.exp(torch.tensor(expected_ce)))
    assert metrics["validation_tokens"] == 3
    assert metrics["perplexity_overflow"] is False
    assert model.training
    assert model.forward_modes == [(False, False), (False, False)]
    assert torch.equal(model.attention_masks[0], torch.tensor([[1, 0, 0]], dtype=torch.bool))
    assert torch.equal(model.attention_masks[1], torch.tensor([[1, 1, 0]], dtype=torch.bool))


def test_validation_rejects_a_split_without_target_tokens_and_restores_mode():
    model = _FixedValidationLM().train()
    with pytest.raises(ValueError, match="no supervised target tokens"):
        evaluate_supervised(
            model,
            _MaskedValidationDataset(zero_mask=True),
            Namespace(batch_size=2, num_workers=0, device="cpu"),
            torch.device("cpu"),
        )
    assert model.training


def test_validation_lazy_buffers_remain_usable_by_ddp_afterward():
    model = _LazyValidationBufferLM().train()
    evaluate_supervised(
        model,
        _MaskedValidationDataset(),
        Namespace(batch_size=2, num_workers=0, device="cpu"),
        torch.device("cpu"),
    )

    assert not torch.is_inference(model.lazy_buffer)


class _DropoutLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 8)
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Linear(8, 3)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.head(self.dropout(self.embedding(input_ids))))


def _assert_nested_state_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_state_equal(left_item, right_item)
    else:
        assert left == right


def test_supervised_resume_matches_uninterrupted_model_optimizer_and_rng(tmp_path):
    identity = {"dataset": "tiny-v1"}
    common_args = {
        "batch_size": 1,
        "num_workers": 0,
        "device": "cpu",
        "lr": 1e-2,
        "epochs": 1,
        "accumulation_steps": 2,
        "grad_clip": 1.0,
        "max_steps": 4,
        "log_interval": 100,
        "save_interval": 2,
        "seed": 123,
        "resume": "",
    }
    interrupted_path = tmp_path / "interrupted.pth"
    continuous_path = tmp_path / "continuous.pth"
    resumed_path = tmp_path / "resumed.pth"

    def checkpoint_callback(target_for_final):
        def save(model, *, optimizer, scaler, training_state, kind):
            del kind
            target = (
                interrupted_path
                if training_state["optimizer_step"] == 2
                else target_for_final
            )
            save_checkpoint(
                str(target),
                model,
                optimizer=optimizer,
                scaler=scaler,
                step=training_state["optimizer_step"],
                stage="resume-test",
                training_state=training_state,
            )

        return save

    torch.manual_seed(77)
    continuous_model = _DropoutLM()
    continuous_stats = train_supervised(
        continuous_model,
        _TripleDataset(size=8),
        Namespace(**common_args),
        torch.device("cpu"),
        checkpoint_callback(continuous_path),
        stage="resume-test",
        run_identity=identity,
    )
    assert interrupted_path.exists()
    assert continuous_path.exists()

    torch.manual_seed(999)
    resumed_model = _DropoutLM()
    resumed_stats = train_supervised(
        resumed_model,
        _TripleDataset(size=8),
        Namespace(**{**common_args, "resume": str(interrupted_path)}),
        torch.device("cpu"),
        checkpoint_callback(resumed_path),
        stage="resume-test",
        run_identity=identity,
    )

    continuous = torch.load(continuous_path, map_location="cpu", weights_only=True)
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=True)
    _assert_nested_state_equal(
        continuous["model_state_dict"],
        resumed["model_state_dict"],
    )
    _assert_nested_state_equal(
        continuous["optimizer_state_dict"],
        resumed["optimizer_state_dict"],
    )
    torch.testing.assert_close(
        continuous["rng_state"]["torch_cpu"],
        resumed["rng_state"]["torch_cpu"],
        rtol=0,
        atol=0,
    )
    assert continuous_stats["optimizer_steps"] == resumed_stats["optimizer_steps"] == 4
    assert continuous_stats["history"][-1]["train_ce"] == pytest.approx(
        resumed_stats["history"][-1]["train_ce"]
    )
