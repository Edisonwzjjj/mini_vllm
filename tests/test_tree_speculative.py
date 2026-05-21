"""Tests for target-model tree speculative verification."""

import pytest

from mini_vllm.attention_patch import paged_ctx
from mini_vllm.config import EngineConfig
from mini_vllm.draft_tree import build_dummy_tree
from mini_vllm.eagle_runner import EagleRunner
from mini_vllm.model_runner import ModelRunner
from mini_vllm.sampling_params import SamplingParams
from mini_vllm.sequence import Sequence


MODEL_PATH = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def model_runner() -> ModelRunner:
    return ModelRunner(
        model_path=MODEL_PATH,
        block_size=16,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.5,
    )


def _make_seq(model_runner: ModelRunner, seq_id: int, prompt: str) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=model_runner.tokenizer.encode(prompt),
        block_size=model_runner.block_size,
        num_layers=model_runner.num_layers,
    )


def _prefill_first_token(model_runner: ModelRunner, seq: Sequence) -> int:
    logits = model_runner.run_prefill([seq])[0]
    token_id = logits.argmax(-1).item()
    seq.output_token_ids.append(token_id)
    return token_id


def test_run_tree_verify_logits_shape_and_invariant(model_runner: ModelRunner) -> None:
    seq = _make_seq(model_runner, seq_id=0, prompt="Hello")
    _prefill_first_token(model_runner, seq)

    old_cached = seq.num_cached_tokens
    old_outputs = list(seq.output_token_ids)
    tree = build_dummy_tree(root_token_id=seq.last_token_id, topk=2, depth=3)

    logits, returned_old_cached, target_cached = model_runner.run_tree_verify(seq, tree)

    assert logits.shape == (1 + len(tree.token_ids), model_runner.model.config.vocab_size)
    assert returned_old_cached == old_cached
    assert target_cached == old_cached + 1 + len(tree.token_ids)
    assert seq.num_cached_tokens == old_cached
    assert seq.output_token_ids == old_outputs


def test_run_tree_verify_clears_tree_mode(model_runner: ModelRunner) -> None:
    seq = _make_seq(model_runner, seq_id=1, prompt="The capital of France is")
    _prefill_first_token(model_runner, seq)
    tree = build_dummy_tree(root_token_id=seq.last_token_id, topk=2, depth=3)

    model_runner.run_tree_verify(seq, tree)

    assert getattr(paged_ctx, "is_tree_verify", False) is False


def test_decode_still_works_after_tree_verify(model_runner: ModelRunner) -> None:
    seq = _make_seq(model_runner, seq_id=2, prompt="Test")
    _prefill_first_token(model_runner, seq)
    tree = build_dummy_tree(root_token_id=seq.last_token_id, topk=2, depth=3)

    model_runner.run_tree_verify(seq, tree)
    logits = model_runner.run_decode([seq])[0]
    next_token = logits.argmax(-1).item()
    seq.output_token_ids.append(next_token)

    assert seq.num_cached_tokens == seq.num_tokens - 1
    assert next_token >= 0


def _run_normal_greedy(model_runner: ModelRunner, prompt: str, max_tokens: int) -> list[int]:
    seq = _make_seq(model_runner, seq_id=100, prompt=prompt)
    _prefill_first_token(model_runner, seq)

    while seq.num_output_tokens < max_tokens:
        logits = model_runner.run_decode([seq])[0]
        token_id = logits.argmax(-1).item()
        seq.output_token_ids.append(token_id)
        if token_id == model_runner.eos_token_id:
            break

    return seq.output_token_ids


def _run_tree_greedy(model_runner: ModelRunner, prompt: str, max_tokens: int) -> list[int]:
    seq = _make_seq(model_runner, seq_id=200, prompt=prompt)
    _prefill_first_token(model_runner, seq)
    runner = EagleRunner(
        EngineConfig(
            model_path=MODEL_PATH,
            enable_eagle=True,
            eagle_mode="tree",
            eagle_topk=2,
            eagle_spec_steps=3,
        ),
        model_runner,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    while not seq.is_finished() and seq.num_output_tokens < max_tokens:
        before = seq.num_output_tokens
        runner.step(seq, sampling_params)
        assert seq.num_cached_tokens == seq.num_tokens - 1
        assert seq.num_output_tokens > before

    return seq.output_token_ids


def test_tree_step_preserves_cache_invariant(model_runner: ModelRunner) -> None:
    seq = _make_seq(model_runner, seq_id=3, prompt="The capital of France is")
    _prefill_first_token(model_runner, seq)
    runner = EagleRunner(
        EngineConfig(model_path=MODEL_PATH, enable_eagle=True, eagle_mode="tree"),
        model_runner,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=6)

    results = runner.step(seq, sampling_params)

    assert results
    assert seq.num_cached_tokens == seq.num_tokens - 1
    assert seq.num_output_tokens <= sampling_params.max_tokens


def test_tree_spec_matches_normal_greedy(model_runner: ModelRunner) -> None:
    prompt = "The capital of France is"
    max_tokens = 6

    normal_tokens = _run_normal_greedy(model_runner, prompt, max_tokens)
    tree_tokens = _run_tree_greedy(model_runner, prompt, max_tokens)

    assert tree_tokens == normal_tokens
