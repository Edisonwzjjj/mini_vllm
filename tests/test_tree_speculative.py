"""Tests for target-model tree speculative verification."""

import pytest

from mini_vllm.attention_patch import paged_ctx
from mini_vllm.config import EngineConfig
from mini_vllm.draft_tree import DraftTree, build_dummy_tree
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


def test_tree_spec_leftmost_full_accept_with_oracle_draft(model_runner: ModelRunner) -> None:
    prompt = "The capital of France is"
    max_tokens = 6
    oracle_tokens = _run_normal_greedy(model_runner, prompt, max_tokens)

    seq = _make_seq(model_runner, seq_id=300, prompt=prompt)
    _prefill_first_token(model_runner, seq)
    assert seq.output_token_ids == oracle_tokens[:1]

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

    def oracle_tree(_seq: Sequence) -> DraftTree:
        token_ids = [_seq.last_token_id] * 7
        token_ids[0] = oracle_tokens[1]
        token_ids[1] = oracle_tokens[2]
        token_ids[2] = oracle_tokens[3]
        return DraftTree(
            token_ids=token_ids,
            parent_indices=[-1, 0, 1, 1, 0, 4, 4],
        )

    runner.draft_tree_fn = oracle_tree
    results = runner.step(seq, SamplingParams(temperature=0.0, max_tokens=max_tokens))

    assert len(results) == 4
    assert seq.output_token_ids == oracle_tokens[:seq.num_output_tokens]
    assert seq.num_cached_tokens == seq.num_tokens - 1


# ---------------------------------------------------------------------------
# Tree spec + PLD multi-branch draft (P0.2)
# ---------------------------------------------------------------------------


def _run_pld_tree_spec(
    model_runner: ModelRunner,
    prompt: str,
    max_tokens: int,
    topk: int = 2,
    depth: int = 3,
    max_ngram: int = 3,
    min_ngram: int = 2,
) -> tuple[list[int], dict]:
    """End-to-end tree-spec runner with PLD multi-branch draft.

    Asserts the chain-spec invariants on every step:
      - num_cached_tokens == num_tokens - 1  (KV invariant)
      - at least 1 token of forward progress per step
    Returns (output_token_ids, metrics_snapshot).
    """
    seq = _make_seq(model_runner, seq_id=700, prompt=prompt)
    _prefill_first_token(model_runner, seq)

    runner = EagleRunner(
        EngineConfig(
            model_path=MODEL_PATH,
            enable_eagle=True,
            eagle_mode="tree",
            eagle_topk=topk,
            eagle_spec_steps=depth,
            eagle_use_pld=True,
            eagle_pld_max_ngram=max_ngram,
            eagle_pld_min_ngram=min_ngram,
        ),
        model_runner,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    while not seq.is_finished() and seq.num_output_tokens < max_tokens:
        before = seq.num_output_tokens
        runner.step(seq, sampling_params)
        assert seq.num_cached_tokens == seq.num_tokens - 1
        assert seq.num_output_tokens > before  # forward-progress invariant

    return seq.output_token_ids, dict(runner.metrics)


def test_tree_spec_pld_matches_normal_greedy(model_runner: ModelRunner) -> None:
    """正确性：tree spec + PLD draft 的输出必须等于纯 greedy decode 的输出。

    考点：
    - leftmost commit + PLD draft 的接受逻辑不会改变 greedy 输出
    - PLD 找不到匹配时降级为 padding tree，verify forward 仍维持正确性
    """
    prompt = "List the days: Monday, Tuesday, Wednesday. List the days: Monday,"
    max_tokens = 10

    normal_tokens = _run_normal_greedy(model_runner, prompt, max_tokens)
    spec_tokens, metrics = _run_pld_tree_spec(model_runner, prompt, max_tokens)

    assert spec_tokens == normal_tokens, (
        f"tree-spec output diverged from greedy:\n"
        f"  normal = {normal_tokens}\n"
        f"  spec   = {spec_tokens}"
    )
    assert metrics["spec_steps"] >= 1, (
        f"expected at least 1 spec step (prompt has strong repetition), "
        f"got metrics={metrics}"
    )


def test_tree_spec_pld_degrades_on_short_prompt(model_runner: ModelRunner) -> None:
    """降级行为：短 prompt 下 PLD 几乎全 miss，但 tree spec 仍维持 greedy 等价性。

    考点：
    - 短 prompt 上 PLD ngram 查询大量失败 → 返回的 7 槽树几乎全是 last_token padding
    - verify forward 在 padding tree 上仍能正确跑（acceptance 很低但 KV 不变量保住）
    - 注：tree-mode 的 _fallback_decode_step 仅在 prompt_lookup_tree 返回 None
      时触发（仅空 token_ids 才会），生产路径几乎不会走 —— 那条路由由
      test_draft_pld_tree_empty 单元测试覆盖。
    """
    prompt = "Hi"
    max_tokens = 6

    normal_tokens = _run_normal_greedy(model_runner, prompt, max_tokens)
    spec_tokens, metrics = _run_pld_tree_spec(model_runner, prompt, max_tokens)

    assert spec_tokens == normal_tokens, (
        f"short-prompt tree-spec output diverged from greedy:\n"
        f"  normal = {normal_tokens}\n"
        f"  spec   = {spec_tokens}"
    )
    assert metrics["spec_steps"] >= 1, (
        f"tree spec should still run (with padding tree), got metrics={metrics}"
    )

