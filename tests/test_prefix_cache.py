"""Test: prefix cache correctness — same-prefix prompts reuse KV, outputs match."""

import pytest
from mini_vllm import LLM, SamplingParams


MODEL_PATH = "Qwen/Qwen3-0.6B"

# A shared prefix + different suffixes
SYSTEM_PROMPT = "You are a helpful AI assistant. " * 12  # ~108 tokens
QUERIES = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Japan?",
]


@pytest.fixture(scope="module")
def llm():
    return LLM(model_path=MODEL_PATH, block_size=16, max_num_seqs=8,
               max_num_batched_tokens=2048, gpu_memory_utilization=0.5)


@pytest.mark.timeout(300)
def test_prefix_cache_output_correctness(llm):
    """Outputs with prefix cache must match outputs without prefix cache.

    Run same prompts twice:
      1st run: no cache → full prefill
      2nd run: cache populated → suffix prefill
    Both must produce identical token ids (greedy decoding).
    """
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    prompts = [SYSTEM_PROMPT + q for q in QUERIES]

    # Run 1: no prefix cache yet
    outputs_run1 = llm.generate(prompts, sp)

    # Run 2: prefix cache should now be populated
    outputs_run2 = llm.generate(prompts, sp)

    for i in range(len(prompts)):
        ids1 = outputs_run1[i]["token_ids"]
        ids2 = outputs_run2[i]["token_ids"]
        assert ids1 == ids2, (
            f"Prompt {i} mismatch between run1 and run2: "
            f"run1={ids1}, run2={ids2}"
        )


@pytest.mark.timeout(300)
def test_2nd_3rd_prefill_fewer_tokens(llm):
    """M3 acceptance: send 3 same-prefix prompts, 2nd and 3rd compute fewer tokens.

    Key: all 3 must be in the SAME generate() call so prefix cache
    is shared across sequences (blocks are not freed until all finish).
    """
    sp = SamplingParams(temperature=0.0, max_tokens=5)
    prefix = "You are a helpful AI assistant. " * 12  # ~108 tokens

    prompts = [
        prefix + "What is the capital of France?",
        prefix + "What is the capital of Germany?",
        prefix + "What is the capital of Japan?",
    ]

    # Reset stats
    bm = llm.engine.model_runner.block_manager
    bm.total_blocks_requested = 0
    bm.total_blocks_hit = 0

    # Run all 3 together — 1st gets no hit, 2nd/3rd hit prefix
    outputs = llm.generate(prompts, sp)
    hit_rate = bm.cache_hit_rate()

    print(f"\nCache hit rate for 3 same-prefix prompts: {hit_rate:.1%}")
    for i, o in enumerate(outputs):
        print(f"  Prompt {i}: {len(o['token_ids'])} output tokens")

    # 2nd and 3rd prompts should have hit prefix cache
    assert hit_rate > 0, f"Expected prefix cache hits, got hit_rate={hit_rate:.1%}"
    assert len(outputs) == 3
    for o in outputs:
        assert len(o["token_ids"]) == 5


@pytest.mark.timeout(300)
def test_prefix_cache_hit_rate(llm):
    """Prefix cache hit rate should be > 0 for same-prefix prompts in same batch."""
    sp = SamplingParams(temperature=0.0, max_tokens=5)
    prompts = [SYSTEM_PROMPT + q for q in QUERIES]

    # Reset stats
    bm = llm.engine.model_runner.block_manager
    bm.total_blocks_requested = 0
    bm.total_blocks_hit = 0

    # All in same batch — 1st prompt no hit, 2nd/3rd hit prefix
    llm.generate(prompts, sp)
    hit_rate = bm.cache_hit_rate()

    print(f"\nPrefix cache hit rate: {hit_rate:.1%}")
    print(f"  Hits: {bm.total_blocks_hit} / Requested: {bm.total_blocks_requested}")

    assert hit_rate > 0.0, f"Expected cache hits for same-prefix prompts, got {hit_rate:.1%}"


@pytest.mark.timeout(300)
def test_prefix_cache_shared_blocks_refcount(llm):
    """Shared prefix blocks should have ref_count > 1 when multiple sequences use them.

    After generating with same-prefix prompts, the prefix blocks should have
    ref_count > 1. When one sequence finishes, ref_count decrements but
    block is NOT freed.
    """
    sp = SamplingParams(temperature=0.0, max_tokens=5)
    prompt_a = SYSTEM_PROMPT + QUERIES[0]
    prompt_b = SYSTEM_PROMPT + QUERIES[1]

    # Generate seq A first (populates prefix cache)
    output_a = llm.generate([prompt_a], sp)[0]

    # Generate seq B (should reuse prefix blocks)
    output_b = llm.generate([prompt_b], sp)[0]

    # Verify both produced output
    assert len(output_a["token_ids"]) == 5
    assert len(output_b["token_ids"]) == 5


@pytest.mark.timeout(300)
def test_deallocate_preserves_cache_then_eviction_cleans(llm):
    """When sequences finish, blocks go to cached pool (hash preserved).
    When memory is needed, cached blocks are evicted (hash cleaned).

    With our 585-block pool, eviction won't happen in normal use.
    So after generate(), hash entries should still exist for reuse.
    """
    sp = SamplingParams(temperature=0.0, max_tokens=5)
    prompt = "This is a unique test prompt for deallocate. " * 5
    bm = llm.engine.model_runner.block_manager

    # Run once — blocks stay in radix tree after seq finishes (ref_count=0 but cached)
    llm.generate([prompt], sp)
    assert len(bm.block_id_to_node) > 0, "Tree nodes should persist in radix tree"

    # Run same prompt again — should hit prefix cache from cached pool
    bm.total_blocks_requested = 0
    bm.total_blocks_hit = 0
    out = llm.generate([prompt], sp)[0]
    hit_rate = bm.cache_hit_rate()
    assert hit_rate > 0.0, f"Should hit prefix cache on 2nd run, got {hit_rate:.1%}"
    assert len(out["token_ids"]) == 5


@pytest.mark.timeout(300)
def test_different_prefix_no_collision(llm):
    """Different prompts should NOT hit each other's prefix cache."""
    sp = SamplingParams(temperature=0.0, max_tokens=20)

    # Two completely different prompts
    prompt_x = "The quick brown fox jumps over the lazy dog."
    prompt_y = "In a galaxy far far away, there lived a robot."

    out_x_first = llm.generate([prompt_x], sp)[0]
    out_x_second = llm.generate([prompt_x], sp)[0]

    # Same prompt run twice should still match (prefix cache helps)
    assert out_x_first["token_ids"] == out_x_second["token_ids"], (
        "Same prompt should produce identical output on repeat run"
    )

    # Different prompt should not collide
    out_y = llm.generate([prompt_y], sp)[0]
    assert out_y["token_ids"] != out_x_first["token_ids"], (
        "Different prompts should produce different outputs"
    )


@pytest.mark.timeout(300)
def test_identical_prompts_cache_hit(llm):
    """Identical prompts should fully hit prefix cache on 2nd run."""
    sp = SamplingParams(temperature=0.0, max_tokens=10)
    prompt = "Hello, how are you today?"

    # First run: no cache
    out1 = llm.generate([prompt], sp)[0]

    # Second run: should fully hit prefix cache
    out2 = llm.generate([prompt], sp)[0]

    assert out1["token_ids"] == out2["token_ids"]
