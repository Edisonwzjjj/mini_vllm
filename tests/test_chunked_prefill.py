"""Test: chunked prefill — long prompts are split into chunks and still produce correct output."""

import pytest
from mini_vllm import LLM, SamplingParams


MODEL_PATH = "Qwen/Qwen3-0.6B"

# Small max_num_batched_tokens to force chunking
CHUNK_SIZE = 128


@pytest.fixture(scope="module")
def llm_small_chunk():
    """Engine with tiny chunk size to force multiple prefill rounds."""
    return LLM(model_path=MODEL_PATH, block_size=16, max_num_seqs=4,
               max_num_batched_tokens=CHUNK_SIZE, gpu_memory_utilization=0.5)


@pytest.fixture(scope="module")
def llm_normal():
    """Normal engine for reference output."""
    return LLM(model_path=MODEL_PATH, block_size=16, max_num_seqs=4,
               max_num_batched_tokens=2048, gpu_memory_utilization=0.5)


@pytest.mark.timeout(300)
def test_chunked_prefill_output_matches_normal(llm_normal, llm_small_chunk):
    """Chunked prefill should produce identical output to normal prefill (greedy)."""
    # ~200 token prompt — forces 2 chunks with chunk_size=128
    prompt = "You are a helpful AI assistant. " * 20 + "What is the capital of France?"
    sp = SamplingParams(temperature=0.0, max_tokens=10)

    out_normal = llm_normal.generate([prompt], sp)[0]
    out_chunked = llm_small_chunk.generate([prompt], sp)[0]

    print(f"\nNormal:  {out_normal['token_ids']}")
    print(f"Chunked: {out_chunked['token_ids']}")

    assert out_normal["token_ids"] == out_chunked["token_ids"], (
        "Chunked prefill output should match normal prefill"
    )


@pytest.mark.timeout(300)
def test_chunked_prefill_long_prompt_completes(llm_small_chunk):
    """Very long prompt (>3 chunks) still completes correctly."""
    # ~400 token prompt — forces 4+ chunks with chunk_size=128
    prompt = "You are a helpful AI assistant. " * 40 + "Tell me about Paris."
    sp = SamplingParams(temperature=0.0, max_tokens=10)

    out = llm_small_chunk.generate([prompt], sp)[0]

    assert len(out["token_ids"]) == 10, (
        f"Expected 10 output tokens, got {len(out['token_ids'])}"
    )


@pytest.mark.timeout(300)
def test_chunked_prefill_multiple_seqs(llm_small_chunk):
    """Multiple long prompts in same batch all complete correctly."""
    prefix = "You are a helpful AI assistant. " * 10  # ~60 tokens
    prompts = [
        prefix + "What is 1+1?",
        prefix + "What is 2+2?",
        prefix + "What is 3+3?",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=5)

    outputs = llm_small_chunk.generate(prompts, sp)

    assert len(outputs) == 3
    for o in outputs:
        assert len(o["token_ids"]) == 5


@pytest.mark.timeout(300)
def test_chunked_prefill_prefix_cache_still_works(llm_small_chunk):
    """Prefix cache should still work with chunked prefill.

    Run same-prefix prompt twice — 2nd run should hit prefix cache.
    """
    prompt = "You are a helpful AI assistant. " * 15 + "Hello!"
    sp = SamplingParams(temperature=0.0, max_tokens=5)

    # Reset cache stats
    bm = llm_small_chunk.engine.model_runner.block_manager
    bm.total_blocks_requested = 0
    bm.total_blocks_hit = 0

    out1 = llm_small_chunk.generate([prompt], sp)[0]

    # 2nd run — prefix cache should hit
    bm.total_blocks_requested = 0
    bm.total_blocks_hit = 0
    out2 = llm_small_chunk.generate([prompt], sp)[0]

    hit_rate = bm.cache_hit_rate()
    print(f"\nPrefix cache hit rate on 2nd run: {hit_rate:.1%}")

    assert hit_rate > 0, "Prefix cache should still work with chunked prefill"
    assert out1["token_ids"] == out2["token_ids"]
