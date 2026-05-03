"""Test: multi-sequence batched prefill + per-sequence decode correctness."""

import pytest
from mini_vllm import LLM, SamplingParams


MODEL_PATH = "Qwen/Qwen3-0.6B"
PROMPTS = [
    "Hello, world.",
    "What is attention?",
    "The capital of France is",
    "Explain quantum computing",
    "Write a haiku about rain",
    "1+1=",
    "The meaning of life is",
    "Once upon a time",
]


@pytest.fixture(scope="module")
def single_outputs():
    """Run each prompt individually, cache results."""
    llm = LLM(model_path=MODEL_PATH, block_size=16, max_num_seqs=8,
              max_num_batched_tokens=2048, gpu_memory_utilization=0.5)
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    results = []
    for prompt in PROMPTS:
        out = llm.generate([prompt], sp)[0]["token_ids"]
        results.append(out)
    return results


@pytest.fixture(scope="module")
def multi_outputs():
    """Run all prompts together with batched prefill."""
    llm = LLM(model_path=MODEL_PATH, block_size=16, max_num_seqs=8,
              max_num_batched_tokens=2048, gpu_memory_utilization=0.5)
    sp = SamplingParams(temperature=0.0, max_tokens=20)
    return llm.generate(PROMPTS, sp)


@pytest.mark.timeout(300)
def test_multi_matches_single(single_outputs, multi_outputs):
    """Each sequence in batch must produce the same tokens as when run alone."""
    for i, (single_ids, multi_result) in enumerate(zip(single_outputs, multi_outputs)):
        multi_ids = multi_result["token_ids"]
        assert single_ids == multi_ids, (
            f"Prompt {i} mismatch: single={single_ids}, multi={multi_ids}"
        )


@pytest.mark.timeout(300)
def test_all_sequences_return(multi_outputs):
    """All 8 prompts must produce output."""
    assert len(multi_outputs) == len(PROMPTS)
    for i, out in enumerate(multi_outputs):
        assert len(out["token_ids"]) == 20, (
            f"Prompt {i}: expected 20 tokens, got {len(out['token_ids'])}"
        )


@pytest.mark.timeout(300)
def test_output_nontrivial(multi_outputs):
    """No empty or all-zero outputs."""
    for out in multi_outputs:
        assert all(tid > 0 for tid in out["token_ids"])
        assert len(out["text"].strip()) > 0
