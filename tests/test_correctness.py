"""Test: mini-vllm output matches HF model.generate (greedy)."""

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from mini_vllm import LLM, SamplingParams


MODEL_PATH = "Qwen/Qwen3-0.6B"
PROMPT = "Hello, world."
MAX_TOKENS = 20
TEMPERATURE = 0.01


@pytest.fixture(scope="module")
def mini_vllm_output():
    """Run mini-vllm once, cache the result for all tests."""
    llm = LLM(
        model_path=MODEL_PATH,
        block_size=16,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.5,
    )
    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    return llm.generate(prompts=[PROMPT], sampling_params=sp)[0]


@pytest.fixture(scope="module")
def hf_output():
    """Run HF model.generate once, cache the result for all tests."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.float32
    ).to("mps")
    inputs = tokenizer(PROMPT, return_tensors="pt").to("mps")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
    # Strip prompt tokens, keep only generated ones
    gen_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
    del model
    return {"token_ids": gen_ids, "text": tokenizer.decode(gen_ids, skip_special_tokens=True)}


@pytest.mark.timeout(180)
def test_decode_matches_hf(mini_vllm_output, hf_output):
    """Mini-vllm output must match HF model.generate token-for-token."""
    mini_ids = mini_vllm_output["token_ids"]
    hf_ids = hf_output["token_ids"]

    print(f"\n[mini-vllm] tokens: {mini_ids}")
    print(f"[HF]        tokens: {hf_ids}")
    print(f"[mini-vllm] text: {mini_vllm_output['text']}")
    print(f"[HF]        text: {hf_output['text']}")

    assert mini_ids == hf_ids, (
        f"Token mismatch!\n"
        f"  mini-vllm: {mini_ids}\n"
        f"  HF:        {hf_ids}\n"
        f"  First diff at position {next(i for i,(a,b) in enumerate(zip(mini_ids,hf_ids)) if a!=b) if mini_ids!=hf_ids else 'N/A'}"
    )


@pytest.mark.timeout(180)
def test_output_length(mini_vllm_output):
    """Should generate exactly max_tokens tokens (no early stop for this prompt)."""
    assert len(mini_vllm_output["token_ids"]) == MAX_TOKENS, (
        f"Expected {MAX_TOKENS} tokens, got {len(mini_vllm_output['token_ids'])}"
    )


@pytest.mark.timeout(180)
def test_output_nontrivial(mini_vllm_output):
    """Output should contain real tokens, not all zeros or padding."""
    token_ids = mini_vllm_output["token_ids"]
    assert all(tid > 0 for tid in token_ids), f"Got zero/negative token ids: {token_ids}"
    assert len(mini_vllm_output["text"].strip()) > 0, "Output text is empty"
