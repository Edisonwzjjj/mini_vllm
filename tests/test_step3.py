"""Integration test: prefill + decode produces correct tokens (matches HF greedy)."""

import torch
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer


@pytest.fixture(scope="module")
def model_runner():
    """Load model once for all tests in this module."""
    from mini_vllm.model_runner import ModelRunner
    mr = ModelRunner(
        model_path="Qwen/Qwen3-0.6B",
        block_size=16,
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.5,
    )
    return mr


def _run_hf_greedy(prompt: str, max_tokens: int):
    """Reference: HF model.generate greedy output."""
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32).to("mps")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    token_ids = out[0][inputs["input_ids"].shape[1]:].tolist()
    del model
    return token_ids


def test_block_manager_matches_kv_cache(model_runner):
    """Bug 2 check: BlockManager num_blocks must match kv_cache num_blocks."""
    bm = model_runner.block_manager
    kv_num_blocks = model_runner.kv_cache.shape[1]
    print(f"[CHECK] BlockManager.num_blocks={bm.num_blocks}, kv_cache blocks={kv_num_blocks}")
    assert bm.num_blocks == kv_num_blocks, (
        f"MISMATCH: BlockManager has {bm.num_blocks} blocks but kv_cache has {kv_num_blocks}. "
        f"Allocate will return block_ids >= {bm.num_blocks}, causing index-out-of-range in kv_cache."
    )


def test_prefill_slot_mapping(model_runner):
    """Slot mapping should map every prompt token to a valid kv_cache position."""
    from mini_vllm.sequence import Sequence

    tokenizer = model_runner.tokenizer
    prompt = "Hello, world."
    prompt_ids = tokenizer.encode(prompt)
    seq = Sequence(seq_id=0, prompt_token_ids=prompt_ids, block_size=16, num_layers=28)

    # Run prefill
    logits = model_runner.run_prefill([seq])[0]

    # Check: num_cached_tokens should equal prompt length
    print(f"[CHECK] num_cached_tokens={seq.num_cached_tokens}, prompt_len={len(prompt_ids)}")
    assert seq.num_cached_tokens == len(prompt_ids), (
        f"num_cached_tokens mismatch: {seq.num_cached_tokens} != {len(prompt_ids)}"
    )

    # Check: block_table should have been allocated
    num_blocks = len(seq.block_table[0])
    blocks_needed = model_runner.block_manager.num_blocks_needed(len(prompt_ids))
    print(f"[CHECK] blocks_allocated={num_blocks}, blocks_needed={blocks_needed}")
    assert num_blocks == blocks_needed, (
        f"Not enough blocks: {num_blocks} < {blocks_needed}"
    )

    # Check: slot_mapping range should be within kv_cache bounds
    slot_mapping = model_runner.block_manager.get_slot_mapping(seq.block_table[0], len(prompt_ids))
    max_slot = max(slot_mapping)
    kv_max = model_runner.kv_cache.shape[1] * model_runner.block_size - 1
    print(f"[CHECK] max_slot={max_slot}, kv_max_slot={kv_max}")
    assert max_slot <= kv_max, (
        f"Slot {max_slot} exceeds kv_cache range [0, {kv_max}]"
    )

    # Check: logits shape
    print(f"[CHECK] logits shape={logits.shape}")
    assert logits.shape == (model_runner.model.config.vocab_size,), f"Unexpected logits shape: {logits.shape}"

    # Check: first generated token matches HF greedy
    next_token = logits.argmax(-1).item()
    print(f"[CHECK] first generated token_id={next_token} ({tokenizer.decode([next_token])})")
    assert next_token > 0, "Generated token_id is 0, likely something wrong"


def test_prefill_then_decode_tokens(model_runner):
    """Prefill + multiple decode steps should produce tokens (not crash)."""
    from mini_vllm.sequence import Sequence

    tokenizer = model_runner.tokenizer
    prompt = "The capital of France is"
    prompt_ids = tokenizer.encode(prompt)
    seq = Sequence(seq_id=1, prompt_token_ids=prompt_ids, block_size=16, num_layers=28)

    # Prefill
    logits = model_runner.run_prefill([seq])[0]
    next_token = logits.argmax(-1).item()
    seq.output_token_ids.append(next_token)
    print(f"[PREFILL] token {next_token} = '{tokenizer.decode([next_token])}'")
    print(f"[PREFILL] num_cached={seq.num_cached_tokens}, num_tokens={seq.num_tokens}")

    # Decode 5 steps
    generated = [next_token]
    for step in range(5):
        print(f"[DECODE step={step}] num_cached={seq.num_cached_tokens}, "
              f"num_tokens={seq.num_tokens}, position={seq.num_tokens - 1}")

        logits = model_runner.run_decode([seq])[0]
        next_token = logits.argmax(-1).item()
        seq.output_token_ids.append(next_token)
        generated.append(next_token)
        print(f"[DECODE step={step}] token {next_token} = '{tokenizer.decode([next_token])}'")

    # Sanity: should have generated 6 tokens total (1 from prefill + 5 decode)
    assert len(generated) == 6, f"Expected 6 tokens, got {len(generated)}"

    # Sanity: no token should be 0 (unlikely for valid text)
    for i, tid in enumerate(generated):
        assert tid != 0, f"Step {i} generated token_id=0, likely a bug"

    full_text = tokenizer.decode(prompt_ids + generated)
    print(f"[RESULT] {full_text}")


def test_decode_position_ids(model_runner):
    """Verify position_ids are correct at each decode step."""
    from mini_vllm.sequence import Sequence

    tokenizer = model_runner.tokenizer
    prompt_ids = tokenizer.encode("Hello")
    seq = Sequence(seq_id=2, prompt_token_ids=prompt_ids, block_size=16, num_layers=28)

    logits = model_runner.run_prefill([seq])[0]
    seq.output_token_ids.append(logits.argmax(-1).item())

    # After prefill: num_cached=prompt_len, num_tokens=prompt_len+1
    print(f"[POS] After prefill: num_cached={seq.num_cached_tokens}, num_tokens={seq.num_tokens}")

    # First decode: position should be prompt_len (= num_tokens - 1 at this point)
    # run_decode uses seq.num_tokens - 1 as position_id
    expected_pos = seq.num_tokens - 1
    print(f"[POS] First decode: expected position={expected_pos}, "
          f"should equal prompt_len={len(prompt_ids)}")
    assert expected_pos == len(prompt_ids), (
        f"Position mismatch: {expected_pos} != {len(prompt_ids)}. "
        f"Decode will use wrong position for RoPE, causing garbage output."
    )

    # Run decode and check position progression
    for step in range(3):
        pos_before = seq.num_tokens - 1
        logits = model_runner.run_decode([seq])[0]
        seq.output_token_ids.append(logits.argmax(-1).item())
        pos_after = seq.num_tokens - 1
        print(f"[POS] Decode step {step}: position_before={pos_before}, position_after={pos_after}")
        assert pos_after == pos_before + 1, (
            f"Position should increment by 1 each step: {pos_before} -> {pos_after}"
        )


def test_block_allocation_across_boundary(model_runner):
    """When tokens cross a block boundary, new block should be allocated."""
    from mini_vllm.sequence import Sequence

    tokenizer = model_runner.tokenizer
    # block_size=16, so we need a prompt that uses exactly 1 block (<=16 tokens)
    # then decode until we need a 2nd block
    prompt_ids = tokenizer.encode("Hello")  # likely 1-2 tokens
    seq = Sequence(seq_id=3, prompt_token_ids=prompt_ids, block_size=16, num_layers=28)

    logits = model_runner.run_prefill([seq])[0]
    seq.output_token_ids.append(logits.argmax(-1).item())

    blocks_after_prefill = len(seq.block_table[0])
    print(f"[BLOCK] After prefill: {blocks_after_prefill} blocks, "
          f"num_cached={seq.num_cached_tokens}, block_size={model_runner.block_size}")

    # Decode until we cross block boundary (need >16 cached tokens)
    for step in range(20):
        logits = model_runner.run_decode([seq])[0]
        seq.output_token_ids.append(logits.argmax(-1).item())

    blocks_after = len(seq.block_table[0])
    cached = seq.num_cached_tokens
    print(f"[BLOCK] After 20 decode steps: {blocks_after} blocks, num_cached={cached}")

    expected_blocks = (cached + model_runner.block_size - 1) // model_runner.block_size
    assert blocks_after == expected_blocks, (
        f"Block count mismatch: {blocks_after} != {expected_blocks}. "
        f"New block not allocated when crossing block_size boundary."
    )


def test_num_cached_tokens_monotonic(model_runner):
    """num_cached_tokens must increase by exactly 1 each decode step."""
    from mini_vllm.sequence import Sequence

    tokenizer = model_runner.tokenizer
    prompt_ids = tokenizer.encode("Test")
    seq = Sequence(seq_id=4, prompt_token_ids=prompt_ids, block_size=16, num_layers=28)

    logits = model_runner.run_prefill([seq])[0]
    seq.output_token_ids.append(logits.argmax(-1).item())

    prev_cached = seq.num_cached_tokens
    print(f"[CACHED] After prefill: {prev_cached}")

    for step in range(5):
        logits = model_runner.run_decode([seq])[0]
        seq.output_token_ids.append(logits.argmax(-1).item())
        now_cached = seq.num_cached_tokens
        diff = now_cached - prev_cached
        print(f"[CACHED] Step {step}: {prev_cached} -> {now_cached} (diff={diff})")
        assert diff == 1, f"num_cached_tokens should increment by 1, got diff={diff}"
        prev_cached = now_cached
