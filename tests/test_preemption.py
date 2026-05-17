"""Test: preemption — when KV cache runs out, sequences are preempted and still complete correctly."""

import pytest
from mini_vllm import LLM, SamplingParams
from mini_vllm.block_manager import RadixTree


MODEL_PATH = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def llm():
    return LLM(model_path=MODEL_PATH, block_size=16, max_num_seqs=4,
               max_num_batched_tokens=2048, gpu_memory_utilization=0.5)


def _shrink_free_blocks(llm, keep=5):
    """Reduce free_blocks to simulate memory pressure."""
    bm = llm.engine.model_runner.block_manager
    bm.free_blocks = bm.free_blocks[:keep]


def _reset_blocks(llm):
    """Restore block_manager to a clean state between tests."""
    bm = llm.engine.model_runner.block_manager
    # Reset block manager to initial state
    bm.free_blocks = list(range(bm.num_blocks))
    bm.radix_tree = RadixTree(bm.block_size)
    bm.block_id_to_node = {}
    bm.total_blocks_requested = 0
    bm.total_blocks_hit = 0


@pytest.mark.timeout(300)
def test_preemption_all_sequences_complete(llm):
    """With very few blocks, preemption kicks in and all sequences still complete."""
    _reset_blocks(llm)
    _shrink_free_blocks(llm, keep=4)

    prompts = [
        "Hello, how are you today?",
        "What is the capital of France?",
        "Tell me a short story about a cat.",
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=10)

    outputs = llm.generate(prompts, sp)

    assert len(outputs) == 3
    for o in outputs:
        assert len(o["token_ids"]) == 10


@pytest.mark.timeout(300)
def test_preemption_output_correctness(llm):
    """Preempted sequence should produce same output as non-preempted (greedy).

    Run the same prompt twice:
      1st: with plenty of blocks → no preemption
      2nd: with very few blocks → preemption may happen
    Both should produce identical token ids.
    """
    _reset_blocks(llm)

    prompt = "The meaning of life is"
    sp = SamplingParams(temperature=0.0, max_tokens=15)

    # Run 1: normal (lots of blocks)
    out_normal = llm.generate([prompt], sp)[0]

    # Run 2: with memory pressure
    _reset_blocks(llm)
    _shrink_free_blocks(llm, keep=3)
    out_pressure = llm.generate([prompt], sp)[0]

    print(f"\nNormal:  {out_normal['token_ids']}")
    print(f"Pressure: {out_pressure['token_ids']}")

    assert out_normal["token_ids"] == out_pressure["token_ids"], (
        "Preempted sequence output should match non-preempted"
    )


@pytest.mark.timeout(300)
def test_preempt_one_frees_blocks(llm):
    """After preempt_one, the victim's blocks are deallocated."""
    _reset_blocks(llm)

    from mini_vllm.sequence import Sequence

    seq = Sequence(seq_id=99, prompt_token_ids=[1, 2, 3, 4], block_size=16,
                   num_layers=llm.engine.model_runner.num_layers)

    # Manually allocate a block
    bm = llm.engine.model_runner.block_manager
    blocks = bm.allocate(2)
    for layer in range(len(seq.block_table)):
        seq.block_table[layer] = list(blocks)
    seq.num_cached_tokens = 4

    llm.engine.scheduler.running_seqs.append(seq)

    free_before = len(bm.free_blocks)
    victim = llm.engine.scheduler.preempt_one()

    assert victim is seq
    assert seq.num_cached_tokens == 0
    assert seq.output_token_ids == []
    # block_table is NOT cleared by preempt_one — it's cleared by _free_seq_resources
    assert seq.block_table[0] != [], "block_table not yet cleared (preempt_one doesn't clear it)"

    # Now free resources (as engine.step does after preempt)
    llm.engine._free_seq_resources(victim)
    assert seq.block_table[0] == [], "block_table cleared after _free_seq_resources"


@pytest.mark.timeout(300)
def test_try_allocate_returns_none_when_full():
    """try_allocate returns None when not enough blocks (without raising)."""
    from mini_vllm.block_manager import BlockManager

    bm = BlockManager(block_size=16, num_blocks=10)
    # Consume 9 blocks
    bm.allocate(9)

    # Only 1 free block left
    result = bm.try_allocate(3)
    assert result is None, "Should return None when not enough blocks"

    # 1 block should still work
    result = bm.try_allocate(1)
    assert result is not None
    assert len(result) == 1
