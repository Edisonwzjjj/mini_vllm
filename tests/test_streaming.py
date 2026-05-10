"""Test: streaming output — generate_stream yields tokens one by one."""

import pytest
from mini_vllm import LLM, SamplingParams


MODEL_PATH = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def llm():
    return LLM(model_path=MODEL_PATH, block_size=16, max_num_seqs=4,
               max_num_batched_tokens=2048, gpu_memory_utilization=0.5)


@pytest.mark.timeout(300)
def test_stream_yields_all_tokens(llm):
    """generate_stream should yield one result per output token per sequence."""
    sp = SamplingParams(temperature=0.0, max_tokens=10)
    prompt = "Hello, how are you?"

    chunks = list(llm.generate_stream([prompt], sp))

    # Should yield exactly 10 chunks (one per output token, last one also has finished=True)
    assert len(chunks) == 10, f"Expected 10 chunks, got {len(chunks)}"

    # Last chunk should mark finished
    assert chunks[-1]["finished"] is True
    # Earlier chunks are not finished
    for c in chunks[:-1]:
        assert c["finished"] is False


@pytest.mark.timeout(300)
def test_stream_output_matches_generate(llm):
    """Streaming output should produce the same tokens as generate()."""
    sp = SamplingParams(temperature=0.0, max_tokens=15)
    prompt = "The meaning of life is"

    # Non-streaming
    out_batch = llm.generate([prompt], sp)[0]

    # Streaming
    chunks = list(llm.generate_stream([prompt], sp))
    stream_token_ids = [c["token_id"] for c in chunks]

    assert out_batch["token_ids"] == stream_token_ids, (
        f"Stream {stream_token_ids} != batch {out_batch['token_ids']}"
    )


@pytest.mark.timeout(300)
def test_stream_multi_sequence(llm):
    """Streaming with multiple sequences yields tokens for each."""
    sp = SamplingParams(temperature=0.0, max_tokens=5)
    prompts = ["What is 1+1?", "What is 2+2?"]

    chunks = list(llm.generate_stream(prompts, sp))

    # Group by seq_id
    by_seq = {}
    for c in chunks:
        by_seq.setdefault(c["seq_id"], []).append(c)

    assert set(by_seq.keys()) == {0, 1}, f"Expected seq_ids 0,1 got {set(by_seq.keys())}"
    for seq_id in [0, 1]:
        seq_chunks = by_seq[seq_id]
        # 5 tokens per sequence (including the finished one)
        assert len(seq_chunks) == 5, f"seq {seq_id}: expected 5 chunks, got {len(seq_chunks)}"
        assert seq_chunks[-1]["finished"] is True


@pytest.mark.timeout(300)
def test_stream_eos_stops_early(llm):
    """If EOS is produced, stream stops for that sequence before max_tokens."""
    sp = SamplingParams(temperature=0.7, max_tokens=100)
    prompt = "Say just the word 'hello' and nothing else."

    chunks = list(llm.generate_stream([prompt], sp))
    token_chunks = [c for c in chunks if not c["finished"]]

    # Should stop well before 100 tokens (EOS or short output)
    assert len(token_chunks) < 100, "Should have stopped before max_tokens"
