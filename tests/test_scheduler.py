"""Test: Scheduler basic operation (M1 single sequence)."""

from mini_vllm.scheduler import Scheduler
from mini_vllm.sequence import Sequence, SequenceStatus


def test_schedule_prefill_then_decode():
    seq = Sequence(seq_id=0, prompt_token_ids=[1, 2, 3, 4, 5], block_size=16, num_layers=28)
    seq.status = SequenceStatus.DECODING  # not finished, not yet cached
    scheduler = Scheduler(max_num_seqs=8, max_num_batched_tokens=2048)

    # First: should prefill (0 cached, 5 prompt tokens)
    out = scheduler.schedule([seq])
    assert out.is_prefill is True
    assert len(out.seqs) == 1


def test_schedule_decode_after_prefill():
    seq = Sequence(seq_id=0, prompt_token_ids=[1, 2, 3, 4, 5], block_size=16, num_layers=28)
    seq.status = SequenceStatus.DECODING
    seq.num_cached_tokens = 5  # all prompt tokens cached
    scheduler = Scheduler(max_num_seqs=8, max_num_batched_tokens=2048)

    out = scheduler.schedule([seq])
    assert out.is_prefill is False


def test_schedule_empty():
    scheduler = Scheduler(max_num_seqs=8, max_num_batched_tokens=2048)
    out = scheduler.schedule([])
    assert len(out.seqs) == 0
