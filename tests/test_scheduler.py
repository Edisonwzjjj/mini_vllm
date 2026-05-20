"""Test: Scheduler basic operation."""

from mini_vllm.block_manager import BlockManager
from mini_vllm.scheduler import Scheduler
from mini_vllm.sequence import Sequence, SequenceStatus


def _make_scheduler() -> Scheduler:
    block_manager = BlockManager(block_size=16, num_blocks=128)
    return Scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=2048,
        block_manager=block_manager,
    )


def test_schedule_prefill_then_decode() -> None:
    seq = Sequence(seq_id=0, prompt_token_ids=[1, 2, 3, 4, 5], block_size=16, num_layers=28)
    seq.status = SequenceStatus.WAITING
    scheduler = _make_scheduler()
    scheduler.add_seqs([seq])

    # First: should prefill a waiting sequence.
    out = scheduler.schedule()
    assert out.is_prefill is True
    assert out.seqs == [seq]


def test_schedule_decode_after_prefill() -> None:
    seq = Sequence(seq_id=0, prompt_token_ids=[1, 2, 3, 4, 5], block_size=16, num_layers=28)
    seq.status = SequenceStatus.DECODING
    seq.num_cached_tokens = 5
    seq.num_prefill_tokens = 5
    scheduler = _make_scheduler()
    scheduler.running_seqs.append(seq)

    out = scheduler.schedule()
    assert out.is_prefill is False
    assert out.seqs == [seq]


def test_schedule_empty() -> None:
    scheduler = _make_scheduler()
    out = scheduler.schedule()
    assert len(out.seqs) == 0
