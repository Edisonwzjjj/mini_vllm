"""Scheduler — decides which sequences to run next. M2 will add continuous batching."""

from typing import List, Tuple
from mini_vllm.sequence import Sequence, SequenceStatus


class SchedulerOutput:
    """Result of one scheduling decision."""

    def __init__(self, seqs: List[Sequence], is_prefill: bool):
        self.seqs = seqs
        self.is_prefill = is_prefill


class Scheduler:
    """M1: trivially schedules one sequence at a time.
    M2: add waiting/running queues and continuous batching.
    """

    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens

    def schedule(self, seqs: List[Sequence]) -> SchedulerOutput:
        """M1: find the first non-finished sequence, run it.
        Prefill if num_cached_tokens < num_prompt_tokens, else decode.
        """
        for seq in seqs:
            if seq.is_finished():
                continue
            if seq.num_cached_tokens < seq.num_prompt_tokens:
                return SchedulerOutput([seq], is_prefill=True)
            else:
                return SchedulerOutput([seq], is_prefill=False)
        # All finished
        return SchedulerOutput([], is_prefill=False)
