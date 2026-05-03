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
        self.running_seqs: List[Sequence] = []  # for M2: currently running sequences
        self.waiting_seqs: List[Sequence] = []  # for M2:
        
    def add_seqs(self, seqs: List[Sequence]):
        """Add new sequences to the waiting queue."""
        for seq in seqs:
            if len(self.waiting_seqs) + len(self.running_seqs) >= self.max_num_seqs:
                break
            self.waiting_seqs.append(seq)

    def schedule(self) -> SchedulerOutput:
        """Schedule the next batch of sequences."""
        if self.waiting_seqs:
            # Prefill: take one sequence from waiting, move to running
            seq = self.waiting_seqs.pop(0)
            self.running_seqs.append(seq)
            return SchedulerOutput(seqs=[seq], is_prefill=True)
        if self.running_seqs:
            # Decode: run ALL running sequences together
            return SchedulerOutput(seqs=list(self.running_seqs), is_prefill=False)
        return SchedulerOutput(seqs=[], is_prefill=False)  # all finished
    
    def postprocess(self, output: Sequence) -> None:
        """Postprocess the output of the scheduler."""
        if output.is_finished():
            self.running_seqs.remove(output)
            
    
