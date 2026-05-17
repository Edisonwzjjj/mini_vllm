"""Scheduler — decides which sequences to run next. M2 will add continuous batching."""

from typing import List, Tuple
from mini_vllm.sequence import Sequence, SequenceStatus
from typing import Optional
from mini_vllm.block_manager import BlockManager

class SchedulerOutput:
    """Result of one scheduling decision."""

    def __init__(self, seqs: List[Sequence], is_prefill: bool):
        self.seqs = seqs
        self.is_prefill = is_prefill


class Scheduler:
    """M1: trivially schedules one sequence at a time.
    M2: add waiting/running queues and continuous batching.
    """

    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int, block_manager: BlockManager):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.running_seqs: List[Sequence] = []  # for M2: currently running sequences
        self.waiting_seqs: List[Sequence] = []  # for M2:
        self.block_manager = block_manager
        
    def add_seqs(self, seqs: List[Sequence]):
        """Add new sequences to the waiting queue."""
        for seq in seqs:
            if len(self.waiting_seqs) + len(self.running_seqs) >= self.max_num_seqs:
                break
            self.waiting_seqs.append(seq)

    def schedule(self) -> SchedulerOutput:
        """Schedule the next batch of sequences."""
        """第一步先继续chunks prefill, 再找完全没prefill的, 最后再找正在 decode 的"""
        prefill_seqs = [s for s in self.running_seqs if s.is_prefill_finished == False]
        if prefill_seqs:
            return SchedulerOutput(seqs=prefill_seqs, is_prefill=True)

        if self.waiting_seqs:
            # Cache-aware scheduling: sort by prefix hit ratio (descending)
            # Higher hit ratio → fewer new blocks needed → schedule first
            self.waiting_seqs.sort(
                key=lambda s: self.block_manager.radix_tree.match_prefix_ratio(s.prompt_token_ids),
                reverse=True,
            )

            # Prefill: take sequences up to limits, move to running
            prefill_seqs = []
            total_tokens = 0
            while self.waiting_seqs:
                if len(self.running_seqs) + len(prefill_seqs) >= self.max_num_seqs:
                    break
                remaining_budget = self.max_num_batched_tokens - total_tokens
                if remaining_budget <= 0:
                    break
                seq = self.waiting_seqs.pop(0)
                self.running_seqs.append(seq)
                prefill_seqs.append(seq)
                chunk_size = min(seq.num_prompt_tokens, remaining_budget)
                total_tokens += chunk_size
            return SchedulerOutput(seqs=prefill_seqs, is_prefill=True)
        if self.running_seqs:
            # Decode: run ALL running sequences together
            return SchedulerOutput(seqs=list(self.running_seqs), is_prefill=False)
        return SchedulerOutput(seqs=[], is_prefill=False)  # all finished
    
    def postprocess(self, output: Sequence) -> None:
        """Postprocess the output of the scheduler."""
        if output.is_finished():
            self.running_seqs.remove(output)
            
    def preempt_one(self) -> Optional[Sequence]:
        """Preempt the newest running sequence (LIFO), move back to waiting head."""
        if not self.running_seqs:
            return None
        victim = self.running_seqs.pop()  # LIFO:踢最新加入的
        victim.num_cached_tokens = 0
        victim.num_prefill_tokens = 0  # 重新从头 prefill
        victim.output_token_ids = []  # 清空：重新 prefill 后从头生成
        # NOTE: block_table is NOT cleared here — engine._free_seq_resources
        # must deallocate blocks first, then clear block_table.
        self.waiting_seqs.insert(0, victim)  # 放回头部，优先重新调度
        return victim
            
