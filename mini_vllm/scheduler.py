"""Scheduler — decides which sequences to run next. M2 will add continuous batching."""

from typing import List, Tuple
from mini_vllm.sequence import Sequence, SequenceStatus
from typing import Optional


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
        """第一步先继续chunks prefill, 再找完全没prefill的, 最后再找正在 decode 的"""
        prefill_seqs = [s for s in self.running_seqs if s.is_prefill_finished == False]
        if prefill_seqs:
            return SchedulerOutput(seqs=prefill_seqs, is_prefill=True)

        if self.waiting_seqs:
            # Prefill: take all waiting sequences up to limits, move to running
            prefill_seqs = []
            total_tokens = 0
            while self.waiting_seqs:
                seq = self.waiting_seqs[0]
                if len(self.running_seqs) + len(prefill_seqs) >= self.max_num_seqs:
                    break
                # chunk_size: 这轮最多能 prefill 多少 token
                remaining_budget = self.max_num_batched_tokens - total_tokens
                if remaining_budget <= 0:
                    break
                self.waiting_seqs.pop(0)
                self.running_seqs.append(seq)
                prefill_seqs.append(seq)
                # 按 chunk 大小计，prompt 再长也只占 chunk 大小的 budget
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
        # 清空 block_table，重新 prefill 时 allocate_with_prefix 会重新填充
        for layer in range(len(victim.block_table)):
            victim.block_table[layer] = []
        self.waiting_seqs.insert(0, victim)  # 放回头部，优先重新调度
        return victim
            
