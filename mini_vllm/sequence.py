"""Sequence state machine — tracks a single request through its lifecycle."""

from enum import Enum, auto
from typing import List


class SequenceStatus(Enum):
    WAITING = auto()
    PREFILL = auto()
    DECODING = auto()
    FINISHED = auto()


class Sequence:
    """One request from prompt to completion.

    Key state:
      - token_ids: all tokens so far (prompt + generated)
      - block_table: list of block IDs this sequence owns
      - num_cached_tokens: how many tokens already have KV cached
      - status: where in the lifecycle
    """

    def __init__(self, seq_id: int, prompt_token_ids: List[int], block_size: int, num_layers: int):
        self.seq_id = seq_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.output_token_ids: List[int] = []
        # Per-layer block table: block_table[layer_idx] = List[int]
        # Each layer has its own list of block IDs.
        self.block_table: List[List[int]] = [[] for _ in range(num_layers)]
        self.num_cached_tokens: int = 0
        self.block_size = block_size
        self.num_layers = num_layers
        self.status = SequenceStatus.WAITING
        
        self.num_prefill_tokens: int = 0
        

    @property
    def token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def last_token_id(self) -> int:
        return self.token_ids[-1]
    
    @property
    def is_prefill_finished(self) -> bool:
        return self.num_prefill_tokens >= len(self.prompt_token_ids)

    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED
    
    def mark_finished(self):
        self.status = SequenceStatus.FINISHED
