"""Block manager — paged KV cache allocation + prefix cache (prefix cache in M3)."""

from typing import Dict, List, Optional, Tuple


class BlockManager:
    """Manages paged KV cache blocks.

    Each block stores KV for `block_size` consecutive tokens.
    Blocks are identified by integer block_id.

    M1 scope: allocate / append / deallocate only (no prefix cache).
    M3 scope: add hash-based prefix cache with ref counting.
    """

    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Free block pool
        self.free_blocks: List[int] = list(range(num_blocks))

        # block_id -> ref_count (M1: always 1, M3: can be >1 for shared prefix)
        self.ref_counts: Dict[int, int] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def allocate(self, num_blocks: int) -> List[int]:
        """Allocate `num_blocks` fresh blocks. Returns list of block_ids."""
        if num_blocks > self.num_free_blocks:
            raise RuntimeError(
                f"Not enough blocks: need {num_blocks}, have {self.num_free_blocks}"
            )
        blocks = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        for b in blocks:
            self.ref_counts[b] = 1
        return blocks

    def deallocate(self, block_ids: List[int]) -> None:
        """Release blocks back to free pool (M3: decrement ref_count first)."""
        for b in block_ids:
            if b not in self.ref_counts:
                continue
            self.ref_counts[b] -= 1
            if self.ref_counts[b] == 0:
                del self.ref_counts[b]
                self.free_blocks.append(b)

    def num_blocks_needed(self, num_tokens: int) -> int:
        """How many blocks needed to cache `num_tokens` tokens?"""
        return (num_tokens + self.block_size - 1) // self.block_size

    def get_slot_mapping(self, block_table_layer: List[int], num_tokens: int) -> List[int]:
        """Compute slot_mapping for one layer: flat indices into the per-layer KV cache.

        slot_mapping[i] = block_table_layer[token_i // block_size] * block_size + token_i % block_size

        Args:
            block_table_layer: the block table for a single layer (list of block_ids)
            num_tokens: number of tokens to map
        """
        slots = []
        for t in range(num_tokens):
            block_idx = t // self.block_size
            offset = t % self.block_size
            block_id = block_table_layer[block_idx]
            slots.append(block_id * self.block_size + offset)
        return slots
