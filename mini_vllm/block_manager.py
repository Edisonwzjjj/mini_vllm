"""Block manager — paged KV cache allocation + prefix cache (prefix cache in M3)."""

import hashlib
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

        # Prefix cache: hash_string -> block_id
        self.hash_to_block_id: Dict[str, int] = {}
        # Reverse index: block_id -> hash_string (for cleanup on eviction)
        self.block_to_hash: Dict[int, str] = {}

        # Cached blocks: ref_count=0 but data preserved for prefix reuse
        # Evicted only when free_blocks runs out
        self.cached_blocks: List[int] = []

        # Cache hit stats
        self.total_blocks_requested: int = 0
        self.total_blocks_hit: int = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def _hash_func(self, data: bytes) -> str:
        """Hash function for prefix cache. Use md5 for speed."""
        return hashlib.md5(data).hexdigest()

    def compute_hashes(self, token_ids: List[int]) -> List[str]:
        """Compute chain hash for each full block in token_ids.

        Hash chain: hash_i = hash(parent_hash || block_i_tokens)
        Only full blocks are hashed (last partial block is skipped).

        Returns:
            List of hash strings, one per full block. Length = num_full_blocks.
        """
        num_full_blocks = len(token_ids) // self.block_size
        hashes = []
        parent_hash = b""  # empty bytes = no parent
        for i in range(num_full_blocks):
            start = i * self.block_size
            block_tokens = token_ids[start:start + self.block_size]
            # Chain: parent_hash || block_tokens
            data = parent_hash + str(tuple(block_tokens)).encode()
            block_hash = self._hash_func(data)
            hashes.append(block_hash)
            parent_hash = block_hash.encode()  # next block's parent
        return hashes

    def allocate_with_prefix(
        self, token_ids: List[int], block_table_layer: List[int]
    ) -> Tuple[List[int], int]:
        """Allocate blocks for a sequence, reusing prefix cache hits.

        For each full block, check if hash hits in prefix cache:
          - Hit + block in use (ref_count > 0): reuse, ref_count++
          - Hit + block cached (ref_count == 0): re-activate from cached pool
          - Miss: allocate new block
        Last partial block always gets a new allocation.
        """
        hashes = self.compute_hashes(token_ids)
        num_full_blocks = len(hashes)
        num_total_blocks = self.num_blocks_needed(len(token_ids))
        cached_tokens = 0

        # 1. Check prefix cache hits for full blocks
        hit_count = 0  # consecutive hits from the start
        for h in hashes:
            if h in self.hash_to_block_id:
                block_id = self.hash_to_block_id[h]
                # Block still valid: either in-use or cached
                if block_id in self.ref_counts or block_id in self.cached_blocks:
                    hit_count += 1
                else:
                    break  # stale entry, treat as miss
            else:
                break  # prefix must be consecutive; break at first miss

        # 2. Reuse hit blocks
        for i in range(hit_count):
            block_id = self.hash_to_block_id[hashes[i]]
            if block_id in self.cached_blocks:
                # Re-activate cached block
                self.cached_blocks.remove(block_id)
                self.ref_counts[block_id] = 1
            else:
                # Block already in use, increment ref_count
                self.ref_counts[block_id] += 1
            block_table_layer.append(block_id)
        cached_tokens = hit_count * self.block_size

        # Track hit rate
        self.total_blocks_requested += num_full_blocks
        self.total_blocks_hit += hit_count

        # 3. Allocate new blocks for the rest (misses + partial block)
        num_new_blocks = num_total_blocks - hit_count
        if num_new_blocks > 0:
            self._ensure_free_blocks(num_new_blocks)
            new_blocks = self.free_blocks[:num_new_blocks]
            self.free_blocks = self.free_blocks[num_new_blocks:]
            for b in new_blocks:
                self.ref_counts[b] = 1
            block_table_layer.extend(new_blocks)

        return block_table_layer, cached_tokens

    def _ensure_free_blocks(self, num_needed: int) -> None:
        """Ensure enough free blocks, evicting cached blocks if necessary."""
        while len(self.free_blocks) < num_needed and self.cached_blocks:
            # Evict the oldest cached block
            block_id = self.cached_blocks.pop(0)
            # Clean up hash entries for evicted block
            if block_id in self.block_to_hash:
                h = self.block_to_hash.pop(block_id)
                self.hash_to_block_id.pop(h, None)
            self.free_blocks.append(block_id)
        if len(self.free_blocks) < num_needed:
            raise RuntimeError(
                f"Not enough blocks: need {num_needed}, "
                f"have {len(self.free_blocks)} free + {len(self.cached_blocks)} cached"
            )

    def allocate(self, num_blocks: int) -> List[int]:
        """Allocate `num_blocks` fresh blocks. No prefix cache lookup."""
        self._ensure_free_blocks(num_blocks)
        blocks = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        for b in blocks:
            self.ref_counts[b] = 1
        return blocks

    def deallocate(self, block_ids: List[int]) -> None:
        """Release blocks: decrement ref_count. When ref_count hits 0,
        move block to cached_blocks (data preserved for prefix reuse).
        Cached blocks are evicted only when free_blocks runs out.
        """
        for b in block_ids:
            if b not in self.ref_counts:
                continue
            self.ref_counts[b] -= 1
            if self.ref_counts[b] == 0:
                del self.ref_counts[b]
                # Don't free immediately — keep in cached pool for reuse
                self.cached_blocks.append(b)

    def cache_hit_rate(self) -> float:
        """Return prefix cache hit rate (hit blocks / total requested)."""
        if self.total_blocks_requested == 0:
            return 0.0
        return self.total_blocks_hit / self.total_blocks_requested

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

    def hash_blocks(self, token_ids: List[int], block_table_layer: List[int]) -> None:
        """Register full blocks into prefix cache after prefill.

        Called AFTER forward pass completes, when KV values are written to cache.
        For each full block, register hash -> block_id and block_id -> hash.
        """
        hashes = self.compute_hashes(token_ids)
        for i in range(len(hashes)):
            block_id = block_table_layer[i]
            self.hash_to_block_id[hashes[i]] = block_id
            self.block_to_hash[block_id] = hashes[i]
            
