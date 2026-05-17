"""Block manager — paged KV cache allocation + prefix cache (radix tree)."""

import heapq
import time
from typing import Dict, List, Optional, Tuple


class BlockManager:
    """Manages paged KV cache blocks with radix-tree prefix cache.

    Each block stores KV for `block_size` consecutive tokens.
    Blocks are identified by integer block_id.

    M1 scope: allocate / append / deallocate only (no prefix cache).
    M3 scope: add radix-tree prefix cache with ref counting.
    """

    def __init__(self, block_size: int, num_blocks: int):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Free block pool
        self.free_blocks: List[int] = list(range(num_blocks))

        # Radix tree for prefix cache
        self.radix_tree = RadixTree(block_size)
        # Reverse index: block_id -> tree node (for deallocate)
        self.block_id_to_node: Dict[int, RadixTreeNode] = {}

        # Eviction heap: min-heap of (last_access_time, id(node), node)
        # Lazy deletion — pop and skip if node is no longer evictable.
        self.eviction_heap: List[Tuple[float, int, RadixTreeNode]] = []

        # Cache hit stats
        self.total_blocks_requested: int = 0
        self.total_blocks_hit: int = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def allocate_with_prefix(
        self, token_ids: List[int], block_table_layer: List[int]
    ) -> Tuple[List[int], int]:
        """Allocate blocks for a sequence, reusing prefix cache hits.

        Uses radix tree match_prefix to find cached blocks:
          - Hit + in use (ref_count > 0): reuse, ref_count++
          - Hit + cached (ref_count == 0): re-activate
          - Miss: allocate new block
        Last partial block always gets a new allocation.
        """
        num_full_blocks = len(token_ids) // self.block_size
        num_total_blocks = self.num_blocks_needed(len(token_ids))
        now = time.monotonic()

        # 1. Match prefix in radix tree
        matched_nodes, cached_tokens = self.radix_tree.match_prefix(token_ids)
        hit_count = len(matched_nodes)

        # 2. Reuse hit blocks
        for node in matched_nodes:
            node.ref_count += 1
            node.last_access_time = now
            block_table_layer.append(node.block_id)

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
                # Temporarily track ref_count=1; will be moved to tree on insert
                node = RadixTreeNode()
                node.block_id = b
                node.ref_count = 1
                node.last_access_time = now
                self.block_id_to_node[b] = node
            block_table_layer.extend(new_blocks)

        return block_table_layer, cached_tokens

    def _ensure_free_blocks(self, num_needed: int) -> None:
        """Ensure enough free blocks, evicting cached tree nodes if necessary."""
        while len(self.free_blocks) < num_needed:
            evict_node = self._pop_eviction_candidate()
            if evict_node is None:
                break
            # evict returns all block_ids freed (including cascaded parents)
            freed_ids = self.radix_tree.evict(evict_node)
            for block_id in freed_ids:
                self.block_id_to_node.pop(block_id, None)
            self.free_blocks.extend(freed_ids)
        if len(self.free_blocks) < num_needed:
            raise RuntimeError(
                f"Not enough blocks: need {num_needed}, "
                f"have {len(self.free_blocks)} free"
            )

    def _push_eviction(self, node: "RadixTreeNode") -> None:
        """Push a node onto the eviction heap."""
        heapq.heappush(self.eviction_heap, (node.last_access_time, id(node), node))

    def _pop_eviction_candidate(self) -> Optional["RadixTreeNode"]:
        """Pop the best eviction candidate from the heap (LRU).

        Uses lazy deletion: skip entries that are no longer evictable
        (ref_count changed, node got children, or already evicted).
        """
        while self.eviction_heap:
            _, _, node = heapq.heappop(self.eviction_heap)
            if node.is_evictable:
                return node
        return None

    def allocate(self, num_blocks: int) -> List[int]:
        """Allocate `num_blocks` fresh blocks. No prefix cache lookup."""
        self._ensure_free_blocks(num_blocks)
        blocks = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        now = time.monotonic()
        for b in blocks:
            node = RadixTreeNode()
            node.block_id = b
            node.ref_count = 1
            node.last_access_time = now
            self.block_id_to_node[b] = node
        return blocks
    
    def try_allocate(self, num_blocks: int) -> Optional[List[int]]:
        """Try to allocate without raising. Returns None if not enough blocks."""
        # Eviction heap size is an upper bound on evictable nodes (lazy deletion)
        if len(self.free_blocks) + len(self.eviction_heap) < num_blocks:
            return None
        self._ensure_free_blocks(num_blocks)
        blocks = self.free_blocks[:num_blocks]
        self.free_blocks = self.free_blocks[num_blocks:]
        return blocks

    def deallocate(self, block_ids: List[int]) -> None:
        """Release blocks: decrement ref_count on tree nodes.

        When ref_count hits 0:
          - If node is in the radix tree: stays for prefix reuse, becomes evictable.
            Push onto eviction heap if it's a leaf.
          - If node is an orphan (not in tree, e.g. from allocate()): free immediately.
        """
        for b in block_ids:
            node = self.block_id_to_node.get(b)
            if node is None or node.ref_count <= 0:
                continue
            node.ref_count -= 1
            if node.ref_count == 0:
                if node.parent is None and node is not self.radix_tree.root:
                    # Orphan node (not in tree): free immediately
                    self.free_blocks.append(b)
                    self.block_id_to_node.pop(b, None)
                elif node.is_leaf:
                    # Tree node became evictable — push onto heap
                    self._push_eviction(node)

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

    def insert_blocks(self, token_ids: List[int], block_table_layer: List[int]) -> None:
        """Register full blocks into radix tree after prefill.

        Called AFTER forward pass completes, when KV values are written to cache.
        Inserts the full path into the tree and links block_id -> node.

        For blocks already in block_id_to_node (orphan nodes from allocate_with_prefix),
        their ref_count is carried over to the tree node.
        """
        num_full_blocks = len(token_ids) // self.block_size
        full_block_ids = block_table_layer[:num_full_blocks]
        nodes = self.radix_tree.insert(token_ids, full_block_ids)
        for node in nodes:
            old = self.block_id_to_node.get(node.block_id)
            if old is not None and old is not node:
                # Orphan node: carry over ref_count
                node.ref_count = old.ref_count
                node.last_access_time = old.last_access_time
            self.block_id_to_node[node.block_id] = node

    def _iter_nodes(self):
        """Iterate all tree nodes (BFS)."""
        stack = [self.radix_tree.root]
        while stack:
            node = stack.pop()
            if node is not self.radix_tree.root:
                yield node
            stack.extend(node.children.values())
            

class RadixTreeNode:
    """Node in the radix tree for block-level prefix cache.

    Each node corresponds to one full block (block_size tokens).
    The root node is virtual: no tokens, no block_id.
    At block granularity, no node splitting is needed
    (unlike SGLang which operates at token granularity).
    """

    __slots__ = ("parent", "children", "block_id", "ref_count", "last_access_time")

    def __init__(self, parent: Optional["RadixTreeNode"] = None):
        self.parent: Optional[RadixTreeNode] = parent
        self.children: Dict[tuple, "RadixTreeNode"] = {}  # key = block_tokens tuple
        self.block_id: int = -1         # KV block id (-1 = root / unassigned)
        self.ref_count: int = 0         # how many sequences reference this node
        self.last_access_time: float = 0.0  # for LRU eviction

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def is_evictable(self) -> bool:
        """Only leaf nodes with ref_count==0 and valid block can be evicted."""
        return self.ref_count == 0 and self.block_id >= 0 and self.is_leaf


class RadixTree:
    """Radix tree for block-level prefix cache.

    Nodes are indexed by block-sized token tuples.
    Supports longest prefix match, insertion, and LRU eviction.
    """

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.root = RadixTreeNode()

    def _token_ids_to_blocks(self, token_ids: List[int]) -> List[tuple]:
        """Split token_ids into block-sized tuples. Only full blocks."""
        blocks = []
        for i in range(0, len(token_ids) // self.block_size * self.block_size, self.block_size):
            blocks.append(tuple(token_ids[i:i + self.block_size]))
        return blocks

    def match_prefix(self, token_ids: List[int]) -> Tuple[List[RadixTreeNode], int]:
        """Find longest prefix match in the tree.

        Returns:
            (matched_nodes, cached_tokens) — list of matched nodes and total cached tokens.
        """
        blocks = self._token_ids_to_blocks(token_ids)
        matched_nodes = []
        node = self.root
        for block_tokens in blocks:
            if block_tokens in node.children:
                child = node.children[block_tokens]
                matched_nodes.append(child)
                node = child
            else:
                break
        cached_tokens = len(matched_nodes) * self.block_size
        return matched_nodes, cached_tokens

    def insert(self, token_ids: List[int], block_ids: List[int]) -> List[RadixTreeNode]:
        """Insert a path into the tree, creating nodes for missing blocks.

        Args:
            token_ids: full token sequence (only full blocks are inserted).
            block_ids: block_id for each full block (same length as full blocks).

        Returns:
            List of nodes created or traversed (one per full block).
        """
        blocks = self._token_ids_to_blocks(token_ids)
        assert len(blocks) == len(block_ids), (
            f"Expected {len(blocks)} block_ids, got {len(block_ids)}"
        )
        nodes = []
        node = self.root
        for block_tokens, block_id in zip(blocks, block_ids):
            if block_tokens in node.children:
                child = node.children[block_tokens]
            else:
                child = RadixTreeNode(parent=node)
                node.children[block_tokens] = child
            child.block_id = block_id
            nodes.append(child)
            node = child
        return nodes

    def evict(self, node: RadixTreeNode) -> List[int]:
        """Evict a leaf node: remove from parent, return freed block_ids.

        Only evictable nodes (leaf, ref_count==0, has block) can be evicted.
        After eviction, check if parent also became evictable leaf (recursive cleanup).
        Returns list of all block_ids freed (including cascaded parents).
        """
        if not node.is_evictable:
            raise RuntimeError("Cannot evict non-evictable node")
        # Detach from parent
        parent = node.parent
        if parent is not None:
            for key, child in parent.children.items():
                if child is node:
                    del parent.children[key]
                    break
        freed = [node.block_id]
        # Recursive cleanup: if parent became an evictable leaf, evict it too
        if parent is not None and parent.is_evictable:
            freed.extend(self.evict(parent))
        return freed

    def match_prefix_ratio(self, token_ids: list[int]) -> float:
        """Return prefix match ratio: cached tokens / total tokens.

        Represents what fraction of tokens don't need to be computed.
        """
        if len(token_ids) == 0:
            return 0.0
        _, cached_tokens = self.match_prefix(token_ids)
        return cached_tokens / len(token_ids)
        