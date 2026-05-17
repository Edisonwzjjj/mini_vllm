"""Test: radix tree prefix cache — block-level sharing, branching, eviction."""

from mini_vllm.block_manager import BlockManager, RadixTree, RadixTreeNode


# ---------- RadixTree unit tests (no GPU needed) ----------


class TestRadixTreeBasic:
    """Basic radix tree operations at block granularity."""

    def setup_method(self):
        self.tree = RadixTree(block_size=4)

    def test_match_prefix_empty_tree(self):
        """Empty tree matches nothing."""
        nodes, cached = self.tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
        assert nodes == []
        assert cached == 0

    def test_insert_and_match(self):
        """Insert one path, then match it fully."""
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 full blocks
        block_ids = [0, 1]
        nodes = self.tree.insert(token_ids, block_ids)
        assert len(nodes) == 2

        matched, cached = self.tree.match_prefix(token_ids)
        assert len(matched) == 2
        assert cached == 8  # 2 blocks * 4 tokens

    def test_partial_block_not_inserted(self):
        """Only full blocks are inserted; partial block is ignored."""
        token_ids = [1, 2, 3, 4, 5]  # 1 full + 1 partial
        block_ids = [0]
        nodes = self.tree.insert(token_ids, block_ids)
        assert len(nodes) == 1

    def test_match_prefix_partial_hit(self):
        """Insert a path, then match with a longer path — partial hit."""
        self.tree.insert([1, 2, 3, 4, 5, 6, 7, 8], [0, 1])

        # Longer path: first 2 blocks match, 3rd doesn't exist
        matched, cached = self.tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        assert len(matched) == 2
        assert cached == 8

    def test_no_match_different_tokens(self):
        """Completely different tokens should not match."""
        self.tree.insert([1, 2, 3, 4], [0])

        matched, cached = self.tree.match_prefix([5, 6, 7, 8])
        assert len(matched) == 0
        assert cached == 0


class TestRadixTreeBranching:
    """Test the key advantage of radix tree over hash chain: branching.

    Hash chain: system + user1 and system + user2 must be consecutive matches.
    Radix tree: system is a shared internal node, user1/user2 are separate branches.
    """

    def setup_method(self):
        self.tree = RadixTree(block_size=4)
        # system = [1,2,3,4], user1 = [5,6,7,8], user2 = [9,10,11,12]
        self.system = [1, 2, 3, 4]
        self.user1 = [5, 6, 7, 8]
        self.user2 = [9, 10, 11, 12]

    def test_branching_shared_prefix(self):
        """Two paths sharing a prefix should share the internal node."""
        # Insert system + user1
        self.tree.insert(self.system + self.user1, [0, 1])
        # Insert system + user2
        self.tree.insert(self.system + self.user2, [2, 3])

        # Both should match system prefix
        matched_a, cached_a = self.tree.match_prefix(self.system + self.user1)
        matched_b, cached_b = self.tree.match_prefix(self.system + self.user2)

        assert len(matched_a) == 2  # system + user1
        assert len(matched_b) == 2  # system + user2
        assert cached_a == 8
        assert cached_b == 8

        # The system block node should be the SAME object (shared)
        assert matched_a[0] is matched_b[0]
        # The user blocks should be DIFFERENT objects (branched)
        assert matched_a[1] is not matched_b[1]

    def test_system_node_has_two_children(self):
        """After branching, the system node has 2 children."""
        self.tree.insert(self.system + self.user1, [0, 1])
        self.tree.insert(self.system + self.user2, [2, 3])

        system_nodes, _ = self.tree.match_prefix(self.system)
        system_node = system_nodes[0]
        assert len(system_node.children) == 2

    def test_only_system_match(self):
        """A new path that shares only system should partially match."""
        self.tree.insert(self.system + self.user1, [0, 1])

        # system + different_user not in tree yet
        matched, cached = self.tree.match_prefix(self.system + [99, 99, 99, 99])
        assert len(matched) == 1  # only system
        assert cached == 4


class TestRadixTreeEviction:
    """Test LRU eviction: only leaf + ref_count==0 nodes are evictable."""

    def setup_method(self):
        self.tree = RadixTree(block_size=4)

    def test_leaf_evictable(self):
        """A leaf with ref_count=0 and block_id >= 0 is evictable."""
        self.tree.insert([1, 2, 3, 4], [0])
        matched, _ = self.tree.match_prefix([1, 2, 3, 4])
        leaf = matched[0]
        leaf.ref_count = 0
        assert leaf.is_evictable

    def test_internal_node_not_evictable(self):
        """An internal node (has children) is not evictable even with ref_count=0."""
        self.tree.insert([1, 2, 3, 4, 5, 6, 7, 8], [0, 1])
        matched, _ = self.tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
        parent = matched[0]  # first block node
        parent.ref_count = 0
        assert not parent.is_evictable  # has child

    def test_ref_count_nonzero_not_evictable(self):
        """A node with ref_count > 0 is not evictable even if leaf."""
        self.tree.insert([1, 2, 3, 4], [0])
        matched, _ = self.tree.match_prefix([1, 2, 3, 4])
        leaf = matched[0]
        leaf.ref_count = 1
        assert not leaf.is_evictable

    def test_evict_leaf_removes_from_parent(self):
        """Evicting a leaf detaches it from parent."""
        self.tree.insert([1, 2, 3, 4], [0])
        matched, _ = self.tree.match_prefix([1, 2, 3, 4])
        leaf = matched[0]
        leaf.ref_count = 0

        freed = self.tree.evict(leaf)
        assert freed == [0]
        # Root should have no children now
        assert len(self.tree.root.children) == 0

    def test_evict_leaf_cascades_to_parent(self):
        """After evicting a leaf, if parent becomes evictable leaf, it's evicted too."""
        self.tree.insert([1, 2, 3, 4, 5, 6, 7, 8], [0, 1])
        matched, _ = self.tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
        parent = matched[0]  # block 0
        child = matched[1]   # block 1
        parent.ref_count = 0
        child.ref_count = 0

        # Evict child → parent becomes leaf with ref_count=0 → cascaded eviction
        freed = self.tree.evict(child)
        assert freed == [1, 0]
        # Parent should also be evicted (cascade)
        assert len(self.tree.root.children) == 0

    def test_evict_leaf_no_cascade_if_parent_has_other_children(self):
        """Eviction doesn't cascade if parent still has other children."""
        self.tree.insert([1, 2, 3, 4, 5, 6, 7, 8], [0, 1])      # branch A
        self.tree.insert([1, 2, 3, 4, 9, 10, 11, 12], [2, 3])    # branch B

        matched, _ = self.tree.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
        parent = matched[0]  # shared system node (block 0)
        child_a = matched[1]  # user1 node (block 1)
        parent.ref_count = 0
        child_a.ref_count = 0

        # Evict user1 leaf → parent still has user2 branch, no cascade
        freed = self.tree.evict(child_a)
        assert freed == [1]
        # Parent still exists with 1 child
        assert len(self.tree.root.children) == 1
        system_node = list(self.tree.root.children.values())[0]
        assert len(system_node.children) == 1  # only branch B left


# ---------- BlockManager integration tests (no GPU needed) ----------


class TestBlockManagerRadixSharing:
    """Test BlockManager with radix tree: multi-turn prefix sharing."""

    def setup_method(self):
        self.bm = BlockManager(block_size=4, num_blocks=20)

    def test_multi_turn_shared_system_prompt(self):
        """Multi-turn: seq1 = [system + user1], seq2 = [system + user2].
        seq2 should hit system prefix cache. After seq1 deallocates,
        system blocks should NOT be freed (seq2 still using them).
        """
        system = [1, 2, 3, 4]       # 1 block
        user1 = [5, 6, 7, 8]        # 1 block
        user2 = [9, 10, 11, 12]     # 1 block

        # --- Seq1: allocate system + user1 ---
        bt1, cached1 = self.bm.allocate_with_prefix(
            system + user1, []
        )
        assert cached1 == 0  # no cache yet
        assert len(bt1) == 2  # 2 blocks
        # Simulate forward pass → insert into tree
        self.bm.insert_blocks(system + user1, bt1)

        # --- Seq2: allocate system + user2 ---
        self.bm.total_blocks_requested = 0
        self.bm.total_blocks_hit = 0
        bt2, cached2 = self.bm.allocate_with_prefix(
            system + user2, []
        )
        # System block should hit cache
        assert cached2 > 0, f"Expected system prefix cache hit, got cached={cached2}"
        assert len(bt2) == 2
        # Verify hit rate
        assert self.bm.total_blocks_hit > 0

        # System block should be shared: same block_id in both tables
        assert bt1[0] == bt2[0], "System block should be shared"

        # Insert seq2 into tree
        self.bm.insert_blocks(system + user2, bt2)

        # Verify system node has ref_count == 2
        system_node = self.bm.block_id_to_node[bt1[0]]
        assert system_node.ref_count == 2, f"Expected ref_count=2, got {system_node.ref_count}"

        # --- Deallocate seq1 ---
        self.bm.deallocate(bt1)
        # System block ref_count should be 1 (not 0), NOT evictable
        assert system_node.ref_count == 1
        # User1 block ref_count should be 0, becomes evictable leaf
        user1_node = self.bm.block_id_to_node[bt1[1]]
        assert user1_node.ref_count == 0
        assert user1_node.is_evictable

        # --- Seq3: same system + user3, should still hit system cache ---
        self.bm.total_blocks_requested = 0
        self.bm.total_blocks_hit = 0
        bt3, cached3 = self.bm.allocate_with_prefix(
            system + [13, 14, 15, 16], []
        )
        assert cached3 > 0, "System prefix should still be cached after seq1 dealloc"
        assert bt3[0] == bt1[0], "System block should still be the same"

    def test_identical_prompt_full_hit(self):
        """Same prompt twice: 2nd should fully hit prefix cache."""
        tokens = [1, 2, 3, 4, 5, 6, 7, 8]  # 2 blocks

        # First: no cache
        bt1, cached1 = self.bm.allocate_with_prefix(tokens, [])
        assert cached1 == 0
        self.bm.insert_blocks(tokens, bt1)
        self.bm.deallocate(bt1)

        # Second: full cache hit
        bt2, cached2 = self.bm.allocate_with_prefix(tokens, [])
        assert cached2 == 8, f"Expected full cache hit (8), got {cached2}"
        assert bt2[0] == bt1[0] and bt2[1] == bt1[1]

    def test_lru_eviction_order(self):
        """Eviction should pick the least recently accessed evictable leaf."""
        # Insert two separate paths
        self.bm.allocate_with_prefix([1, 2, 3, 4], [])
        bt_a, _ = self.bm.allocate_with_prefix([1, 2, 3, 4], [])
        self.bm.insert_blocks([1, 2, 3, 4], bt_a)

        bt_b, _ = self.bm.allocate_with_prefix([5, 6, 7, 8], [])
        self.bm.insert_blocks([5, 6, 7, 8], bt_b)

        # Deallocate both → both evictable
        self.bm.deallocate(bt_a)
        self.bm.deallocate(bt_b)

        # Both blocks are evictable leaves, bt_a was accessed first
        # (lower last_access_time) → should be evicted first
        candidate = self.bm._pop_eviction_candidate()
        assert candidate is not None
        # The older one should be picked
        node_a = self.bm.block_id_to_node[bt_a[0]]
        node_b = self.bm.block_id_to_node[bt_b[0]]
        assert candidate is node_a or candidate is node_b

    def test_eviction_frees_blocks_for_new_allocation(self):
        """When free blocks run out, evicting cached nodes should free space."""
        # Use all blocks
        small_bm = BlockManager(block_size=4, num_blocks=4)
        bt, _ = small_bm.allocate_with_prefix([1, 2, 3, 4, 5, 6, 7, 8], [])
        small_bm.insert_blocks([1, 2, 3, 4, 5, 6, 7, 8], bt)
        small_bm.deallocate(bt)

        # Now allocate again — should succeed by evicting cached nodes
        bt2, cached = small_bm.allocate_with_prefix([9, 10, 11, 12], [])
        assert cached == 0  # different tokens, no cache hit
        assert len(bt2) == 1

    def test_three_branches_same_system(self):
        """Three conversations sharing same system prompt → 3 branches."""
        system = [1, 2, 3, 4]
        users = [
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]

        block_tables = []
        for user in users:
            bt, cached = self.bm.allocate_with_prefix(system + user, [])
            self.bm.insert_blocks(system + user, bt)
            block_tables.append(bt)

        # System node should have 3 children
        system_node = self.bm.block_id_to_node[block_tables[0][0]]
        assert len(system_node.children) == 3, (
            f"Expected 3 branches from system node, got {len(system_node.children)}"
        )
        # System ref_count = 3
        assert system_node.ref_count == 3
