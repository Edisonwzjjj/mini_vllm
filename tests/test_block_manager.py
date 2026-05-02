"""Test: BlockManager basic operations."""

from mini_vllm.block_manager import BlockManager


def test_allocate_deallocate():
    bm = BlockManager(block_size=16, num_blocks=10)
    blocks = bm.allocate(3)
    assert len(blocks) == 3
    assert bm.num_free_blocks == 7
    bm.deallocate(blocks)
    assert bm.num_free_blocks == 10


def test_num_blocks_needed():
    bm = BlockManager(block_size=16, num_blocks=100)
    assert bm.num_blocks_needed(1) == 1
    assert bm.num_blocks_needed(16) == 1
    assert bm.num_blocks_needed(17) == 2
    assert bm.num_blocks_needed(32) == 2


def test_slot_mapping():
    bm = BlockManager(block_size=4, num_blocks=10)
    block_table = [2, 5]  # 2 blocks, block_size=4
    # token 0→ block 2, offset 0 → slot 8
    # token 1→ block 2, offset 1 → slot 9
    # token 2→ block 2, offset 2 → slot 10
    # token 3→ block 2, offset 3 → slot 11
    # token 4→ block 5, offset 0 → slot 20
    # token 5→ block 5, offset 1 → slot 21
    slots = bm.get_slot_mapping(block_table, 6)
    assert slots == [8, 9, 10, 11, 20, 21]


def test_allocate_too_many():
    bm = BlockManager(block_size=16, num_blocks=5)
    with pytest.raises(RuntimeError, match="Not enough blocks"):
        bm.allocate(6)


import pytest
