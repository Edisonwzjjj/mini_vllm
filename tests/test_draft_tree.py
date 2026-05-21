"""Tests for draft tree utilities used by tree speculative decoding."""

import torch
import pytest

from mini_vllm.attention_patch import _build_tree_verify_mask
from mini_vllm.draft_tree import DraftTree, build_dummy_tree


def test_dummy_tree_structure() -> None:
    tree = build_dummy_tree(root_token_id=42, topk=2, depth=3)

    assert tree.token_ids == [42] * 7
    assert tree.parent_indices == [-1, 0, 1, 1, 0, 4, 4]


def test_draft_tree_rejects_invalid_parent_order() -> None:
    with pytest.raises(AssertionError):
        DraftTree(token_ids=[1, 2], parent_indices=[1, -1])


def test_ancestors() -> None:
    tree = build_dummy_tree(root_token_id=42)

    assert tree.ancestors(0) == [0]
    assert tree.ancestors(1) == [0, 1]
    assert tree.ancestors(2) == [0, 1, 2]
    assert tree.ancestors(3) == [0, 1, 3]
    assert tree.ancestors(4) == [0, 4]
    assert tree.ancestors(5) == [0, 4, 5]
    assert tree.ancestors(6) == [0, 4, 6]


def test_children() -> None:
    tree = build_dummy_tree(root_token_id=42)

    assert tree.children(-1) == [0]
    assert tree.children(0) == [1, 4]
    assert tree.children(1) == [2, 3]
    assert tree.children(4) == [5, 6]
    assert tree.children(2) == []


def test_tree_mask_allows_only_ancestors_and_self() -> None:
    tree = build_dummy_tree(root_token_id=42)
    mask = tree.build_tree_mask(torch.device("cpu"))

    assert mask.shape == (7, 7)

    # Node 2 path: 0 -> 1 -> 2.
    assert mask[2, 0].item() == 0.0
    assert mask[2, 1].item() == 0.0
    assert mask[2, 2].item() == 0.0
    assert torch.isneginf(mask[2, 3]).item()
    assert torch.isneginf(mask[2, 4]).item()
    assert torch.isneginf(mask[2, 6]).item()

    # Node 5 path: 0 -> 4 -> 5.
    assert mask[5, 0].item() == 0.0
    assert mask[5, 4].item() == 0.0
    assert mask[5, 5].item() == 0.0
    assert torch.isneginf(mask[5, 1]).item()
    assert torch.isneginf(mask[5, 2]).item()
    assert torch.isneginf(mask[5, 6]).item()


def test_tree_verify_mask_adds_prefix_and_last_token_offsets() -> None:
    tree = build_dummy_tree(root_token_id=42)
    tree_mask = tree.build_tree_mask(torch.device("cpu"))
    prefix_len = 3

    mask = _build_tree_verify_mask(prefix_len, tree_mask, torch.device("cpu"))

    # num_draft=7, suffix=[last_token]+drafts has 8 rows,
    # keys=[prefix(3)]+[last_token]+drafts has 11 columns.
    assert mask.shape == (1, 1, 8, 11)

    # All query rows can attend to all prefix tokens.
    assert mask[0, 0, 0, 0].item() == 0.0
    assert mask[0, 0, 0, 2].item() == 0.0
    assert mask[0, 0, 7, 0].item() == 0.0
    assert mask[0, 0, 7, 2].item() == 0.0

    # Row 0 is last_token. It can attend itself but no draft token.
    last_token_col = prefix_len
    assert mask[0, 0, 0, last_token_col].item() == 0.0
    assert torch.isneginf(mask[0, 0, 0, last_token_col + 1]).item()
    assert torch.isneginf(mask[0, 0, 0, last_token_col + 7]).item()

    # Draft node 2 row can attend last_token and draft path 0 -> 1 -> 2.
    draft_2_row = 1 + 2
    draft_0_col = prefix_len + 1 + 0
    draft_1_col = prefix_len + 1 + 1
    draft_2_col = prefix_len + 1 + 2
    draft_3_col = prefix_len + 1 + 3
    draft_4_col = prefix_len + 1 + 4
    assert mask[0, 0, draft_2_row, last_token_col].item() == 0.0
    assert mask[0, 0, draft_2_row, draft_0_col].item() == 0.0
    assert mask[0, 0, draft_2_row, draft_1_col].item() == 0.0
    assert mask[0, 0, draft_2_row, draft_2_col].item() == 0.0
    assert torch.isneginf(mask[0, 0, draft_2_row, draft_3_col]).item()
    assert torch.isneginf(mask[0, 0, draft_2_row, draft_4_col]).item()

    # Draft node 5 row can attend last_token and draft path 0 -> 4 -> 5.
    draft_5_row = 1 + 5
    draft_5_col = prefix_len + 1 + 5
    assert mask[0, 0, draft_5_row, last_token_col].item() == 0.0
    assert mask[0, 0, draft_5_row, draft_0_col].item() == 0.0
    assert mask[0, 0, draft_5_row, draft_4_col].item() == 0.0
    assert mask[0, 0, draft_5_row, draft_5_col].item() == 0.0
    assert torch.isneginf(mask[0, 0, draft_5_row, draft_1_col]).item()
    assert torch.isneginf(mask[0, 0, draft_5_row, draft_2_col]).item()
