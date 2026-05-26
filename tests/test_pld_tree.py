"""Unit tests for prompt_lookup_tree (option-A topology: 7 slots).

The 4 invariants we want to lock in:
  1. Strong match → 7 nodes, parent_indices == _PARENT_INDICES, tokens come
     from PLD (no last_token padding in the result).
  2. No match → 7 nodes, all slots padded with last_token.
  3. Weak/partial match → 7 nodes, PLD-derived tokens for slots it can fill,
     last_token padding for the rest.
  4. parent_indices is always exactly [-1, 0, 1, 1, 0, 4, 4].
"""
from mini_vllm.draft_pld import _PARENT_INDICES, prompt_lookup_tree
from mini_vllm.draft_tree import DraftTree


def test_draft_pld_tree_all_match():
    # token_ids = [1, 2, 1, 2, 1, 2, 1]; PLD should find rich continuations.
    # Trace:
    #   slot 0 (root,        query=[1,2,1]) → 2
    #   slot 1 (parent=0,    query=[2,1,2]) → 1   (only 1 distinct cont)
    #   slot 2 (parent=1,    query=[1,2,1]) → 2
    #   slot 3 (parent=1)    pad with last_token=1
    #   slot 4 (parent=0)    pad with last_token=1
    #   slot 5 (parent=4,    query=[1,2,1]) → 2
    #   slot 6 (parent=4)    pad with last_token=1
    tree = prompt_lookup_tree([1, 2, 1, 2, 1, 2, 1])
    assert tree == DraftTree(
        token_ids=[2, 1, 2, 1, 1, 2, 1],
        parent_indices=[-1, 0, 1, 1, 0, 4, 4],
    )


def test_draft_pld_tree_no_match():
    # token_ids = [1, 2, 3, 4, 5]; no repeated bigram → all PLD queries miss.
    # All 7 slots padded with last_token (= 5).
    tree = prompt_lookup_tree([1, 2, 3, 4, 5])
    assert tree == DraftTree(
        token_ids=[5, 5, 5, 5, 5, 5, 5],
        parent_indices=[-1, 0, 1, 1, 0, 4, 4],
    )


def test_draft_pld_tree_partial_match():
    # token_ids = [1, 2, 3, 1, 2]; some bigrams hit, deeper levels fall back.
    # Trace:
    #   slot 0 (root,        query=[1,2]   via fallback) → 3
    #   slot 1 (parent=0,    query=[1,2,3]) → 1
    #   slot 2 (parent=1,    query=[2,3,1]) → 2
    #   slot 3, 4, 5, 6 all pad with last_token = 2
    tree = prompt_lookup_tree([1, 2, 3, 1, 2])
    assert tree == DraftTree(
        token_ids=[3, 1, 2, 2, 2, 2, 2],
        parent_indices=[-1, 0, 1, 1, 0, 4, 4],
    )


def test_draft_pld_tree_empty():
    # Empty input is the only case that returns None — caller skips spec.
    assert prompt_lookup_tree([]) is None


def test_draft_pld_tree_parent_indices_always_canonical():
    # Invariant 4: parent_indices never deviates from the option-A topology,
    # regardless of input.
    for token_ids in [
        [1, 2, 1, 2, 1, 2, 1],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 1, 2],
        [7, 7, 7],
        [100],
    ]:
        tree = prompt_lookup_tree(token_ids)
        assert tree is not None, f"got None for {token_ids}"
        assert tree.parent_indices == _PARENT_INDICES
        assert len(tree.token_ids) == 7
