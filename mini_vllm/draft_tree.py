"""Draft tree data structures for tree speculative decoding."""

from dataclasses import dataclass

import torch


@dataclass
class DraftTree:
    """Draft tokens plus parent links.

    The implicit root is seq.last_token_id and is not included in token_ids.
    parent_indices[i] == -1 means token_ids[i]'s parent is the implicit root.
    """

    token_ids: list[int]
    parent_indices: list[int]

    def __post_init__(self) -> None:
        assert len(self.token_ids) == len(self.parent_indices)
        for i, parent in enumerate(self.parent_indices):
            assert parent == -1 or parent < i

    def ancestors(self, idx: int) -> list[int]:
        """Return ancestor node indices plus self, from root child to idx."""
        ancestors: list[int] = []
        while idx != -1:
            ancestors.append(idx)
            idx = self.parent_indices[idx]
        return ancestors[::-1]

    def children(self, idx: int) -> list[int]:
        """Return child node indices for a parent node index.

        idx == -1 means children of the implicit root / seq.last_token_id.
        """
        return [
            i for i, parent_idx in enumerate(self.parent_indices)
            if parent_idx == idx
        ]

    def build_tree_mask(self, device: torch.device) -> torch.Tensor:
        """Build draft-to-draft attention mask.

        mask[query_idx, key_idx] = 0.0 if key is query's ancestor or self.
        mask[query_idx, key_idx] = -inf otherwise.
        """
        n = len(self.token_ids)
        mask = torch.full((n, n), float("-inf"), device=device)
        for i in range(n):
            for j in self.ancestors(i):
                mask[i, j] = 0.0
        return mask


def build_dummy_tree(root_token_id: int, topk: int = 2, depth: int = 3) -> DraftTree:
    assert topk == 2
    assert depth == 3

    # DFS pre-order:
    #
    # implicit root / seq.last_token_id
    # └── 0
    #     ├── 1
    #     │   ├── 2
    #     │   └── 3
    #     └── 4
    #         ├── 5
    #         └── 6
    token_ids = [root_token_id] * 7
    parent_indices = [-1, 0, 1, 1, 0, 4, 4]
    return DraftTree(token_ids=token_ids, parent_indices=parent_indices)
