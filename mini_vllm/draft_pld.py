"""Prompt Lookup Decoding (PLD) draft generators.

Zero-cost speculative draft: search the existing token sequence for the most
recent occurrence of the last n-gram, and use the tokens that followed it as
the draft. Falls back from max_ngram down to min_ngram.

Two flavours:
- prompt_lookup_draft: chain (single linear sequence)
- prompt_lookup_tree:  tree (multiple sibling continuations, fixed topology)

Reference: https://github.com/apoorvumang/prompt-lookup-decoding
"""
from __future__ import annotations

from mini_vllm.draft_tree import DraftTree


# Fixed topology (option A): same as build_dummy_tree.
# Slot indices follow DFS pre-order:
#
#   root (implicit, = seq.last_token_id) ── 0 ─┬── 1 ─┬── 2
#                                              │      └── 3
#                                              └── 4 ─┬── 5
#                                                     └── 6
#
# Parent of slot i is _PARENT_INDICES[i]; -1 means the implicit root.
_PARENT_INDICES = [-1, 0, 1, 1, 0, 4, 4]
_NUM_SLOTS = len(_PARENT_INDICES)


def prompt_lookup_draft(
    token_ids: list[int],
    draft_len: int,
    max_ngram: int = 3,
    min_ngram: int = 2,
) -> list[int]:
    """Look up the last n-gram in token_ids and return the next draft_len tokens.

    Returns up to draft_len tokens, or [] on miss / insufficient sequence length.
    """
    n = len(token_ids)
    for gram_size in range(max_ngram, min_ngram - 1, -1):
        if n < gram_size + 1:
            continue
        query = token_ids[n - gram_size:]
        for idx in range(n - gram_size - 1, -1, -1):
            if token_ids[idx:idx + gram_size] == query:
                end = idx + gram_size
                return token_ids[end:end + draft_len]
    return []


def _find_topk_distinct_continuations(
    token_ids: list[int],
    query: list[int],
    topk: int,
    max_ngram: int,
    min_ngram: int,
) -> list[int]:
    """Search token_ids for occurrences of query (with ngram fallback) and
    return up to topk distinct continuation tokens, ordered by recency.

    Strategy: try gram_size = min(len(query), max_ngram) down to min_ngram,
    each time searching all matches right-to-left and collecting distinct
    continuation tokens. Returns as soon as we have topk distinct tokens.
    """
    n = len(token_ids)
    seen: set[int] = set()
    out: list[int] = []

    upper = min(len(query), max_ngram)
    for gram_size in range(upper, min_ngram - 1, -1):
        if gram_size > len(query):
            continue
        if n < gram_size + 1:
            continue
        # Use the last gram_size tokens of the (possibly longer) query.
        sub_query = query[-gram_size:]
        for idx in range(n - gram_size - 1, -1, -1):
            end = idx + gram_size
            if end >= n:
                continue
            if token_ids[idx:end] == sub_query:
                cont = token_ids[end]
                if cont not in seen:
                    seen.add(cont)
                    out.append(cont)
                    if len(out) >= topk:
                        return out
    return out


def prompt_lookup_tree(
    token_ids: list[int],
    topk: int = 2,
    depth: int = 3,
    max_ngram: int = 3,
    min_ngram: int = 2,
) -> DraftTree | None:
    """Build a multi-branch draft tree by running PLD per parent slot.

    Topology is fixed to the option-A layout (7 slots, parent_indices =
    [-1, 0, 1, 1, 0, 4, 4]). topk and depth must match that layout.

    For each parent slot we issue one PLD query whose key is
        prompt_suffix + tokens-along-ancestor-chain(parent)
    and collect up to topk distinct continuation tokens to fill that parent's
    child slots. Slots that cannot be filled by PLD are padded with the last
    prompt token (low expected acceptance, but verify forward stays correct).

    Returns None only for an empty token_ids; otherwise always returns a fully
    populated 7-slot tree (padded with last_token where PLD misses).
    """
    assert topk == 2 and depth == 3, (
        "prompt_lookup_tree currently only supports the option-A topology "
        "(topk=2, depth=3, 7 slots)."
    )
    if not token_ids:
        return None

    last_token = token_ids[-1]
    slot_tokens: list[int | None] = [None] * _NUM_SLOTS

    # Group children by parent slot. Order matters: we must fill a parent's
    # token before computing the query for its children's group.
    # Iteration order is by parent slot index ascending; -1 (root) goes first.
    groups: list[tuple[int, list[int]]] = [(-1, [0])]
    for parent_slot in range(_NUM_SLOTS):
        kids = [i for i, p in enumerate(_PARENT_INDICES) if p == parent_slot]
        if kids:
            groups.append((parent_slot, kids))

    for parent_slot, child_slots in groups:
        # Build the query for this parent group.
        # Ancestor tokens (in tree order) are appended to the prompt suffix.
        ancestor_chain: list[int] = []
        if parent_slot != -1:
            cur = parent_slot
            chain_rev: list[int] = []
            while cur != -1:
                t = slot_tokens[cur]
                if t is None:
                    # Parent slot was never filled; bail out for this group.
                    chain_rev = []
                    ancestor_chain = []
                    break
                chain_rev.append(t)
                cur = _PARENT_INDICES[cur]
            ancestor_chain = chain_rev[::-1]

        # The query searches for the prompt's tail joined with the ancestor
        # tokens. We try the largest practical n-gram first; we don't pad
        # the prompt suffix beyond what's available.
        prompt_tail_size = max(max_ngram - len(ancestor_chain), 0)
        prompt_tail = token_ids[-prompt_tail_size:] if prompt_tail_size else []
        query = prompt_tail + ancestor_chain
        # If query is shorter than min_ngram, PLD has nothing to match on.
        if len(query) < min_ngram:
            candidates: list[int] = []
        else:
            candidates = _find_topk_distinct_continuations(
                token_ids=token_ids,
                query=query,
                topk=len(child_slots),
                max_ngram=max_ngram,
                min_ngram=min_ngram,
            )

        # Root-level miss: fall through to padding (whole tree = last_token).
        # Caller still gets a usable 7-slot tree; acceptance will likely be 0
        # or 1, but verify forward stays correct and we make progress.

        # Fill child slots: PLD-derived tokens first, then pad with last_token.
        for i, child_slot in enumerate(child_slots):
            if i < len(candidates):
                slot_tokens[child_slot] = candidates[i]
            else:
                slot_tokens[child_slot] = last_token

    # Sanity: every slot should now be filled.
    assert all(t is not None for t in slot_tokens)
    return DraftTree(
        token_ids=[t for t in slot_tokens],  # type: ignore[misc]
        parent_indices=list(_PARENT_INDICES),
    )
