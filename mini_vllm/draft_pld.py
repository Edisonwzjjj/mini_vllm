"""Prompt Lookup Decoding (PLD) draft generator.

Zero-cost speculative draft: search the existing token sequence for the most
recent occurrence of the last n-gram, and use the tokens that followed it as
the draft. Falls back from max_ngram down to min_ngram.

Reference: https://github.com/apoorvumang/prompt-lookup-decoding
"""


def prompt_lookup_draft(
    token_ids: list[int],
    draft_len: int,
    max_ngram: int = 3,
    min_ngram: int = 2,
) -> list[int]:
    """Look up the last n-gram in token_ids and return the next draft_len tokens.

    Args:
        token_ids: full sequence so far (prompt + output).
        draft_len: maximum number of draft tokens to return.
        max_ngram: largest n-gram length to try first.
        min_ngram: smallest n-gram length before giving up.

    Returns:
        A list of up to draft_len tokens. Empty list on miss or insufficient
        sequence length.
    """
    n = len(token_ids)
    # Try larger n-grams first; fall back to smaller ones on miss.
    for gram_size in range(max_ngram, min_ngram - 1, -1):
        # Need at least gram_size query tokens plus 1 earlier position to search.
        if n < gram_size + 1:
            continue

        query = token_ids[n - gram_size:]
        # Search [0 .. n - gram_size - 1] from right to left for the most
        # recent occurrence. A match at idx means
        # token_ids[idx : idx + gram_size] == query, and the continuation
        # starts at idx + gram_size.
        for idx in range(n - gram_size - 1, -1, -1):
            if token_ids[idx:idx + gram_size] == query:
                end = idx + gram_size
                return token_ids[end:end + draft_len]

    return []
