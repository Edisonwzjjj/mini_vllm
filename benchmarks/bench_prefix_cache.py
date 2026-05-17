"""Benchmark: multi-turn conversation prefix cache hit rate.

Compares radix tree vs simulated hash chain behavior:
  - Hash chain: only consecutive prefix hits (linear chain, no branching)
  - Radix tree: branching prefix sharing (system prompt shared across conversations)

No GPU needed — pure BlockManager logic.
"""

from mini_vllm.block_manager import BlockManager, RadixTree


def simulate_hash_chain_hit_rate(
    block_size: int, conversations: list[list[list[int]]]
) -> dict:
    """Simulate hash chain prefix cache behavior.

    Hash chain: hash(parent_hash || block_tokens) → block_id
    Must be consecutive hits from the start. After a branch, different
    suffixes cannot share the system prefix blocks.

    In the hash chain model, if seq1 = [system + user1] and seq2 = [system + user2],
    and they are in the same batch (both still in-use), seq2 CAN share system blocks
    because the hash chain for both includes the same system tokens.

    But if seq1 has already finished and been deallocated, the system blocks
    are in the cached pool. Seq2 can still hit them via hash lookup.

    The key limitation: hash chain can't share a prefix that BRANCHED.
    It can only share prefixes that are IDENTICAL up to the branching point,
    AND the branching point must be at a block boundary.

    Actually, hash chain CAN share the system prefix — the limitation is:
    it cannot reuse blocks AFTER the branch point if different sequences
    diverge at a non-block-aligned boundary. But at block granularity,
    both approaches are equivalent for same-prefix scenarios.

    The real difference: radix tree naturally supports partial prefix sharing
    when blocks from previous conversations are still cached.
    """
    num_blocks = 1000
    bm = BlockManager(block_size=block_size, num_blocks=num_blocks)

    # Track hash chain state
    hash_to_block_id = {}
    block_to_hash = {}
    ref_counts = {}
    cached_blocks = []

    total_requested = 0
    total_hit = 0

    for conversation in conversations:
        # Each conversation: [system, user1, user2, ...]
        # All turns share the same system prefix
        for turn_tokens in conversation:
            # Compute hash chain for this turn
            full_blocks = len(turn_tokens) // block_size
            total_requested += full_blocks

            # Check consecutive prefix hits (hash chain style)
            parent_hash = b""
            hit_count = 0
            for i in range(full_blocks):
                start = i * block_size
                block_tokens = tuple(turn_tokens[start:start + block_size])
                import hashlib
                data = parent_hash + str(block_tokens).encode()
                h = hashlib.md5(data).hexdigest()
                if h in hash_to_block_id:
                    bid = hash_to_block_id[h]
                    if bid in ref_counts or bid in cached_blocks:
                        hit_count += 1
                        parent_hash = h.encode()
                    else:
                        break
                else:
                    break

            total_hit += hit_count

            # Simulate allocation: reuse hit blocks, allocate new for rest
            allocated = []
            # Reuse hits
            parent_hash = b""
            for i in range(hit_count):
                start = i * block_size
                block_tokens = tuple(turn_tokens[start:start + block_size])
                data = parent_hash + str(block_tokens).encode()
                h = hashlib.md5(data).hexdigest()
                bid = hash_to_block_id[h]
                if bid in cached_blocks:
                    cached_blocks.remove(bid)
                    ref_counts[bid] = 1
                else:
                    ref_counts[bid] = ref_counts.get(bid, 0) + 1
                allocated.append(bid)
                parent_hash = h.encode()

            # Allocate new blocks for misses + partial
            num_total = (len(turn_tokens) + block_size - 1) // block_size
            for i in range(hit_count, num_total):
                bid = len(allocated)  # simple allocation
                allocated.append(bid)
                ref_counts[bid] = 1

            # Register hashes for full blocks (after forward pass)
            parent_hash = b""
            for i in range(full_blocks):
                start = i * block_size
                block_tokens = tuple(turn_tokens[start:start + block_size])
                data = parent_hash + str(block_tokens).encode()
                h = hashlib.md5(data).hexdigest()
                bid = allocated[i]
                hash_to_block_id[h] = bid
                block_to_hash[bid] = h
                parent_hash = h.encode()

            # Deallocate all blocks for this turn
            for bid in allocated:
                if bid in ref_counts:
                    ref_counts[bid] -= 1
                    if ref_counts[bid] == 0:
                        del ref_counts[bid]
                        cached_blocks.append(bid)

    return {
        "total_requested": total_requested,
        "total_hit": total_hit,
        "hit_rate": total_hit / total_requested if total_requested > 0 else 0.0,
    }


def radix_tree_hit_rate(
    block_size: int, conversations: list[list[list[int]]]
) -> dict:
    """Measure radix tree prefix cache hit rate for the same scenario."""
    num_blocks = 1000
    bm = BlockManager(block_size=block_size, num_blocks=num_blocks)

    for conversation in conversations:
        for turn_tokens in conversation:
            bt, _ = bm.allocate_with_prefix(turn_tokens, [])
            bm.insert_blocks(turn_tokens, bt)
            bm.deallocate(bt)

    return {
        "total_requested": bm.total_blocks_requested,
        "total_hit": bm.total_blocks_hit,
        "hit_rate": bm.cache_hit_rate(),
    }


def make_conversations(
    num_users: int,
    num_turns: int,
    system_len: int,  # in tokens
    user_len: int,    # in tokens per turn
    block_size: int,
) -> list[list[list[int]]]:
    """Generate multi-turn conversations sharing a system prompt.

    Returns list of conversations, each conversation is a list of turns,
    each turn is a list of token_ids.

    System prompt is the same across all users. Each user's turns are different.
    Turn tokens = system + unique_user_tokens for turn 1,
                  system + previous_user_tokens + new_user_tokens for later turns.
    """
    # Shared system prompt
    system_tokens = list(range(100, 100 + system_len))

    conversations = []
    for user_id in range(num_users):
        turns = []
        history = []
        for turn_id in range(num_turns):
            # Each turn: system + history + new user message
            user_tokens = list(
                range(10000 + user_id * 1000 + turn_id * 100,
                      10000 + user_id * 1000 + turn_id * 100 + user_len)
            )
            history.extend(user_tokens)
            turn_tokens = system_tokens + list(history)
            turns.append(turn_tokens)
        conversations.append(turns)
    return conversations


def run_benchmark():
    """Run the benchmark and print results."""
    block_size = 16
    scenarios = [
        {"num_users": 3, "num_turns": 1, "system_len": 64, "user_len": 32,
         "desc": "3 users, 1 turn each, shared system (64 tokens)"},
        {"num_users": 5, "num_turns": 1, "system_len": 128, "user_len": 32,
         "desc": "5 users, 1 turn each, shared system (128 tokens)"},
        {"num_users": 3, "num_turns": 3, "system_len": 64, "user_len": 16,
         "desc": "3 users, 3 turns each, shared system (64 tokens)"},
        {"num_users": 10, "num_turns": 1, "system_len": 128, "user_len": 16,
         "desc": "10 users, 1 turn, shared system (128 tokens)"},
    ]

    print("=" * 70)
    print("Multi-turn Conversation Prefix Cache Benchmark")
    print("Radix Tree vs Hash Chain")
    print("=" * 70)

    for s in scenarios:
        convs = make_conversations(
            s["num_users"], s["num_turns"], s["system_len"],
            s["user_len"], block_size
        )

        radix = radix_tree_hit_rate(block_size, convs)
        hash_chain = simulate_hash_chain_hit_rate(block_size, convs)

        print(f"\n  {s['desc']}")
        print(f"  {'':>20} {'Radix Tree':>12} {'Hash Chain':>12}")
        print(f"  {'':>20} {'-'*12:>12} {'-'*12:>12}")
        print(f"  {'Blocks requested':>20} {radix['total_requested']:>12} {hash_chain['total_requested']:>12}")
        print(f"  {'Blocks hit':>20} {radix['total_hit']:>12} {hash_chain['total_hit']:>12}")
        print(f"  {'Hit rate':>20} {radix['hit_rate']:>11.1%} {hash_chain['hit_rate']:>11.1%}")

    print("\n" + "=" * 70)

    # Key insight: for multi-turn conversations, radix tree and hash chain
    # should have the same hit rate in this simple scenario because
    # each turn includes the full history (system + previous turns).
    # The difference shows up when:
    # 1. Same prefix is reused across DIFFERENT conversations (both work)
    # 2. A prefix is partially shared (radix tree can share, hash chain can't)
    # 3. Block boundaries cause hash chain to miss (radix tree is unaffected)

    # Scenario: branching prefix — same system, diverging first block of user
    print("\nBranching scenario (same system, diverging user in same batch):")
    system = list(range(100, 164))  # 64 tokens = 4 blocks
    # User1 shares first 2 blocks with user2, then diverges
    user_shared = list(range(200, 232))  # 32 tokens = 2 blocks
    user1_suffix = list(range(300, 316))  # 16 tokens = 1 block
    user2_suffix = list(range(400, 416))  # 16 tokens = 1 block

    conv_branch = [
        [system + user_shared + user1_suffix],  # user 1: 7 blocks
        [system + user_shared + user2_suffix],  # user 2: shares 6 blocks
    ]

    radix = radix_tree_hit_rate(block_size, conv_branch)
    hash_chain = simulate_hash_chain_hit_rate(block_size, conv_branch)

    print(f"  {'':>20} {'Radix Tree':>12} {'Hash Chain':>12}")
    print(f"  {'Blocks hit':>20} {radix['total_hit']:>12} {hash_chain['total_hit']:>12}")
    print(f"  {'Hit rate':>20} {radix['hit_rate']:>11.1%} {hash_chain['hit_rate']:>11.1%}")

    print("\nNote: In this scenario, both radix tree and hash chain achieve")
    print("the same hit rate because the shared prefix is identical and")
    print("consecutive. The radix tree advantage appears in more complex")
    print("scenarios: EAGLE speculative decoding (tree-structured draft),")
    print("RL rollout with varying responses, or partial block sharing.")


if __name__ == "__main__":
    run_benchmark()
