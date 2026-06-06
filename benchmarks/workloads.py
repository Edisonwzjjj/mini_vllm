"""Prompt workload generators for mini-vllm benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Workload:
    name: str
    description: str
    prompts: list[str]


def _repeat_to_count(items: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    out = []
    while len(out) < count:
        out.extend(items)
    return out[:count]


def synthetic_repetition_prompts(count: int) -> list[str]:
    """PLD-friendly prompts with literal repeated continuations."""
    base = [
        (
            "List the days of the week: Monday, Tuesday, Wednesday, Thursday, "
            "Friday, Saturday, Sunday. List the days of the week again: "
            "Monday, Tuesday,"
        ),
        (
            "Shopping list:\n- apple\n- banana\n- cherry\n- date\n- elderberry\n"
            "Shopping list:\n- apple\n- banana\n- cherry\n-"
        ),
        (
            "Q: What is 2+2? A: 4.\nQ: What is 3+3? A: 6.\n"
            "Q: What is 4+4? A: 8.\nQ: What is 5+5? A:"
        ),
        (
            "def add(a, b): return a + b\n"
            "def sub(a, b): return a - b\n"
            "def mul(a, b): return a * b\n"
            "def div(a, b): return"
        ),
    ]
    return _repeat_to_count(base, count)


def agent_tool_prompts(count: int) -> list[str]:
    """Agent-like prompts: shared system prompt, tool schema, varied user tasks."""
    system = (
        "You are an autonomous coding assistant. Follow the policy exactly. "
        "When a tool call is needed, respond with compact JSON only. "
        "Available tools are inspect_file, run_tests, edit_file, and summarize. "
        "Tool schema: {\"tool\": string, \"args\": object, \"rationale\": string}.\n"
    )
    history = (
        "Conversation history:\n"
        "User: Please inspect the scheduler and find why decode stalls.\n"
        "Assistant: {\"tool\":\"inspect_file\",\"args\":{\"path\":\"mini_vllm/scheduler.py\"},"
        "\"rationale\":\"Need scheduler state transitions.\"}\n"
        "User: Now check whether prefix cache changes the scheduling order.\n"
        "Assistant: {\"tool\":\"inspect_file\",\"args\":{\"path\":\"mini_vllm/block_manager.py\"},"
        "\"rationale\":\"Need prefix cache hit accounting.\"}\n"
    )
    tasks = [
        "User: Add a benchmark for TTFT and output the next tool call.",
        "User: Investigate a failing FP8 KV test and output the next tool call.",
        "User: Compare chain speculative decoding with tree speculative decoding.",
        "User: Find the minimal file to edit for cache-aware scheduling metrics.",
        "User: Explain why long prefill can starve decode and choose a tool.",
    ]
    return _repeat_to_count([system + history + task for task in tasks], count)


def code_agent_prompts(count: int) -> list[str]:
    """Code-heavy prompts with repeated structures and local corpus flavor."""
    prelude = (
        "Repository excerpt:\n"
        "class Scheduler:\n"
        "    def schedule(self):\n"
        "        if self.waiting and self.can_prefill():\n"
        "            return prefill_batch\n"
        "        return decode_batch\n\n"
        "class BlockManager:\n"
        "    def allocate_with_prefix(self, token_ids, block_table):\n"
        "        matched_nodes, cached_tokens = self.radix_tree.match_prefix(token_ids)\n"
        "        return block_table, cached_tokens\n\n"
    )
    tasks = [
        "Write the next pytest that verifies waiting requests do not livelock.",
        "Write the next benchmark case for repeated system prompts.",
        "Write the next metric collection helper for prefix cache hit rate.",
        "Write the next doc paragraph explaining paged KV slot mapping.",
    ]
    return _repeat_to_count([prelude + task for task in tasks], count)


def long_context_prompts(count: int) -> list[str]:
    """Long-prefill, short-decode prompts for KV pressure and TTFT."""
    paragraph = (
        "Inference serving needs to balance prefill throughput and decode latency. "
        "The request contains a repeated system prompt, tool schemas, retrieved "
        "documents, and prior turns. Prefix caching can reuse stable tokens, but "
        "new user content still consumes KV blocks. "
    )
    base = []
    for i in range(4):
        base.append((paragraph * (16 + i * 4)) + f"\nQuestion {i}: summarize the routing decision in one sentence.")
    return _repeat_to_count(base, count)


_BUILDERS: dict[str, Callable[[int], list[str]]] = {
    "synthetic": synthetic_repetition_prompts,
    "agent": agent_tool_prompts,
    "code": code_agent_prompts,
    "long-context": long_context_prompts,
}


def build_workload(name: str, count: int) -> Workload:
    if name not in _BUILDERS:
        valid = ", ".join(sorted(_BUILDERS))
        raise ValueError(f"Unknown workload {name!r}; expected one of: {valid}")
    descriptions = {
        "synthetic": "Literal repetition; useful for PLD-friendly speculative decoding.",
        "agent": "Shared system prompt, tool schema, and multi-turn agent-style JSON tasks.",
        "code": "Code/repository-shaped prompts with repeated implementation patterns.",
        "long-context": "Long prefill and short decode; useful for KV pressure and TTFT.",
    }
    return Workload(
        name=name,
        description=descriptions[name],
        prompts=_BUILDERS[name](count),
    )


def workload_names() -> list[str]:
    return sorted(_BUILDERS)
