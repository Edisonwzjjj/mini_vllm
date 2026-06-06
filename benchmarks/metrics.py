"""Shared metric helpers for benchmark scripts."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    method: str
    workload: str
    num_requests: int
    prompt_tokens: int
    output_tokens: int
    max_tokens: int
    elapsed_s: float
    ttft_s: float | None
    tpot_s: float | None
    tokens_per_s: float
    peak_memory_gb: float | None
    kv_cache_gb: float | None
    kv_cache_dtype: str
    num_blocks: int | None
    num_free_blocks: int | None
    prefix_hit_rate: float | None
    prefix_blocks_hit: int | None
    prefix_blocks_requested: int | None
    cuda_graph_captures: int
    cuda_graph_replays: int
    spec_steps: int
    fallback_steps: int
    draft_tokens_proposed: int
    draft_tokens_accepted: int
    bonus_tokens: int
    acceptance_rate: float | None
    avg_accept_per_spec_step: float | None
    bonus_rate: float | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def now_s() -> float:
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def reset_peak_memory() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024**3


def cuda_device_label() -> str:
    import torch

    if not torch.cuda.is_available():
        return "cuda unavailable"
    props = torch.cuda.get_device_properties(0)
    return f"{props.name} ({props.total_memory / 1024**3:.1f} GiB)"


def environment_info() -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError:
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": "not installed",
            "cuda_available": False,
            "cuda_device": "cuda unavailable",
        }

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_device_label(),
    }


def count_prompt_tokens(tokenizer: Any, prompts: list[str]) -> int:
    return sum(len(tokenizer.encode(prompt)) for prompt in prompts)


def kv_cache_size_gb(model_runner: Any) -> float | None:
    kv_cache = getattr(model_runner, "kv_cache", None)
    if kv_cache is None:
        return None
    return kv_cache.nelement() * kv_cache.element_size() / 1024**3


def block_manager_stats(model_runner: Any) -> dict[str, Any]:
    bm = getattr(model_runner, "block_manager", None)
    if bm is None:
        return {
            "num_blocks": None,
            "num_free_blocks": None,
            "prefix_hit_rate": None,
            "prefix_blocks_hit": None,
            "prefix_blocks_requested": None,
        }
    requested = getattr(bm, "total_blocks_requested", None)
    hit = getattr(bm, "total_blocks_hit", None)
    hit_rate = bm.cache_hit_rate() if hasattr(bm, "cache_hit_rate") else None
    return {
        "num_blocks": getattr(bm, "num_blocks", None),
        "num_free_blocks": getattr(bm, "num_free_blocks", None),
        "prefix_hit_rate": hit_rate,
        "prefix_blocks_hit": hit,
        "prefix_blocks_requested": requested,
    }


def spec_metrics(engine: Any) -> dict[str, Any]:
    runner = getattr(engine, "spec_runner", None)
    if runner is None:
        return {
            "spec_steps": 0,
            "fallback_steps": 0,
            "draft_tokens_proposed": 0,
            "draft_tokens_accepted": 0,
            "bonus_tokens": 0,
            "acceptance_rate": None,
            "avg_accept_per_spec_step": None,
            "bonus_rate": None,
        }
    metrics = dict(getattr(runner, "metrics", {}))
    proposed = metrics.get("draft_tokens_proposed", 0)
    accepted = metrics.get("draft_tokens_accepted", 0)
    spec_steps = metrics.get("spec_steps", 0)
    bonus = metrics.get("bonus_tokens", 0)
    return {
        "spec_steps": spec_steps,
        "fallback_steps": metrics.get("fallback_steps", 0),
        "draft_tokens_proposed": proposed,
        "draft_tokens_accepted": accepted,
        "bonus_tokens": bonus,
        "acceptance_rate": accepted / proposed if proposed else None,
        "avg_accept_per_spec_step": accepted / spec_steps if spec_steps else None,
        "bonus_rate": bonus / spec_steps if spec_steps else None,
    }


def compute_tpot(elapsed_s: float, ttft_s: float | None, output_tokens: int) -> float | None:
    if ttft_s is None or output_tokens <= 1:
        return None
    return max(elapsed_s - ttft_s, 0.0) / (output_tokens - 1)


def write_jsonl(path: str | Path, rows: list[BenchmarkResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(row.to_json() + "\n")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[BenchmarkResult]) -> str:
    headers = [
        "method",
        "workload",
        "req",
        "out_tok",
        "tok/s",
        "TTFT(s)",
        "TPOT(s)",
        "peakGB",
        "kvGB",
        "hit%",
        "accept%",
        "fallback",
        "graphs",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        hit_pct = None if r.prefix_hit_rate is None else r.prefix_hit_rate * 100
        acc_pct = None if r.acceptance_rate is None else r.acceptance_rate * 100
        values = [
            r.method,
            r.workload,
            r.num_requests,
            r.output_tokens,
            f"{r.tokens_per_s:.2f}",
            _fmt(r.ttft_s),
            _fmt(r.tpot_s),
            _fmt(r.peak_memory_gb, 2),
            _fmt(r.kv_cache_gb, 2),
            _fmt(hit_pct, 1),
            _fmt(acc_pct, 1),
            r.fallback_steps,
            f"{r.cuda_graph_captures}/{r.cuda_graph_replays}",
        ]
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
    return "\n".join(lines)
