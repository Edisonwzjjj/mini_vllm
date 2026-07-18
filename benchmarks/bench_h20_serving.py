"""H20-oriented serving benchmark for mini-vllm.

The script is intentionally hardware-agnostic: it runs on any CUDA device, but
the output is shaped for H20 experiments. It measures one method/workload at a
time, emits JSONL for later plotting, and prints a markdown summary table for
vault/README notes.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

from metrics import (
    BenchmarkResult,
    aggregate_ttft_tpot,
    block_manager_stats,
    count_prompt_tokens,
    environment_info,
    kv_cache_size_gb,
    markdown_table,
    now_s,
    peak_memory_gb,
    reset_peak_memory,
    spec_metrics,
    write_jsonl,
)
from workloads import build_workload, workload_names


MODEL_PATH = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
DEFAULT_METHODS = ["greedy", "chain-pld", "tree-pld"]


@dataclass(frozen=True)
class MethodConfig:
    name: str
    enable_eagle: bool = False
    eagle_mode: str = "chain"
    eagle_draft_len: int = 4
    eagle_topk: int = 2
    eagle_spec_steps: int = 3
    eagle_use_pld: bool = False


METHODS = {
    "greedy": MethodConfig(name="greedy"),
    "chain-pld": MethodConfig(
        name="chain-pld",
        enable_eagle=True,
        eagle_mode="chain",
        eagle_draft_len=4,
        eagle_use_pld=True,
    ),
    "tree-pld": MethodConfig(
        name="tree-pld",
        enable_eagle=True,
        eagle_mode="tree",
        eagle_topk=2,
        eagle_spec_steps=3,
        eagle_use_pld=True,
    ),
    "tree-dummy": MethodConfig(
        name="tree-dummy",
        enable_eagle=True,
        eagle_mode="tree",
        eagle_topk=2,
        eagle_spec_steps=3,
        eagle_use_pld=False,
    ),
}


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [method for method in methods if method not in METHODS]
    if unknown:
        valid = ", ".join(sorted(METHODS))
        raise argparse.ArgumentTypeError(f"Unknown methods {unknown}; expected subset of: {valid}")
    return methods


def make_llm(args: argparse.Namespace, method: MethodConfig) -> LLM:
    from mini_vllm import LLM

    return LLM(
        model_path=args.model_path,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        deterministic=args.deterministic,
        kv_cache_dtype=args.kv_dtype,
        kv_scale=args.kv_scale,
        kv_scale_calib_tokens=args.kv_scale_calib_tokens,
        enable_eagle=method.enable_eagle,
        eagle_mode=method.eagle_mode,
        eagle_draft_len=method.eagle_draft_len,
        eagle_topk=method.eagle_topk,
        eagle_spec_steps=method.eagle_spec_steps,
        eagle_use_pld=method.eagle_use_pld,
        eagle_pld_max_ngram=args.pld_max_ngram,
        eagle_pld_min_ngram=args.pld_min_ngram,
    )


def run_warmup(args: argparse.Namespace, llm) -> None:
    """Run one short, discarded generation to trigger lazy one-time costs
    (first CUDA Graph capture, cuDNN/cuBLAS algo autotune, etc.) before the
    timed run. Uses a prompt that never overlaps with the real workload, so
    it doesn't inflate the real run's prefix-cache hit rate.

    Per the execution handbook: without this, the first CUDA Graph capture
    (which is much slower than a replay) could land inside the timed window.
    """
    if args.skip_warmup:
        return
    from mini_vllm import SamplingParams

    warmup_sp = SamplingParams(temperature=0.0, max_tokens=4)
    llm.generate(["mini-vllm benchmark warmup pass, ignore this output."], warmup_sp)


def run_full(
    args: argparse.Namespace,
    method: MethodConfig,
    workload_name: str,
    prompts: list[str],
    sp: SamplingParams,
) -> BenchmarkResult:
    import torch

    llm = make_llm(args, method)
    tokenizer = llm.engine.model_runner.tokenizer
    prompt_tokens = count_prompt_tokens(tokenizer, prompts)

    run_warmup(args, llm)

    reset_peak_memory()
    t0 = now_s()
    outputs = llm.generate(prompts, sp)
    elapsed = now_s() - t0

    output_tokens = sum(len(item["token_ids"]) for item in outputs)
    mr = llm.engine.model_runner
    bm_stats = block_manager_stats(mr)
    spec = spec_metrics(llm.engine)
    tokens_per_s = output_tokens / elapsed if elapsed > 0 else 0.0
    kv_dtype = str(mr.kv_cache.dtype).replace("torch.", "")

    # True per-request TTFT/TPOT: each output dict already carries its own
    # ttft_s/tpot_s, recorded from real per-request timestamps inside
    # LLMEngine.generate() (see engine.py) — not from a separate stream pass
    # against a fresh engine, and not derived from batch-level elapsed time.
    latency = aggregate_ttft_tpot(outputs)

    result = BenchmarkResult(
        method=method.name,
        workload=workload_name,
        num_requests=len(prompts),
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        max_tokens=sp.max_tokens,
        elapsed_s=elapsed,
        tokens_per_s=tokens_per_s,
        ttft_s_mean=latency["ttft_s_mean"],
        ttft_s_p50=latency["ttft_s_p50"],
        ttft_s_p90=latency["ttft_s_p90"],
        tpot_s_mean=latency["tpot_s_mean"],
        tpot_s_p50=latency["tpot_s_p50"],
        tpot_s_p90=latency["tpot_s_p90"],
        peak_memory_gb=peak_memory_gb(),
        kv_cache_gb=kv_cache_size_gb(mr),
        kv_cache_dtype=kv_dtype,
        num_blocks=bm_stats["num_blocks"],
        num_free_blocks=bm_stats["num_free_blocks"],
        prefix_hit_rate=bm_stats["prefix_hit_rate"],
        prefix_blocks_hit=bm_stats["prefix_blocks_hit"],
        prefix_blocks_requested=bm_stats["prefix_blocks_requested"],
        cuda_graph_captures=mr.decode_graph_capture_count,
        cuda_graph_replays=mr.decode_graph_replay_count,
        spec_steps=spec["spec_steps"],
        fallback_steps=spec["fallback_steps"],
        draft_tokens_proposed=spec["draft_tokens_proposed"],
        draft_tokens_accepted=spec["draft_tokens_accepted"],
        bonus_tokens=spec["bonus_tokens"],
        acceptance_rate=spec["acceptance_rate"],
        avg_accept_per_spec_step=spec["avg_accept_per_spec_step"],
        bonus_rate=spec["bonus_rate"],
    )

    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_method(
    args: argparse.Namespace,
    method_name: str,
    workload_name: str,
    prompts: list[str],
) -> BenchmarkResult:
    from mini_vllm import SamplingParams

    method = METHODS[method_name]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    print(f"\n--- {method.name} / workload={workload_name} / requests={len(prompts)} ---")
    result = run_full(args, method, workload_name, prompts, sp)
    print(
        f"tokens={result.output_tokens} elapsed={result.elapsed_s:.2f}s "
        f"tok/s={result.tokens_per_s:.2f} peak={result.peak_memory_gb}GB "
        f"graphs={result.cuda_graph_captures}/{result.cuda_graph_replays}"
    )
    if result.ttft_s_mean is not None:
        print(
            f"latency: TTFT mean={result.ttft_s_mean:.3f}s p50={result.ttft_s_p50:.3f}s "
            f"p90={result.ttft_s_p90:.3f}s | TPOT mean={result.tpot_s_mean:.4f}s "
            f"p50={result.tpot_s_p50:.4f}s p90={result.tpot_s_p90:.4f}s"
        )
    if result.draft_tokens_proposed:
        print(
            f"spec: steps={result.spec_steps} accepted={result.draft_tokens_accepted}/"
            f"{result.draft_tokens_proposed} fallback={result.fallback_steps}"
        )
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--workload", choices=workload_names(), default="synthetic")
    parser.add_argument("--num-requests", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--methods", type=parse_methods, default=DEFAULT_METHODS,
                        help="Comma-separated subset of greedy,chain-pld,tree-pld,tree-dummy")
    parser.add_argument("--json-out", default=None, help="Append results to this JSONL file.")
    parser.add_argument("--markdown-out", default=None, help="Write markdown summary table.")
    parser.add_argument(
        "--skip-ttft", action="store_true",
        help="Deprecated no-op, kept for backward compatibility with existing scripts. "
             "TTFT/TPOT are now always measured from real per-request timestamps inside "
             "the single generate() call (no separate stream pass needed).",
    )
    parser.add_argument(
        "--skip-warmup", action="store_true",
        help="Skip the discarded warmup generation. Not recommended for timed runs: "
             "without it, one-time costs (first CUDA Graph capture, cuBLAS/cuDNN algo "
             "autotune) can land inside the timed window and inflate elapsed_s/TTFT.",
    )

    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic fp32 attention path. Slower; disables FP8 KV.")
    parser.add_argument("--kv-dtype", choices=["auto", "fp32", "bf16", "fp8_e4m3"], default="bf16")
    parser.add_argument("--kv-scale", type=float, default=None)
    parser.add_argument("--kv-scale-calib-tokens", type=int, default=4096)
    parser.add_argument("--pld-max-ngram", type=int, default=3)
    parser.add_argument("--pld-min-ngram", type=int, default=2)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.kv_dtype == "fp8_e4m3" and args.deterministic:
        parser.error("fp8_e4m3 KV requires non-deterministic mode; remove --deterministic.")

    workload = build_workload(args.workload, args.num_requests)
    print("Environment:")
    for key, value in environment_info().items():
        print(f"  {key}: {value}")
    print(f"Workload: {workload.name} — {workload.description}")
    print(f"Methods: {', '.join(args.methods)}")

    rows = [
        run_method(args, method_name, workload.name, workload.prompts)
        for method_name in args.methods
    ]

    table = markdown_table(rows)
    print("\nSummary:")
    print(table)

    if args.json_out:
        write_jsonl(args.json_out, rows)
        print(f"\nWrote JSONL: {args.json_out}")
    if args.markdown_out:
        path = Path(args.markdown_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(table + "\n", encoding="utf-8")
        print(f"Wrote markdown: {path}")


if __name__ == "__main__":
    main()
