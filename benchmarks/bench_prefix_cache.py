"""Benchmark: prefix cache throughput — shared system prompt + multiple queries.

Measures whether prefix cache provides speedup vs no-cache when prompts
share a common prefix (the typical chat use case).

Config: 200-token system prompt + 10 different user queries, max_tokens=100
"""

import time
from transformers import AutoTokenizer
from mini_vllm import LLM, SamplingParams

MODEL_PATH = "Qwen/Qwen3-0.6B"
SYSTEM_PROMPT_REPEATS = 22  # ~200 tokens with "You are a helpful AI assistant. "
MAX_TOKENS = 100
NUM_QUERIES = 10

QUERIES = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about the ocean.",
    "What is 2+2?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "Explain gravity to a 5-year-old.",
    "What is the largest planet in our solar system?",
    "How does photosynthesis work?",
    "What is the meaning of life?",
]


def bench_prefix_cache(tokenizer):
    """Run same-prefix prompts twice: first populates cache, second benefits from it."""
    system = "You are a helpful AI assistant. " * SYSTEM_PROMPT_REPEATS
    prompts = [system + q for q in QUERIES]
    prompt_lens = [len(tokenizer.encode(p)) for p in prompts]
    system_len = len(tokenizer.encode(system))
    print(f"System prompt: {system_len} tokens")
    print(f"Total prompt lengths: min={min(prompt_lens)}, max={max(prompt_lens)}, "
          f"avg={sum(prompt_lens)/len(prompt_lens):.0f}")

    llm = LLM(model_path=MODEL_PATH, block_size=16, max_num_seqs=16,
              max_num_batched_tokens=4096, gpu_memory_utilization=0.5)
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    # --- Batch 1: no cache (cold start) ---
    print("\n--- Batch 1: No prefix cache (cold) ---")
    t0 = time.time()
    outputs_1 = llm.generate(prompts, sp)
    time_cold = time.time() - t0
    tokens_1 = sum(len(o["token_ids"]) for o in outputs_1)
    tps_cold = tokens_1 / time_cold
    print(f"Tokens: {tokens_1}, Time: {time_cold:.2f}s, Throughput: {tps_cold:.1f} tokens/s")

    # Check cache state
    bm = llm.engine.model_runner.block_manager
    print(f"Prefix cache entries: {len(bm.hash_to_block_id)}")

    # --- Batch 2: with prefix cache (warm) ---
    print("\n--- Batch 2: With prefix cache (warm) ---")
    t0 = time.time()
    outputs_2 = llm.generate(prompts, sp)
    time_warm = time.time() - t0
    tokens_2 = sum(len(o["token_ids"]) for o in outputs_2)
    tps_warm = tokens_2 / time_warm
    print(f"Tokens: {tokens_2}, Time: {time_warm:.2f}s, Throughput: {tps_warm:.1f} tokens/s")

    # --- Verify correctness ---
    print("\n--- Correctness check ---")
    all_match = True
    for i in range(len(prompts)):
        if outputs_1[i]["token_ids"] != outputs_2[i]["token_ids"]:
            print(f"MISMATCH at prompt {i}!")
            all_match = False
    if all_match:
        print("All outputs match between cold and warm runs.")

    # --- Summary ---
    speedup = tps_warm / tps_cold
    saved_pct = (1 - time_warm / time_cold) * 100
    print(f"\n{'='*55}")
    print(f"{'Method':<30} {'tokens/s':>10} {'Speedup':>8}")
    print(f"{'-'*55}")
    print(f"{'M3 no cache (cold)':<30} {tps_cold:>10.1f} {'1.0x':>8}")
    print(f"{'M3 with cache (warm)':<30} {tps_warm:>10.1f} {f'{speedup:.1f}x':>8}")
    print(f"{'='*55}")
    print(f"Time saved: {saved_pct:.1f}%")
    if speedup >= 1.5:
        print(f"PASS: Prefix cache speedup {speedup:.1f}x >= 1.5x target")
    else:
        print(f"BELOW TARGET: Prefix cache speedup {speedup:.1f}x < 1.5x target")

    return speedup


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    bench_prefix_cache(tokenizer)
