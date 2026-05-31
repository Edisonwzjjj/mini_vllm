"""Benchmark: HF serial vs mini-vllm M2 batched throughput."""

import argparse
import gc
import time
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from mini_vllm import LLM, SamplingParams

MODEL_PATH = "Qwen/Qwen3-0.6B"
NUM_REQUESTS = 32
MIN_PROMPT_LEN = 50
MAX_PROMPT_LEN = 500
MAX_TOKENS = 100


def generate_random_prompts(tokenizer, n, min_len, max_len):
    """Generate n random prompts of varying token lengths."""
    random.seed(42)
    prompts = []
    for _ in range(n):
        target_len = 250
        # Use repeating words to reach target length
        words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                 "and", "a", "in", "to", "of", "is", "it", "that", "was", "for"]
        tokens = []
        while len(tokens) < target_len:
            tokens.append(random.choice(words))
        text = " ".join(tokens)
        # Verify token count, trim if needed
        ids = tokenizer.encode(text)
        if len(ids) > target_len:
            text = tokenizer.decode(ids[:target_len])
        prompts.append(text)
    return prompts


def bench_hf_serial(prompts, tokenizer, model, max_tokens):
    """HF model.generate, one prompt at a time."""
    total_tokens = 0
    t0 = time.time()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        gen_len = out.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += gen_len
    elapsed = time.time() - t0
    return total_tokens, elapsed


def bench_mini_vllm(prompts, max_tokens, kv_dtype, kv_scale, kv_scale_calib_tokens,
                    max_num_seqs, gpu_memory_utilization):
    """mini-vllm batched prefill + per-sequence decode."""
    llm = LLM(model_path=MODEL_PATH, block_size=16, max_num_seqs=max_num_seqs,
              max_num_batched_tokens=4096, gpu_memory_utilization=gpu_memory_utilization,
              deterministic=False, kv_cache_dtype=kv_dtype,
              kv_scale=kv_scale, kv_scale_calib_tokens=kv_scale_calib_tokens)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    t0 = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - t0
    total_tokens = sum(len(o["token_ids"]) for o in outputs)
    mr = llm.engine.model_runner
    print(
        f"CUDA graphs: captures={mr.decode_graph_capture_count}, "
        f"replays={mr.decode_graph_replay_count}, "
        f"keys={list(mr.decode_graphs.keys())}"
    )
    return total_tokens, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-dtype", choices=["bf16", "fp8_e4m3"], default="bf16")
    parser.add_argument("--kv-scale", type=float, default=None)
    parser.add_argument("--kv-scale-calib-tokens", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    args = parser.parse_args()
    gpu_memory_utilization = (
        args.gpu_memory_utilization
        if args.gpu_memory_utilization is not None
        else (0.25 if args.kv_dtype == "fp8_e4m3" else 0.5)
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"Generating {NUM_REQUESTS} random prompts (token len {MIN_PROMPT_LEN}-{MAX_PROMPT_LEN})...")
    prompts = generate_random_prompts(tokenizer, NUM_REQUESTS, MIN_PROMPT_LEN, MAX_PROMPT_LEN)
    prompt_lens = [len(tokenizer.encode(p)) for p in prompts]
    print(f"Prompt lengths: min={min(prompt_lens)}, max={max(prompt_lens)}, "
          f"avg={sum(prompt_lens)/len(prompt_lens):.0f}")

    # --- HF serial ---
    print("\n--- HF model.generate (serial) ---")
    device = "cuda" if torch.cuda.is_available() else "mps"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=dtype
    ).to(device).eval()
    hf_tokens, hf_time = bench_hf_serial(prompts, tokenizer, model, MAX_TOKENS)
    hf_tps = hf_tokens / hf_time
    print(f"Total tokens: {hf_tokens}, Time: {hf_time:.2f}s, Throughput: {hf_tps:.1f} tokens/s")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        print(f"CUDA free memory before mini-vllm: {free_gb:.2f} GB")

    # --- mini-vllm M2 ---
    print(
        f"\n--- mini-vllm M2 (batched prefill + decode, kv={args.kv_dtype}, "
        f"max_num_seqs={args.max_num_seqs}, gpu_mem={gpu_memory_utilization}) ---"
    )
    mv_tokens, mv_time = bench_mini_vllm(
        prompts, MAX_TOKENS, args.kv_dtype, args.kv_scale, args.kv_scale_calib_tokens,
        args.max_num_seqs, gpu_memory_utilization
    )
    mv_tps = mv_tokens / mv_time
    print(f"Total tokens: {mv_tokens}, Time: {mv_time:.2f}s, Throughput: {mv_tps:.1f} tokens/s")

    # --- Summary ---
    speedup = mv_tps / hf_tps
    print(f"\n{'='*50}")
    print(f"{'Method':<30} {'tokens/s':>10} {'Speedup':>8}")
    print(f"{'-'*50}")
    print(f"{'HF generate (serial)':<30} {hf_tps:>10.1f} {'1.0x':>8}")
    print(f"{f'mini-vllm M2 ({args.kv_dtype})':<30} {mv_tps:>10.1f} {f'{speedup:.1f}x':>8}")
    print(f"{'='*50}")
    if speedup >= 4.0:
        print("PASS: M2 throughput >= 4x serial HF")
    else:
        print(f"BELOW TARGET: M2 throughput {speedup:.1f}x < 4x target")


if __name__ == "__main__":
    main()
