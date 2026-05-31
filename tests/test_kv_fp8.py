"""CUDA-only tests for FP8 KV cache storage."""

import pytest
import torch

from mini_vllm import LLM, SamplingParams
from mini_vllm.config import EngineConfig
from mini_vllm.kv_quant import FP8_DTYPE, dequantize_from_fp8, quantize_to_fp8


MODEL_PATH = "Qwen/Qwen3-0.6B"
CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 KV cache requires CUDA")
TEST_KV_SCALE = 0.01


@CUDA_ONLY
def test_fp8_quant_roundtrip_bf16():
    x = torch.linspace(-2.0, 2.0, steps=2048, device="cuda", dtype=torch.bfloat16)
    x_fp8 = quantize_to_fp8(x, TEST_KV_SCALE)
    y = dequantize_from_fp8(x_fp8, TEST_KV_SCALE, torch.bfloat16)

    assert x_fp8.dtype == FP8_DTYPE
    mean_rel_err = ((x.float() - y.float()).abs() / x.float().abs().clamp_min(1e-3)).mean()
    assert mean_rel_err.item() < 0.05


@CUDA_ONLY
@pytest.mark.timeout(300)
def test_fp8_requires_nondeterministic():
    with pytest.raises(ValueError, match="deterministic=False"):
        LLM(
            model_path=MODEL_PATH,
            deterministic=True,
            kv_cache_dtype="fp8_e4m3",
            kv_scale=TEST_KV_SCALE,
        )


@CUDA_ONLY
@pytest.mark.timeout(300)
def test_fp8_kv_engine_runs():
    llm = LLM(
        model_path=MODEL_PATH,
        block_size=16,
        max_num_seqs=2,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.3,
        deterministic=False,
        kv_cache_dtype="fp8_e4m3",
        kv_scale=TEST_KV_SCALE,
    )
    out = llm.generate(["Hello, world.", "What is attention?"], SamplingParams(temperature=0.0, max_tokens=3))

    assert llm.engine.model_runner.kv_cache.dtype == FP8_DTYPE
    assert len(out) == 2
    assert all(len(item["token_ids"]) == 3 for item in out)


@CUDA_ONLY
@pytest.mark.timeout(300)
def test_fp8_kv_close_to_bf16_first_token():
    prompt = ["The capital of France is"]
    sp = SamplingParams(temperature=0.0, max_tokens=1)
    bf16_llm = LLM(
        model_path=MODEL_PATH,
        block_size=16,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.3,
        deterministic=False,
        kv_cache_dtype="bf16",
    )
    bf16_ids = bf16_llm.generate(prompt, sp)[0]["token_ids"]
    del bf16_llm
    torch.cuda.empty_cache()

    fp8_llm = LLM(
        model_path=MODEL_PATH,
        block_size=16,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.3,
        deterministic=False,
        kv_cache_dtype="fp8_e4m3",
        kv_scale=TEST_KV_SCALE,
    )
    fp8_ids = fp8_llm.generate(prompt, sp)[0]["token_ids"]

    assert fp8_ids == bf16_ids


@CUDA_ONLY
@pytest.mark.timeout(300)
def test_fp8_kv_with_cuda_graph():
    llm = LLM(
        model_path=MODEL_PATH,
        block_size=16,
        max_num_seqs=1,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.3,
        deterministic=False,
        kv_cache_dtype="fp8_e4m3",
        kv_scale=TEST_KV_SCALE,
    )
    out = llm.generate(["Write a haiku about rain"], SamplingParams(temperature=0.0, max_tokens=4))
    mr = llm.engine.model_runner

    assert len(out[0]["token_ids"]) == 4
    assert mr.decode_graph_capture_count >= 1


def test_engine_config_rejects_bad_kv_dtype():
    with pytest.raises(ValueError, match="kv_cache_dtype"):
        EngineConfig(model_path=MODEL_PATH, kv_cache_dtype="fp8")
