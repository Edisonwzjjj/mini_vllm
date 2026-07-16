"""Global configuration for mini-vllm engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EngineConfig:
    model_path: str
    block_size: int = 16
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 2048
    gpu_memory_utilization: float = 0.5
    dtype: str = "auto"  # "auto" | "float16" | "float32"
    deterministic: bool = True
    kv_cache_dtype: str = "auto"  # "auto" | "fp32" | "bf16" | "fp8_e4m3"
    kv_scale: float | None = None
    kv_scale_calib_tokens: int = 4096
    enable_eagle: bool = False
    eagle_draft_len: int = 4
    eagle_verify_greedy_only: bool = True
    eagle_mode: str = "chain"
    eagle_topk: int = 2
    eagle_spec_steps: int = 3
    eagle_use_pld: bool = False
    eagle_pld_max_ngram: int = 3
    eagle_pld_min_ngram: int = 2

    def __post_init__(self) -> None:
        valid_kv_dtypes = {"auto", "fp32", "bf16", "fp8_e4m3"}
        if self.kv_cache_dtype not in valid_kv_dtypes:
            raise ValueError(f"kv_cache_dtype must be one of {sorted(valid_kv_dtypes)}, got {self.kv_cache_dtype!r}")
        if self.kv_scale is not None and self.kv_scale <= 0:
            raise ValueError("kv_scale must be positive")
        if self.kv_scale_calib_tokens <= 0:
            raise ValueError("kv_scale_calib_tokens must be positive")
