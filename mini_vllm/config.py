"""Global configuration for mini-vllm engine."""

from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    model_path: str
    block_size: int = 16
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 2048
    gpu_memory_utilization: float = 0.5
    dtype: str = "auto"  # "auto" | "float16" | "float32"
