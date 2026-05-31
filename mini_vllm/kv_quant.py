"""FP8 KV cache quant/dequant helpers."""

import torch


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0


def quantize_to_fp8(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Convert a floating tensor to scaled FP8 E4M3 storage."""
    if scale <= 0:
        raise ValueError("kv_scale must be positive")
    return (x.float() / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


def dequantize_from_fp8(x_fp8: torch.Tensor, scale: float, out_dtype: torch.dtype) -> torch.Tensor:
    """Convert scaled FP8 E4M3 storage back to compute dtype."""
    if scale <= 0:
        raise ValueError("kv_scale must be positive")
    return x_fp8.to(out_dtype) * scale


def calibrate_scale(absmax: float, headroom: float = 0.95) -> float:
    """Build a symmetric per-tensor scale from an observed absolute max."""
    if absmax <= 0:
        raise ValueError("Cannot calibrate kv_scale from non-positive absmax")
    return float(absmax) / (FP8_MAX * headroom)
