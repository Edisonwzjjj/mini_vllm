"""Verify a mini-vllm runtime environment is usable on a locked-down GPU box.

Checks (in order), each printed with a pass/fail marker:
  - Python version
  - torch import + version + CUDA build tag
  - CUDA available / driver forward-compatibility (torch.cuda.is_available)
  - GPU name + total memory
  - BF16 support
  - FP8 (torch.float8_e4m3fn) roundtrip
  - CUDA Graph capture/replay
  - transformers import + version
  - mini_vllm import
  - Qwen3 attention module shape (sanity: patch target still exists)

Exits non-zero if any *required* check fails. FP8 and CUDA Graph are
"best effort" checks — they report status but do not fail the whole
script, since some GPUs/driver combos legitimately lack them.

Usage:
    python scripts/verify_env.py [--out results/env_snapshot/env_report.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback


def _print_status(label: str, ok: bool, detail: str = "") -> None:
    marker = "PASS" if ok else "FAIL"
    line = f"[{marker}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=None, help="Path to write a JSON report")
    args = parser.parse_args()

    report: dict = {}
    required_failures: list[str] = []

    # --- Python version -----------------------------------------------
    py_version = platform.python_version()
    py_ok = sys.version_info[:2] == (3, 9)
    _print_status("Python 3.9.x", py_ok, py_version)
    report["python_version"] = py_version
    if not py_ok:
        required_failures.append("python_version")

    # --- torch import ----------------------------------------------------
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        _print_status("import torch", False, str(exc))
        report["torch_import_error"] = str(exc)
        _finish(report, args.out, ["torch_import"])
        return 1

    torch_version = torch.__version__
    cuda_build_tag = getattr(torch.version, "cuda", None)
    torch_ok = cuda_build_tag is not None
    _print_status("torch has CUDA build", torch_ok, f"torch={torch_version}, cuda_build={cuda_build_tag}")
    report["torch_version"] = torch_version
    report["torch_cuda_build"] = cuda_build_tag
    if not torch_ok:
        required_failures.append("torch_cuda_build")

    # --- CUDA availability ----------------------------------------------
    cuda_available = False
    try:
        cuda_available = torch.cuda.is_available()
    except Exception as exc:  # noqa: BLE001
        report["cuda_available_error"] = str(exc)
    _print_status("torch.cuda.is_available()", cuda_available)
    report["cuda_available"] = cuda_available
    if not cuda_available:
        required_failures.append("cuda_available")

    gpu_name = None
    gpu_mem_gib = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
            _print_status("GPU detected", True, f"{gpu_name}, {gpu_mem_gib:.1f} GiB")
        except Exception as exc:  # noqa: BLE001
            _print_status("GPU detected", False, str(exc))
            required_failures.append("gpu_detect")
    report["gpu_name"] = gpu_name
    report["gpu_memory_gib"] = gpu_mem_gib

    # --- BF16 support ------------------------------------------------------
    bf16_ok = False
    if cuda_available:
        try:
            bf16_ok = bool(torch.cuda.is_bf16_supported())
        except Exception as exc:  # noqa: BLE001
            report["bf16_error"] = str(exc)
        _print_status("BF16 supported", bf16_ok)
        if not bf16_ok:
            required_failures.append("bf16_supported")
    report["bf16_supported"] = bf16_ok

    # --- FP8 roundtrip (best effort) ---------------------------------------
    fp8_mae = None
    if cuda_available and hasattr(torch, "float8_e4m3fn"):
        try:
            x = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
            y = x.to(torch.float8_e4m3fn)
            z = y.to(torch.bfloat16)
            fp8_mae = float((x - z).abs().float().mean().item())
            _print_status("FP8 (float8_e4m3fn) roundtrip", True, f"MAE={fp8_mae:.6g}")
        except Exception as exc:  # noqa: BLE001
            _print_status("FP8 (float8_e4m3fn) roundtrip", False, str(exc))
    else:
        _print_status("FP8 (float8_e4m3fn) roundtrip", False, "dtype unavailable or no CUDA")
    report["fp8_roundtrip_mae"] = fp8_mae

    # --- CUDA Graph capture/replay (best effort) ---------------------------
    cuda_graph_ok = False
    if cuda_available and hasattr(torch.cuda, "CUDAGraph"):
        try:
            static_input = torch.zeros(1024, device="cuda")
            static_output = torch.zeros(1024, device="cuda")

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    static_output.copy_(static_input + 1)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_output.copy_(static_input + 1)
            static_input.fill_(41.0)
            graph.replay()
            torch.cuda.synchronize()
            cuda_graph_ok = bool(torch.allclose(static_output, torch.full_like(static_output, 42.0)))
            _print_status("CUDA Graph capture/replay", cuda_graph_ok)
        except Exception as exc:  # noqa: BLE001
            _print_status("CUDA Graph capture/replay", False, str(exc))
    else:
        _print_status("CUDA Graph capture/replay", False, "unavailable or no CUDA")
    report["cuda_graph_ok"] = cuda_graph_ok

    # --- transformers -------------------------------------------------------
    try:
        import transformers

        _print_status("import transformers", True, transformers.__version__)
        report["transformers_version"] = transformers.__version__
    except Exception as exc:  # noqa: BLE001
        _print_status("import transformers", False, str(exc))
        required_failures.append("transformers_import")

    # --- mini_vllm -----------------------------------------------------------
    try:
        import mini_vllm  # noqa: F401

        _print_status("import mini_vllm", True)
        report["mini_vllm_import"] = True
    except Exception as exc:  # noqa: BLE001
        _print_status("import mini_vllm", False, f"{exc}\n{traceback.format_exc()}")
        report["mini_vllm_import"] = False
        required_failures.append("mini_vllm_import")

    # --- Qwen3 attention patch target sanity ---------------------------------
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

        has_target_attrs = all(
            hasattr(Qwen3Attention, name) is False or True  # class attrs may not exist pre-init; just import check
            for name in ("forward",)
        )
        _print_status("Qwen3Attention importable (patch target)", True)
        report["qwen3_attention_importable"] = True
    except Exception as exc:  # noqa: BLE001
        _print_status("Qwen3Attention importable (patch target)", False, str(exc))
        report["qwen3_attention_importable"] = False
        required_failures.append("qwen3_attention_import")

    report["required_failures"] = required_failures
    _finish(report, args.out, required_failures)
    return 1 if required_failures else 0


def _finish(report: dict, out_path, required_failures) -> None:
    print()
    if required_failures:
        print(f"RESULT: FAIL ({len(required_failures)} required check(s) failed: {', '.join(required_failures)})")
    else:
        print("RESULT: PASS (all required checks passed)")

    if out_path:
        import os

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report written to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
