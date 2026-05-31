# KV cache FP8 量化 — 实现清单

mini-vllm 在 4090 上加 FP8 KV cache 的最小改动 patch。目标：
- KV 显存减半（fp32 → fp8，4×；或对照 bf16 → fp8，2×）
- 同等显存下 `num_blocks` 翻倍 → 同 batch 下序列更长 / 同长度下 bsz 更高
- 不破现有 90 个测试的 bit-exact greedy 等价（deterministic 模式仍走 fp32 KV）
- 不引入新算子依赖（用 `torch.float8_e4m3fn` + naive cast，第二阶段再上 fused dequant）

> 这是落地 patch，不是研究计划。每一项都标了文件和大致行数。

---

## 0. 设计前提

1. **范围仅限 KV cache 存储 dtype**。Q / 算 attention / 写出还都是 bf16，FP8 只占 cache。等价于 vLLM 的 `--kv-cache-dtype fp8_e4m3`。
2. **保留 fp32 KV 路径**。`deterministic=True`（pytest 默认）继续走 fp32，绝大多数测试不动。FP8 是 `deterministic=False` 下的一条新路径。
3. **per-tensor static scale**（v1）。先用一个全局 scale，简单粗暴，profile 期跑几条 prompt 取 absmax。后面再升级 per-head / per-block。
4. **gather → bf16 时 dequant**。写入路径 bf16→fp8，读取路径 fp8→bf16，dequant 在现有 `_read_kv_from_blocks` 里就地做，attention kernel 一行不动。
5. **CUDA Graph 兼容**。FP8 cache 仍是 contiguous tensor，capture / replay 行为不变；scale 是固定 scalar，无 dynamic shape。

---

## 1. Config 入口

`mini_vllm/config.py`（在 `EngineConfig` 末尾）

```python
kv_cache_dtype: str = "auto"          # "auto" | "fp32" | "bf16" | "fp8_e4m3"
kv_scale: float | None = None         # None ⇒ profile 时自动取 absmax
kv_scale_calib_tokens: int = 4096     # absmax 校准要看多少 token
```

校验：`fp8_e4m3` 仅在 `deterministic=False` 且 `device == "cuda"` 时可用，否则报错（不要静默降级——会让 bench 数字撒谎）。

---

## 2. 分配 FP8 cache

`mini_vllm/model_runner.py:172` `allocate_kv_cache`

改两行 + 加一段：

```python
def allocate_kv_cache(self):
    kv_dtype = self._resolve_kv_dtype()        # 新方法，下面定义
    bytes_per_elem = torch.tensor([], dtype=kv_dtype).element_size()
    ...
    kv_cache = torch.zeros(..., dtype=kv_dtype, device=self.device)
    print(f"KV cache: ... ({kv_dtype}, scale={self.kv_scale})")
    return kv_cache

def _resolve_kv_dtype(self) -> torch.dtype:
    cfg = self.kv_cache_dtype
    if cfg == "fp8_e4m3":
        assert not self.deterministic, "fp8_e4m3 KV requires deterministic=False"
        assert self.device == "cuda"
        return torch.float8_e4m3fn
    if cfg == "bf16":
        return torch.bfloat16
    if cfg == "fp32" or cfg == "auto":
        return torch.float32
    raise ValueError(cfg)
```

需要在 `__init__` 里把 `kv_cache_dtype / kv_scale` 从 EngineConfig 透传进来（`engine.py` 里已经有 EngineConfig → ModelRunner 的桥接，跟着加 2 个 kwarg）。

---

## 3. Quant / dequant helper

新文件 `mini_vllm/kv_quant.py`（约 60 行）

```python
"""FP8 KV cache quant/dequant helpers.

v1: per-tensor symmetric scale, e4m3 (range ≈ ±448).
"""
import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0   # E4M3 saturated max

def quantize_to_fp8(x: torch.Tensor, scale: float) -> torch.Tensor:
    """bf16 / fp16 / fp32 → fp8_e4m3 with per-tensor scale.

    Stored value = clamp(x / scale, -FP8_MAX, FP8_MAX).to(fp8)
    """
    return (x.float() / scale).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

def dequantize_from_fp8(x_fp8: torch.Tensor, scale: float, out_dtype: torch.dtype) -> torch.Tensor:
    """fp8 → out_dtype. scale is a python float (graph-friendly)."""
    return x_fp8.to(out_dtype) * scale

def calibrate_scale(absmax: float, headroom: float = 0.95) -> float:
    """absmax 来自 profile 阶段记录的 K/V 绝对值最大值。"""
    return float(absmax) / (FP8_MAX * headroom)
```

注意：scale 必须是 python `float`，不是 0-d tensor，否则 CUDA Graph capture 会把它当成可变输入。

---

## 4. 写入路径加 quant

`mini_vllm/attention_patch.py:80` `_write_prefill_kv` 和 `:100` `_write_decode_kv`

每一处赋值前判断 cache dtype：

```python
if kv_cache.dtype == torch.float8_e4m3fn:
    K_write = quantize_to_fp8(K_write, paged_ctx.kv_scale)
    V_write = quantize_to_fp8(V_write, paged_ctx.kv_scale)
```

`paged_ctx.kv_scale` 在 `model_runner.py` 里注入（每次 forward 之前 `paged_ctx.kv_scale = self.kv_scale`）。

注意 `_write_decode_kv` 里是逐 token 索引写的（`[i, :, 0, :]`），quantize 一次切片再写，不要在循环里反复调。把 `key_states` 整体 quant 一次再分发，性能远好于 per-token quant。

---

## 5. 读取路径加 dequant

`mini_vllm/attention_patch.py:122` `_read_kv_from_blocks` 和 `:337` `_build_decode_kv_batch`

```python
K_blocks = kv_cache[layer_idx, block_table, 0]
V_blocks = kv_cache[layer_idx, block_table, 1]
if kv_cache.dtype == torch.float8_e4m3fn:
    K_blocks = dequantize_from_fp8(K_blocks, paged_ctx.kv_scale, paged_ctx.compute_dtype)
    V_blocks = dequantize_from_fp8(V_blocks, paged_ctx.kv_scale, paged_ctx.compute_dtype)
# 后面 reshape / permute 不变
```

`compute_dtype = bf16`（即 query_states.dtype）。在 forward 入口设置一次 `paged_ctx.compute_dtype`。

attention kernel 这一层完全不动 —— gather + dequant 一步完成，喂给 SDPA / 手写 matmul 的 K/V 都是 bf16。这里也是为啥 v1 只省显存不省带宽：从 HBM 读出来仍是 fp8（带宽减半），但算还在 bf16；带宽红利会在 attention 里体现，绝对加速看 attention 占比。

---

## 6. Scale 校准

`mini_vllm/model_runner.py` 加 `_calibrate_kv_scale`：

- 启动后跑 N 条短 prompt（用一份小 calibration set，比如 16 条 wikitext 摘要）
- 临时关掉 quant（或者用 fp32 cache 跑），hook 每层 K/V 的 `.abs().amax()`
- 记最大值 → `calibrate_scale(absmax)` → 注入 `self.kv_scale`
- 整个 profile 跑完后切到 fp8 cache

`kv_scale` 从 EngineConfig 显式传也可以（用户已经测好），优先级：显式 > calibration > 报错（不要默认 1.0）。

校准只跑一次，跑完打印一行 `kv_scale=0.123 (absmax=400.0)` 就完事，别建文件别建持久化，复杂度收住。

---

## 7. CUDA Graph 兼容性

现有 `decode_graphs` dict 按 `(bsz, kv_capacity)` 录制。

- KV cache 本体是 module-level 大 tensor，dtype 切到 fp8 不影响 capture（地址不变）
- `paged_ctx.kv_scale` 是 python float，capture 期间不变 → 进 graph 是 constant，OK
- `paged_ctx.compute_dtype` 同理
- `_read_kv_from_blocks` 里那个 `dequantize_from_fp8` 调用会被 capture 进 graph，replay 时直接复用 —— 这是免费的红利

**唯一风险**：如果 calibration scale 算出来后用户重启了 engine 但忘传，自动 fallback 路径会拿不到 scale。所以 step 6 的"不要默认 1.0"很关键。

---

## 8. 测试

`tests/test_kv_fp8.py`（新文件，约 5 个测试）

1. `test_fp8_quant_roundtrip_bf16` — 纯 helper 单测：bf16 张量 → quant → dequant 误差 < 一个相对 tolerance（比如 mean rel err < 0.05）
2. `test_fp8_kv_engine_runs` — 加载 Qwen3-0.6B，`kv_cache_dtype="fp8_e4m3"`，跑 5 条 prompt，能出 token 不崩
3. `test_fp8_kv_close_to_bf16` — 同 prompt 同 sampling，fp8 KV 和 bf16 KV greedy token 序列前 N（比如 8）个相同，或 perplexity 差距 < 阈值
4. `test_fp8_requires_nondeterministic` — `deterministic=True + fp8_e4m3` 必须 raise
5. `test_fp8_kv_with_cuda_graph` — graph 路径下 fp8 KV 也能跑（只要 deterministic=False + cuda）

关键：**这 5 个测试都标 `@pytest.mark.skipif(not cuda, …)`**，CPU/MPS 跳过；不要污染默认 90 个测试。

---

## 9. Bench

`benchmarks/bench_throughput.py` 加一行 CLI flag `--kv-dtype {bf16,fp8_e4m3}`，对比表：

| Config | KV size | bsz=16 throughput | bsz max |
|---|---|---|---|
| bf16 KV | X GB | A tok/s | M |
| fp8_e4m3 KV | X/2 GB | B tok/s | 2M |

预期：bsz=16 时 throughput 涨 5-15%（attention HBM 带宽减半，但 Q/output 还是 bf16，attention 本身只占总时延一部分）。**真正的红利在更大 bsz 和更长 context 下**，因为 KV 大小是瓶颈时，能塞下的 bsz 直接翻倍。

不要承诺加速幅度，bench 跑出来什么写什么。

---

## 10. 文档

写一篇 `FP8_KV_NOTES.md`（mini-vllm 现有 notes 风格），记录：

- 为啥 v1 选 per-tensor scale（简单 + 4090 性能足够）
- E4M3 vs E5M2 选 E4M3 的原因（dynamic range 不够大但精度足够 KV，KV 不像 activation 有大 outlier）
- absmax calibration 的局限（per-tensor 偏保守，per-head 是后续优化项）
- bench 实测数字
- 已知坑：哪些算子还不支持 fp8（比如老 PyTorch < 2.1）

这是给未来面试讲的素材，不是给用户看的 README。

---

## 实施顺序（建议 1 天）

1. (30min) Config + helper 文件 — step 1, 3
2. (1h) allocate_kv_cache 改 dtype，先用手填 scale=1.0 跑通 dummy 路径 — step 2
3. (1h) 写入路径 quant — step 4
4. (1h) 读取路径 dequant，跑通 1 条 prompt — step 5
5. (30min) 加 5 个测试，先跑能过 — step 8
6. (1h) calibration，profile 出真 scale — step 6
7. (30min) bench 出对比数字 — step 9
8. (30min) 写 notes — step 10

**坑预警**：

- `torch.float8_e4m3fn` 不支持很多 op（比如 `.transpose` / `.permute` 需要 PyTorch ≥ 2.1，`reshape` 要看版本），任何 reshape/permute 之前一定先 dequant。所以 step 5 里 dequant 必须放在 reshape **之前**，不是之后。
- `.element_size()` 在 fp8 上等于 1，KV cache 显存预算自动会变成原来的 1/4（fp32 比对）/ 1/2（bf16 比对），`num_blocks` 会自动多。验证一遍 print 出来的 num_blocks。
- E4M3 没有 inf，溢出会 saturate 到 ±448，所以 `clamp_` 不能省，否则下溢/溢出后值变 NaN / 0 都遇到过。
- CUDA Graph 录制时 `paged_ctx.kv_scale` 必须已经是真值，否则 graph 里冻结的是错的 scale。

---

## v2 路线（不在本次 patch）

- per-head scale（精度 +1 bit 量级）
- per-block scale（block 粒度小，calibration 几乎免费）
- E5M2 KV（dynamic range 大，长 context 友好）
- fused dequant + attention（写 Triton kernel，省一遍 HBM round-trip）
- KV 写入也走 quantized matmul（W8A8 路线，已经离开"只 quant KV"的范畴）

v2 上 5090 之前不动。
