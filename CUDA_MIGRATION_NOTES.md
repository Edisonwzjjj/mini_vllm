# mini-vLLM CUDA 迁移记录

## 背景

今天把 `mini-vllm` 从原先偏 `MPS` 的实现迁移到 `CUDA` 环境，并修复了 CUDA 下的测试、显存和 benchmark 问题。

最终 correctness 测试结果：

```bash
python -m pytest > /tmp/pytest_cuda.log 2>&1
# 90 passed
```

## 今天的主要修改

### 1. 设备选择从 MPS 迁移到 CUDA

原代码里有不少隐含假设：默认设备是 `mps`，部分测试也硬编码 `.to("mps")`。

这在 Linux + NVIDIA GPU 上会直接失败。

修改：

- `ModelRunner` 优先使用 `cuda`。
- `tests/test_step3.py` 中 HF reference 改成 `cuda if available else mps`。
- `benchmarks/bench_throughput.py` 中 HF baseline 不再硬编码 `.to("mps")`。

### 2. CUDA 显存分配策略

CUDA 下 pytest 会在一个 Python 进程中创建多个 module-scoped `LLM/ModelRunner`。如果每个实例都分配很大的 KV cache，会导致后面的测试或 benchmark OOM。

修改：

- CUDA 下用 `torch.cuda.mem_get_info()` 获取当前 free memory。
- `deterministic=True` 测试模式下限制 KV cache 最大值，避免测试中多个模型实例互相挤爆显存。
- benchmark 中 HF baseline 跑完后执行：

```python
del model
gc.collect()
torch.cuda.empty_cache()
```

然后再构造 `mini-vllm`。

### 3. 修复 prefill attention mask bug

原来的 prefill additive mask 用 `1.0` 表示可见位置：

```python
mask = mask.masked_fill(mask == 0, float("-inf"))
```

但 additive attention mask 的语义是：

- 可见位置应该是 `0.0`
- 不可见位置应该是 `-inf`

否则相当于给所有可见 token 的 attention score 额外加 `1.0`。

修复为：

```python
mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
```

这是一个真实代码 bug，只是在 MPS 上没有明显暴露。

### 4. 引入 `deterministic` 开关

CUDA 下不同执行路径可能触发不同 kernel、不同 GEMM shape、不同 accumulation order。数学上等价的路径，浮点结果不一定 bit-exact。

例如：

- single prefill vs batched prefill
- single decode vs batched decode
- full prefill vs chunked prefill
- full prefill vs prefix-cache suffix prefill
- normal decode vs speculative verify

对于 greedy decoding：

```python
next_token = logits.argmax()
```

如果 top-1/top-2 logits 很接近，极小数值差异就可能导致 token 分叉。

因此新增：

```python
deterministic: bool = True
```

含义：

- `deterministic=True`：测试/correctness 模式，保证 token-level 输出稳定。
- `deterministic=False`：benchmark/fast path 模式，启用真正 batched 路径和 CUDA fused SDPA。

### 5. correctness 模式行为

当 `deterministic=True`：

- multi-seq prefill 会拆成逐个 seq prefill。
- multi-seq decode 会拆成逐个 seq decode。
- prefix-cache/chunked prefill 会重新计算当前 prompt prefix，保证和 full prefill 对齐。
- dummy speculative/tree draft 默认 fallback 到 normal decode，除非使用 PLD 或 oracle draft。
- paged attention 使用显式 `fp32 matmul -> softmax -> matmul`，避免 CUDA fused kernel 数值漂移。

### 6. fast path / benchmark 模式行为

当 `deterministic=False`：

- batched prefill 保持 batched。
- batched decode 保持 batched。
- prefix-cache/chunked prefill 使用 suffix-only compute。
- attention 使用 CUDA fused `scaled_dot_product_attention`。
- dummy speculative/tree path 不强制 fallback。

benchmark 中已经使用：

```python
deterministic=False
```

## 当前结论

### 哪些是代码问题？

明确的代码问题：

1. 设备选择硬编码/默认 MPS，不适配 CUDA。
2. CUDA KV cache 分配策略不合理，导致 OOM。
3. additive attention mask 使用 `1.0` 作为可见位置，这是错误的。
4. 默认假设 CUDA 上不同执行路径 bit-exact 一致，这个假设不成立。

### 测试有没有问题？

现有测试作为 `deterministic=True` 的 correctness test 是合理的：它要求 greedy token ids 完全一致。

但如果直接拿同一套 exact token equality 测试去测 `deterministic=False` fast path，就过于严格。

fast path 更合理的测试标准应该是：

- 不崩溃；
- 输出长度正确；
- block/KV/cache invariant 正确；
- logits 在容忍范围内接近；
- throughput 有提升。

精确 token ids 相等应该只作为 deterministic mode 的要求。

## 为什么 benchmark 只有约 1.8x？

当前观测：

```text
HF serial:     ~39.8 tokens/s
mini-vLLM M2:  ~73.7 tokens/s
speedup:       ~1.8x
```

原因：

1. HF `generate()` 本身已经很强，内部用了 KV cache 和 CUDA kernels。
2. 当前 `mini-vllm` decode 每步仍会重组历史 KV。
3. 没有真正的 fused paged-attention CUDA/Triton kernel。
4. `MAX_TOKENS=100` 时 decode 阶段占大头。
5. prefill 受 `max_num_batched_tokens` 限制，16 个请求没有一次性全部进入同一 batch。

所以当前实现只是“逻辑上 batch 了”，还不是 vLLM 级别的高性能 paged attention。

## `run_decode` 里的 KV 复制在哪里？

`run_decode()` 本身看起来没有直接复制整个 KV，它主要做三件事：

1. 为每个 seq 计算新 token 要写入的 slot。
2. 把 `block_tables`、`slot_mapping`、`num_cached_after` 写入 `paged_ctx`。
3. 调用 HF model forward。

代码位置：

```text
mini_vllm/model_runner.py:215
```

关键逻辑：

```python
paged_ctx.kv_cache = self.kv_cache
paged_ctx.slot_mapping = [info["slot"] for info in decode_infos]
paged_ctx.block_tables = [info["block_table"] for info in decode_infos]
paged_ctx.num_cached_after = [info["num_cached_after"] for info in decode_infos]
paged_ctx.is_prefill = False
outputs = self.model(input_ids=input_ids, position_ids=position_ids)
```

真正的 KV gather/copy 发生在 attention monkey patch 里。

调用链是：

```text
ModelRunner.run_decode()
  -> self.model(...)
    -> patched Qwen3Attention.forward
      -> _decode_attention()
        -> _write_decode_kv()
        -> _build_decode_kv_batch()
          -> _read_kv_from_blocks()
```

相关代码位置：

```text
mini_vllm/attention_patch.py:306
```

`_build_decode_kv_batch()` 会先创建连续的 `K_batch/V_batch`：

```python
K_batch = kv_cache.new_zeros(
    meta["bsz"], meta["num_kv_heads"], max_cached, meta["head_dim"]
)
V_batch = kv_cache.new_zeros(
    meta["bsz"], meta["num_kv_heads"], max_cached, meta["head_dim"]
)
```

然后对 batch 中每个 seq：

```python
K_full, V_full = _read_kv_from_blocks(...)
K_batch[i, :, :num_cached_list[i], :] = K_full
V_batch[i, :, :num_cached_list[i], :] = V_full
```

这里就是每个 decode step 都在做的 KV gather/copy。

`_read_kv_from_blocks()` 的位置：

```text
mini_vllm/attention_patch.py:119
```

它从 paged KV cache 中按 `block_table` 取出 blocks：

```python
K_blocks = kv_cache[layer_idx, block_table, 0]
V_blocks = kv_cache[layer_idx, block_table, 1]
```

然后 reshape 成连续的：

```python
K_full = K_blocks.permute(...).reshape(...)[:, :total_len, :]
V_full = V_blocks.permute(...).reshape(...)[:, :total_len, :]
```

最后再 copy 到 `K_batch/V_batch`。

这也是当前 decode 慢的核心原因之一：

- 真正 vLLM 的 paged attention kernel 会直接根据 block table 读取 KV。
- 当前实现每步都先把 paged KV 重组为连续 KV batch，再调用 attention。
- 所以 batch decode 虽然逻辑上并行，但有大量 gather/copy 开销。

## CUDA Graph TODO

目标：先降低 decode 阶段 Python overhead 和 CUDA kernel launch overhead。

### Step 1：固定 decode shape

CUDA Graph 需要静态 shape。先固定：

- `batch_size`，例如 `max_num_seqs=16`
- decode input shape：`(batch_size, 1)`
- 最大 context/cache 长度
- dtype 和 device

已经完成的 seq 不要从 batch 中移除，而是保留 placeholder row，并通过 mask 屏蔽。

### Step 2：预分配 decode buffer

在 `ModelRunner` 里预分配：

```python
decode_input_ids       # (max_num_seqs, 1)
decode_position_ids    # (max_num_seqs, 1)
decode_slot_mapping    # (max_num_seqs,)
decode_num_cached_after# (max_num_seqs,)
```

后续每步 decode 只用 `.copy_()` 更新，不再每步 `torch.tensor(...)`。

### Step 3：去掉 graph 内动态分配

CUDA Graph capture 内不能有频繁动态分配。需要避免：

- `torch.tensor(list, device=...)`
- `new_zeros(...)`
- 动态 `torch.cat`
- shape 随 batch/context 改变

### Step 4：拆分 prepare 和 replay

把 decode 拆成：

1. Python prepare：
   - 更新 seq metadata；
   - 计算 slot；
   - 更新预分配 buffer。

2. CUDA graph replay：
   - 用固定 buffer 跑 model forward；
   - 输出 logits。

### Step 5：先只 graph normal decode

不要一开始 graph prefill。prefill 变长更复杂。

第一阶段只做：

```text
run_decode(batch_size=N, seq_len=1)
```

可以先捕获固定 batch size，例如：

- graph for batch 1
- graph for batch 2
- graph for batch 4
- graph for batch 8
- graph for batch 16

或者只捕获一个 max-batch graph，通过 mask 处理 inactive rows。

### Step 6：短期优化 KV gather/copy

CUDA Graph 只能减少 launch/Python overhead，不能消除 KV gather/copy。

短期：

- 预分配 `K_batch/V_batch`；
- 每步 in-place 填充；
- 避免 `new_zeros`；
- 避免重复创建 mask。

长期：

- 写真正的 paged attention CUDA/Triton kernel；
- 让 attention 直接根据 `block_table` 读 `kv_cache`，不要重组成连续 `K_batch/V_batch`。

### Step 7：warmup 后 capture

示例流程：

```python
for _ in range(3):
    decode_forward_static()
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_outputs = decode_forward_static()
```

replay：

```python
static_input_ids.copy_(new_input_ids)
static_position_ids.copy_(new_position_ids)
graph.replay()
```

### Step 8：验证 correctness

先小规模验证：

1. eager decode logits vs graph decode logits：`torch.allclose`。
2. greedy token ids 在固定 prompt 上一致。
3. 多 seq 中某些 seq 提前结束时 mask 正确。
4. 连续 decode 100 steps，显存不增长。

### Step 9：加入 benchmark 对比

在 `bench_throughput.py` 中增加第三列：

```text
HF serial
mini-vLLM eager fast path
mini-vLLM CUDA graph decode
```

统计：

- total tokens/s
- prefill time
- decode time
- peak GPU memory

### Step 10：先 profile 再写 kernel

CUDA Graph 不一定解决所有性能问题。

如果 profile 显示主要时间在 KV gather/copy，那么 CUDA Graph 提升有限，下一步应该是 paged attention kernel。
