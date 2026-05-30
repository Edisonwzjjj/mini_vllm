# mini-vLLM CUDA Graph 优化学习笔记

> 这份笔记记录本轮围绕 `run_decode()` 做 CUDA Graph 前置优化的 10 个步骤，以及为什么它能把 `mini-vLLM` 相比 HF serial baseline 的吞吐从约 `1.8x` 提升到约 `2.5x`。
>
> 关键词：`CUDA Graph`、`decode`、`KV cache`、`paged attention`、`deterministic`、`fast path`、`benchmark`

---

## 0. 当前结论

### Q：本轮优化前后吞吐变化是什么？

之前 fast path benchmark 大约是：

```text
HF serial:     ~39.8 tokens/s
mini-vLLM M2:  ~73.7 tokens/s
speedup:       ~1.8x
```

经过 10 步 decode/CUDA Graph 前置优化后：

```text
HF serial:     39.9 tokens/s
mini-vLLM M2:  99.1 tokens/s
speedup:       2.5x
```

也就是：

```text
73.7 -> 99.1 tokens/s
约 +34% mini-vLLM 吞吐提升
```

最终 benchmark 中观察到：

```text
CUDA graphs: captures=8, replays=99
keys=[
  (16, 496), (16, 512), (16, 528), (16, 544),
  (16, 560), (16, 576), (16, 592), (16, 608)
]
```

说明 CUDA Graph 确实被 capture/replay 了。

---

## 1. 为什么 HF serial 已经不弱？

### Q：为什么我们预期有 4x，但一开始只有 1.8x？

因为 benchmark 里的 HF baseline 是：

```python
model.generate(..., do_sample=False)
```

HF `generate()` 虽然是逐 prompt serial，但它不是 naive 实现。它内部已经用了：

- CUDA kernel；
- KV cache；
- optimized attention；
- optimized generate loop；
- transformers 内部的缓存结构。

所以它不是“每生成一个 token 都重新算完整 prompt”的弱 baseline。

如果之前推算 4x 时默认 HF serial 非常低效，那么会高估 mini-vLLM 的相对加速。

---

## 2. mini-vLLM 为什么仍然能比 HF serial 快？

### Q：mini-vLLM 加速 HF serial 的主要原理是什么？

主要来自两个方面：

### 2.1 多请求 batching

HF baseline 是：

```text
prompt 1 generate 100 tokens
prompt 2 generate 100 tokens
...
prompt 16 generate 100 tokens
```

也就是串行处理 16 个请求。

mini-vLLM 则尝试把多个请求放在一起：

```text
prefill batch
then decode batch step 1
then decode batch step 2
...
```

这样 GPU 可以在一次 forward 中处理多个 sequence，提高设备利用率。

### 2.2 paged KV cache

mini-vLLM 的设计类似 vLLM：

- 每个 sequence 的 KV cache 不要求连续；
- 用 block table 管理每个 sequence 拥有哪些 KV blocks；
- decode 时根据 block table 读取对应 KV。

理论上，这能让多请求调度和 KV 管理更灵活。

---

## 3. 为什么当前实现还达不到真正 vLLM 的速度？

### Q：既然用了 paged KV cache，为什么没有 4x 以上？

因为当前实现仍然不是 fused paged attention。

真正 vLLM 的核心优势是：

```text
attention kernel 直接根据 block table 读取 paged KV cache
```

而当前实现是：

```text
paged KV cache
  -> 每步 decode 先 gather/copy 成连续 K_batch/V_batch
  -> 再调用 attention
```

也就是说，当前 decode 里仍有大量 KV 重组开销。

相关代码在：

```text
mini_vllm/attention_patch.py::_build_decode_kv_batch()
```

核心逻辑：

```python
K_batch[i, :, :num_cached, :] = K_full
V_batch[i, :, :num_cached, :] = V_full
```

这个过程每层、每步都会发生。

---

## 4. CUDA Graph 主要解决什么问题？

### Q：CUDA Graph 能消除 KV gather/copy 吗？

不能。

CUDA Graph 主要减少：

- Python 调度 overhead；
- CUDA kernel launch overhead；
- 每步重复构造 tensor / launch kernel 的成本；
- allocator 相关 overhead。

但它不会自动把下面这个操作变快很多：

```python
K_batch[i, :, :num_cached, :] = K_full
```

如果 KV gather/copy 是主要瓶颈，那么 CUDA Graph 只能部分优化。

真正要进一步提速，需要：

- fused paged attention CUDA/Triton kernel；
- 或者避免每步重组连续 `K_batch/V_batch`。

---

# 本轮 10 步优化详解

---

## Step 1：预分配 `decode_input_ids / decode_position_ids`

### Q：原来哪里慢？

原来的 `run_decode()` 每步都会创建 tensor：

```python
input_ids = torch.tensor(
    [[seq.last_token_id] for seq in seqs], device=self.device
)
position_ids = torch.tensor(
    [[seq.num_tokens - 1] for seq in seqs], device=self.device
)
```

这会导致：

- 每个 decode step 都有 tensor allocation；
- tensor 地址不稳定；
- 不适合 CUDA Graph capture；
- Python list -> CUDA tensor 的构造开销重复发生。

### Q：改成了什么？

在 `ModelRunner._init_decode_buffers()` 中预分配：

```python
self.decode_input_ids = torch.empty(
    self.max_num_seqs, 1, dtype=torch.long, device=self.device
)
self.decode_position_ids = torch.empty(
    self.max_num_seqs, 1, dtype=torch.long, device=self.device
)
```

每步 decode 只更新已有 buffer：

```python
self.decode_input_ids[i, 0] = seq.last_token_id
self.decode_position_ids[i, 0] = seq.num_tokens - 1
```

### Q：为什么这一步能加速？

它减少了每步动态分配和 tensor 构造。更重要的是，它为 CUDA Graph 做准备：Graph replay 时需要稳定的 input tensor。

---

## Step 2：预分配 `slot_mapping / num_cached_after`

### Q：这两个东西是什么？

`slot_mapping` 表示当前 decode token 要写入 KV cache 的位置：

```text
slot = block_id * block_size + offset
```

`num_cached_after` 表示本次 decode 写入后，每个 sequence 已经缓存了多少 token 的 KV。

### Q：原来有什么问题？

原来每步构造 Python list：

```python
paged_ctx.slot_mapping = [info["slot"] for info in decode_infos]
paged_ctx.num_cached_after = [info["num_cached_after"] for info in decode_infos]
```

attention 内部又可能把它转 tensor。

### Q：改成了什么？

新增长期 buffer：

```python
self.decode_slot_mapping = torch.empty(
    self.max_num_seqs, dtype=torch.long, device=self.device
)
self.decode_num_cached_after = torch.empty(
    self.max_num_seqs, dtype=torch.long, device=self.device
)
```

每步只填值：

```python
self.decode_slot_mapping[i] = slot
self.decode_num_cached_after[i] = total_tokens_after
```

### Q：为什么这一步能加速？

减少 metadata tensor 的反复创建，并让 decode metadata 变成稳定 CUDA buffer。

---

## Step 3：预分配 `K_batch / V_batch / decode_mask`

### Q：原来最大的问题是什么？

`_build_decode_kv_batch()` 每层、每步都会新建：

```python
K_batch = kv_cache.new_zeros(...)
V_batch = kv_cache.new_zeros(...)
```

`_build_decode_mask()` 每步也会新建：

```python
mask = torch.zeros(...)
```

这些分配非常频繁。

### Q：改成了什么？

新增：

```python
self.decode_k_batch = None
self.decode_v_batch = None
self.decode_mask = None
self.decode_kv_capacity = 0
```

并通过：

```python
_ensure_decode_attention_buffers(max_cached)
```

按需扩容。

### Q：为什么 capacity 要按 block size 向上取整？

因为 KV cache 以 block 为单位组织。按 block size 对齐：

```python
capacity = ceil(max_cached / block_size) * block_size
```

可以减少频繁扩容。

例如：

```text
max_cached = 501 -> capacity = 512
max_cached = 502 -> capacity = 512
...
max_cached = 512 -> capacity = 512
max_cached = 513 -> capacity = 528
```

### Q：为什么这一步能加速？

减少 allocator 开销，也让 `K_batch/V_batch/mask` 的 tensor 对象更稳定。

---

## Step 4：集中 CPU metadata，减少 `.item()` / `.tolist()`

### Q：为什么 `.item()` / `.tolist()` 有问题？

如果 tensor 在 GPU 上，调用：

```python
x.item()
x.tolist()
```

通常会触发 GPU -> CPU 同步。

这会破坏异步流水，并且不适合 CUDA Graph。

### Q：原来在哪里发生？

attention patch 中：

```python
num_cached_values = num_cached_list.tolist()
slot = int(slots[i].item())
```

这些操作可能每层都发生。

### Q：改成了什么？

在 `run_decode()` 中一次性维护 Python list：

```python
slot_mapping_values.append(slot)
num_cached_after_values.append(total_tokens_after)
```

然后传入 `paged_ctx`：

```python
paged_ctx.slot_mapping_values = slot_mapping_values
paged_ctx.num_cached_after_values = num_cached_after_values
```

attention 内优先使用这些 CPU list，避免每层反复从 GPU tensor 同步。

### Q：为什么这一步能加速？

它把同步点集中到 decode 准备阶段，减少 attention 每层的隐藏同步。

---

## Step 5：预分配 block table tensor buffer

### Q：block table 是什么？

每个 sequence 的 KV cache 由多个 block 组成。`block_table` 记录这个 sequence 使用哪些 block。

例如：

```python
seq.block_table[0] = [12, 13, 20, 21]
```

表示这个 sequence 的 token KV 分布在这些 block 中。

### Q：原来有什么问题？

原来 `block_table` 是 Python list：

```python
paged_ctx.block_tables = [seq.block_table[0], ...]
```

这对 CUDA Graph 不友好。

### Q：改成了什么？

预分配：

```python
self.decode_block_tables = torch.empty(
    self.max_num_seqs,
    self.kv_cache.shape[1],
    dtype=torch.long,
    device=self.device,
)
self.decode_num_blocks = torch.empty(
    self.max_num_seqs,
    dtype=torch.long,
    device=self.device,
)
```

每步把 Python block table copy 进去：

```python
self.decode_block_tables[i, :num_blocks].copy_(block_table_tensor)
```

### Q：为什么这一步能加速？

这一步本身不一定立即加速，因为仍有 copy。但它把结构从 Python list 迁移到稳定 CUDA buffer，是 CUDA Graph 和未来 fused paged attention 的前置条件。

---

## Step 6：使用完整 capacity buffer，减少动态 shape

### Q：为什么动态 shape 是问题？

CUDA Graph 和 tensor shape 绑定。

如果每步 decode 都用：

```python
K_batch[:, :, :max_cached, :]
```

那么随着 decode step 增长，`max_cached` 不断变化，graph 无法复用。

### Q：改成了什么？

现在传完整 capacity：

```python
paged_ctx.decode_k_batch = self.decode_k_batch[:bsz]
paged_ctx.decode_v_batch = self.decode_v_batch[:bsz]
paged_ctx.decode_mask = self.decode_mask[:bsz]
```

mask 负责屏蔽无效区域：

```python
mask[i, 0, 0, num_cached:k_len] = -inf
```

### Q：为什么这一步能加速？

它让 shape 更稳定，从而 CUDA Graph 可以复用。

### Q：代价是什么？

会多计算一点 padding KV，因为 attention 的 K/V 长度是 capacity，而不是准确的 `max_cached`。

这是典型 tradeoff：

```text
固定 shape / 可 graph replay
vs
更少 padding 计算
```

---

## Step 7：抽出 `_decode_forward_static()`

### Q：为什么要抽这个函数？

CUDA Graph capture 需要一个清晰的 forward 入口。

原来 forward 逻辑混在 `run_decode()` 中。现在抽成：

```python
def _decode_forward_static(self, bsz: int):
    input_ids = self.decode_input_ids[:bsz]
    position_ids = self.decode_position_ids[:bsz]
    with torch.no_grad():
        return self.model(input_ids=input_ids, position_ids=position_ids)
```

### Q：为什么这一步能加速？

它本身不直接加速，但把将来 capture 的代码边界明确了。

后面 CUDA Graph 捕获的就是这个函数。

---

## Step 8：加入 graph cache skeleton / dispatcher

### Q：为什么需要 graph cache？

不同 shape 需要不同 CUDA Graph。

当前 key 是：

```python
(bsz, decode_kv_capacity)
```

因为这两个会影响 tensor shape。

### Q：新增了什么？

```python
self.decode_graphs = {}
```

以及：

```python
def _decode_graph_key(self, bsz: int):
    return (bsz, self.decode_kv_capacity)
```

和：

```python
def _can_use_decode_graph(self, bsz: int):
    return self.device == "cuda" and not self.deterministic ...
```

### Q：为什么 deterministic 模式不启用 graph？

因为 correctness test 依赖稳定 token-level 输出。CUDA Graph fast path 不是测试优先路径。

所以只在：

```python
deterministic=False
```

时启用。

---

## Step 9：最小 CUDA Graph capture/replay

### Q：capture 做了什么？

第一次遇到某个 graph key 时：

```python
for _ in range(3):
    self._decode_forward_static(bsz)
torch.cuda.synchronize()

with torch.cuda.graph(graph):
    static_outputs = self._decode_forward_static(bsz)
```

然后存起来：

```python
self.decode_graphs[key] = (graph, static_outputs)
```

### Q：replay 做了什么？

后续同 shape：

```python
graph.replay()
return static_outputs
```

### Q：为什么返回 logits 要 clone？

graph replay 的 output tensor 是静态 storage。下一次 replay 会覆盖它。

所以取 logits 时：

```python
outputs.logits[i, -1, :].clone()
```

避免后续使用时被覆盖。

### Q：为什么 capture 时没有用 fused SDPA？

某些 fused SDPA kernel 在 graph capture 下不稳定或不支持。capture 时临时用显式 matmul attention，更安全。

---

## Step 10：增加 graph 观测指标

### Q：为什么要加统计？

没有统计时，我们不知道：

- graph 有没有 capture；
- replay 了多少次；
- key 是否太碎；
- shape 是否稳定。

### Q：新增了什么？

```python
self.decode_graph_capture_count = 0
self.decode_graph_replay_count = 0
```

benchmark 中打印：

```python
CUDA graphs: captures=..., replays=..., keys=...
```

### Q：这次观察到什么？

```text
captures=8
replays=99
keys=[(16, 496), ..., (16, 608)]
```

说明：

- graph 确实在工作；
- replay 次数不少；
- 但 key 仍然较碎，因为 `decode_kv_capacity` 随上下文长度增长。

---

# 为什么优化后从 1.8x 到 2.5x？

### Q：主要提升来自哪里？

来自 decode 阶段减少了：

- Python tensor allocation；
- CUDA tensor allocation；
- metadata 同步；
- kernel launch overhead；
- 部分 Python 调度 overhead。

### Q：为什么不是巨大提升？

因为主要瓶颈之一还在：

```python
K_batch[i, :, :num_cached, :] = K_full
V_batch[i, :, :num_cached, :] = V_full
```

也就是每步每层仍然在做 KV gather/copy。

CUDA Graph 可以减少 launch 和 Python overhead，但不能消除这个 copy。

---

# 后续最值得做什么？

## 方案 A：固定 graph capacity，减少 captures

当前 key：

```text
(16, 496), (16, 512), ..., (16, 608)
```

如果 benchmark 里最大上下文已知：

```text
max prompt 495 + max tokens 100 ≈ 595
```

可以一开始就分配：

```text
capacity = 608 或 640
```

这样 graph key 可能变成一个：

```text
(16, 640)
```

目标：

```text
captures=1
replays≈100
```

## 方案 B：减少 KV gather/copy

短期：

- 优化 `_build_decode_kv_batch()`；
- 尽量减少 advanced indexing；
- 减少 Python loop；
- 尝试更好的 tensorized gather。

长期：

- 写真正的 paged attention CUDA/Triton kernel；
- 直接根据 block table 读 `kv_cache`；
- 不再构造连续 `K_batch/V_batch`。

## 方案 C：profile decode 阶段

下一步优化前应该拆分时间：

- prefill time；
- decode preparation time；
- graph replay time；
- KV gather/copy time；
- sampling time。

否则容易优化错方向。

---

# 一句话总结

这轮 10 步优化的核心是：

> 把 decode 从“每步动态创建一堆 Python/CUDA 对象”逐步改成“复用稳定 buffer，并用 CUDA Graph replay 固定 decode forward”，因此吞吐从约 `1.8x` 提升到 `2.5x`；但真正的 4x+ 还需要减少 KV gather/copy，最好实现 fused paged attention。
