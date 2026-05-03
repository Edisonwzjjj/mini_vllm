# M2 学习笔记 — 今天踩的坑和学到的

## 核心概念

### 1. 连续批处理（Continuous Batching）

M1 只跑一条序列：prefill → decode → decode → ... → finish → 下一条。

M2 让多条序列"交织"执行：
- 新序列进来 → prefill（入队）
- 已经在跑的 → decode（一起跑）
- 结束的 → 释放资源，腾坑位给 waiting 的

关键：**scheduler 是引擎的大脑**，决定谁跑、跑什么。

### 2. Batched Prefill 的本质

把多条 prompt 的 token 拼成一长条，一次 forward 搞定。核心难点：

**序列间不能互相看到**。如果直接拼接 + is_causal=True，后面的序列会 attend 到前面的序列，输出就错了。解决：block-diagonal causal mask — 每个序列内部 causal，序列之间完全隔离。

### 3. Batched Decode 的本质

把多条序列的"下一个 token"拼成 batch，一次 forward 搞定 Q/K/V 投影和 MLP。难点：

**每条序列的 KV cache 长度不同**。SDPA 要求 batch 内 KV 长度一致，所以需要 padding + mask。但 padding 会浪费计算和显存。

### 4. Paged KV Cache 的开销

读取 KV 时要遍历 block_table，把散落在不同 block 的数据拼接起来。这个操作在 Python 循环里很慢：
- 16 序列 × 28 层 × ~20 blocks = ~9000 次 cat 操作/step
- 在 CUDA 上可以被 kernel 并行化，MPS 上不行

---

## 今天踩的坑

### 坑 1: temperature=0.01 ≠ greedy

**现象**: 多序列跑时 prompt 1 的输出和单条跑不一致，第 9 个 token 开始分叉。

**根因**: temperature=0.01 时 softmax 概率分布很尖锐但不唯一，multinomial 采样仍有随机性。两条路径的 logits 有微小浮点差异（MPS 精度问题），被采样放大成选了不同 token。

**解决**: 对比正确性必须用 greedy（temperature=0 → argmax）。temperature=0.01 只在"近似 greedy 但允许微小随机"的场景有意义。

### 坑 2: `torch.cat(dim=0)` vs `dim=1` 拼接方向

**现象**: 拼接后的 input_ids shape 是 `(3, seq_len)` 而非 `(1, total_len)`。

**根因**: batch 维是 dim=0，序列维是 dim=1。拼接多条 prompt 到一条长序列应该沿 dim=1（序列方向）。

**教训**: 拼接前想清楚目标 shape。batched prefill 目标是 `(1, total_len)`，不是 `(batch, seq_len)`。

### 坑 3: 嵌套列表 vs 展平列表

**现象**: `slot_mapping = [[0,1,2,3], [16,17,18,19]]` 传给 attention_patch，只处理了第一个子列表。

**根因**: attention_patch 期望 `List[int]`，传入的是 `List[List[int]]`。

**教训**: 接口约定要明确。多序列拼接时，per-seq 的数据结构要展平。

### 坑 4: `reshape(num_kv_heads, -1)` 压平了不该压的维度

**现象**: `IndexError: too many indices for tensor of dimension 2`。

**根因**: `K_blocks.permute(1,0,2,3)` shape 是 `(num_kv_heads, n_blocks, block_size, head_dim)`。`reshape(num_kv_heads, -1)` 把后三维全压成了一维 `(num_kv_heads, n_blocks*block_size*head_dim)`，丢失了 token 和 head_dim 的分界。

**解决**: `reshape(num_kv_heads, -1, head_dim)` 保留 head_dim 维度。

**教训**: reshape 时 `-1` 只能用于一个维度，其他维度必须显式指定以确保正确。

### 坑 5: Python 循环是 MPS 上的性能杀手

**现象**: batched decode 反而比 HF serial 慢。

**根因**: 每步 16 序列 × 28 层 = 448 次 Python 循环迭代（读 block、cat、SDPA）。每次迭代的 Python 开销（解释器、类型检查、tensor 元数据创建）在 MPS 上远大于实际 GPU 计算。

**教训**: 在 MPS 上，减少 Python 循环次数比减少计算量更重要。CUDA 上这个开销被 GPU 并行化掩盖了。

---

## 今天学到的设计决策

### Scheduler 为什么先 prefill 再 decode？

因为 prefill 是"入队"操作 — 新序列必须先 prefill 才能 decode。如果先 decode，running 里的序列占着坑位，waiting 里的永远进不来。极端情况：如果所有 running 序列都在 decode，没有新序列进来，waiting 队列会饿死。

### 为什么 batched prefill 用 block-diagonal mask 而不是 padding？

两种方案都能实现"序列间隔离"：
- Padding：所有序列 pad 到最长，浪费计算
- Block-diagonal：直接构建精确 mask，无浪费

Prompt 通常短（几十到几百 token），padding 浪费不大，但 block-diagonal 更干净。

### 为什么 MPS 上达不到 4x 加速？

4x 目标是为 CUDA GPU 设计的。CUDA 的核心优势：
1. 大量并行核心可以同时处理 16 条序列的 attention
2. Kernel launch 开销低
3. Paged KV cache 的散读操作可以用自定义 kernel 优化

MPS（Apple Silicon GPU）的局限：
1. 并行度低，小任务 GPU 利用率不足
2. Python 循环的 CPU 开销无法被 GPU 掩盖
3. 没有 custom kernel 支持，只能用标准 PyTorch op
