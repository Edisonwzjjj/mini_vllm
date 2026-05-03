# Debug 方法论 — 推理引擎开发中怎么找 bug

> 不是"出了错怎么办"，而是"怎么系统性地定位问题"。

---

## 总原则：二分法 + 最小复现

遇到 bug 时，不要猜。按这个流程：

```
1. 缩小范围：能不能用最简单的输入复现？
2. 二分定位：哪一步开始出错的？
3. 打印中间态：slot_mapping、block_table、logits 对不对？
4. 对比参照：和已知正确的输出（HF generate）比差在哪？
```

---

## 场景 1: 输出和 HF 不一致

**这是最常见的 bug，也是最关键的。**

### Step 1: 确认是采样问题还是 KV cache 问题

```python
# 用 greedy（temperature=0）排除采样随机性
sp = SamplingParams(temperature=0.0, max_tokens=20)
```

如果 greedy 一致但 temperature=0.01 不一致 → 采样问题，不是 KV cache 问题。
如果 greedy 也不一致 → 继续排查。

### Step 2: 逐 token 对比，找到第一个分叉点

```python
single_ids = llm.generate([prompt], sp)[0]["token_ids"]
multi_ids = llm.generate([prompt], sp)[0]["token_ids"]  # 换成你要对比的方式

for i, (a, b) in enumerate(zip(single_ids, multi_ids)):
    if a != b:
        print(f"First diff at token {i}: single={a}, multi={b}")
        break
```

分叉点告诉你在哪一步 decode 出了问题。如果第 1 个 token 就错了 → prefill 有 bug。如果前 N 个对、后面错了 → decode 某一步的 KV 读写出问题。

### Step 3: 打印 logits 看差异大小

```python
# 在 model_runner 的 forward 后打印
print(f"logits diff: {(logits_a - logits_b).abs().max().item()}")
```

如果差异极小（< 1e-5）→ 浮点精度问题。
如果差异大（> 0.1）→ 逻辑 bug。

### Step 4: 检查 KV cache 的读写

这是最可能的 bug 来源。打印每一步的关键状态：

```python
# 在 attention_patch 的 decode 路径加打印
print(f"layer={layer_idx}, bid={bid}, off={off}, num_cached={num_cached}")
```

常见 KV cache bug：
- position_ids 算错（应该是 `num_tokens - 1`，不是 0）
- slot_mapping 算错（block_id * block_size + offset）
- num_cached_tokens 没更新或更新时机不对
- 写入和读取的 block 不一致

---

## 场景 2: 多序列输出错误（跨序列污染）

**现象**: 序列 A 的输出里出现了序列 B 的内容。

### 根本原因: KV cache 的 block 被共享了

排查方法：

```python
# 检查不同序列的 block_table 是否有重叠
for i, seq in enumerate(seqs):
    print(f"seq {i}: block_table={seq.block_table[0]}")

# 检查 BlockManager 是否把同一个 block 分配了两次
# 在 allocate() 里加断言
```

常见问题：
- BlockManager 的 free_blocks 用了错的数据结构（比如 list 可以有重复）
- 序列结束后没有 deallocate，block 被新序列复用但旧数据还在

---

## 场景 3: Shape 错误 / IndexError

**这是最好 debug 的，因为 Python 会告诉你具体哪一行。**

### 方法: 打印每一步的 shape

```python
# 在出错的函数里加
print(f"tensor shape: {x.shape}, expected: (bsz, num_heads, seq_len, head_dim)")
```

常见的 shape 错误：
- `reshape` 把不该压平的维度压了 → 检查 reshape 参数
- `permute` 顺序错了 → 画出来每一步的 shape
- batch 维缺失或多余 → 检查 `unsqueeze(0)` / `squeeze()`

### 小技巧: 先用固定小输入验证

```python
# 不要用真实模型跑，先用小 tensor 验证逻辑
kv = torch.zeros(2, 10, 2, 4, 16, 64)  # 2 layers, 10 blocks, ...
bt = [0, 3]
result = kv[0, bt, 0]
print(result.shape)  # 应该是 (2, 4, 16, 64)
```

---

## 场景 4: 性能不达标

**不要盲目优化，先找瓶颈。**

### Step 1: 加计时，看哪一步慢

```python
import time
t0 = time.time()
logits = self.model_runner.run_prefill(seqs)
t1 = time.time()
print(f"prefill: {t1-t0:.3f}s")

for seq in seqs:
    t0 = time.time()
    logits = self.model_runner.run_decode(seq)
    t1 = time.time()
    print(f"decode seq {seq.seq_id}: {t1-t0:.3f}s")
```

### Step 2: 区分 Python 开销 vs GPU 计算

```python
# 看 GPU 时间
torch.mps.synchronize()  # CUDA 用 torch.cuda.synchronize()
t0 = time.time()
# ... 你的操作 ...
torch.mps.synchronize()
t1 = time.time()
```

如果 synchronize 前后的时间差很大 → Python 开销是瓶颈。
如果差很小 → GPU 计算是瓶颈，要优化算法。

### Step 3: 数操作次数

```python
# 粗算每步有多少次 Python 循环
print(f"每步循环次数: {len(seqs) * num_layers} (序列数 × 层数)")
```

如果 > 100 → Python 开销可能占主导，考虑向量化。

---

## 今天实际的 debug 案例

### 案例: prompt 1 多序列输出不一致

1. **发现**: 3 条 prompt 中 prompt 1 的 token_ids 不匹配
2. **二分**: 用 temperature=0.0 greedy 后全部匹配 → 采样随机性是原因
3. **根因**: temperature=0.01 时 multinomial 采样，两条路径 logits 有 ~1e-6 浮点差异，被采样放大
4. **结论**: 对比正确性必须用 greedy

### 案例: reshape IndexError

1. **发现**: `IndexError: too many indices for tensor of dimension 2`
2. **定位**: `K_blocks.permute(1,0,2,3).reshape(num_kv_heads, -1)` 结果是 2D，但后面按 3D 索引
3. **根因**: `reshape(num_kv_heads, -1)` 把 `(num_kv_heads, n_blocks, block_size, head_dim)` 后三维全压平了
4. **修复**: `reshape(num_kv_heads, -1, self.head_dim)` 保留 head_dim

### 案例: tensor 索引行为异常

1. **发现**: `kv_cache[layer_idx, bt_tensor, 0]` 结果 shape 不对
2. **验证**: 用小 tensor 单独测试 → `bt = [0, 4]` (list) 工作正常
3. **根因**: PyTorch 混合 int + tensor 索引时有隐式行为差异
4. **修复**: 用 Python list 而非 torch.tensor 做索引

---

## Debug 的心智模型

把推理引擎想象成一条流水线：

```
Tokenizer → Sequence → Scheduler → ModelRunner → Attention Patch → KV Cache
                ↑                                       ↑
              状态机                              读写正确性是核心
```

每个环节都可能出问题。Debug 时从两头往中间缩：
1. 输入对不对？（tokenizer encode）
2. 输出对不对？（和 HF 对比）
3. 中间状态对不对？（slot_mapping, block_table, num_cached_tokens）
4. 哪一步开始不对？（二分法）
