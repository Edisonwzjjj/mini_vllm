# M2 Progress Changelog

> 记录从 M1 完成到 M2 当前的每次改动、原因、效果。

---

## 1. Scheduler: 单条 → waiting/running 队列

**改了什么**: `scheduler.py` 新增 `waiting_seqs` / `running_seqs` 队列，`add_seqs()` 一次加入，`schedule()` 批量 prefill 后 decode 全部 running。

**为什么改**: M1 的 scheduler 只能挑一条序列跑，无法并发。连续批处理的核心就是"多序列同时 decode"。

**踩的坑**:
- `add_seqs()` 在 `step()` 里每次调用 → 序列被重复加入。修复：移到 `generate()` 只调一次。
- decode 时只返回 `running_seqs[0]` → 只跑一条没加速。修复：返回 `list(self.running_seqs)`。
- `postprocess()` 没被调用 → 完成的序列不释放坑位。修复：在 step() 里调 postprocess。

---

## 2. Engine: 适配多序列 step()

**改了什么**: `step()` 不再接收 seqs 参数，改从 scheduler 内部队列取。prefill 和 decode 分开处理。

**为什么改**: M1 的 step 把 seqs 外部传入 + 每次调 add_seqs，逻辑混乱。队列由 scheduler 自己管，engine 只调 schedule()。

---

## 3. model.eval() + greedy 采样修复

**改了什么**: `model_runner.py` 加 `.eval()`，sample_token 加 temperature=0 走 argmax。

**为什么改**: 多序列跑时 prompt 1 输出不一致。排查发现是 temperature=0.01 不是真正 greedy，multinomial 采样放大了微小浮点差异。改成 greedy（argmax）后三条 prompt 全部 match。

**教训**: 对比正确性必须用 greedy（temperature=0），temperature=0.01 的随机性会引入误判。

---

## 4. Batched Prefill: 拼接 prompt + block-diagonal causal mask

**改了什么**:
- `run_prefill(seqs: list[Sequence])` 把多条 prompt 的 input_ids 拼成 `(1, total_len)`
- position_ids 每条序列从 0 开始：`[0,1,2,3, 0,1,2,3, 0,1,2,3,4]`
- slot_mapping 展平为一维
- 构建 block-diagonal causal mask（每个序列内部 causal，序列之间隔离）
- logits 提取每条序列最后一个位置

**为什么改**: 逐条 prefill 需要 N 次 forward，拼接后只需 1 次。

**踩的坑**:
- input_ids 用 `torch.cat(dim=0)` 拼出 `(3, seq_len)` 而非 `(1, total_len)` → 应该用 dim=1 或先 extend 再 tensor
- slot_mapping 是嵌套列表 `[[0,1,2,3], [16,17,18,19]]` → 需要 flatten
- mask 用 `torch.zeros(total_len)` 创建 1D → 需要 `(total_len, total_len)` 2D
- forward 用的是循环最后一次的 input_ids → 应该用拼接后的
- num_cached_tokens 在 forward 前更新 → forward 失败后状态不一致，应在 forward 后
- logits 索引用 `logits[start:end]` → 应该是 `logits[0, last_pos, :]`

---

## 5. Attention Patch: 支持 batched prefill mask

**改了什么**: `paged_attention_forward` 的 SDPA 调用从固定 `is_causal=True` 改为根据场景选择：
- prefill + mask → 用 attn_mask
- prefill 无 mask → is_causal=True
- decode → 用 decode_mask

**为什么改**: batched prefill 时序列间不能互相看到，必须用自定义 mask 替代标准 causal。

---

## 6. Batched Decode: 从逐条到 batch forward

**改了什么**:
- `run_decode(seqs: list[Sequence])` 把多条序列的 last_token 拼成 `(batch_size, 1)`
- paged_ctx 新增 `block_tables`（per-seq）和 `num_cached_after`（per-seq）
- attention_patch decode 路径：逐条写 KV + 读取 KV 做 padded batch SDPA

**为什么改**: 逐条 decode 每步 N 次 forward，batched 后只需 1 次。Q/K/V 投影和 MLP 都是 batch 的，主要收益在这里。

**踩的坑**:
- `kv_cache[layer_idx, bt_tensor, 0]` 在 bt 是 torch.tensor 时索引行为异常 → 改用 Python list 索引
- `reshape(num_kv_heads, -1)` 把最后三维都压平 → 应该 `reshape(num_kv_heads, -1, head_dim)` 保留 head_dim
- padded approach 每层分配大零张量 → GC 开销大，改用 `kv_cache.new_zeros()`

---

## 7. 性能优化尝试

| 方案 | tokens/s | 说明 |
|------|----------|------|
| 逐条 decode | 11.9 | 16 次 forward/step |
| batched decode + per-seq SDPA | 17.5 | 1 次 forward，但 448 次 SDPA/step |
| batched decode + padded SDPA | 15.3 | 1 次 forward + 1 次 SDPA，但 padding 开销大 |
| batched decode + permute 优化 | 17.5 | 用 permute+reshape 替代 for+cat |

**结论**: MPS 上 paged KV cache 的 Python 循环开销是瓶颈。CUDA 上这些操作可被 GPU 并行化，预期收益远大于 MPS。

---

## 当前文件改动汇总

| 文件 | 主要改动 |
|------|----------|
| `scheduler.py` | waiting/running 队列，批量 prefill 调度 |
| `engine.py` | step() 适配 prefill/decode 分支，add_seqs 只调一次 |
| `model_runner.py` | run_prefill 批量化，run_decode 批量化，model.eval()，debug print 条件化 |
| `attention_patch.py` | prefill mask 支持，decode padded batch SDPA，permute 优化 |
| `sampling_params.py` | 移除空 Sampling 函数 |
| `__init__.py` | 导出 LLM, SamplingParams |
| `tests/test_correctness.py` | 8 条 prompt 单条 vs 批量对比 |
| `benchmarks/bench_throughput.py` | HF serial vs mini-vllm 吞吐对比 |
