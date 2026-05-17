# mini-vllm · 需求文档

> 你的第一个推理引擎。不是从零造 vLLM，而是**复刻它的"调度 + KV 管理"层**，把模型本身当黑盒。
>
> **目标**：用约 500 行 Python 实现一个能跑、能 batch、有 prefix cache、有抢占的简化推理引擎。完成后你对推理引擎的"控制平面"会有手感，不再是看源码。
>
> **预算**：2 周，每天 1-2 小时。
> **代码量**：~500 行（不含模型加载，那部分用 transformers）。
> **语言**：Python + PyTorch。

---

## 0. 这个项目的边界（先看清不做什么）

### ✅ 你要做的（控制平面）
- Sequence 状态机
- BlockManager（分页 + prefix cache 的 hash 索引）
- Scheduler（continuous batching，prefill/decode 切换，抢占）
- KV Cache 的逻辑布局（block_table、slot_mapping）
- 主循环 + 简单的 `generate()` API

### ❌ 你**不**要做的（数据平面）
- 不写 attention kernel（用 PyTorch 的 `scaled_dot_product_attention`）
- 不写 Triton/CUDA
- 不做 tensor parallel（单卡）
- 不做 CUDA graph
- 不做 chunked prefill（M3 选做）
- 不做量化、speculative decoding、MoE
- 不接 OpenAI API、不写 server（直接 Python 函数调用）

> **理由**：这些都是数据平面或高级优化，不影响你理解"调度+内存管理"这个核心。砍掉它们让你能 2 周完工，否则会陷入泥潭。

---

## 1. 模型选择

**统一用** `Qwen3-0.6B`，原因：
- 你已经下载过（`~/huggingface/Qwen3-0.6B/`）
- 你刚读完它的 nano-vllm 实现，对它的结构最熟
- 单卡能跑、显存压力小、迭代快

**怎么用**：通过 `transformers.AutoModelForCausalLM.from_pretrained` 加载，然后**自己接管 KV Cache 管理**（不用 HF 的 `past_key_values` 接口）。

> 这是项目的关键技术挑战之一：**如何让 transformers 的模型用你自己管的 KV Cache**。提示：HF 模型的 attention 模块支持传 `past_key_value`，但格式不是分页的。你有两条路：
> - **路 A（推荐）**：monkey-patch 模型每一层的 attention forward，替换成你自己的 attention 实现（用 `scaled_dot_product_attention`），从你的 BlockManager 里组装 K/V
> - **路 B**：自己写一个 Qwen3 的 forward（参考 nano-vllm `models/qwen3.py`），但这样代码量翻倍
>
> 选 A。具体怎么 patch 是你的练习。

---

## 2. 目录结构（建议）

```
mini-vllm/
├── README.md
├── REQUIREMENTS.md          # 本文件
├── mini_vllm/
│   ├── __init__.py
│   ├── config.py            # 全局配置
│   ├── sequence.py          # Sequence 状态机
│   ├── block_manager.py     # 分页 KV + prefix cache
│   ├── scheduler.py         # 调度器
│   ├── model_runner.py      # 接 HF 模型, 管 KV Cache 显存
│   ├── attention_patch.py   # monkey-patch Qwen3 attention
│   └── engine.py            # 主入口 (LLMEngine)
├── tests/
│   ├── test_block_manager.py
│   ├── test_scheduler.py
│   └── test_correctness.py  # 对比 HF generate 的输出
├── benchmarks/
│   └── bench_throughput.py
├── examples/
│   └── basic.py
└── pyproject.toml
```

---

## 3. 接口定义（外部 API）

写完后，下面这段代码必须能跑：

```python
from mini_vllm import LLM, SamplingParams

llm = LLM(
    model_path="~/huggingface/Qwen3-0.6B/",
    block_size=16,                # 故意比 nano-vllm 小, 让 prefix cache 更易触发
    max_num_seqs=8,
    max_num_batched_tokens=2048,
    gpu_memory_utilization=0.5,   # 留点显存给开发调试
)

sp = SamplingParams(temperature=0.7, max_tokens=64)
outputs = llm.generate(
    prompts=["Hello, world.", "What is attention?", "Hello, world."],
    sampling_params=sp,
)
for o in outputs:
    print(o["text"])
```

**返回格式**：和 nano-vllm 完全一致（list of `{"text": str, "token_ids": list[int]}`）。

---

## 4. 里程碑

### 🎯 M1：单序列能跑通（3-4 天）

**目标**：能 prefill + decode 一个 prompt，输出和 HF `model.generate` 数值一致（greedy 或固定 seed）。

**必须实现**：
- `Sequence`：含 `token_ids`、`block_table`、`num_cached_tokens`、状态机
- `BlockManager`：能 allocate / append / deallocate，**暂不做 prefix cache**
- `ModelRunner`：分配 KV Cache 大显存块，monkey-patch attention，实现 prefill 和 decode 两个 forward
- `LLMEngine`：单序列跑通

**验收标准**：
1. `examples/basic.py` 跑一条 prompt 能输出结果
2. `tests/test_correctness.py`：固定温度=0.01（近似贪心），输出 token 序列**和 HF 原版** `model.generate` 完全一致
3. 打印 `block_table` 和 `slot_mapping`，能肉眼验证正确

**关键决策点（你要自己想清楚）**：
- KV Cache 的 shape 怎么设计？参考 nano-vllm `model_runner.py:115` 但不要照抄
- attention forward 里，怎么从 block_table + 当前 token 数算出"应该读 KV Cache 的哪些 slot"？
- decode 时新算的 K、V 怎么写进 cache？（提示：用 `index_put_` 或直接索引赋值，不需要 Triton）
- monkey-patch 的最小侵入点在哪？（提示：Qwen3 的 `Qwen3Attention.forward`）

**你会撞的墙**：
- KV Cache 的 dtype 和模型 dtype 不一致 → forward 出 NaN
- forget 把 Q/K rotary embed → 输出乱码但不报错（最难 debug 的 bug）
- decode 时 position id 算错（应该是 `len(seq) - 1`，不是 0）
- attention mask 在 prefill 时漏了 causal → 数值对不上

---

### 🎯 M2：多序列 + 连续批处理（4-5 天）

**目标**：能并发 8 条不同长度的请求，throughput 比 M1 单条循环至少 4x。

**新增实现**：
- `Scheduler`：waiting / running 两个队列，`schedule()` 返回 `(seqs, is_prefill)`
- prefill 时把多条 prompt 拼成一长条送入模型（**这一步是性能关键**）
- decode 时把多条序列的 last_token 拼成 batch
- 每条序列独立的 `block_table`，attention 时各自从 cache 读自己那部分
- `postprocess`：判断 EOS、达到 max_tokens、释放 KV

**验收标准**：
1. 同时发 8 条不同长度的 prompt，全部正确返回
2. benchmark：和"用 HF generate 串行跑 8 条"对比，吞吐至少快 4x（理想 6-8x）
3. 打印每一步的 batch 状态：`[prefill 3 seqs, 142 tokens]` / `[decode 8 seqs]`

**关键决策点**：
- 调度策略：先 prefill 还是先 decode？为什么？（提示：抄 nano-vllm 的策略，但你要能解释）
- batch 内 sequence 长度不一，attention 怎么处理？两种方案：
  - **方案 1**：用 padding + attention mask（简单，浪费）
  - **方案 2**：用 `scaled_dot_product_attention` 的 `is_causal=True` + 各自处理（复杂，无浪费）
  - 给自己的提示：M2 用方案 1，M3 再优化
- decode 时 batch=8 但每条长度不同，KV 怎么读？（提示：每条序列单独索引自己的 block_table）

**你会撞的墙**：
- 多条序列的 `slot_mapping` 算错（最常见 bug，建议先写一个 `print_block_layout(seqs)` 调试函数）
- 拼 batch 后忘记把 position id 也对应拼好
- 一个序列 finish 后没正确从 running 队列移除，下一步崩溃
- KV Cache 不够用时没有抢占逻辑，直接 OOM
- scheduler 在没活干的时候死循环

---

### 🎯 M3：Prefix Cache → Radix Tree（已完成）

**目标**：用 Radix Tree 替换 hash chain，实现前缀缓存 + LRU 驱逐 + Cache-Aware 调度。

**已实现**：
- `RadixTreeNode`：parent/children(block_tokens tuple 做 key)/block_id/ref_count/last_access_time
- `RadixTree`：match_prefix / insert / evict（级联清理） / match_prefix_ratio
- `BlockManager`：radix_tree 替换 hash_to_block_id，eviction_heap (O(log N))，insert_blocks 替换 hash_blocks
- `Scheduler`：cache-aware 排序（按 match_prefix_ratio 降序），block_manager 引用
- 修复 `preempt_one` bug（先 deallocate 再清 block_table）

**关键认知**：
- **Block 粒度下 radix tree 和 hash chain 命中率完全相同**（benchmark 验证）
- Radix tree 的真正优势需要 token 粒度（split_node）+ EAGLE bigram key
- LRU 驱逐 > FIFO 驱逐：热门 system prompt 不被误驱

**验收**：
- 所有 34 个测试通过（含 19 个新 radix tree 测试）
- Benchmark：3/5/10 用户多轮对话 hit rate = hash chain hit rate

---

### 🌟 M4（已完成）：抢占 / Chunked Prefill / 流式输出

**已实现**：
- 抢占：preempt_one (LIFO) + _free_seq_resources
- Chunked Prefill：max_num_batched_tokens 限制 + 续接 chunk
- 流式输出：generate_stream() yield token

---

### 🌟 M5：EAGLE 投机解码（进行中）

**目标**：实现 EAGLE-style 树形投机解码，理解 draft model / tree attention / greedy 验证 / radix tree draft KV 复用。

**简化约定**：
- Greedy 验证（不做 stochastic sampling）
- 小 draft tree：top-2, depth 3（最多 7 draft token + 1 bonus）
- 纯 PyTorch tree attention（无自定义 CUDA kernel）
- Block 粒度（不做 token 粒度 split_node）
- 单序列先行

**EAGLE Decode 循环**：
```
1. DRAFT:   Draft model（1 层 + fc 投影）生成树形候选 token
2. VERIFY:  Target model 一次 forward 验证所有 draft token（tree attention）
3. POST:    BFS 遍历树，greedy 接受匹配的 token，拒绝的释放 KV
4. EXTEND:  用 accepted token 跑 draft model，准备下一轮输入
```

**新增文件**：
- `draft_model.py`：EAGLE draft model（1 层，fc=Linear(hidden*2, hidden)，共享 embed+lm_head）
- `draft_tree.py`：DraftTree 数据类、tree mask 构建、greedy 验证
- `eagle_runner.py`：draft→verify→post_verify→draft_extend 循环

**修改文件**：
- `attention_patch.py`：加 tree verify 分支（tree_mask 替代 causal mask）
- `model_runner.py`：hidden state 捕获、run_verify、draft model 初始化
- `engine.py`：EAGLE decode 路径
- `sequence.py`：draft 字段（draft_tokens, draft_parent_indices, hidden_state）
- `config.py`：EAGLE 配置（enable_eagle, eagle_topk, eagle_spec_steps）
- `block_manager.py`：DraftRadixTree（bigram key 复用 draft KV）

**验收标准**：
1. Draft tree mask 正确：每个 draft token 只 attend 到祖先路径
2. Greedy 验证：accepted tokens 与 target model 独立 greedy decode 一致
3. EAGLE decode 吞吐 > 原始 decode（每个 step 产出 >1 token）
4. Draft KV 复用：相同 bigram 在下一轮 draft 时命中 radix tree

---

## 5. 性能基准

写一个 `benchmarks/bench_throughput.py`，对比三方：

```
配置: 16 条请求, prompt 长度 [50, 500] 随机, max_tokens 100

|                          | tokens/s | 加速比 |
|--------------------------|----------|--------|
| HF generate (串行)        |   xxx    |  1.0x  |
| mini-vllm M2 (无 prefix)  |   xxx    |  ?x    |
| mini-vllm M3 (有 prefix)  |   xxx    |  ?x    |
| nano-vllm (参考)          |   xxx    |  ?x    |
```

合格线：M2 ≥ 4x，M3 ≥ 5x。低于这个数说明实现有问题，不要放过。

---

## 6. 测试策略

**最关键的一类测试**：和 HF `model.generate` 的**输出一致性**。

```python
# tests/test_correctness.py
def test_decode_matches_hf():
    prompt = "Hello"
    sp = SamplingParams(temperature=0.01, max_tokens=20)

    out_hf = run_with_hf(prompt, sp)
    out_mini = llm.generate([prompt], sp)[0]["token_ids"]

    assert out_hf == out_mini, f"Mismatch: {out_hf} vs {out_mini}"
```

**为什么 temperature=0.01 而不是真正的 greedy**：你和 HF 用同样的采样函数才能比，但 nano-vllm 不支持 greedy。设小温度近似。

**进阶测试**：
- batch 一致性：单条跑 vs batch 里跑，结果应该完全相同（没相同就是有 bug）
- prefix cache 一致性：开/关 prefix cache，结果应该完全相同
- 抢占一致性（M4）：被抢占重做的序列，结果应该和不被抢占一致

---

## 7. 我会怎么 review

每个里程碑你说"做完了"时，我会问你这些问题（提前看一眼，做的时候带着问题）：

**M1 review 问题**：
1. 你的 KV Cache 总显存是多少？怎么算的？
2. 给我看 prefill 一次的 `slot_mapping`，解释每个数字怎么算出来的
3. 如果我把 block_size 从 16 改成 32，吞吐会变化吗？为什么？
4. monkey-patch 的代码贴出来，解释为什么这么改最少侵入

**M2 review 问题**：
1. 你的 scheduler 为什么先 prefill 再 decode，反过来会怎样？
2. 8 条序列的 batch，attention 计算复杂度是 8 倍单条还是更多？
3. 一个序列 EOS 提前结束，剩下 7 条会受影响吗？
4. 你的实现里有没有任何"等待"逻辑？（应该没有，全异步）

**M3 review 问题**：
1. 你的 hash 怎么形成前缀链？画一下数据流
2. 两条序列共享 block，第一条结束时怎么保证不影响第二条？
3. 如果两个不同 prompt 的 hash 冲突了，你怎么发现的？
4. prefix cache 在什么场景下完全没用？什么场景下收益最大？

---

## 8. 学习记录建议

每个里程碑结束写一份**自己的复盘**到 Obsidian（用我们已经做好的 obsidian skill）：

```
- 这个 milestone 做了什么
- 撞到的最难的 3 个 bug，根因是什么
- 哪个设计决策当时纠结很久，最后选了什么、为什么
- 如果重来一遍，会怎么做不同
```

这种复盘的价值 > 代码本身。两周后回看，你会发现这是你成长曲线最陡的一段。

---

## 9. 资源清单

**必读**：
- nano-vllm 的 `block_manager.py`、`scheduler.py`、`model_runner.py`（你已经读过）
- PyTorch 文档：`scaled_dot_product_attention`、`index_put_`
- transformers 文档：Qwen3 模型代码（找到 `Qwen3Attention` 类）

**参考但不照抄**：
- vLLM 论文 *"Efficient Memory Management for LLM Serving with PagedAttention"*

**禁止**：
- 不要看 vLLM 原版的 scheduler / block_manager 源码（太复杂会把你带偏）
- 不要让我或任何 LLM 直接给你写代码 —— 卡住可以问思路、问是不是某个方向，但代码自己写

---

## 10. 第一步行动

今天/明天开工时，按这个顺序：

1. 在 `~/Desktop/mini-vllm` 用 `uv` 或 `pip` 建好环境（torch + transformers + xxhash）
2. 写 `examples/basic.py`，先用纯 HF `model.generate` 跑通一条 prompt（baseline，10 行代码）
3. 写 `tests/test_correctness.py` 的"和 HF 对比"框架（先让它跑通 HF 自己 vs 自己，永远是 pass）
4. 开始 M1：先写 `Sequence`，最简单，10 分钟
5. 然后 `BlockManager` 的 allocate/deallocate（先不要 prefix cache）
6. 然后是最难的 `ModelRunner` + monkey-patch

每完成一个文件，跑一下测试，绿了再往下。**不要憋大招然后一起跑，一定 N 段调试地狱**。

---

## 11. 求助原则

- **可以问我**："我的 slot_mapping 应该是 [10, 11, 12] 还是 [10, 26, 42]，哪个对？"
- **可以问我**："monkey-patch attention 时我应该 patch forward 还是整个 module？"
- **可以问我**："这个 bug 卡了 1 小时了，根因可能在哪几个方向？"

- **不要问我**："给我写一下 BlockManager"
- **不要问我**："把整个 scheduler 的代码贴给我"

我的角色是 mentor 和 reviewer，不是代码生成器。这是为你好——你已经会编程了，缺的就是"自己撞墙的次数"。

---

**祝顺利。算完第一题来对答案。**
