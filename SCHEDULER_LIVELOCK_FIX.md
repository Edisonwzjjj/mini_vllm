# Scheduler 活锁问题修复记录

## 背景

在运行下面命令时：

```bash
cd /root/autodl-tmp/mini_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python min_test3.py
```

发现现象：

- 5 个 prompt 一个一个跑没有问题；
- 5 个 prompt 一起传给 `generate(PROMPTS, sp)` 时会卡住；
- CPU 占用较高；
- GPU 利用率接近 0；
- 日志停在 batch generate 的第一个 prefill 后不再前进。

相关测试脚本中 `LLM` 初始化为：

```python
llm = LLM(
    model_path="Qwen/Qwen3-0.6B",
    block_size=16,
    max_num_seqs=1,
    max_num_batched_tokens=4096,
    gpu_memory_utilization=0.5,
)
```

关键点是：

```python
max_num_seqs=1
```

但后面一次性提交了 5 个 prompts：

```python
out = llm.engine.generate(PROMPTS, sp)
```

---

## 问题现象

修复前日志大致停在：

```text
--- all 5 prompts at once ---
[PREFILL] 1 seqs, total_suffix_tokens=32, suffix_lengths=[32], prefix_lengths=[0]
```

之后没有继续输出。

用 `faulthandler` 定位后，发现主线程卡在 scheduler 逻辑附近：

```text
mini_vllm/block_manager.py:287 in match_prefix
mini_vllm/block_manager.py:345 in match_prefix_ratio
mini_vllm/scheduler.py:43 in schedule
mini_vllm/engine.py:79 in step
mini_vllm/engine.py:123 in generate
```

表面看像是 prefix cache 或 radix tree 卡住，但真正原因是 scheduler 进入了活锁状态。

---

## 根因分析

### Q：为什么单个 prompt 没问题？

单个 prompt 时：

```text
waiting_seqs = 1
running_seqs = 0
max_num_seqs = 1
```

scheduler 可以把这个 seq 从 waiting 移到 running，然后 prefill、decode、finish，流程正常。

### Q：为什么 5 个 prompt 一起会卡住？

当一次性提交 5 个 prompts，且 `max_num_seqs=1` 时，状态变化如下：

1. 第一个 seq 进入 running：

```text
running_seqs = 1
waiting_seqs = 4
max_num_seqs = 1
```

2. 第一个 seq prefill 完成后，还没 decode 完，scheduler 下一次调度时发现 `waiting_seqs` 非空，于是进入 waiting 分支。

原来的逻辑类似：

```python
if self.waiting_seqs:
    prefill_seqs = []
    total_tokens = 0
    while self.waiting_seqs:
        if len(self.running_seqs) + len(prefill_seqs) >= self.max_num_seqs:
            break
        ...
    return SchedulerOutput(seqs=prefill_seqs, is_prefill=True)
```

这时：

```python
len(self.running_seqs) == 1
len(prefill_seqs) == 0
self.max_num_seqs == 1
```

所以条件成立：

```python
len(self.running_seqs) + len(prefill_seqs) >= self.max_num_seqs
# 1 + 0 >= 1 -> True
```

`while` 立即 `break`，导致：

```python
prefill_seqs = []
```

但原代码仍然返回：

```python
SchedulerOutput(seqs=[], is_prefill=True)
```

### Q：为什么这会导致活锁？

`engine.step()` 看到 scheduler 返回空 seqs：

```python
if not scheduler_output.seqs:
    return []
```

于是本轮什么都不做。

但是此时：

```text
running seq 没有 decode
waiting seq 没有 prefill
没有 seq finished
```

`generate()` 的循环条件是：

```python
while not all(seq.is_finished() for seq in seqs):
    self.step(sampling_params)
```

因为没有任何 seq 完成，所以循环继续。

下一轮 scheduler 又进入同样状态：

```text
running_seqs = 1
waiting_seqs = 4
max_num_seqs = 1
```

又返回空 `prefill_seqs`。

最终形成活锁：

```text
waiting 非空 -> 尝试 prefill -> running 已满 -> prefill_seqs 空 -> 返回空任务 -> engine 不做事 -> 状态不变 -> 重复
```

---

## 修复方案

修复位置：

```text
mini_vllm/scheduler.py
```

修复思路：

> 如果 waiting queue 非空，但 running 已经满了，不能返回空 prefill；应该让 running seqs 继续 decode，直到它们完成并释放容量。

修复后的逻辑：

```python
# Prefill: take sequences up to limits, move to running
prefill_seqs = []
total_tokens = 0
while self.waiting_seqs:
    if len(self.running_seqs) + len(prefill_seqs) >= self.max_num_seqs:
        break
    remaining_budget = self.max_num_batched_tokens - total_tokens
    if remaining_budget <= 0:
        break
    seq = self.waiting_seqs.pop(0)
    self.running_seqs.append(seq)
    prefill_seqs.append(seq)
    chunk_size = min(seq.num_prompt_tokens, remaining_budget)
    total_tokens += chunk_size
if prefill_seqs:
    return SchedulerOutput(seqs=prefill_seqs, is_prefill=True)
# Waiting queue exists, but running is already full. Decode running
# sequences so they can finish and free capacity for waiting ones.
```

然后继续走后面的 decode 分支：

```python
if self.running_seqs:
    return SchedulerOutput(seqs=list(self.running_seqs), is_prefill=False)
```

---

## 修复前后行为对比

### 修复前

```text
running_seqs = 1
waiting_seqs = 4
max_num_seqs = 1
```

scheduler 返回：

```python
SchedulerOutput(seqs=[], is_prefill=True)
```

engine 什么都不做，状态永远不变。

### 修复后

同样状态下，scheduler 不再返回空 prefill，而是继续 decode running seq：

```python
SchedulerOutput(seqs=[running_seq], is_prefill=False)
```

running seq 会继续生成 token，最终完成并释放容量。之后 waiting queue 中的下一个 seq 就可以进入 running。

---

## 验证结果

### 1. 单元测试

运行：

```bash
cd /root/autodl-tmp/mini_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/test_scheduler.py tests/test_preemption.py -q
```

结果：

```text
7 passed
```

### 2. 复现脚本

运行：

```bash
cd /root/autodl-tmp/mini_vllm
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 180 python min_test3.py
```

修复后输出：

```text
--- all 5 prompts at once ---
[PREFILL] 1 seqs, total_suffix_tokens=32, suffix_lengths=[32], prefix_lengths=[0]
[PREFILL] 1 seqs, total_suffix_tokens=33, suffix_lengths=[33], prefix_lengths=[0]
[PREFILL] 1 seqs, total_suffix_tokens=53, suffix_lengths=[53], prefix_lengths=[0]
[PREFILL] 1 seqs, total_suffix_tokens=40, suffix_lengths=[40], prefix_lengths=[0]
[PREFILL] 1 seqs, total_suffix_tokens=10, suffix_lengths=[10], prefix_lengths=[0]
  prompt 0: 128 tokens
  prompt 1: 128 tokens
  prompt 2: 128 tokens
  prompt 3: 128 tokens
  prompt 4: 128 tokens
```

说明 5 个 prompts 一起提交时不再卡死。

---

## 经验总结

### Q：这个 bug 属于什么类型？

这是 scheduler 活锁，不是 CUDA OOM，也不是模型 forward 卡住。

活锁特征：

- 程序仍在运行；
- CPU 可能持续占用；
- GPU 利用率很低；
- 状态没有前进；
- 没有 exception。

### Q：为什么容易误判？

因为日志最后停在 prefill，且 GPU 没动，很容易误以为：

- CUDA Graph 卡住；
- HF model forward 卡住；
- prefix cache 查找卡住；
- radix tree 死循环。

但实际是 scheduler 一直返回空任务。

### Q：scheduler 设计时要注意什么？

一个重要原则：

> 如果系统里还有未完成请求，scheduler 不应该返回空任务，除非真的没有任何可执行工作。

具体到这里：

- waiting queue 非空不代表一定能 prefill；
- 如果 running 已满，应该先 decode running；
- decode 让 running 完成后，waiting 才能继续进入。

### Q：这个问题和 `max_num_seqs` 有什么关系？

`max_num_seqs=1` 最容易触发。

因为只要有一个 running seq，waiting queue 就无法再调入新的 prefill。

但类似问题在更大 `max_num_seqs` 下也可能出现，例如：

```text
running_seqs 已满
waiting_seqs 非空
```

如果 scheduler 仍然优先 waiting 分支并返回空 prefill，就会活锁。

---

## 后续建议

可以补一个专门的 scheduler regression test：

```python
def test_schedule_decode_when_waiting_exists_but_running_full():
    scheduler = Scheduler(max_num_seqs=1, ...)
    running = Sequence(...)
    waiting = Sequence(...)
    running.num_prefill_tokens = running.num_prompt_tokens
    scheduler.running_seqs.append(running)
    scheduler.waiting_seqs.append(waiting)

    out = scheduler.schedule()

    assert out.is_prefill is False
    assert out.seqs == [running]
```

这个测试可以防止以后 scheduler 重构时重新引入同类活锁。
