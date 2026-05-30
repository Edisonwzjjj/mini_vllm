from mini_vllm import LLM, SamplingParams

llm = LLM(
      model_path="Qwen/Qwen3-0.6B",
      block_size=16,
      max_num_seqs=1,
      max_num_batched_tokens=4096,
      gpu_memory_utilization=0.5,
  )
sp = SamplingParams(temperature=0.0, max_tokens=128)  # bench 真实长度

  # bench 的第一个 prompt
PROMPTS = [
      "List the days of the week: Monday, Tuesday, Wednesday, Thursday, Friday, "
      "Saturday, Sunday. List the days of the week again: Monday, Tuesday,",

      "def add(a, b): return a + b\n"
      "def sub(a, b): return a - b\n"
      "def mul(a, b): return a * b\n"
      "def div(a, b): return",

      "Shopping list:\n- apple\n- banana\n- cherry\n- date\n- elderberry\n"
      "Shopping list:\n- apple\n- banana\n- cherry\n-",

      "Q: What is 2+2? A: 4.\nQ: What is 3+3? A: 6.\nQ: What is 4+4? A: 8.\n"
      "Q: What is 5+5? A:",

      "Once upon a time in a small village there lived",
  ]

  # 先一个一个跑确认
for i, p in enumerate(PROMPTS):
    print(f"--- prompt {i} ---")
    out = llm.engine.generate([p], sp)
    print(f"  out tokens: {len(out[0]['token_ids'])}")
    print(f"  preview: {out[0]['text'][:60]!r}")

print("\n--- all 5 prompts at once ---")
out = llm.engine.generate(PROMPTS, sp)
for i, o in enumerate(out):
    print(f"  prompt {i}: {len(o['token_ids'])} tokens")