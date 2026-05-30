from mini_vllm import LLM, SamplingParams
llm = LLM(
      model_path="Qwen/Qwen3-0.6B",
      block_size=16,
      max_num_seqs=1,
      max_num_batched_tokens=4096,
      gpu_memory_utilization=0.5,
  )
sp = SamplingParams(temperature=0.0, max_tokens=10)

print("Test 1: short prompt")
out = llm.engine.generate(["Hello"], sp)
print("OK:", out[0]["text"])

print("Test 2: medium prompt")
out = llm.engine.generate(["The capital of France is"], sp)
print("OK:", out[0]["text"])