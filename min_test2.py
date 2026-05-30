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
prompt = (
      "List the days of the week: Monday, Tuesday, Wednesday, Thursday, Friday, "
      "Saturday, Sunday. List the days of the week again: Monday, Tuesday,"
  )
print(f"prompt len: {len(prompt)} chars")
out = llm.engine.generate([prompt], sp)
print(f"output len: {len(out[0]['token_ids'])} tokens")
print("text preview:", out[0]["text"][:100])