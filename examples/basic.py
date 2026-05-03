"""Basic usage example — M1 single sequence."""

from mini_vllm import LLM, SamplingParams

llm = LLM(
    model_path="Qwen/Qwen3-0.6B",
    block_size=16,
    max_num_seqs=8,
    max_num_batched_tokens=2048,
    gpu_memory_utilization=0.5,
)

sp = SamplingParams(temperature=0.01, max_tokens=20)
outputs = llm.generate(prompts=["Hello, world."], sampling_params=sp)
for o in outputs:
    print(o["text"])
