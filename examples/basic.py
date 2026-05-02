"""Basic usage example — will work after M1 is complete."""

# This is the target API from REQUIREMENTS.md.
# After M1, uncomment and run:

# from mini_vllm import LLM, SamplingParams

# llm = LLM(
#     model_path="~/huggingface/Qwen3-0.6B/",
#     block_size=16,
#     max_num_seqs=8,
#     max_num_batched_tokens=2048,
#     gpu_memory_utilization=0.5,
# )

# sp = SamplingParams(temperature=0.01, max_tokens=20)
# outputs = llm.generate(prompts=["Hello, world."], sampling_params=sp)
# for o in outputs:
#     print(o["text"])
import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = 'Qwen/Qwen3-0.6B'
cache = '/Users/zijunwang/huggingface'
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache)
model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir=cache, dtype=torch.float32).to('mps')
input = tokenizer("Hello, world.", return_tensors="pt").to('mps')
output = model.generate(**input, max_new_tokens=20, do_sample=False)
print(tokenizer.decode(output[0]))