"""LLMEngine — main entry point orchestrating scheduler + model runner + block manager."""

# M1 TODO: implement this module
#
# The engine loop (M1, single sequence):
#   while not all finished:
#       scheduler_output = scheduler.schedule(seqs)
#       if scheduler_output.is_prefill:
#           logits = model_runner.run_prefill(scheduler_output.seqs)
#       else:
#           logits = model_runner.run_decode(scheduler_output.seqs)
#       sample next token from logits
#       append token to sequence
#       update block_table if needed
#       check EOS / max_tokens
#
# Also provide a convenience LLM class that wraps LLMEngine.
