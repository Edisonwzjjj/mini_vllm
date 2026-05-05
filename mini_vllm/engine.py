"""LLMEngine — main entry point orchestrating scheduler + model runner + block manager."""

import torch
from .scheduler import Scheduler
from .model_runner import ModelRunner
from .sequence import Sequence
from .sampling_params import SamplingParams
from .config import EngineConfig


def sample_token(logits: torch.Tensor, sampling_params: SamplingParams) -> int:
    """Sample a single token from logits.

    logits: (1, vocab_size) or (vocab_size,)
    """
    if logits.dim() == 2:
        logits = logits[0]  # (vocab_size,)

    temperature = sampling_params.temperature
    if temperature < 1e-6:
        return logits.argmax().item()

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


class LLMEngine:
    """Engine loop: schedule → forward → sample → append → check done."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.model_runner = ModelRunner(
            model_path=config.model_path,
            block_size=config.block_size,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            gpu_memory_utilization=config.gpu_memory_utilization,
        )
        self.scheduler = Scheduler(config.max_num_seqs, config.max_num_batched_tokens)

    def _free_seq_resources(self, seq: Sequence) -> None:
        """Deallocate a finished sequence's blocks."""
        for layer in range(self.model_runner.num_layers):
            self.model_runner.block_manager.deallocate(seq.block_table[layer])

    def step(self, sampling_params: SamplingParams):
        """One iteration: schedule → forward → sample → append → check done."""
        scheduler_output = self.scheduler.schedule()
        if not scheduler_output.seqs:
            return []  # all finished

        seqs = scheduler_output.seqs
        finished = []

        if scheduler_output.is_prefill:
            # Batched prefill: returns one logits per sequence
            logits_list = self.model_runner.run_prefill(seqs)
            for seq, logits in zip(seqs, logits_list):
                next_token = sample_token(logits, sampling_params)
                seq.output_token_ids.append(next_token)
                if (next_token == self.model_runner.eos_token_id
                        or seq.num_output_tokens >= sampling_params.max_tokens):
                    seq.mark_finished()
                    finished.append(seq)
                    self.scheduler.postprocess(seq)
                    self._free_seq_resources(seq)
        else:
            # Batched decode: returns one logits per sequence
            logits_list = self.model_runner.run_decode(seqs)
            for seq, logits in zip(seqs, logits_list):
                next_token = sample_token(logits, sampling_params)
                seq.output_token_ids.append(next_token)
                if (next_token == self.model_runner.eos_token_id
                        or seq.num_output_tokens >= sampling_params.max_tokens):
                    seq.mark_finished()
                    finished.append(seq)
                    self.scheduler.postprocess(seq)
                    self._free_seq_resources(seq)

        return finished

    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[dict]:
        """Tokenize prompts → run engine loop → return results."""
        # 1. Tokenize and create Sequence objects
        seqs = []
        for i, prompt in enumerate(prompts):
            prompt_token_ids = self.model_runner.tokenizer.encode(prompt)
            seq = Sequence(
                seq_id=i,
                prompt_token_ids=prompt_token_ids,
                block_size=self.model_runner.block_size,
                num_layers=self.model_runner.num_layers,
            )
            seqs.append(seq)

        # 2. Add all sequences to the scheduler ONCE
        self.scheduler.add_seqs(seqs)

        # 3. Main loop until all sequences are finished
        while not all(seq.is_finished() for seq in seqs):
            self.step(sampling_params)

        # 3. Convert output token IDs back to text
        outputs = []
        for seq in seqs:
            output_text = self.model_runner.tokenizer.decode(
                seq.output_token_ids, skip_special_tokens=True
            )
            outputs.append({
                "text": output_text,
                "token_ids": seq.output_token_ids,
            })
        return outputs


class LLM:
    """Convenience wrapper — matches the public API in REQUIREMENTS.md."""

    def __init__(self, model_path: str, block_size: int = 16,
                 max_num_seqs: int = 8, max_num_batched_tokens: int = 2048,
                 gpu_memory_utilization: float = 0.5):
        config = EngineConfig(
            model_path=model_path,
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self.engine = LLMEngine(config)

    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[dict]:
        return self.engine.generate(prompts, sampling_params)
