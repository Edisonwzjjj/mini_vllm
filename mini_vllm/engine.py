"""LLMEngine — main entry point orchestrating scheduler + model runner + block manager."""

import torch
from .scheduler import Scheduler
from .model_runner import ModelRunner
from .sequence import Sequence
from .sampling_params import SamplingParams
from .config import EngineConfig
from .eagle_runner import EagleRunner

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
            deterministic=config.deterministic,
        )
        self.scheduler = Scheduler(config.max_num_seqs, config.max_num_batched_tokens,
                                  self.model_runner.block_manager)
        self.spec_runner = EagleRunner(config, self.model_runner) if config.enable_eagle else None

    def _free_seq_resources(self, seq: Sequence) -> None:
        """Deallocate a sequence's blocks and clear its block_table."""
        for layer in range(self.model_runner.num_layers):
            self.model_runner.block_manager.deallocate(seq.block_table[layer])
            seq.block_table[layer] = []

    def _sample_and_append(self, seq, logits, sampling_params):
        """Sample a token, append to seq, return StepOutput dict."""
        token_id = sample_token(logits, sampling_params)
        seq.output_token_ids.append(token_id)

        is_finished = (
            token_id == self.model_runner.eos_token_id
            or seq.num_output_tokens >= sampling_params.max_tokens
        )
        if is_finished:
            seq.mark_finished()
            self.scheduler.postprocess(seq)
            self._free_seq_resources(seq)

        text = self.model_runner.tokenizer.decode([token_id], skip_special_tokens=True)
        return {
            "seq_id": seq.seq_id,
            "token_id": token_id,
            "text": text,
            "finished": is_finished,
        }

    def step(self, sampling_params: SamplingParams) -> list[dict]:
        """One iteration: schedule → forward → sample → append.

        Returns list of {seq_id, token_id, text, finished} for each token
        produced this step. Empty list if no work to do.
        """
        scheduler_output = self.scheduler.schedule()
        if not scheduler_output.seqs:
            return []

        seqs = scheduler_output.seqs
        results = []

        if scheduler_output.is_prefill:
            logits_list = self.model_runner.run_prefill(seqs)
            for seq, logits in zip(seqs, logits_list):
                if not seq.is_prefill_finished:
                    continue
                results.append(self._sample_and_append(seq, logits, sampling_params))
        else:
            if (
                self.spec_runner is not None
                and self.spec_runner.draft_len > 0
                and len(seqs) == 1  # spec_runner only supports batch_size=1 for now
            ):
                seq = seqs[0]
                results.extend(self.spec_runner.step(seq, sampling_params))
                if seq.is_finished():
                    self.scheduler.postprocess(seq)
                    self._free_seq_resources(seq)
            else:
                logits_list = self.model_runner.run_decode(seqs)
                if logits_list is None:
                    victim = self.scheduler.preempt_one()
                    if victim is None:
                        raise RuntimeError("OOM: no sequence to preempt")
                    self._free_seq_resources(victim)
                    print(f"[PREEMPT] seq_id={victim.seq_id}, released blocks")
                    return []
                for seq, logits in zip(seqs, logits_list):
                    results.append(self._sample_and_append(seq, logits, sampling_params))

        return results

    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[dict]:
        """Tokenize prompts → run engine loop → return results."""
        seqs = self._create_sequences(prompts)
        self.scheduler.add_seqs(seqs)
        step_count = 0
        max_steps = (sampling_params.max_tokens + 50) * len(seqs) + 100
        
        while not all(seq.is_finished() for seq in seqs):
            self.step(sampling_params)
            step_count += 1
            if step_count > max_steps:
                unfinished = [s for s in seqs if not s.is_finished()]
                raise RuntimeError(f"Engine stuck after {step_count} steps (max={max_steps}). "
                f"Unfinished: " + ", ".join(
                f"id={s.seq_id} out={s.num_output_tokens} "
                f"prefill_done={s.is_prefill_finished} status={s.status}"
                for s in unfinished))

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

    def generate_stream(self, prompts: list[str], sampling_params: SamplingParams):
        """Tokenize prompts → run engine loop → yield each token as it's produced.

        Yields {seq_id, token_id, text, finished} dicts, one per token per sequence.
        """
        seqs = self._create_sequences(prompts)
        self.scheduler.add_seqs(seqs)

        while not all(seq.is_finished() for seq in seqs):
            step_results = self.step(sampling_params)
            for r in step_results:
                yield r

    def _create_sequences(self, prompts: list[str]) -> list[Sequence]:
        """Tokenize prompts and create Sequence objects."""
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
        return seqs


class LLM:
    """Convenience wrapper — matches the public API in REQUIREMENTS.md."""

    def __init__(self, model_path: str, block_size: int = 16,
                 max_num_seqs: int = 8, max_num_batched_tokens: int = 2048,
                 gpu_memory_utilization: float = 0.5,
                 enable_eagle: bool = False,
                 eagle_draft_len: int = 4,
                 eagle_mode: str = "chain",
                 eagle_topk: int = 2,
                 eagle_spec_steps: int = 3,
                 eagle_use_pld: bool = False,
                 eagle_pld_max_ngram: int = 3,
                 eagle_pld_min_ngram: int = 2,
                 deterministic: bool = True):
        config = EngineConfig(
            model_path=model_path,
            block_size=block_size,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_eagle=enable_eagle,
            eagle_draft_len=eagle_draft_len,
            eagle_mode=eagle_mode,
            eagle_topk=eagle_topk,
            eagle_spec_steps=eagle_spec_steps,
            eagle_use_pld=eagle_use_pld,
            eagle_pld_max_ngram=eagle_pld_max_ngram,
            eagle_pld_min_ngram=eagle_pld_min_ngram,
            deterministic=deterministic,
        )
        self.engine = LLMEngine(config)

    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[dict]:
        return self.engine.generate(prompts, sampling_params)

    def generate_stream(self, prompts: list[str], sampling_params: SamplingParams):
        """Yield each token as it's produced. Usage:

            for chunk in llm.generate_stream(["Hello"], sp):
                print(chunk["text"], end="", flush=True)
        """
        return self.engine.generate_stream(prompts, sampling_params)
