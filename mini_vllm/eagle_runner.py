"""Chain speculative decoding runner for M5 EAGLE groundwork."""

from dataclasses import dataclass
from typing import Callable

import torch

from .config import EngineConfig
from .model_runner import ModelRunner
from .sampling_params import SamplingParams
from .sequence import Sequence


@dataclass
class VerifyResult:
    logits: torch.Tensor
    old_num_cached: int
    target_num_cached: int


class EagleRunner:
    """Greedy chain speculative decoder.

    First M5 version: single sequence, dummy draft tokens, target-model verify.
    """

    def __init__(self, config: EngineConfig, model_runner: ModelRunner):
        self.draft_len = config.eagle_draft_len
        self.verify_greedy_only = config.eagle_verify_greedy_only
        self.model_runner = model_runner
        self.draft_token_fn: Callable[[Sequence], list[int]] | None = None

    def _draft_tokens(self, seq: Sequence) -> list[int]:
        if self.draft_token_fn is not None:
            return self.draft_token_fn(seq)
        return [seq.last_token_id] * self.draft_len

    def _trim_kv_blocks(self, seq: Sequence, keep_num_cached: int) -> None:
        needed_blocks = self.model_runner.block_manager.num_blocks_needed(keep_num_cached)
        for layer in range(self.model_runner.num_layers):
            extra_blocks = seq.block_table[layer][needed_blocks:]
            if extra_blocks:
                self.model_runner.block_manager.deallocate(extra_blocks)
                seq.block_table[layer] = seq.block_table[layer][:needed_blocks]

    def step(self, seq: Sequence, sampling_params: SamplingParams) -> list[dict]:
        if self.verify_greedy_only:
            assert sampling_params.temperature < 1e-6, (
                "EagleRunner only supports greedy decoding for verification"
            )

        remaining_tokens = sampling_params.max_tokens - seq.num_output_tokens
        if remaining_tokens <= 0:
            seq.mark_finished()
            return []

        draft_tokens = self._draft_tokens(seq)[: self.draft_len]
        if not draft_tokens:
            return []

        logits, old_num_cached, _ = self.model_runner.run_chain_verify(seq, draft_tokens)
        greedy_tokens = logits.argmax(dim=-1).tolist()

        accepted = 0
        for i, draft_token in enumerate(draft_tokens):
            if greedy_tokens[i] != draft_token:
                break
            accepted += 1

        if accepted == len(draft_tokens):
            tokens_to_append = draft_tokens + [greedy_tokens[len(draft_tokens)]]
        else:
            tokens_to_append = draft_tokens[:accepted] + [greedy_tokens[accepted]]

        tokens_to_append = tokens_to_append[:remaining_tokens]
        new_cached = old_num_cached + len(tokens_to_append)
        seq.num_cached_tokens = new_cached
        self._trim_kv_blocks(seq, new_cached)

        results = []
        for token_id in tokens_to_append:
            seq.output_token_ids.append(token_id)
            is_finished = (
                token_id == self.model_runner.eos_token_id
                or seq.num_output_tokens >= sampling_params.max_tokens
            )
            if is_finished:
                seq.mark_finished()

            results.append({
                "seq_id": seq.seq_id,
                "token_id": token_id,
                "text": self.model_runner.tokenizer.decode([token_id], skip_special_tokens=True),
                "finished": is_finished,
            })
            if is_finished:
                break

        return results
