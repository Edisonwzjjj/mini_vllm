"""Chain speculative decoding runner for M5 EAGLE groundwork."""

from dataclasses import dataclass
from typing import Callable

import torch

from .config import EngineConfig
from .draft_tree import DraftTree, build_dummy_tree
from .model_runner import ModelRunner
from .sampling_params import SamplingParams
from .sequence import Sequence
from .draft_pld import prompt_lookup_draft

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
        self.mode = config.eagle_mode
        self.topk = config.eagle_topk
        self.spec_steps = config.eagle_spec_steps
        self.model_runner = model_runner
        self.draft_token_fn: Callable[[Sequence], list[int]] | None = None
        self.draft_tree_fn: Callable[[Sequence], DraftTree] | None = None
        self.use_pld_default = config.eagle_use_pld
        self.pld_max_ngram = config.eagle_pld_max_ngram
        self.pld_min_ngram = config.eagle_pld_min_ngram
        self.metrics: dict[str, int] = {
            "spec_steps": 0,
            "draft_tokens_proposed": 0,
            "draft_tokens_accepted": 0,
            "bonus_tokens": 0,
            "fallback_steps": 0,
        }

    def reset_metrics(self) -> None:
        for k in self.metrics:
            self.metrics[k] = 0

    @property
    def acceptance_rate(self) -> float:
        proposed = self.metrics["draft_tokens_proposed"]
        if proposed == 0:
            return 0.0
        return self.metrics["draft_tokens_accepted"] / proposed

    def _draft_tokens(self, seq: Sequence) -> list[int]:
        if self.draft_token_fn is not None:
            return self.draft_token_fn(seq)
        if self.use_pld_default:
            return prompt_lookup_draft(
                token_ids=seq.token_ids,
                draft_len=self.draft_len,
                max_ngram=self.pld_max_ngram,
                min_ngram=self.pld_min_ngram
            )
        return [seq.last_token_id] * self.draft_len

    def _trim_kv_blocks(self, seq: Sequence, keep_num_cached: int) -> None:
        needed_blocks = self.model_runner.block_manager.num_blocks_needed(keep_num_cached)
        for layer in range(self.model_runner.num_layers):
            extra_blocks = seq.block_table[layer][needed_blocks:]
            if extra_blocks:
                self.model_runner.block_manager.deallocate(extra_blocks)
                seq.block_table[layer] = seq.block_table[layer][:needed_blocks]

    def _append_tokens(
        self,
        seq: Sequence,
        tokens_to_append: list[int],
        old_num_cached: int,
        sampling_params: SamplingParams,
    ) -> list[dict]:
        remaining_tokens = sampling_params.max_tokens - seq.num_output_tokens
        tokens_to_append = tokens_to_append[:remaining_tokens]

        results = []
        appended = 0
        for token_id in tokens_to_append:
            seq.output_token_ids.append(token_id)
            appended += 1
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

        new_cached = old_num_cached + appended
        seq.num_cached_tokens = new_cached
        self._trim_kv_blocks(seq, new_cached)
        return results

    def _check_greedy(self, sampling_params: SamplingParams) -> None:
        if self.verify_greedy_only:
            assert sampling_params.temperature < 1e-6, (
                "EagleRunner only supports greedy decoding for verification"
            )

    def step(self, seq: Sequence, sampling_params: SamplingParams) -> list[dict]:
        self._check_greedy(sampling_params)
        if self.mode == "tree":
            return self._tree_step(seq, sampling_params)
        return self._chain_step(seq, sampling_params)

    def _chain_step(self, seq: Sequence, sampling_params: SamplingParams) -> list[dict]:
        remaining_tokens = sampling_params.max_tokens - seq.num_output_tokens
        if remaining_tokens <= 0:
            seq.mark_finished()
            return []

        draft_tokens = self._draft_tokens(seq)[: self.draft_len]
        if not draft_tokens:
            return self._fallback_decode_step(seq, sampling_params)

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

        self.metrics["spec_steps"] += 1
        self.metrics["draft_tokens_proposed"] += len(draft_tokens)
        self.metrics["draft_tokens_accepted"] += accepted
        if accepted == len(draft_tokens):
            self.metrics["bonus_tokens"] += 1

        return self._append_tokens(seq, tokens_to_append, old_num_cached, sampling_params)

    def _fallback_decode_step(self, seq: Sequence, sampling_params: SamplingParams) -> list[dict]:
        """Decode one token via the normal path when PLD finds no draft.

        Note: model_runner.run_decode() advances seq.num_cached_tokens by 1
        internally (it caches seq.last_token_id's KV). We then append the
        sampled token, which restores the chain-spec invariant
        num_cached_tokens == num_tokens - 1.
        """
        self.metrics["fallback_steps"] += 1

        logits = self.model_runner.run_decode([seq])[0]
        greedy_token = int(logits.argmax(dim=-1).item())

        # run_decode already set num_cached_tokens = num_tokens (KV for the
        # previous last_token is now committed). After we append greedy_token,
        # num_tokens grows by 1, so num_cached == num_tokens - 1 again.
        # We bypass _append_tokens because it would overwrite num_cached_tokens.
        remaining = sampling_params.max_tokens - seq.num_output_tokens
        if remaining <= 0:
            return []

        seq.output_token_ids.append(greedy_token)
        is_finished = (
            greedy_token == self.model_runner.eos_token_id
            or seq.num_output_tokens >= sampling_params.max_tokens
        )
        if is_finished:
            seq.mark_finished()

        return [{
            "seq_id": seq.seq_id,
            "token_id": greedy_token,
            "text": self.model_runner.tokenizer.decode([greedy_token], skip_special_tokens=True),
            "finished": is_finished,
        }]
        
        
    def _is_tree_node_accepted(
        self,
        tree: DraftTree,
        greedy_tokens: list[int],
        node_idx: int,
    ) -> bool:
        parent_idx = tree.parent_indices[node_idx]
        pred_pos = 0 if parent_idx == -1 else 1 + parent_idx
        return greedy_tokens[pred_pos] == tree.token_ids[node_idx]

    def _accepted_leftmost_path(
        self,
        tree: DraftTree,
        greedy_tokens: list[int],
    ) -> list[int]:
        path: list[int] = []
        parent_idx = -1
        while True:
            children = tree.children(parent_idx)
            if not children:
                break

            # First version only commits the leftmost child. This keeps the
            # accepted path a contiguous DFS-prefix in the KV cache.
            child_idx = children[0]
            if not self._is_tree_node_accepted(tree, greedy_tokens, child_idx):
                break

            path.append(child_idx)
            parent_idx = child_idx
        return path

    def _tree_step(self, seq: Sequence, sampling_params: SamplingParams) -> list[dict]:
        remaining_tokens = sampling_params.max_tokens - seq.num_output_tokens
        if remaining_tokens <= 0:
            seq.mark_finished()
            return []

        tree = self._draft_tree(seq)
        logits, old_num_cached, _ = self.model_runner.run_tree_verify(seq, tree)
        greedy_tokens = logits.argmax(dim=-1).tolist()

        accepted_path = self._accepted_leftmost_path(tree, greedy_tokens)
        accepted_tokens = [tree.token_ids[i] for i in accepted_path]
        pred_pos = 0 if not accepted_path else 1 + accepted_path[-1]
        tokens_to_append = accepted_tokens + [greedy_tokens[pred_pos]]

        self.metrics["spec_steps"] += 1
        # For tree, "proposed" counts the leftmost-path nodes we actually
        # consider as candidates for commit (matching the chain-style metric).
        leftmost_chain_len = self._leftmost_chain_len(tree)
        self.metrics["draft_tokens_proposed"] += leftmost_chain_len
        self.metrics["draft_tokens_accepted"] += len(accepted_path)
        if len(accepted_path) == leftmost_chain_len and leftmost_chain_len > 0:
            self.metrics["bonus_tokens"] += 1

        return self._append_tokens(seq, tokens_to_append, old_num_cached, sampling_params)

    def _leftmost_chain_len(self, tree: DraftTree) -> int:
        length = 0
        parent_idx = -1
        while True:
            children = tree.children(parent_idx)
            if not children:
                break
            length += 1
            parent_idx = children[0]
        return length

    def _draft_tree(self, seq: Sequence) -> DraftTree:
        if self.draft_tree_fn is not None:
            return self.draft_tree_fn(seq)
        return build_dummy_tree(root_token_id=seq.last_token_id, topk=self.topk, depth=self.spec_steps)