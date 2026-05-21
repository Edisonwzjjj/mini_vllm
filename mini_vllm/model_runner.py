"""Model runner — loads HF model, allocates KV cache, runs forward passes."""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mini_vllm.attention_patch import apply_attention_patch, paged_ctx
from mini_vllm.sequence import Sequence
from mini_vllm.block_manager import BlockManager
from mini_vllm.draft_tree import DraftTree
class ModelRunner:
    def __init__(self, model_path: str, block_size: int, max_num_seqs: int,
                 max_num_batched_tokens: int, gpu_memory_utilization: float):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.float32
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.gpu_memory_utilization = gpu_memory_utilization

        # Read model dimensions from config
        cfg = self.model.config
        self.num_layers = cfg.num_hidden_layers
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.eos_token_id = cfg.eos_token_id
        
        self.kv_cache = self.allocate_kv_cache()
        num_blocks = self.kv_cache.shape[1]
        self.block_manager = BlockManager(self.block_size, num_blocks)
        apply_attention_patch(self.model)


    def allocate_kv_cache(self):
        """Allocate per-layer paged KV cache tensors.

        Shape per layer: (num_blocks, 2, num_kv_heads, block_size, head_dim)
          - 2 = K and V
          - Each layer has its own block pool and block_table.
        On MPS we can't query available VRAM precisely, so use a fixed pool size.
        """
        # Each block (per layer) stores: 2 * num_kv_heads * block_size * head_dim floats
        bytes_per_block = (
            2 * self.num_kv_heads * self.block_size * self.head_dim * 4
        )
        # Rough budget: ~2 GB for KV cache on a typical Mac
        kv_budget_bytes = 2 * 1024**3
        num_blocks = kv_budget_bytes // (bytes_per_block * self.num_layers)

        # One tensor per layer
        kv_cache = torch.zeros(
            self.num_layers, num_blocks, 2,
            self.num_kv_heads, self.block_size, self.head_dim,
            dtype=torch.float32, device=self.device,
        )
        total_gb = kv_cache.nelement() * 4 / 1024**3
        print(f"KV cache: {num_blocks} blocks/layer × {self.num_layers} layers, {total_gb:.2f} GB")
        return kv_cache

    def run_prefill(self, seqs: list[Sequence]):
        # 1. Allocate blocks with prefix cache & build suffix-only inputs
        all_input_ids = []
        all_position_ids = []
        all_slots = []
        suffix_lengths = []
        prefix_lengths = []
        prefill_seqs = []  # sequences that need suffix prefill

        for seq in seqs:
            chunk_start = seq.num_prefill_tokens
            # chunk_end: 这轮最多 prefill 到哪
            # TODO: 多 seq 时按剩余 budget 分配，目前先按 max_num_batched_tokens
            chunk_end = min(seq.num_prompt_tokens, chunk_start + self.max_num_batched_tokens)

            if chunk_start == 0:
                # 第一轮 chunk: 可以命中跨序列 prefix cache
                bt_layer, cached_tokens = self.block_manager.allocate_with_prefix(
                    seq.prompt_token_ids[:chunk_end], seq.block_table[0]
                )
                for layer in range(1, self.num_layers):
                    seq.block_table[layer] = list(bt_layer)
                seq.block_table[0] = bt_layer
                prefix_len = cached_tokens
            else:
                # 续接 chunk: block_table 里已有之前的 block，只需追加新 block
                num_blocks_needed = self.block_manager.num_blocks_needed(chunk_end)
                cur_blocks = len(seq.block_table[0])
                if cur_blocks < num_blocks_needed:
                    new_blocks = self.block_manager.allocate(num_blocks_needed - cur_blocks)
                    for layer in range(self.num_layers):
                        seq.block_table[layer].extend(new_blocks)
                prefix_len = chunk_start  # 之前的 token 都已有 KV，等价于 prefix

            suffix_len = chunk_end - max(prefix_len, chunk_start)

            # When suffix_len=0 (all tokens cached), we still need logits at
            # the last position. Force at least 1 suffix token through prefill
            # so it's computed correctly (decode path can't handle no output tokens).
            if suffix_len == 0 and prefix_len > 0:
                prefix_len -= 1
                suffix_len = 1

            prefill_seqs.append(seq)

            suffix_token_ids = seq.prompt_token_ids[max(prefix_len, chunk_start):chunk_end]
            suffix_positions = list(range(max(prefix_len, chunk_start), chunk_end))

            full_slot_mapping = self.block_manager.get_slot_mapping(
                seq.block_table[0], chunk_end
            )
            suffix_slot_mapping = full_slot_mapping[max(prefix_len, chunk_start):chunk_end]

            all_input_ids.extend(suffix_token_ids)
            all_position_ids.extend(suffix_positions)
            all_slots.extend(suffix_slot_mapping)
            suffix_lengths.append(suffix_len)
            prefix_lengths.append(prefix_len)

        total_suffix_len = sum(suffix_lengths)

        input_ids = torch.tensor([all_input_ids], device=self.device)
        position_ids = torch.tensor([all_position_ids], device=self.device)

        # 2. Build mask for suffix prefill
        # Each suffix token attends to: all prefix tokens + causal within suffix
        # Mask shape: (total_suffix_len, sum(prefix_len + suffix_len))
        total_cols = sum(pl + sl for pl, sl in zip(prefix_lengths, suffix_lengths))
        mask = torch.zeros(total_suffix_len, total_cols, device=self.device)
        row_offset = 0
        col_offset = 0
        for prefix_len, suffix_len in zip(prefix_lengths, suffix_lengths):
            # Suffix rows → all prefix cols (fully visible)
            mask[row_offset:row_offset+suffix_len, col_offset:col_offset+prefix_len] = 1.0
            # Suffix rows → suffix cols (causal)
            mask[row_offset:row_offset+suffix_len,
                 col_offset+prefix_len:col_offset+prefix_len+suffix_len] = torch.tril(
                torch.ones(suffix_len, suffix_len, device=self.device)
            )
            row_offset += suffix_len
            col_offset += prefix_len + suffix_len
        mask = mask.masked_fill(mask == 0, float('-inf'))

        # 3. Set paged context
        paged_ctx.kv_cache = self.kv_cache
        paged_ctx.slot_mapping = all_slots
        paged_ctx.is_prefill = True
        paged_ctx.attn_mask = mask
        # New fields for suffix prefill
        paged_ctx.prefix_lengths = prefix_lengths
        paged_ctx.suffix_lengths = suffix_lengths
        paged_ctx.block_tables = [seq.block_table[0] for seq in prefill_seqs]

        print(f"[PREFILL] {len(prefill_seqs)} seqs, total_suffix_tokens={total_suffix_len}, "
              f"suffix_lengths={suffix_lengths}, prefix_lengths={prefix_lengths}")
        if os.environ.get("MINI_VLLM_DEBUG"):
            print(f"  slot_mapping={all_slots}")

        # 4. Forward
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, position_ids=position_ids)

        # 5. Update num_cached_tokens & register hashes AFTER forward
        for seq in prefill_seqs:
            chunk_end = min(seq.num_prompt_tokens, seq.num_prefill_tokens + self.max_num_batched_tokens)
            seq.num_cached_tokens = chunk_end
            seq.num_prefill_tokens = chunk_end
            # Only insert full blocks that have been computed so far
            self.block_manager.insert_blocks(seq.prompt_token_ids[:chunk_end], seq.block_table[0])

        # 6. Extract last-position logits for each prefill sequence
        logits = outputs.logits  # (1, total_suffix_len, vocab_size)
        result_logits = []
        start = 0
        for length in suffix_lengths:
            last_pos = start + length - 1
            result_logits.append(logits[0, last_pos, :])
            start += length

        return result_logits
        

    def run_decode(self, seqs: list[Sequence]):
        # 1. Allocate blocks & compute per-sequence slot
        decode_infos = []
        for seq in seqs:
            total_tokens_after = seq.num_cached_tokens + 1
            num_blocks_needed = self.block_manager.num_blocks_needed(total_tokens_after)
            cur_blocks = len(seq.block_table[0])
            if cur_blocks < num_blocks_needed:
                new_blocks = self.block_manager.try_allocate(num_blocks_needed - cur_blocks)
                if new_blocks is None:
                    return []  # signal OOM to caller for potential preemption
                for layer in range(self.num_layers):
                    seq.block_table[layer].extend(new_blocks)

            new_token_pos = seq.num_cached_tokens
            block_idx = new_token_pos // self.block_size
            offset = new_token_pos % self.block_size
            block_id = seq.block_table[0][block_idx]
            slot = block_id * self.block_size + offset

            decode_infos.append({
                "slot": slot,
                "block_table": seq.block_table[0],
                "num_cached_after": total_tokens_after,
            })

        # 2. Build batch input
        input_ids = torch.tensor(
            [[seq.last_token_id] for seq in seqs], device=self.device
        )  # (batch_size, 1)
        position_ids = torch.tensor(
            [[seq.num_tokens - 1] for seq in seqs], device=self.device
        )  # (batch_size, 1)

        # 3. Set paged context
        paged_ctx.kv_cache = self.kv_cache
        paged_ctx.slot_mapping = [info["slot"] for info in decode_infos]
        paged_ctx.block_tables = [info["block_table"] for info in decode_infos]
        paged_ctx.num_cached_after = [info["num_cached_after"] for info in decode_infos]
        paged_ctx.is_prefill = False
        # Clear prefix cache fields to avoid stale data
        paged_ctx.prefix_lengths = None
        paged_ctx.suffix_lengths = None

        if os.environ.get("MINI_VLLM_DEBUG"):
            print(f"[DECODE] {len(seqs)} seqs")

        # 4. Forward
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, position_ids=position_ids)

        # 5. Update num_cached_tokens & extract logits
        logits_list = []
        for i, seq in enumerate(seqs):
            seq.num_cached_tokens += 1
            logits_list.append(outputs.logits[i, -1, :])  # (vocab_size,)

        return logits_list


    def run_tree_verify(self, seq: Sequence, draft_tree: DraftTree) -> tuple[torch.Tensor, int, int]:
        """Run target-model verification for tree speculative decoding.

        The input suffix is [last_token] + draft tree tokens. This writes KV for
        the suffix with tree attention and leaves commit or rollback of
        seq.num_cached_tokens to EagleRunner.
        """
        verify_token_ids = [seq.last_token_id] + draft_tree.token_ids
        old_num_cached = seq.num_cached_tokens
        target_num_cached = old_num_cached + len(verify_token_ids)

        num_blocks_needed = self.block_manager.num_blocks_needed(target_num_cached)
        cur_blocks = len(seq.block_table[0])
        if cur_blocks < num_blocks_needed:
            new_blocks = self.block_manager.try_allocate(num_blocks_needed - cur_blocks)
            if new_blocks is None:
                raise RuntimeError("OOM during tree speculative verification")
            for layer in range(self.num_layers):
                seq.block_table[layer].extend(new_blocks)

        full_slot_mapping = self.block_manager.get_slot_mapping(
            seq.block_table[0], target_num_cached
        )
        suffix_slot_mapping = full_slot_mapping[old_num_cached:target_num_cached]

        input_ids = torch.tensor([verify_token_ids], device=self.device)
        position_ids = torch.tensor(
            [list(range(old_num_cached, target_num_cached))], device=self.device
        )

        paged_ctx.kv_cache = self.kv_cache
        paged_ctx.slot_mapping = suffix_slot_mapping
        paged_ctx.is_prefill = True
        paged_ctx.is_tree_verify = True
        paged_ctx.attn_mask = None
        paged_ctx.prefix_lengths = [old_num_cached]
        paged_ctx.suffix_lengths = [len(verify_token_ids)]
        paged_ctx.block_tables = [seq.block_table[0]]
        paged_ctx.tree_mask = draft_tree.build_tree_mask(torch.device(self.device))

        try:
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, position_ids=position_ids)
        finally:
            paged_ctx.is_tree_verify = False

        return outputs.logits[0], old_num_cached, target_num_cached


    def run_chain_verify(self, seq: Sequence, draft_token_ids: list[int]) -> tuple[torch.Tensor, int, int]:
        """Run target-model verification for chain speculative decoding.

        The input suffix is [last_token] + draft tokens. This writes KV for the
        suffix, returns logits for each suffix position, and leaves commit or
        rollback of seq.num_cached_tokens to EagleRunner.
        """
        verify_token_ids = [seq.last_token_id] + draft_token_ids
        old_num_cached = seq.num_cached_tokens
        target_num_cached = old_num_cached + len(verify_token_ids)

        num_blocks_needed = self.block_manager.num_blocks_needed(target_num_cached)
        cur_blocks = len(seq.block_table[0])
        if cur_blocks < num_blocks_needed:
            new_blocks = self.block_manager.try_allocate(num_blocks_needed - cur_blocks)
            if new_blocks is None:
                raise RuntimeError("OOM during chain speculative verification")
            for layer in range(self.num_layers):
                seq.block_table[layer].extend(new_blocks)

        full_slot_mapping = self.block_manager.get_slot_mapping(
            seq.block_table[0], target_num_cached
        )
        suffix_slot_mapping = full_slot_mapping[old_num_cached:target_num_cached]

        input_ids = torch.tensor([verify_token_ids], device=self.device)
        position_ids = torch.tensor(
            [list(range(old_num_cached, target_num_cached))], device=self.device
        )

        paged_ctx.kv_cache = self.kv_cache
        paged_ctx.slot_mapping = suffix_slot_mapping
        paged_ctx.is_prefill = True
        paged_ctx.attn_mask = None
        paged_ctx.prefix_lengths = [old_num_cached]
        paged_ctx.suffix_lengths = [len(verify_token_ids)]
        paged_ctx.block_tables = [seq.block_table[0]]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, position_ids=position_ids)

        return outputs.logits[0], old_num_cached, target_num_cached
        