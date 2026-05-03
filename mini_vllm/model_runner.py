"""Model runner — loads HF model, allocates KV cache, runs forward passes."""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mini_vllm.attention_patch import apply_attention_patch, paged_ctx
from mini_vllm.sequence import Sequence
from mini_vllm.block_manager import BlockManager

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
        # 1. Allocate blocks & build per-sequence input_ids / position_ids / slot_mapping
        all_input_ids = []
        all_position_ids = []
        all_slots = []
        seq_lengths = []

        for seq in seqs:
            cur_blocks = len(seq.block_table[0])
            num_blocks_needed = self.block_manager.num_blocks_needed(seq.num_prompt_tokens)
            if cur_blocks < num_blocks_needed:
                new_blocks = self.block_manager.allocate(num_blocks_needed - cur_blocks)
                for layer in range(self.num_layers):
                    seq.block_table[layer].extend(new_blocks)
            block_tables = seq.block_table[0]
            slot_mapping = self.block_manager.get_slot_mapping(block_tables, seq.num_prompt_tokens)

            all_input_ids.extend(seq.prompt_token_ids)
            all_position_ids.extend(range(seq.num_prompt_tokens))
            all_slots.extend(slot_mapping)
            seq_lengths.append(seq.num_prompt_tokens)

        total_len = sum(seq_lengths)

        input_ids = torch.tensor([all_input_ids], device=self.device)
        position_ids = torch.tensor([all_position_ids], device=self.device)

        # 2. Build block-diagonal causal mask
        mask = torch.zeros(total_len, total_len, device=self.device)
        start = 0
        for length in seq_lengths:
            end = start + length
            mask[start:end, start:end] = torch.tril(torch.ones(length, length, device=self.device))
            start = end
        mask = mask.masked_fill(mask == 0, float('-inf'))

        # 3. Set paged context
        paged_ctx.kv_cache = self.kv_cache
        paged_ctx.slot_mapping = all_slots      # flat list
        paged_ctx.num_cached_tokens = total_len
        paged_ctx.is_prefill = True
        paged_ctx.attn_mask = mask               # (total_len, total_len)

        print(f"[PREFILL] {len(seqs)} seqs, total_tokens={total_len}, "
              f"seq_lengths={seq_lengths}")
        if os.environ.get("MINI_VLLM_DEBUG"):
            print(f"  slot_mapping={all_slots}")

        # 4. Forward
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, position_ids=position_ids)

        # 5. Update num_cached_tokens AFTER forward
        for seq in seqs:
            seq.num_cached_tokens += seq.num_prompt_tokens

        # 6. Extract last-position logits for each sequence
        logits = outputs.logits  # (1, total_len, vocab_size)
        result_logits = []
        start = 0
        for length in seq_lengths:
            last_pos = start + length - 1
            result_logits.append(logits[0, last_pos, :])  # (vocab_size,)
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
                new_blocks = self.block_manager.allocate(num_blocks_needed - cur_blocks)
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
        