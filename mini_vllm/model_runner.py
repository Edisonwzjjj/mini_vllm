"""Model runner — loads HF model, allocates KV cache, runs forward passes."""

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
        ).to(self.device)
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

    def run_prefill(self, seqs: Sequence):
        num_blocks_needed = self.block_manager.num_blocks_needed(seqs.num_prompt_tokens)
        cur_blocks = len(seqs.block_table[0])
        if cur_blocks < num_blocks_needed:
            new_blocks = self.block_manager.allocate(num_blocks_needed - cur_blocks)
            for layer in range(self.num_layers):
                seqs.block_table[layer].extend(new_blocks)
        block_tables = seqs.block_table[0]
        slot_mapping = self.block_manager.get_slot_mapping(block_tables, seqs.num_prompt_tokens)
        input_ids = torch.tensor([seqs.prompt_token_ids], device=self.device)
        position_ids = torch.arange(len(seqs.prompt_token_ids), device=self.device).unsqueeze(0)
        
        paged_ctx.kv_cache = self.kv_cache
        paged_ctx.block_table = block_tables        # List[int]
        paged_ctx.slot_mapping = slot_mapping        # List[int]
        paged_ctx.num_cached_tokens = seqs.num_prompt_tokens
        paged_ctx.is_prefill = True

        print(f"[PREFILL] prompt_len={seqs.num_prompt_tokens}, "
              f"block_table={block_tables}, slot_mapping={slot_mapping}")

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, position_ids=position_ids)
        seqs.num_cached_tokens += len(seqs.prompt_token_ids)
        return outputs.logits[:, -1, :]
        

    def run_decode(self, seqs: Sequence):
        total_tokens_after = seqs.num_cached_tokens + 1
        num_blocks_needed = self.block_manager.num_blocks_needed(total_tokens_after)
        cur_blocks = len(seqs.block_table[0])
        if cur_blocks < num_blocks_needed:
            new_blocks = self.block_manager.allocate(num_blocks_needed - cur_blocks)
            for layer in range(self.num_layers):
                seqs.block_table[layer].extend(new_blocks)
        
        new_token_pos = seqs.num_cached_tokens
        block_idx = new_token_pos // self.block_size
        offsets = new_token_pos % self.block_size
        block_id = seqs.block_table[0][block_idx] 
        slot = block_id * self.block_size + offsets
        
        slot_mapping = [slot]
        
        input_ids = torch.tensor([[seqs.last_token_id]], device=self.device)
        position_ids = torch.tensor([[seqs.num_tokens - 1]], device=self.device)
        
        paged_ctx.kv_cache = self.kv_cache
        paged_ctx.block_table = seqs.block_table[0]      # List[int]
        paged_ctx.slot_mapping = slot_mapping            # List[int], 长度=1
        paged_ctx.num_cached_tokens = total_tokens_after # 旧缓存 + 新写入的 1 个
        paged_ctx.is_prefill = False

        print(f"[DECODE] pos={seqs.num_cached_tokens}, "
              f"block_table={seqs.block_table[0]}, slot_mapping={slot_mapping}")

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, position_ids=position_ids)
        seqs.num_cached_tokens += 1
        return outputs.logits[:, -1, :]
        