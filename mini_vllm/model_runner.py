"""Model runner — loads HF model, allocates KV cache, runs forward passes."""

from __future__ import annotations

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mini_vllm.attention_patch import apply_attention_patch, paged_ctx
from mini_vllm.sequence import Sequence
from mini_vllm.block_manager import BlockManager
from mini_vllm.draft_tree import DraftTree
from mini_vllm.kv_quant import calibrate_scale


class ModelRunner:
    def __init__(self, model_path: str, block_size: int, max_num_seqs: int,
                 max_num_batched_tokens: int, gpu_memory_utilization: float,
                 deterministic: bool = True, kv_cache_dtype: str = "auto",
                 kv_scale: float | None = None, kv_scale_calib_tokens: int = 4096):
        self.device = ("cuda" if torch.cuda.is_available() else "mps" )
        self.deterministic = deterministic
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_scale = float(kv_scale) if kv_scale is not None else None
        self.kv_scale_calib_tokens = kv_scale_calib_tokens
        self._validate_kv_config()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.compute_dtype = next(self.model.parameters()).dtype
        self.block_size = block_size
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.gpu_memory_utilization = gpu_memory_utilization

        # Read model dimensions from config
        cfg = self.model.config
        self.num_layers = self._required_config_int(cfg, "num_hidden_layers")
        self.num_heads = self._required_config_int(cfg, "num_attention_heads")
        self.num_kv_heads = self._resolve_num_kv_heads(cfg)
        self.head_dim = self._resolve_head_dim(cfg)
        self.eos_token_id = cfg.eos_token_id
        self.eos_token_ids = self._normalize_eos_token_ids(cfg.eos_token_id)
        self._validate_model_layout()
        self._validate_kv_config()
        if self.kv_cache_dtype == "fp8_e4m3" and self.kv_scale is None:
            self.kv_scale = self._calibrate_kv_scale()
        
        self.kv_cache = self.allocate_kv_cache()
        num_blocks = self.kv_cache.shape[1]
        self.block_manager = BlockManager(self.block_size, num_blocks)
        self._init_decode_buffers()
        apply_attention_patch(self.model)

    @staticmethod
    def _required_config_int(cfg, name: str) -> int:
        value = getattr(cfg, name, None)
        if value is None:
            raise ValueError(f"Model config is missing required field {name!r}")
        return int(value)

    def _resolve_num_kv_heads(self, cfg) -> int:
        value = getattr(cfg, "num_key_value_heads", None)
        if value is None:
            value = getattr(cfg, "num_attention_heads", None)
        if value is None:
            raise ValueError("Model config is missing num_key_value_heads/num_attention_heads")
        return int(value)

    def _resolve_head_dim(self, cfg) -> int:
        value = getattr(cfg, "head_dim", None)
        if value is not None:
            return int(value)
        hidden_size = getattr(cfg, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Model config is missing head_dim and hidden_size")
        hidden_size = int(hidden_size)
        if hidden_size % self.num_heads != 0:
            raise ValueError(
                f"Cannot infer head_dim: hidden_size={hidden_size} "
                f"is not divisible by num_attention_heads={self.num_heads}"
            )
        return hidden_size // self.num_heads

    @staticmethod
    def _normalize_eos_token_ids(eos_token_id) -> set[int]:
        if eos_token_id is None:
            return set()
        if isinstance(eos_token_id, (list, tuple, set)):
            return {int(token_id) for token_id in eos_token_id}
        return {int(eos_token_id)}

    def is_eos(self, token_id: int) -> bool:
        return int(token_id) in self.eos_token_ids

    def _validate_model_layout(self) -> None:
        if not hasattr(self.model, "model") or not hasattr(self.model.model, "layers"):
            raise ValueError(
                "mini-vllm currently expects HuggingFace causal LM models with "
                "`model.layers` and `layer.self_attn`, such as Qwen3/Qwen3-Coder."
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_kv_heads})"
            )
        missing = []
        first_attn = self.model.model.layers[0].self_attn
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            if not hasattr(first_attn, name):
                missing.append(name)
        if missing:
            raise ValueError(
                "mini-vllm attention patch only supports Qwen-style attention; "
                f"missing fields on first self_attn: {missing}"
            )

    def _validate_kv_config(self) -> None:
        valid = {"auto", "fp32", "bf16", "fp8_e4m3"}
        if self.kv_cache_dtype not in valid:
            raise ValueError(f"Unsupported kv_cache_dtype: {self.kv_cache_dtype}")
        if self.kv_scale is not None and self.kv_scale <= 0:
            raise ValueError("kv_scale must be positive")
        if self.kv_scale_calib_tokens <= 0:
            raise ValueError("kv_scale_calib_tokens must be positive")
        if self.kv_cache_dtype == "fp8_e4m3":
            if self.deterministic:
                raise ValueError("fp8_e4m3 KV requires deterministic=False")
            if self.device != "cuda":
                raise ValueError("fp8_e4m3 KV requires CUDA")
            if not hasattr(torch, "float8_e4m3fn"):
                raise ValueError("fp8_e4m3 KV requires torch.float8_e4m3fn support")

    def _resolve_kv_dtype(self) -> torch.dtype:
        if self.kv_cache_dtype == "fp8_e4m3":
            if self.kv_scale is None:
                raise ValueError("fp8_e4m3 KV requires kv_scale or successful calibration")
            return torch.float8_e4m3fn
        if self.kv_cache_dtype == "bf16":
            return torch.bfloat16
        if self.kv_cache_dtype in {"auto", "fp32"}:
            return torch.float32
        raise ValueError(f"Unsupported kv_cache_dtype: {self.kv_cache_dtype}")

    def _calibrate_kv_scale(self) -> float:
        absmax = 0.0

        def collect_absmax(_module, _inputs, output):
            nonlocal absmax
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                absmax = max(absmax, float(tensor.detach().float().abs().max().item()))

        hooks = []
        for layer in self.model.model.layers:
            attn = layer.self_attn
            hooks.append(attn.k_norm.register_forward_hook(collect_absmax))
            hooks.append(attn.v_proj.register_forward_hook(collect_absmax))

        seed = (
            "The quick brown fox jumps over the lazy dog. "
            "KV cache calibration observes representative key and value activations. "
            "Attention mechanisms store previous tokens for efficient autoregressive decoding. "
        )
        token_ids = self.tokenizer.encode(seed)
        while len(token_ids) < self.kv_scale_calib_tokens:
            token_ids.extend(token_ids)
        token_ids = token_ids[:self.kv_scale_calib_tokens]

        max_chunk = 512
        try:
            with torch.no_grad():
                for start in range(0, len(token_ids), max_chunk):
                    chunk = token_ids[start:start + max_chunk]
                    input_ids = torch.tensor([chunk], dtype=torch.long, device=self.device)
                    position_ids = torch.arange(len(chunk), dtype=torch.long, device=self.device).unsqueeze(0)
                    self.model(input_ids=input_ids, position_ids=position_ids)
        finally:
            for hook in hooks:
                hook.remove()

        scale = calibrate_scale(absmax)
        print(f"KV FP8 calibration: kv_scale={scale:.6g} (absmax={absmax:.6g}, tokens={len(token_ids)})")
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return scale

    def _set_kv_runtime_context(self) -> None:
        paged_ctx.kv_scale = self.kv_scale
        paged_ctx.compute_dtype = self.compute_dtype


    def _init_decode_buffers(self):
        """Preallocate fixed-capacity decode input buffers.

        CUDA Graph capture needs stable tensor objects and preferably stable
        shapes. This first step removes per-token `torch.tensor(...)`
        allocations from decode input construction. We still return a dynamic
        slice for now; later steps will replay a fixed max-batch graph.
        """
        self.decode_input_ids = torch.empty(
            self.max_num_seqs, 1, dtype=torch.long, device=self.device
        )
        self.decode_position_ids = torch.empty(
            self.max_num_seqs, 1, dtype=torch.long, device=self.device
        )
        self.decode_slot_mapping = torch.empty(
            self.max_num_seqs, dtype=torch.long, device=self.device
        )
        self.decode_num_cached_after = torch.empty(
            self.max_num_seqs, dtype=torch.long, device=self.device
        )
        self.decode_block_tables = torch.empty(
            self.max_num_seqs, self.kv_cache.shape[1], dtype=torch.long, device=self.device
        )
        self.decode_num_blocks = torch.empty(
            self.max_num_seqs, dtype=torch.long, device=self.device
        )
        self.decode_k_batch = None
        self.decode_v_batch = None
        self.decode_mask = None
        self.decode_kv_capacity = 0
        # CUDA Graph cache skeleton. Later steps will store captured graphs here.
        # Keyed by (batch_size, kv_capacity), because both affect tensor shapes.
        self.decode_graphs = {}
        self.decode_graph_replay_count = 0
        self.decode_graph_capture_count = 0
        self._is_capturing_decode_graph = False

    def _ensure_decode_attention_buffers(self, max_cached: int) -> None:
        """Grow reusable decode attention buffers when needed.

        `_build_decode_kv_batch()` used to allocate fresh `K_batch`, `V_batch`,
        and decode masks every layer and every decode step. This is expensive
        and not graph-friendly. Here we allocate persistent buffers and grow
        them only when the active context length exceeds the previous capacity.
        """
        if self.decode_kv_capacity >= max_cached:
            return

        capacity = ((max_cached + self.block_size - 1) // self.block_size) * self.block_size
        dtype = self.compute_dtype
        self.decode_k_batch = torch.empty(
            self.max_num_seqs, self.num_kv_heads, capacity, self.head_dim,
            dtype=dtype, device=self.device,
        )
        self.decode_v_batch = torch.empty(
            self.max_num_seqs, self.num_kv_heads, capacity, self.head_dim,
            dtype=dtype, device=self.device,
        )
        self.decode_mask = torch.empty(
            self.max_num_seqs, 1, 1, capacity,
            dtype=dtype, device=self.device,
        )
        self.decode_kv_capacity = capacity

    def _prepare_decode_inputs(self, seqs: list[Sequence]):
        """Fill preallocated decode buffers and return active batch views."""
        bsz = len(seqs)
        if bsz > self.max_num_seqs:
            raise ValueError(f"decode batch size {bsz} > max_num_seqs {self.max_num_seqs}")
        for i, seq in enumerate(seqs):
            self.decode_input_ids[i, 0] = seq.last_token_id
            self.decode_position_ids[i, 0] = seq.num_tokens - 1
        return self.decode_input_ids[:bsz], self.decode_position_ids[:bsz]

    def _decode_forward_static(self, bsz: int):
        """Forward entry point for decode.

        This is still eager execution. The point of this wrapper is to isolate
        the exact model call that a later step will warm up, capture with
        `torch.cuda.CUDAGraph`, and replay.
        """
        input_ids = self.decode_input_ids[:bsz]
        position_ids = self.decode_position_ids[:bsz]
        with torch.no_grad():
            return self.model(input_ids=input_ids, position_ids=position_ids)

    def _decode_graph_key(self, bsz: int) -> tuple[int, int]:
        """Shape key for a future CUDA Graph decode replay."""
        return (bsz, self.decode_kv_capacity)

    def _can_use_decode_graph(self, bsz: int) -> bool:
        """Whether this decode step is eligible for CUDA Graph replay."""
        return (
            self.device == "cuda"
            and not self.deterministic
            and bsz > 0
            and self.decode_kv_capacity > 0
        )

    def _capture_decode_graph(self, bsz: int):
        """Capture one eager decode forward as a CUDA Graph for this shape key."""
        key = self._decode_graph_key(bsz)
        for _ in range(3):
            self._decode_forward_static(bsz)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        paged_ctx.is_capturing_decode_graph = True
        self._is_capturing_decode_graph = True
        try:
            with torch.cuda.graph(graph):
                static_outputs = self._decode_forward_static(bsz)
        finally:
            paged_ctx.is_capturing_decode_graph = False
            self._is_capturing_decode_graph = False
        self.decode_graphs[key] = (graph, static_outputs)
        self.decode_graph_capture_count += 1

    def _decode_forward(self, bsz: int):
        """Decode forward dispatcher with minimal CUDA Graph replay support."""
        if not self._can_use_decode_graph(bsz):
            return self._decode_forward_static(bsz)

        key = self._decode_graph_key(bsz)
        if key not in self.decode_graphs:
            self._capture_decode_graph(bsz)

        graph, static_outputs = self.decode_graphs[key]
        graph.replay()
        self.decode_graph_replay_count += 1
        return static_outputs


    def allocate_kv_cache(self):
        """Allocate per-layer paged KV cache tensors.

        Shape per layer: (num_blocks, 2, num_kv_heads, block_size, head_dim)
          - 2 = K and V
          - Each layer has its own block pool and block_table.
        On CUDA we use gpu_memory_utilization * free_memory to size the cache.
        On MPS we can't query available VRAM precisely, so use a fixed pool size.

        Default/auto keeps float32 KV for deterministic bit-exact tests. Fast
        mode can opt into bf16 or fp8_e4m3 storage through kv_cache_dtype.
        """
        kv_dtype = self._resolve_kv_dtype()
        bytes_per_elem = torch.tensor([], dtype=kv_dtype).element_size()
        # Each block (per layer) stores: 2 * num_kv_heads * block_size * head_dim values
        bytes_per_block = (
            2 * self.num_kv_heads * self.block_size * self.head_dim * bytes_per_elem
        )
        if self.device == "cuda":
            # Keep the cache bounded during pytest: several module-scoped LLMs
            # are created in one Python process, and an oversized first cache
            # can starve later model loads. 256 blocks is still far above what
            # these tests need; preemption tests shrink the pool manually.
            free_vram, _total_vram = torch.cuda.mem_get_info()
            kv_budget_bytes = int(free_vram * self.gpu_memory_utilization)
            if self.deterministic:
                kv_budget_bytes = min(kv_budget_bytes, 900 * 1024**2)
            kv_budget_bytes = max(kv_budget_bytes, 256 * 1024**2)  # at least 256MB
        else:
            # Rough budget: ~2 GB for KV cache on a typical Mac
            kv_budget_bytes = 2 * 1024**3
        num_blocks = max(int(kv_budget_bytes // (bytes_per_block * self.num_layers)), 16)

        # One tensor per layer
        kv_cache = torch.zeros(
            self.num_layers, num_blocks, 2,
            self.num_kv_heads, self.block_size, self.head_dim,
            dtype=kv_dtype, device=self.device,
        )
        total_gb = kv_cache.nelement() * bytes_per_elem / 1024**3
        scale_info = f", scale={self.kv_scale:.6g}" if self.kv_scale is not None else ""
        print(f"KV cache: {num_blocks} blocks/layer × {self.num_layers} layers, "
              f"{total_gb:.2f} GB ({kv_dtype}{scale_info})")
        return kv_cache

    def run_prefill(self, seqs: list[Sequence]):
        # Run each sequence independently. This avoids CUDA GEMM/attention
        # algorithm differences between packed multi-seq prefill and single-seq
        # prefill, which can flip greedy argmaxes. Prefix-cache statistics still
        # work because each sequence inserts its blocks before the next one.
        if self.deterministic and len(seqs) > 1:
            return [self.run_prefill([seq])[0] for seq in seqs]

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

            prefill_seqs.append(seq)

            full_slot_mapping = self.block_manager.get_slot_mapping(
                seq.block_table[0], chunk_end
            )

            # In deterministic mode, recompute the whole current prompt prefix
            # whenever prefix cache/chunked prefill would otherwise use a
            # different attention path. Fast mode keeps suffix-only prefill.
            if self.deterministic and (prefix_len > 0 or chunk_start > 0):
                compute_start = 0
                effective_prefix_len = 0
            else:
                compute_start = max(prefix_len, chunk_start)
                effective_prefix_len = prefix_len

            suffix_token_ids = seq.prompt_token_ids[compute_start:chunk_end]
            suffix_positions = list(range(compute_start, chunk_end))
            suffix_slot_mapping = full_slot_mapping[compute_start:chunk_end]
            suffix_len = len(suffix_token_ids)

            all_input_ids.extend(suffix_token_ids)
            all_position_ids.extend(suffix_positions)
            all_slots.extend(suffix_slot_mapping)
            suffix_lengths.append(suffix_len)
            prefix_lengths.append(effective_prefix_len)

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
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)

        # 3. Set paged context
        self._set_kv_runtime_context()
        paged_ctx.kv_cache = self.kv_cache
        paged_ctx.deterministic = self.deterministic
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
        # Run decode independently per sequence for deterministic equivalence
        # with single-request greedy decoding. CUDA batched GEMMs can differ
        # enough to flip argmax on near-tied logits.
        if self.deterministic and len(seqs) > 1:
            out = []
            for seq in seqs:
                logits = self.run_decode([seq])
                if logits is None:
                    return None
                out.append(logits[0])
            return out

        # 1. Allocate blocks & compute per-sequence slot
        slot_mapping_values = []
        num_cached_after_values = []
        block_table_views = []
        for i, seq in enumerate(seqs):
            total_tokens_after = seq.num_cached_tokens + 1
            num_blocks_needed = self.block_manager.num_blocks_needed(total_tokens_after)
            cur_blocks = len(seq.block_table[0])
            if cur_blocks < num_blocks_needed:
                new_blocks = self.block_manager.try_allocate(num_blocks_needed - cur_blocks)
                if new_blocks is None:
                    return None  # signal OOM to caller for potential preemption
                for layer in range(self.num_layers):
                    seq.block_table[layer].extend(new_blocks)

            new_token_pos = seq.num_cached_tokens
            block_idx = new_token_pos // self.block_size
            offset = new_token_pos % self.block_size
            block_id = seq.block_table[0][block_idx]
            slot = block_id * self.block_size + offset
            slot_mapping_values.append(slot)
            num_cached_after_values.append(total_tokens_after)
            self.decode_slot_mapping[i] = slot
            self.decode_num_cached_after[i] = total_tokens_after
            num_blocks = len(seq.block_table[0])
            self.decode_num_blocks[i] = num_blocks
            block_table_tensor = torch.as_tensor(
                seq.block_table[0], dtype=torch.long, device=self.device
            )
            self.decode_block_tables[i, :num_blocks].copy_(block_table_tensor)
            block_table_views.append(self.decode_block_tables[i, :num_blocks])

        # 2. Fill preallocated decode input buffers
        self._prepare_decode_inputs(seqs)

        # 3. Ensure reusable attention buffers are large enough for this step
        bsz = len(seqs)
        max_cached = max(num_cached_after_values)
        self._ensure_decode_attention_buffers(max_cached)

        # 4. Set paged context
        self._set_kv_runtime_context()
        paged_ctx.kv_cache = self.kv_cache
        paged_ctx.deterministic = self.deterministic
        paged_ctx.slot_mapping = self.decode_slot_mapping[:bsz]
        paged_ctx.slot_mapping_values = slot_mapping_values
        paged_ctx.block_tables = block_table_views
        paged_ctx.num_cached_after = self.decode_num_cached_after[:bsz]
        paged_ctx.num_cached_after_values = num_cached_after_values
        paged_ctx.decode_k_batch = self.decode_k_batch[:bsz]
        paged_ctx.decode_v_batch = self.decode_v_batch[:bsz]
        paged_ctx.decode_mask = self.decode_mask[:bsz]
        paged_ctx.decode_kv_capacity = self.decode_kv_capacity
        paged_ctx.is_prefill = False
        # Clear prefix cache fields to avoid stale data
        paged_ctx.prefix_lengths = None
        paged_ctx.suffix_lengths = None

        if os.environ.get("MINI_VLLM_DEBUG"):
            print(f"[DECODE] {len(seqs)} seqs")

        # 5. Forward through the decode dispatcher
        outputs = self._decode_forward(bsz)

        # 5. Update num_cached_tokens & extract logits
        logits_list = []
        for i, seq in enumerate(seqs):
            seq.num_cached_tokens += 1
            logits_list.append(outputs.logits[i, -1, :].clone())  # (vocab_size,)

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

        self._set_kv_runtime_context()
        paged_ctx.kv_cache = self.kv_cache
        paged_ctx.deterministic = self.deterministic
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

        self._set_kv_runtime_context()
        paged_ctx.kv_cache = self.kv_cache
        paged_ctx.deterministic = self.deterministic
        paged_ctx.slot_mapping = suffix_slot_mapping
        paged_ctx.is_prefill = True
        paged_ctx.attn_mask = None
        paged_ctx.prefix_lengths = [old_num_cached]
        paged_ctx.suffix_lengths = [len(verify_token_ids)]
        paged_ctx.block_tables = [seq.block_table[0]]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, position_ids=position_ids)

        return outputs.logits[0], old_num_cached, target_num_cached
        