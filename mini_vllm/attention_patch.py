"""Attention monkey-patch — replace Qwen3Attention.forward to use our paged KV cache."""

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    repeat_kv,
)


class PageContext:
    """Runtime context set by ModelRunner before each forward call."""
    kv_cache: torch.Tensor      # (num_layers, num_blocks, 2, num_kv_heads, block_size, head_dim)
    block_table: list    # List[int] — used for single-seq decode only
    block_tables: list  # List[List[int]] — per-seq block tables for batched decode
    slot_mapping: list           # List[int] — flat list of slot indices
    num_cached_tokens: int = 0  # used for prefill
    num_cached_after: list  # List[int] — per-seq cached counts for batched decode
    is_prefill: bool = True
    attn_mask: torch.Tensor # (total_len, total_len) block-diagonal causal mask for prefill 
    prefix_lengths: list 
    suffix_lengths: list


paged_ctx = PageContext()


def paged_attention_forward(
    self,                    # Qwen3Attention instance (HF original, has weights)
    hidden_states,           # [1, seq_len, hidden_size]
    position_embeddings,     # (cos, sin) from Qwen3RotaryEmbedding
    attention_mask=None,     # not used in M1
    past_key_values=None,    # not used — we manage our own cache
    **kwargs,
):
    seq_len = hidden_states.size(1)
    layer_idx = self.layer_idx

    # ========== 1. Q/K/V projections (same as original) ==========
    bsz, _, _ = hidden_states.size()
    num_heads = self.config.num_attention_heads       # 16
    num_kv_heads = self.config.num_key_value_heads    # 8
    hidden_size = self.config.hidden_size             # 1024

    # Q and K: proj → reshape → norm → transpose
    # V: proj → reshape → transpose (NO norm for V)
    q_shape = (bsz, seq_len, num_heads, self.head_dim)        # 16 query heads
    kv_shape = (bsz, seq_len, num_kv_heads, self.head_dim)    # 8 kv heads

    query_states = self.q_norm(
        self.q_proj(hidden_states).view(q_shape)
    ).transpose(1, 2)  # [1, 16, seq_len, 128]

    key_states = self.k_norm(
        self.k_proj(hidden_states).view(kv_shape)
    ).transpose(1, 2)  # [1, 8, seq_len, 128]

    value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)
    # [1, 8, seq_len, 128]

    # ========== 2. RoPE (same as original) ==========
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # ========== 3. Paged KV Cache read/write ==========
    kv_cache = paged_ctx.kv_cache
    slot_mapping = paged_ctx.slot_mapping        # List[int]
    num_cached = paged_ctx.num_cached_tokens
    is_prefill = paged_ctx.is_prefill
    block_size = kv_cache.shape[4]               # block_size dim
    has_prefix_hit = getattr(paged_ctx, 'prefix_lengths', None) is not None and any(
        pl > 0 for pl in (paged_ctx.prefix_lengths or [])
    )
    if is_prefill:
        # --- Write suffix K/V to cache (both paths need this) ---
        slots = torch.tensor(slot_mapping, device=kv_cache.device)
        K_write = key_states[0].transpose(0, 1)      # [seq_len, num_kv_heads, head_dim]
        V_write = value_states[0].transpose(0, 1)

        for t in range(seq_len):
            bid = slots[t].item() // block_size
            off = slots[t].item() % block_size
            kv_cache[layer_idx, bid, 0, :, off, :] = K_write[t]   # K
            kv_cache[layer_idx, bid, 1, :, off, :] = V_write[t]   # V

        if has_prefix_hit:
            # ---- SUFFIX PREFILL: read ALL KV from cache, per-sequence SDPA ----
            prefix_lens = paged_ctx.prefix_lengths
            suffix_lens = paged_ctx.suffix_lengths
            bt_list = paged_ctx.block_tables

            attn_outputs = []
            q_offset = 0
            for i in range(len(prefix_lens)):
                prefix_len = prefix_lens[i]
                suffix_len = suffix_lens[i]
                total_len = prefix_len + suffix_len

                # Extract Q for this sequence
                Q_i = query_states[:, :, q_offset:q_offset+suffix_len, :]
                # (1, num_heads, suffix_len, head_dim)

                # Read ALL KV from cache using block_table
                bt = bt_list[i]
                K_blocks = kv_cache[layer_idx, bt, 0]
                V_blocks = kv_cache[layer_idx, bt, 1]
                K_full = K_blocks.permute(1, 0, 2, 3).reshape(
                    num_kv_heads, -1, self.head_dim
                )[:, :total_len, :]
                V_full = V_blocks.permute(1, 0, 2, 3).reshape(
                    num_kv_heads, -1, self.head_dim
                )[:, :total_len, :]
                K_i = K_full.unsqueeze(0)  # (1, num_kv_heads, total_len, head_dim)
                V_i = V_full.unsqueeze(0)

                # GQA expand
                K_exp = repeat_kv(K_i, self.num_key_value_groups)
                V_exp = repeat_kv(V_i, self.num_key_value_groups)

                # Per-sequence mask: suffix attends to all prefix + causal within suffix
                seq_mask = torch.full(
                    (1, 1, suffix_len, total_len), float('-inf'), device=kv_cache.device
                )
                seq_mask[0, 0, :, :prefix_len] = 0.0  # attend to prefix
                causal = torch.tril(torch.ones(suffix_len, suffix_len, device=kv_cache.device)).bool()
                seq_mask[0, 0, :, prefix_len:][causal] = 0.0  # causal within suffix

                attn_out = F.scaled_dot_product_attention(
                    Q_i, K_exp, V_exp, attn_mask=seq_mask
                )  # (1, num_heads, suffix_len, head_dim)
                attn_outputs.append(attn_out)
                q_offset += suffix_len

            attn_output = torch.cat(attn_outputs, dim=2)
            # (1, num_heads, total_suffix_len, head_dim)
        else:
            # ---- ORIGINAL PREFILL: use just-computed K/V ----
            K_attn = key_states
            V_attn = value_states

            K_expanded = repeat_kv(K_attn, self.num_key_value_groups)
            V_expanded = repeat_kv(V_attn, self.num_key_value_groups)

            if paged_ctx.attn_mask is not None:
                mask = paged_ctx.attn_mask.unsqueeze(0).unsqueeze(0).expand(
                    -1, num_heads, -1, -1
                )
                attn_output = F.scaled_dot_product_attention(
                    query_states, K_expanded, V_expanded,
                    attn_mask=mask,
                )
            else:
                attn_output = F.scaled_dot_product_attention(
                    query_states, K_expanded, V_expanded,
                    is_causal=True,
                )
    else:
        # --- Batched decode: write K/V, then padded batch SDPA ---
        slots = torch.tensor(slot_mapping, device=kv_cache.device)
        block_tables = paged_ctx.block_tables       # List[List[int]]
        num_cached_list = paged_ctx.num_cached_after # List[int]
        max_cached = max(num_cached_list)

        # Write each sequence's new K/V to its cache slot
        for i in range(bsz):
            bid = slots[i].item() // block_size
            off = slots[i].item() % block_size
            kv_cache[layer_idx, bid, 0, :, off, :] = key_states[i, :, 0, :]
            kv_cache[layer_idx, bid, 1, :, off, :] = value_states[i, :, 0, :]

        # Read K/V from cache into padded batch tensor
        K_batch = kv_cache.new_zeros(bsz, num_kv_heads, max_cached, self.head_dim)
        V_batch = kv_cache.new_zeros(bsz, num_kv_heads, max_cached, self.head_dim)
        for i in range(bsz):
            bt = block_tables[i]
            K_blocks = kv_cache[layer_idx, bt, 0]   # (n_blocks, num_kv_heads, block_size, head_dim)
            V_blocks = kv_cache[layer_idx, bt, 1]
            K_full = K_blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, self.head_dim)[:, :num_cached_list[i], :]
            V_full = V_blocks.permute(1, 0, 2, 3).reshape(num_kv_heads, -1, self.head_dim)[:, :num_cached_list[i], :]
            K_batch[i, :, :num_cached_list[i], :] = K_full
            V_batch[i, :, :num_cached_list[i], :] = V_full

        K_attn = K_batch
        V_attn = V_batch

    # ========== 4. GQA expand + Attention ==========
    if is_prefill and has_prefix_hit:
        # Already computed attn_output in suffix prefill path above
        pass
    elif is_prefill:
        # Original prefill: already computed above
        pass
    else:
        # Decode: GQA expand + padded batch SDPA
        K_expanded = repeat_kv(K_attn, self.num_key_value_groups)
        V_expanded = repeat_kv(V_attn, self.num_key_value_groups)

        decode_mask = torch.zeros(bsz, 1, 1, max_cached, device=kv_cache.device)
        for i in range(bsz):
            decode_mask[i, 0, 0, num_cached_list[i]:] = float('-inf')
        attn_output = F.scaled_dot_product_attention(
            query_states, K_expanded, V_expanded,
            attn_mask=decode_mask,
        )

    # ========== 5. Output projection (same as original) ==========
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(
        bsz, seq_len, num_heads * self.head_dim
    )
    return self.o_proj(attn_output), None


def apply_attention_patch(model):
    """Replace each layer's attention forward with our paged version."""
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.layer_idx = layer_idx
        attn.forward = paged_attention_forward.__get__(attn, type(attn))
