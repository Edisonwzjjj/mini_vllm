"""Attention monkey-patch — replace Qwen3Attention.forward to use our paged KV cache."""

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    repeat_kv,
)


class PageContext:
    """Runtime context set by ModelRunner before each forward call."""
    kv_cache: torch.Tensor = None       # (num_layers, num_blocks, 2, num_kv_heads, block_size, head_dim)
    block_table: list = None            # List[int] — M1: same for all layers
    slot_mapping: list = None           # List[int] — M1: same for all layers
    num_cached_tokens: int = 0
    is_prefill: bool = True


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
    block_table = paged_ctx.block_table          # List[int]
    slot_mapping = paged_ctx.slot_mapping        # List[int]
    num_cached = paged_ctx.num_cached_tokens
    is_prefill = paged_ctx.is_prefill
    block_size = kv_cache.shape[4]               # block_size dim

    if is_prefill:
        # --- Write K, V into cache via slot_mapping ---
        slots = torch.tensor(slot_mapping, device=kv_cache.device)
        K_write = key_states[0].transpose(0, 1)      # [seq_len, num_kv_heads, head_dim]
        V_write = value_states[0].transpose(0, 1)

        for t in range(seq_len):
            bid = slots[t].item() // block_size
            off = slots[t].item() % block_size
            kv_cache[layer_idx, bid, 0, :, off, :] = K_write[t]   # K
            kv_cache[layer_idx, bid, 1, :, off, :] = V_write[t]   # V

        # Prefill: attention uses the just-computed K, V (they ARE the full sequence)
        K_attn = key_states
        V_attn = value_states
    else:
        # --- Write 1 new token K, V into cache ---
        slots = torch.tensor(slot_mapping, device=kv_cache.device)
        bid = slots[0].item() // block_size
        off = slots[0].item() % block_size
        kv_cache[layer_idx, bid, 0, :, off, :] = key_states[0, :, 0, :]
        kv_cache[layer_idx, bid, 1, :, off, :] = value_states[0, :, 0, :]

        # --- Read full K, V from cache via block_table ---
        K_parts = []
        V_parts = []
        for bid in block_table:
            K_parts.append(kv_cache[layer_idx, bid, 0])   # [num_kv_heads, block_size, head_dim]
            V_parts.append(kv_cache[layer_idx, bid, 1])

        K_full = torch.cat(K_parts, dim=1)[:, :num_cached, :]  # [num_kv_heads, num_cached, head_dim]
        V_full = torch.cat(V_parts, dim=1)[:, :num_cached, :]

        K_attn = K_full.unsqueeze(0)   # [1, num_kv_heads, num_cached, head_dim]
        V_attn = V_full.unsqueeze(0)

    # ========== 4. GQA expand + Attention ==========
    K_expanded = repeat_kv(K_attn, self.num_key_value_groups)  # 2 groups → [1, 16, kv_len, 128]
    V_expanded = repeat_kv(V_attn, self.num_key_value_groups)

    attn_output = F.scaled_dot_product_attention(
        query_states, K_expanded, V_expanded,
        is_causal=is_prefill,   # prefill: need causal mask; decode: 1 token, no mask needed
    )  # [1, 16, seq_len, 128]

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
