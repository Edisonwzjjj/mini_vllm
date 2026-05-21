"""Attention monkey-patch — replace Qwen3Attention.forward to use our paged KV cache."""

from typing import Any, Literal

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    repeat_kv,
)


AttentionMode = Literal["prefill", "suffix_prefill", "decode", "tree_verify"]
AttentionMeta = dict[str, int]


class PageContext:
    """Runtime context set by ModelRunner before each forward call."""
    kv_cache: torch.Tensor      # (num_layers, num_blocks, 2, num_kv_heads, block_size, head_dim)
    block_table: list[int]    # used for single-seq decode only
    block_tables: list[list[int]]  # per-seq block tables for batched decode
    slot_mapping: list[int]           # flat list of slot indices
    num_cached_tokens: int = 0  # used for prefill
    num_cached_after: list[int]  # per-seq cached counts for batched decode
    is_prefill: bool = True
    attn_mask: torch.Tensor # (total_len, total_len) block-diagonal causal mask for prefill
    prefix_lengths: list[int]
    suffix_lengths: list[int]
    tree_mask: torch.Tensor  # (num_draft, num_draft), used by tree verify


paged_ctx = PageContext()


def _get_attention_mode(ctx: PageContext) -> AttentionMode:
    if getattr(ctx, "is_tree_verify", False):
        return "tree_verify"

    if ctx.is_prefill:
        prefix_lengths = getattr(ctx, "prefix_lengths", None)
        if prefix_lengths is not None and any(pl > 0 for pl in prefix_lengths):
            return "suffix_prefill"
        return "prefill"

    return "decode"


def _project_qkv(self: Any, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, AttentionMeta]:
    bsz, seq_len, _ = hidden_states.size()
    num_heads = self.config.num_attention_heads
    num_kv_heads = self.config.num_key_value_heads

    q_shape = (bsz, seq_len, num_heads, self.head_dim)
    kv_shape = (bsz, seq_len, num_kv_heads, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(q_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(kv_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)

    meta = {
        "bsz": bsz,
        "seq_len": seq_len,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": self.head_dim,
        "layer_idx": self.layer_idx,
    }
    return query_states, key_states, value_states, meta


def _apply_qwen_rope(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = position_embeddings
    return apply_rotary_pos_emb(query_states, key_states, cos, sin)


def _write_prefill_kv(
    kv_cache: torch.Tensor,
    layer_idx: int,
    slot_mapping: list[int],
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_size: int,
) -> None:
    slots = torch.tensor(slot_mapping, device=kv_cache.device)
    K_write = key_states[0].transpose(0, 1)
    V_write = value_states[0].transpose(0, 1)

    for t in range(K_write.size(0)):
        slot = slots[t].item()
        bid = slot // block_size
        off = slot % block_size
        kv_cache[layer_idx, bid, 0, :, off, :] = K_write[t]
        kv_cache[layer_idx, bid, 1, :, off, :] = V_write[t]


def _write_decode_kv(
    kv_cache: torch.Tensor,
    layer_idx: int,
    slot_mapping: list[int],
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_size: int,
    bsz: int,
) -> None:
    slots = torch.tensor(slot_mapping, device=kv_cache.device)

    for i in range(bsz):
        slot = slots[i].item()
        bid = slot // block_size
        off = slot % block_size
        kv_cache[layer_idx, bid, 0, :, off, :] = key_states[i, :, 0, :]
        kv_cache[layer_idx, bid, 1, :, off, :] = value_states[i, :, 0, :]


def _read_kv_from_blocks(
    kv_cache: torch.Tensor,
    layer_idx: int,
    block_table: list[int],
    total_len: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    K_blocks = kv_cache[layer_idx, block_table, 0]
    V_blocks = kv_cache[layer_idx, block_table, 1]
    K_full = K_blocks.permute(1, 0, 2, 3).reshape(
        num_kv_heads, -1, head_dim
    )[:, :total_len, :]
    V_full = V_blocks.permute(1, 0, 2, 3).reshape(
        num_kv_heads, -1, head_dim
    )[:, :total_len, :]
    return K_full, V_full


def _build_suffix_prefill_mask(prefix_len: int, suffix_len: int, device: torch.device) -> torch.Tensor:
    total_len = prefix_len + suffix_len
    mask = torch.full((1, 1, suffix_len, total_len), float("-inf"), device=device)
    mask[0, 0, :, :prefix_len] = 0.0

    causal = torch.tril(torch.ones(suffix_len, suffix_len, device=device)).bool()
    suffix_mask = torch.full((suffix_len, suffix_len), float("-inf"), device=device)
    suffix_mask[causal] = 0.0
    mask[0, 0, :, prefix_len:] = suffix_mask
    return mask


def _build_decode_mask(
    bsz: int,
    max_cached: int,
    num_cached_list: list[int],
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(bsz, 1, 1, max_cached, device=device)
    for i in range(bsz):
        mask[i, 0, 0, num_cached_list[i]:] = float("-inf")
    return mask


def _prefill_attention(
    self: Any,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    meta: AttentionMeta,
) -> torch.Tensor:
    kv_cache = paged_ctx.kv_cache
    block_size = kv_cache.shape[4]

    _write_prefill_kv(
        kv_cache,
        meta["layer_idx"],
        paged_ctx.slot_mapping,
        key_states,
        value_states,
        block_size,
    )

    K_expanded = repeat_kv(key_states, self.num_key_value_groups)
    V_expanded = repeat_kv(value_states, self.num_key_value_groups)

    if paged_ctx.attn_mask is not None:
        mask = paged_ctx.attn_mask.unsqueeze(0).unsqueeze(0).expand(
            -1, meta["num_heads"], -1, -1
        )
        return F.scaled_dot_product_attention(
            query_states,
            K_expanded,
            V_expanded,
            attn_mask=mask,
        )

    return F.scaled_dot_product_attention(
        query_states,
        K_expanded,
        V_expanded,
        is_causal=True,
    )


def _suffix_prefill_attention(
    self: Any,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    meta: AttentionMeta,
) -> torch.Tensor:
    kv_cache = paged_ctx.kv_cache
    block_size = kv_cache.shape[4]

    _write_prefill_kv(
        kv_cache,
        meta["layer_idx"],
        paged_ctx.slot_mapping,
        key_states,
        value_states,
        block_size,
    )

    attn_outputs: list[torch.Tensor] = []
    q_offset = 0
    for prefix_len, suffix_len, block_table in zip(
        paged_ctx.prefix_lengths,
        paged_ctx.suffix_lengths,
        paged_ctx.block_tables,
    ):
        total_len = prefix_len + suffix_len
        Q_i = query_states[:, :, q_offset:q_offset + suffix_len, :]
        K_full, V_full = _read_kv_from_blocks(
            kv_cache,
            meta["layer_idx"],
            block_table,
            total_len,
            meta["num_kv_heads"],
            meta["head_dim"],
        )
        K_i = K_full.unsqueeze(0)
        V_i = V_full.unsqueeze(0)

        K_expanded = repeat_kv(K_i, self.num_key_value_groups)
        V_expanded = repeat_kv(V_i, self.num_key_value_groups)
        mask = _build_suffix_prefill_mask(prefix_len, suffix_len, kv_cache.device)
        attn_outputs.append(
            F.scaled_dot_product_attention(
                Q_i,
                K_expanded,
                V_expanded,
                attn_mask=mask,
            )
        )
        q_offset += suffix_len

    return torch.cat(attn_outputs, dim=2)


def _build_decode_kv_batch(
    kv_cache: torch.Tensor,
    layer_idx: int,
    block_tables: list[list[int]],
    num_cached_list: list[int],
    meta: AttentionMeta,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    max_cached = max(num_cached_list)
    K_batch = kv_cache.new_zeros(
        meta["bsz"], meta["num_kv_heads"], max_cached, meta["head_dim"]
    )
    V_batch = kv_cache.new_zeros(
        meta["bsz"], meta["num_kv_heads"], max_cached, meta["head_dim"]
    )

    for i, block_table in enumerate(block_tables):
        K_full, V_full = _read_kv_from_blocks(
            kv_cache,
            layer_idx,
            block_table,
            num_cached_list[i],
            meta["num_kv_heads"],
            meta["head_dim"],
        )
        K_batch[i, :, :num_cached_list[i], :] = K_full
        V_batch[i, :, :num_cached_list[i], :] = V_full

    return K_batch, V_batch, max_cached


def _decode_attention(
    self: Any,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    meta: AttentionMeta,
) -> torch.Tensor:
    kv_cache = paged_ctx.kv_cache
    block_size = kv_cache.shape[4]
    block_tables = paged_ctx.block_tables
    num_cached_list = paged_ctx.num_cached_after

    _write_decode_kv(
        kv_cache,
        meta["layer_idx"],
        paged_ctx.slot_mapping,
        key_states,
        value_states,
        block_size,
        meta["bsz"],
    )

    K_batch, V_batch, max_cached = _build_decode_kv_batch(
        kv_cache,
        meta["layer_idx"],
        block_tables,
        num_cached_list,
        meta,
    )
    K_expanded = repeat_kv(K_batch, self.num_key_value_groups)
    V_expanded = repeat_kv(V_batch, self.num_key_value_groups)
    mask = _build_decode_mask(meta["bsz"], max_cached, num_cached_list, kv_cache.device)

    return F.scaled_dot_product_attention(
        query_states,
        K_expanded,
        V_expanded,
        attn_mask=mask,
    )


def _tree_verify_attention(
    self: Any,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    meta: AttentionMeta,
) -> torch.Tensor:
    kv_cache = paged_ctx.kv_cache
    block_size = kv_cache.shape[4]

    assert meta["bsz"] == 1
    assert len(paged_ctx.prefix_lengths) == 1
    assert len(paged_ctx.suffix_lengths) == 1
    assert len(paged_ctx.block_tables) == 1

    prefix_len = paged_ctx.prefix_lengths[0]
    suffix_len = paged_ctx.suffix_lengths[0]
    total_len = prefix_len + suffix_len
    tree_mask = paged_ctx.tree_mask.to(device=kv_cache.device, dtype=query_states.dtype)
    assert suffix_len == 1 + tree_mask.size(0)

    _write_prefill_kv(
        kv_cache,
        meta["layer_idx"],
        paged_ctx.slot_mapping,
        key_states,
        value_states,
        block_size,
    )

    K_full, V_full = _read_kv_from_blocks(
        kv_cache,
        meta["layer_idx"],
        paged_ctx.block_tables[0],
        total_len,
        meta["num_kv_heads"],
        meta["head_dim"],
    )
    K_i = K_full.unsqueeze(0)
    V_i = V_full.unsqueeze(0)

    K_expanded = repeat_kv(K_i, self.num_key_value_groups)
    V_expanded = repeat_kv(V_i, self.num_key_value_groups)
    mask = _build_tree_verify_mask(prefix_len, tree_mask, kv_cache.device)

    return F.scaled_dot_product_attention(
        query_states,
        K_expanded,
        V_expanded,
        attn_mask=mask,
    )


def _output_projection(self: Any, attn_output: torch.Tensor, meta: AttentionMeta) -> torch.Tensor:
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(
        meta["bsz"], meta["seq_len"], meta["num_heads"] * meta["head_dim"]
    )
    return self.o_proj(attn_output)


def paged_attention_forward(
    self: Any,                    # Qwen3Attention instance (HF original, has weights)
    hidden_states: torch.Tensor,           # [1, seq_len, hidden_size]
    position_embeddings: tuple[torch.Tensor, torch.Tensor],     # (cos, sin) from Qwen3RotaryEmbedding
    attention_mask: torch.Tensor | None = None,     # not used in M1
    past_key_values: Any | None = None,    # not used — we manage our own cache
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    query_states, key_states, value_states, meta = _project_qkv(self, hidden_states)
    query_states, key_states = _apply_qwen_rope(
        query_states, key_states, position_embeddings
    )

    mode = _get_attention_mode(paged_ctx)
    if mode == "prefill":
        attn_output = _prefill_attention(self, query_states, key_states, value_states, meta)
    elif mode == "suffix_prefill":
        attn_output = _suffix_prefill_attention(self, query_states, key_states, value_states, meta)
    elif mode == "decode":
        attn_output = _decode_attention(self, query_states, key_states, value_states, meta)
    elif mode == "tree_verify":
        attn_output = _tree_verify_attention(self, query_states, key_states, value_states, meta)
    else:
        raise ValueError(f"Unknown attention mode: {mode}")

    return _output_projection(self, attn_output, meta), None


def apply_attention_patch(model: Any) -> None:
    """Replace each layer's attention forward with our paged version."""
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.layer_idx = layer_idx
        attn.forward = paged_attention_forward.__get__(attn, type(attn))

def _build_tree_verify_mask(
    prefix_len: int,
    tree_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    assert tree_mask.dim() == 2
    assert tree_mask.size(0) == tree_mask.size(1)

    num_draft = tree_mask.size(0)
    suffix_len = 1 + num_draft
    total_len = prefix_len + suffix_len
    mask = torch.full(
        (1, 1, suffix_len, total_len),
        float("-inf"),
        device=device,
        dtype=tree_mask.dtype,
    )

    # All suffix queries can attend to prefix tokens.
    mask[:, :, :, :prefix_len] = 0.0

    # Row 0 is the last accepted token. It can attend to prefix + itself.
    mask[:, :, 0, prefix_len] = 0.0

    # Draft rows can attend to the last accepted token.
    mask[:, :, 1:, prefix_len] = 0.0

    # Draft rows attend to draft ancestors/self according to tree_mask.
    mask[:, :, 1:, prefix_len + 1:] = tree_mask.to(device=device)
    return mask