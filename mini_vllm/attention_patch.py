"""Attention monkey-patch — replace Qwen3Attention.forward to use our paged KV cache."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from mini_vllm.kv_quant import FP8_DTYPE, dequantize_from_fp8, quantize_to_fp8


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
    kv_scale: float | None = None
    compute_dtype: torch.dtype = torch.bfloat16
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
    if kv_cache.dtype == FP8_DTYPE:
        kv_scale = getattr(paged_ctx, "kv_scale", None)
        if kv_scale is None:
            raise ValueError("fp8_e4m3 KV requires paged_ctx.kv_scale")
        K_write = quantize_to_fp8(K_write, kv_scale)
        V_write = quantize_to_fp8(V_write, kv_scale)

    for t in range(K_write.size(0)):
        slot = slots[t].item()
        bid = slot // block_size
        off = slot % block_size
        kv_cache[layer_idx, bid, 0, :, off, :] = K_write[t]
        kv_cache[layer_idx, bid, 1, :, off, :] = V_write[t]


def _write_decode_kv(
    kv_cache: torch.Tensor,
    layer_idx: int,
    slot_mapping: list[int] | torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    block_size: int,
    bsz: int,
) -> None:
    slots = slot_mapping
    K_write = key_states[:, :, 0, :]
    V_write = value_states[:, :, 0, :]
    if kv_cache.dtype == FP8_DTYPE:
        kv_scale = getattr(paged_ctx, "kv_scale", None)
        if kv_scale is None:
            raise ValueError("fp8_e4m3 KV requires paged_ctx.kv_scale")
        K_write = quantize_to_fp8(K_write, kv_scale)
        V_write = quantize_to_fp8(V_write, kv_scale)

    for i in range(bsz):
        if isinstance(slots, torch.Tensor):
            slot = int(slots[i].item())
        else:
            slot = int(slots[i])
        bid = slot // block_size
        off = slot % block_size
        kv_cache[layer_idx, bid, 0, :, off, :] = K_write[i]
        kv_cache[layer_idx, bid, 1, :, off, :] = V_write[i]


def _read_kv_from_blocks(
    kv_cache: torch.Tensor,
    layer_idx: int,
    block_table: list[int] | torch.Tensor,
    total_len: int,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(block_table, torch.Tensor):
        block_table = block_table.to(device=kv_cache.device, dtype=torch.long)
    if kv_cache.dtype == FP8_DTYPE:
        # Some torch/CUDA builds don't implement advanced (fancy) indexing
        # for float8_e4m3fn ("index_cuda"/"index_select_cuda" not implemented).
        # View as uint8 (same byte layout) for the gather, then view back.
        kv_cache_u8 = kv_cache.view(torch.uint8)
        K_blocks = kv_cache_u8[layer_idx, block_table, 0].view(FP8_DTYPE)
        V_blocks = kv_cache_u8[layer_idx, block_table, 1].view(FP8_DTYPE)
    else:
        K_blocks = kv_cache[layer_idx, block_table, 0]
        V_blocks = kv_cache[layer_idx, block_table, 1]
    if kv_cache.dtype == FP8_DTYPE:
        kv_scale = getattr(paged_ctx, "kv_scale", None)
        if kv_scale is None:
            raise ValueError("fp8_e4m3 KV requires paged_ctx.kv_scale")
        compute_dtype = getattr(paged_ctx, "compute_dtype", torch.bfloat16)
        K_blocks = dequantize_from_fp8(K_blocks, kv_scale, compute_dtype)
        V_blocks = dequantize_from_fp8(V_blocks, kv_scale, compute_dtype)
    K_full = K_blocks.permute(1, 0, 2, 3).reshape(
        num_kv_heads, -1, head_dim
    )[:, :total_len, :]
    V_full = V_blocks.permute(1, 0, 2, 3).reshape(
        num_kv_heads, -1, head_dim
    )[:, :total_len, :]
    return K_full, V_full


def _sdpa(
    query_states: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """Deterministic fp32 attention used by all paged-attention paths.

    CUDA's fused SDPA kernels can pick different algorithms for full prefill,
    suffix prefill, batched decode, and single decode. Small numeric differences
    are enough to flip greedy argmaxes in these tests. Use the same explicit
    fp32 matmul/softmax path everywhere so chunked/full, batched/single, and
    speculative/normal decoding are numerically aligned.
    """
    if not getattr(paged_ctx, "deterministic", True):
        q_dtype = query_states.dtype
        if K.dtype != q_dtype:
            K = K.to(q_dtype)
        if V.dtype != q_dtype:
            V = V.to(q_dtype)
        if attn_mask is not None and attn_mask.dtype != q_dtype:
            attn_mask = attn_mask.to(q_dtype)
        if getattr(paged_ctx, "is_capturing_decode_graph", False):
            return torch.matmul(
                torch.softmax(
                    torch.matmul(query_states, K.transpose(-2, -1)) / (query_states.shape[-1] ** 0.5)
                    + (attn_mask if attn_mask is not None else 0),
                    dim=-1,
                ),
                V,
            )
        return F.scaled_dot_product_attention(
            query_states, K, V, attn_mask=attn_mask, is_causal=is_causal
        )

    q_dtype = query_states.dtype
    Q = query_states.to(torch.float32)
    K = K.to(torch.float32)
    V = V.to(torch.float32)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
    if is_causal:
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        causal = torch.ones(q_len, k_len, device=scores.device, dtype=torch.bool).tril(
            diagonal=k_len - q_len
        )
        scores = scores.masked_fill(~causal, float("-inf"))
    if attn_mask is not None:
        scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype)

    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, V)
    return out.to(q_dtype)


def _build_suffix_prefill_mask(prefix_len: int, suffix_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    total_len = prefix_len + suffix_len
    mask = torch.full((1, 1, suffix_len, total_len), float("-inf"), device=device, dtype=dtype)
    mask[0, 0, :, :prefix_len] = 0.0

    causal = torch.tril(torch.ones(suffix_len, suffix_len, device=device)).bool()
    suffix_mask = torch.full((suffix_len, suffix_len), float("-inf"), device=device, dtype=dtype)
    suffix_mask[causal] = 0.0
    mask[0, 0, :, prefix_len:] = suffix_mask
    return mask


def _build_decode_mask(
    bsz: int,
    k_len: int,
    num_cached_list: list[int] | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    num_cached_values = getattr(paged_ctx, "num_cached_after_values", None)
    if num_cached_values is None:
        if isinstance(num_cached_list, torch.Tensor):
            num_cached_values = num_cached_list.tolist()
        else:
            num_cached_values = num_cached_list

    mask = getattr(paged_ctx, "decode_mask", None)
    if mask is None:
        mask = torch.empty(bsz, 1, 1, k_len, device=device, dtype=dtype)
    else:
        mask = mask[:bsz]
        if mask.dtype != dtype:
            mask = mask.to(dtype=dtype)
        k_len = mask.shape[-1]

    mask.zero_()
    for i in range(bsz):
        mask[i, 0, 0, int(num_cached_values[i]):k_len] = float("-inf")
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
        return _sdpa(
            query_states,
            K_expanded,
            V_expanded,
            attn_mask=mask,
        )

    return _sdpa(
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
        mask = _build_suffix_prefill_mask(prefix_len, suffix_len, kv_cache.device, K_expanded.dtype)
        attn_outputs.append(
            _sdpa(
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
    block_tables: list[list[int] | torch.Tensor],
    num_cached_list: list[int] | torch.Tensor,
    meta: AttentionMeta,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    num_cached_values = getattr(paged_ctx, "num_cached_after_values", None)
    if num_cached_values is None:
        if isinstance(num_cached_list, torch.Tensor):
            num_cached_values = num_cached_list.tolist()
        else:
            num_cached_values = num_cached_list
    max_cached = max(num_cached_values)

    K_batch = getattr(paged_ctx, "decode_k_batch", None)
    V_batch = getattr(paged_ctx, "decode_v_batch", None)
    if K_batch is None or V_batch is None:
        compute_dtype = getattr(paged_ctx, "compute_dtype", torch.bfloat16)
        K_batch = torch.empty(
            meta["bsz"], meta["num_kv_heads"], max_cached, meta["head_dim"],
            dtype=compute_dtype, device=kv_cache.device,
        )
        V_batch = torch.empty(
            meta["bsz"], meta["num_kv_heads"], max_cached, meta["head_dim"],
            dtype=compute_dtype, device=kv_cache.device,
        )
        k_len = max_cached
    else:
        K_batch = K_batch[:meta["bsz"]]
        V_batch = V_batch[:meta["bsz"]]
        k_len = K_batch.shape[2]

    for i, block_table in enumerate(block_tables):
        num_cached = int(num_cached_values[i])
        K_full, V_full = _read_kv_from_blocks(
            kv_cache,
            layer_idx,
            block_table,
            num_cached,
            meta["num_kv_heads"],
            meta["head_dim"],
        )
        K_batch[i, :, :num_cached, :] = K_full
        V_batch[i, :, :num_cached, :] = V_full

    return K_batch, V_batch, k_len


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
        getattr(paged_ctx, "slot_mapping_values", paged_ctx.slot_mapping),
        key_states,
        value_states,
        block_size,
        meta["bsz"],
    )

    K_batch, V_batch, k_len = _build_decode_kv_batch(
        kv_cache,
        meta["layer_idx"],
        block_tables,
        num_cached_list,
        meta,
    )
    K_expanded = repeat_kv(K_batch, self.num_key_value_groups)
    V_expanded = repeat_kv(V_batch, self.num_key_value_groups)
    mask = _build_decode_mask(meta["bsz"], k_len, num_cached_list, kv_cache.device, K_expanded.dtype)

    return _sdpa(
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
    """Tree speculative verify attention.

    Supports batching (bsz > 1): each sequence in the batch has its own
    prefix_len/suffix_len/tree_mask, mirroring how _suffix_prefill_attention
    loops per-sequence. This is what allows tree-PLD speculative decoding to
    run under concurrent serving instead of being restricted to batch_size=1.
    """
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

    # Backward-compatible single-sequence path (paged_ctx.tree_mask) vs the
    # batched path (paged_ctx.tree_masks, a list — one per sequence).
    tree_masks = getattr(paged_ctx, "tree_masks", None)
    if tree_masks is None:
        tree_masks = [paged_ctx.tree_mask]

    attn_outputs: list[torch.Tensor] = []
    q_offset = 0
    for prefix_len, suffix_len, block_table, tree_mask in zip(
        paged_ctx.prefix_lengths,
        paged_ctx.suffix_lengths,
        paged_ctx.block_tables,
        tree_masks,
    ):
        total_len = prefix_len + suffix_len
        tree_mask = tree_mask.to(device=kv_cache.device, dtype=query_states.dtype)
        assert suffix_len == 1 + tree_mask.size(0)

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
        mask = _build_tree_verify_mask(prefix_len, tree_mask, kv_cache.device)
        attn_outputs.append(
            _sdpa(
                Q_i,
                K_expanded,
                V_expanded,
                attn_mask=mask,
            )
        )
        q_offset += suffix_len

    return torch.cat(attn_outputs, dim=2)


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