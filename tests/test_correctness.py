"""Test: mini-vllm output matches HF model.generate (greedy)."""

import pytest


@pytest.mark.timeout(120)
def test_decode_matches_hf():
    """Single sequence greedy decode should match HF output token-for-token."""
    # TODO: implement after M1 is done
    # 1. Load mini-vllm, generate with temperature=0.01
    # 2. Load HF model, generate with do_sample=False
    # 3. Assert token sequences match exactly
    pass
