"""Sampling parameters for generation."""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 16
