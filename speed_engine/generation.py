"""
Generation control: boredom stopper (EQ-BOREDOM), simulated cloud engine,
router speedup (EQ-ROUTER), speculative decode models (EQ-SPEC-*).
"""

from __future__ import annotations

import math
import time
from typing import AsyncIterator, Iterator, List, Optional


# ---------------------------------------------------------------------------
# EQ-BOREDOM — stop when prob < threshold for patience steps after min_tokens
# ---------------------------------------------------------------------------
class BoredomStopper:
    """Stop when top-1 probability stays low (§8th equation)."""

    def __init__(self, threshold: float = 0.15, patience: int = 3, min_tokens: int = 5):
        self.threshold = threshold
        self.patience = patience
        self.min_tokens = min_tokens
        self.low_count = 0
        self.token_count = 0

    def should_stop(self, logprob: float) -> bool:
        prob = math.exp(logprob)
        self.token_count += 1
        if prob < self.threshold:
            self.low_count += 1
        else:
            self.low_count = 0
        return self.token_count >= self.min_tokens and self.low_count >= self.patience

    def reset(self) -> None:
        self.low_count = 0
        self.token_count = 0


# ---------------------------------------------------------------------------
# EQ-ROUTER — S = 1 / ((1-f) + f/rho); KV special case S = total/new
# ---------------------------------------------------------------------------
def router_speedup(f: float, rho: float) -> float:
    if not 0 <= f <= 1:
        raise ValueError("f must be in [0,1]")
    if rho <= 0:
        raise ValueError("rho must be positive")
    return 1.0 / ((1 - f) + f / rho)


def kv_cache_speedup(cached_tokens: int, new_tokens: int) -> float:
    total = cached_tokens + new_tokens
    if total == 0:
        return 1.0
    return total / new_tokens


# ---------------------------------------------------------------------------
# EQ-SPEC-SINGLE — E = (1-alpha^(gamma+1))/(1-alpha); speedup = E*T/(gamma*(D+T))
# ---------------------------------------------------------------------------
def single_draft_speedup(alpha: float, gamma: int, t: float = 1.0, d: float = 0.1) -> float:
    if alpha == 1:
        e = gamma + 1
    else:
        e = (1 - alpha ** (gamma + 1)) / (1 - alpha)
    iteration_time = gamma * (d + t)
    return e * t / iteration_time if iteration_time > 0 else float("inf")


def dual_draft_speedup(alpha: float, gamma: int, t: float = 1.0, d: float = 0.1) -> float:
    p = 2 * alpha - alpha**2
    if p == 1:
        e = gamma + 1
    else:
        e = (1 - p ** (gamma + 1)) / (1 - p)
    iteration_time = gamma * d + 2 * gamma * t
    return e * t / iteration_time if iteration_time > 0 else float("inf")


# ---------------------------------------------------------------------------
# Simulated cloud engine — TTFB = T_prefill + T_decode(1) with keep-alive (§2nd)
# ---------------------------------------------------------------------------
class SimulatedCloudEngine:
    """
    Simulates cloud LLM with configurable base latency (for benchmarks).
    Uses dynamic max_tokens and optional one-liner cap.
    """

    def __init__(
        self,
        base_latency_ms: float = 25.0,
        ms_per_token: float = 2.0,
        one_liner_mode: bool = False,
    ):
        self.base_latency_ms = base_latency_ms
        self.ms_per_token = ms_per_token
        self.one_liner_mode = one_liner_mode
        self._warm = False

    def warm_keepalive(self) -> None:
        """First request builds cache; subsequent skip system prefill (EQ-TTFB-CACHE)."""
        self._warm = True

    def generate_tokens(self, prompt: str, max_tokens: int) -> List[str]:
        if self.one_liner_mode:
            max_tokens = min(max_tokens, 18)
        # Simulate decode: split response from prompt context
        answer = self._answer_from_prompt(prompt, max_tokens)
        words = answer.split()
        return words[:max_tokens]

    def _answer_from_prompt(self, prompt: str, max_tokens: int) -> str:
        if "capital of France" in prompt.lower():
            return "The capital of France is Paris."
        if "bm25" in prompt.lower():
            return "BM25 is a bag-of-words retrieval function used in search."
        if "kv cache" in prompt.lower():
            return "KV cache stores key-value pairs to avoid recomputing attention."
        return "Based on the provided context, here is a concise answer to your question."

    def latency_ms(self, num_tokens: int) -> float:
        restore = 0.5 if self._warm else self.base_latency_ms
        return restore + num_tokens * self.ms_per_token

    def stream_sync(self, prompt: str, max_tokens: int) -> Iterator[str]:
        delay = self.latency_ms(1) / 1000.0 / max(max_tokens, 1)
        for tok in self.generate_tokens(prompt, max_tokens):
            time.sleep(delay)
            yield tok

    async def stream_async(self, prompt: str, max_tokens: int) -> AsyncIterator[str]:
        for tok in self.stream_sync(prompt, max_tokens):
            yield tok
