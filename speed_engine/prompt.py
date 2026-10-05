"""
Prompt optimisation: minimal prompt (EQ-MINIMAL-PROMPT), dynamic tokens (EQ-DYNAMIC-TOKENS),
one-liner latency model (EQ-ONE-LINER), cost delta (EQ-COST-DELTA), chunk prefill (EQ-SINGLE-CHUNK).
"""

from __future__ import annotations

import re
from typing import Callable, List, Optional


_TRIGGER_RE = re.compile(
    r"\b(?:explain|describe|elaborate|illustrate|how\s+to|what\s+is\s+the\s+difference|compare|versus)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# EQ-MINIMAL-PROMPT — speedup_prefill = L_orig / L_min = 1 + B/C
# ---------------------------------------------------------------------------
def minimal_prompt_equation(
    original_prompt: str,
    minimal_prompt: str,
    tokenize_func: Callable[[str], int],
    alpha: float = 0.001,
) -> dict:
    l_orig = tokenize_func(original_prompt)
    l_min = tokenize_func(minimal_prompt)
    b = l_orig - l_min
    c = l_min
    t_orig = alpha * l_orig
    t_min = alpha * l_min
    speedup = t_orig / t_min if t_min > 0 else float("inf")
    return {
        "original_tokens": l_orig,
        "minimal_tokens": l_min,
        "boilerplate_tokens": b,
        "core_tokens": c,
        "speedup_factor": speedup,
        "time_saved_sec": t_orig - t_min,
    }


def build_minimal_prompt(
    query: str,
    chunks: List[str],
    one_liner: bool = False,
    tiny_prefix: str = "Answer concisely.\n",
) -> str:
    """Minimal RAG format: chunk + Q/A (§12th equation)."""
    parts = []
    if one_liner:
        parts.append("Answer in one sentence.\n")
    elif tiny_prefix:
        parts.append(tiny_prefix)
    if chunks:
        parts.append(chunks[0] + "\n\n")
    parts.append(f"Q: {query}\nA:")
    return "".join(parts)


def build_standard_prompt(query: str, chunks: List[str], system: str = "You are a helpful assistant.") -> str:
    p = f"system: {system}\n"
    if chunks:
        p += f"context:\n{chunks[0]}\n"
    p += f"user: {query}\nassistant:"
    return p


# ---------------------------------------------------------------------------
# EQ-DYNAMIC-TOKENS — max_new = min(max_cap, base + multiplier * word_count)
# ---------------------------------------------------------------------------
def compute_dynamic_max_tokens(
    query: str,
    base: int = 20,
    multiplier: float = 2.0,
    max_cap: int = 256,
    long_answer_cap: int = 150,
    floor: int = 10,
    max_query_words: int = 50,
) -> int:
    """Dynamic token budget (§9th equation)."""
    if not isinstance(query, str) or not query.strip():
        return base
    words = query.strip().split()
    word_count = min(len(words), max_query_words)
    if _TRIGGER_RE.search(query):
        return min(long_answer_cap, max_cap)
    dynamic = base + int(multiplier * word_count)
    return max(floor, min(dynamic, max_cap))


# ---------------------------------------------------------------------------
# EQ-ONE-LINER — latency = overhead + tokens * time_per_token
# ---------------------------------------------------------------------------
def simulate_generation_latency(
    output_tokens: int,
    time_per_token_ms: float = 10.0,
    overhead_ms: float = 50.0,
) -> float:
    return overhead_ms + output_tokens * time_per_token_ms


def one_liner_speedup(
    normal_tokens: int = 100,
    one_liner_tokens: int = 18,
    time_per_token_ms: float = 10.0,
    overhead_ms: float = 50.0,
) -> float:
    lat_n = simulate_generation_latency(normal_tokens, time_per_token_ms, overhead_ms)
    lat_o = simulate_generation_latency(one_liner_tokens, time_per_token_ms, overhead_ms)
    return lat_n / lat_o if lat_o > 0 else float("inf")


# ---------------------------------------------------------------------------
# EQ-COST-DELTA — cost_delta = c_prompt*L_add - c_output*delta_L_out
# ---------------------------------------------------------------------------
def cost_delta(
    l_add: int,
    delta_l_out: int,
    c_prompt_per_token: float = 0.0,
    c_output_per_token: float = 1.0,
) -> float:
    return c_prompt_per_token * l_add - c_output_per_token * delta_l_out


# ---------------------------------------------------------------------------
# EQ-SINGLE-CHUNK / EQ-PREFILL-CHUNK
# N_prompt = N_sys + N_query + N_ctx * (chunk_limit / total_chunks)
# ---------------------------------------------------------------------------
def estimate_prefill_ms(
    n_sys: int,
    n_query: int,
    n_ctx: int,
    chunk_limit: int,
    total_chunks: int = 5,
    alpha: float = 0.4,
    beta: float = 100.0,
) -> float:
    if total_chunks == 0:
        effective_ctx = 0
    else:
        effective_ctx = n_ctx * (chunk_limit / total_chunks)
    n_prompt = n_sys + n_query + effective_ctx
    return alpha * n_prompt + beta


def predict_prefill_time_and_speedup(
    l_sys: int,
    l_query: int,
    c: float,
    n_old: int = 5,
    n_new: int = 1,
    alpha_ms_per_token: float = 0.8,
) -> dict:
    tokens_old = l_sys + l_query + n_old * c
    tokens_new = l_sys + l_query + n_new * c
    prefill_old = alpha_ms_per_token * tokens_old
    prefill_new = alpha_ms_per_token * tokens_new
    return {
        "prefill_old_ms": prefill_old,
        "prefill_new_ms": prefill_new,
        "delta_ms": prefill_old - prefill_new,
        "speedup_ratio": prefill_old / prefill_new if prefill_new > 0 else float("inf"),
    }
