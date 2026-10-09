"""Length-based token budgets (ported from the legacy speed_engine).

EQ-DYNAMIC-TOKENS: max_new = min(max_cap, base + multiplier * word_count),
with a long-answer cap for explainer triggers and a floor so tiny caps
never strangle a reply.  Short questions get short budgets (less
generation = less latency); explainers keep room to finish.

Unlike confidence-based budgets (routing_math.confidence_token_budget),
this keys off question length alone, so it is safe to apply wherever
the caller did NOT specify a budget.
"""

from __future__ import annotations

import re

_TRIGGER_RE = re.compile(
    r"\b(?:explain|describe|elaborate|illustrate|how\s+to|what\s+is\s+the\s+difference|compare|versus)\b",
    re.IGNORECASE,
)

# Reasoning markers: thinking models burn ~10x the visible budget in
# hidden chain-of-thought (measured: qwen3 needs 2000+ hidden tokens on
# GSM8K), so ANY small cap truncates the answer into garbage.  These
# queries get NO dynamic cap (None) -- the configured default applies.
# A short budget here is not "faster", it is a shorter wrong answer.
_REASON_RE = re.compile(
    r"\b(?:prove|proof|why|derive|derivation|show\s+that|theorem)\b",
    re.IGNORECASE,
)


def dynamic_max_tokens(
    query: str,
    base: int = 20,
    multiplier: float = 2.0,
    max_cap: int = 256,
    long_answer_cap: int = 150,
    floor: int = 10,
    max_query_words: int = 50,
) -> int | None:
    """Default generation budget from question length.

    Returns ``base`` for empty input; explainer triggers get the long
    cap; reasoning markers return None (no dynamic cap -- the caller
    must fall back to the configured default, never to a small number).
    Everything else scales with word count inside [floor, max_cap].
    """
    if not isinstance(query, str) or not query.strip():
        return base
    if _REASON_RE.search(query):
        return None
    words = query.strip().split()
    word_count = min(len(words), max_query_words)
    if _TRIGGER_RE.search(query):
        return min(long_answer_cap, max_cap)
    dynamic = base + int(multiplier * word_count)
    return max(floor, min(dynamic, max_cap))
