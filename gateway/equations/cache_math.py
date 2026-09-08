"""
Cache-management optimization module.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Cache-management optimizations ported from the user's optimization catalog.

These replace the exact-match-only `SimpleCache` with learned/semantic scoring so
a near-miss query can be served from the closest stored answer and entries are
evicted by value, not just recency.

Pure stdlib. With no live embedding model available on a query, callers fall back
to `hashed_embedding`/`cosine_similarity` from `gateway.opt_core`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Shared – cosine + decay helpers (depend only on math / stdlib)
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(x):
        return hi if x > 0 else lo
    return max(lo, min(hi, x))


def cos_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 on empty/mismatched (never NaN)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if denom == 0.0:
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)) / denom)


def temporal_decay(now: float, created_at: float, half_life: float = 3600.0) -> float:
    """Eq T1 — exponential temporal decay in (0,1]: 2^{-dt/half_life}."""
    half_life = half_life if half_life and half_life > 0 else 3600.0
    dt = max(0.0, now - created_at)
    return math.pow(0.5, dt / half_life)


# ---------------------------------------------------------------------------
# Adaptive similarity threshold  (Tài liệu (8): threshold optimizations)
# ---------------------------------------------------------------------------


class AdaptiveThreshold:
    """Eq T2 — adaptive similarity gate bu + mu*sigma over streaming Recall@K
    scores. Mirror of 'adaptive threshold mu+z*sigma' in the docs."""

    def __init__(self, base: float = 0.92, mu_weight: float = 1.0,
                 sigma_mult: float = 2.0, min_threshold: float = 0.70,
                 max_threshold: float = 0.99):
        self.base = base
        self.mu_weight = mu_weight
        self.sigma_mult = sigma_mult
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.scores: List[float] = []

    def observe(self, sim: float) -> None:
        self.scores.append(_clamp(sim))
        # Keep a windowed tail so the threshold tracks recent traffic.
        if len(self.scores) > 500:
            self.scores = self.scores[-500:]

    def threshold(self) -> float:
        if not self.scores:
            return self.base
        n = len(self.scores)
        mu = sum(self.scores) / n
        var = sum((s - mu) ** 2 for s in self.scores) / n
        sigma = math.sqrt(var)
        t = self.base + self.mu_weight * (mu - self.base) + self.sigma_mult * sigma
        return _clamp(t, self.min_threshold, self.max_threshold)

    def hit(self, sim: float, extra_margin: float = 0.02) -> bool:
        return sim >= (self.threshold() - extra_margin)

    def reset(self) -> None:
        self.scores.clear()


# ---------------------------------------------------------------------------
# Entry value / eviction
# ---------------------------------------------------------------------------

class EntryScore:
    """Eq T3 — composite cache-entry score combining recency, frequency,
    cost (tokens saved) and semantic demand."""

    def __init__(self, age_half_life: float = 3600.0):
        self.age_half_life = age_half_life

    def score(self, *, freq: int = 1, last_access: float, now: float,
              token_cost: int = 1, demand: float = 0.0) -> float:
        """score = blend of normalized frequency, recency decay, token cost and
        exogenous demand.  Used for {'utility-based' rather than LRU} eviction."""

        def _norm_freq(f: int) -> float:
            return _clamp(math.log1p(max(0, f)) / math.log1p(100))

        def _norm_tokens(t: int) -> float:
            return _clamp(math.log1p(max(0, t)) / math.log1p(4096))

        recency = temporal_decay(now, last_access, self.age_half_life)
        value = (0.45 * recency
                 + 0.30 * _norm_freq(freq)
                 + 0.15 * _norm_tokens(token_cost)
                 + 0.10 * _clamp(demand))
        return _clamp(value)


def logistic_eviction_probability(value: float, cap_value: float = 0.5,
                                 steepness: float = 12.0) -> float:
    """Eq T4 — logistic probability an entry is evicted given its score.
    Lower score -> higher eviction probability: p = sigmoid(k*(cap-v))."""
    steepness = steepness if steepness and steepness > 0 else 12.0
    z = steepness * (cap_value - value)
    try:
        p = 1.0 / (1.0 + math.exp(-z))
    except OverflowError:
        p = 1.0 if z > 0 else 0.0
    return _clamp(p)


# ---------------------------------------------------------------------------
# KL staleness (answer drift)
# ---------------------------------------------------------------------------


def kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """KL(P||Q) over two probability vectors; 0.0 when aligned."""
    if not p or not q or len(p) != len(q):
        return 0.0
    total = 0.0
    for a, b in zip(p, q):
        if a <= 0:
            continue
        b = b if b > 0 else 1e-12
        total += a * math.log(a / b)
    return max(0.0, total)


def staleness_score(old_dist: Sequence[float], new_dist: Sequence[float],
                    ttl: float = 3600.0, age: float = 0.0) -> float:
    """Eq T5 — composite answer-staleness = KL drift scaled by entry age.
    Higher = more stale -> candidate for refresh/recompute."""
    drift = kl_divergence(old_dist, new_dist)
    age_factor = _clamp(age / ttl) if ttl else _clamp(age / 3600.0)
    return _clamp(drift * (0.5 + 0.5 * age_factor))


# ---------------------------------------------------------------------------
# Semantic cache hit + answer blending
# ---------------------------------------------------------------------------


class SemanticCacheMath:
    """Eq T6..T8 — two-stage similarity lookup with entry-value-aware ranking
    and optional answer blending of the top-k neighbors.

    Deterministic. No external model: embeddings come from the caller (which may
    use `hashed_embedding`)."""

    def __init__(self, k: int = 3, blend_alpha: float = 0.7,
                 mu_sigma_threshold: Optional[AdaptiveThreshold] = None):
        self.k = max(1, k)
        self.blend_alpha = _clamp(blend_alpha)
        self.threshold = mu_sigma_threshold or AdaptiveThreshold(base=0.92)
        self.entries: Dict[str, Dict] = {}

    # -- two-stage ranking ---------------------------------------------------
    def rank(self, query_emb: Sequence[float],
             candidate_embeddings: Dict[str, Sequence[float]],
             now: float) -> List[Tuple[str, float]]:
        """Stage 1: coarse filtered by cosine; returns (key, score) desc."""
        scored: List[Tuple[str, float]] = []
        for key, emb in candidate_embeddings.items():
            sim = cos_sim(query_emb, emb)
            if sim >= self.threshold.threshold():
                scored.append((key, sim))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[: self.k]

    def blend_answers(self, neighbors: Sequence[Tuple[str, str, float]],
                      exact: Optional[str]) -> str:
        """Eq T8 — weighted answer blending over top-k. If ``exact`` provided and
        non-empty, it wins unconditionally (serving the identical answer preserves
        correctness); otherwise weight each neighbor by Alpha-similarity."""
        if exact and exact.strip():
            return exact
        if not neighbors:
            return ""
        numer = 0.0
        score_sum = 0.0
        for _key, text, sim in neighbors:
            w = _clamp(sim) ** _clamp(self.blend_alpha)
            numer += w * len(text)
            score_sum += w
        if score_sum <= 0:
            return neighbors[0][1]
        target = numer / score_sum
        # Return the neighbor whose length is closest to the blended target.
        return min(neighbors, key=lambda t: abs(len(t[1]) - target))[1]

    def insert_score(self, value: float, now: float, token_cost: int = 1) -> None:
        """Eq T9 — insertion cost/benefit gate: only admit entries whose value is
        high enough to justify occupying scarce cache space."""
        benefit = EntryScore(age_half_life=3600.0).score(
            freq=1, last_access=now, now=now, token_cost=token_cost,
            demand=value)
        return benefit


# ---------------------------------------------------------------------------
# Answer/value-aware eviction policy driver
# ---------------------------------------------------------------------------


def evict_scores(entries: Sequence[Tuple[str, Dict]], now: float,
                 keep_ratio: float = 0.7, age_half_life: float = 3600.0) -> List[str]:
    """Eq T10 — computes per-entry value and returns the keys to evict
    (lowest-value first) up to (1 - keep_ratio) of the store."""
    scorer = EntryScore(age_half_life=age_half_life)
    keyed: List[Tuple[str, float]] = []
    for key, meta in entries:
        value = scorer.score(
            freq=int(meta.get("freq", 1)),
            last_access=float(meta.get("last_access", 0)),
            now=now,
            token_cost=int(meta.get("token_cost", 1)),
            demand=float(meta.get("demand", 0.0)),
        )
        keyed.append((key, value))
    keyed.sort(key=lambda t: t[1])
    evict_n = max(0, len(keyed) - int(len(keyed) * float(keep_ratio)))
    return [k for k, _ in keyed[:evict_n]]


__all__ = [
    "cos_sim", "temporal_decay", "AdaptiveThreshold", "EntryScore",
    "logistic_eviction_probability", "kl_divergence", "staleness_score",
    "SemanticCacheMath", "evict_scores",
]