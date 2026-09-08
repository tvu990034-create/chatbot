"""
optimization module.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model-routing / load-aware / budget / consensus-optimizations family ported from
the user's documentation.

This module covers:
  - #600  final router -> select_model
  - #10   EWMA / load-aware gating
  - #593  budget-aware router
  - #230  request routing (latency-budget / deadline partitioning)
  - CALM #45 and LayerSkip #50  difficulty-based routing
  - Zipf cache sizing
  - confidence-gated token budgets
  - resource-constrained consensus / hysteresis / load surcharge

Pure stdlib. The only cross-module import is ``estimate_tokens`` from
``gateway.opt_core`` (consistent with CONVENTIONS.md). No training loop, no GPU,
no model weights, no network.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from gateway.opt_core import estimate_tokens

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp to [lo, hi]; non-finite values resolve to a bound (no NaN)."""
    if not math.isfinite(x):
        return hi if x > 0 else lo
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# 1. EWMA load tracking  (#10 EWMA / load-aware gating)
# ---------------------------------------------------------------------------


class LoadBalancer:
    """Eq R1 — per-model Exponential Weighted Moving Average (EWMA) of in-flight
    / queue depth: ``ewma = alpha*cur + (1-alpha)*ewma``.

    Deterministic. ``recommend()`` returns the least-loaded model (lowest EWMA).
    """

    # Eq R1 — EWMA load tracker (source: "a-tier" / "s-tier" (#10 load-aware gating))
    def __init__(self, alpha: float = 0.5, models: Optional[Sequence[str]] = None):
        self.alpha = _clamp(float(alpha), 0.0, 1.0)
        self._load: Dict[str, float] = {m: 0.0 for m in (models or [])}

    def observe(self, model: str, queue_depth: float) -> None:
        """Update the EWMA for one model with a fresh queue-depth sample."""
        cur = max(0.0, float(queue_depth))
        prev = self._load.get(model, 0.0)
        self._load[model] = self.alpha * cur + (1.0 - self.alpha) * prev

    def load(self, model: str) -> float:
        """Current EWMA load for a model (0.0 if never observed)."""
        return float(self._load.get(model, 0.0))

    def global_load(self) -> float:
        """Mean EWMA across all known models (0.0 if none)."""
        vals = list(self._load.values())
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def recommend(self) -> Optional[str]:
        """Least-loaded known model (lowest EWMA). None if none tracked."""
        if not self._load:
            return None
        return min(self._load, key=self._load.get)

    def reset(self) -> None:
        self._load = {m: 0.0 for m in self._load}

    def stats(self) -> Dict[str, float]:
        return dict(self._load)


# ---------------------------------------------------------------------------
# 2. Load-aware gating (mirror of select_model residency logic)
# ---------------------------------------------------------------------------


def select_idle_model(loaded, candidate, state_loads, min_dwell, now, last_switch):
    """Eq R2 — load-aware gating mirroring ``opt_core.select_model`` residency
    logic, standalone and pure.

    Returns ``candidate`` when it should take over, otherwise the resident
    ``loaded`` model. The resident model wins if (a) a switch would violate the
    min-dwell residency window AND the expected output is short, or (b) the
    candidate is heavily queued relative to the resident.

    Provenance: #600 (final router -> select_model), #10 (load-aware gating).
    """
    # (source: "#600 select_model / #10 load-aware gating")
    if loaded is None:
        loaded = ""
    if candidate is None:
        candidate = ""
    if not candidate:
        return loaded if loaded else candidate
    if not loaded or loaded == candidate:
        return candidate

    dwell_ok = (now - last_switch) >= max(0.0, min_dwell)
    if not dwell_ok:
        # Keep resident unless we have *very* strong reason to move.
        cq = int(state_loads.get(candidate, 0))
        lq = int(state_loads.get(loaded, 0))
        if cq > lq + 2:
            return candidate
        return loaded

    # Dwell satisfied: prefer candidate unless it is heavily queued behind the
    # resident model.
    cq = int(state_loads.get(candidate, 0))
    lq = int(state_loads.get(loaded, 0))
    if cq > lq + 2:
        return loaded
    return candidate


# ---------------------------------------------------------------------------
# 3. Budget-aware router  (#593 budget-aware router)
# ---------------------------------------------------------------------------


class BudgetAwareRouter:
    """Eq R3 — token-budget-aware router between {fast_model, strong_model,
    fallback}. ``route`` picks a model from the per-token linear cost model and a
    total token budget; ``best_model_for_budget`` picks the model maximizing
    value within a budget (greedy knapsack-lite, small ``n`` so O(n*B) is not
    needed).
    """

    # Eq R3 — budget-aware routing (source: "#593 budget-aware router")
    def __init__(
        self,
        fast_model: str = "fast",
        strong_model: str = "strong",
        fallback: str = "fallback",
        cost_per_token: Optional[Dict[str, float]] = None,
        value: Optional[Dict[str, float]] = None,
        fallback_token_threshold: int = 256,
    ):
        self.fast_model = fast_model
        self.strong_model = strong_model
        self.fallback = fallback
        self.cost_per_token = dict(cost_per_token or {
            fast_model: 0.01,
            strong_model: 0.05,
            fallback: 0.001,
        })
        self.value = dict(value or {
            fast_model: 0.4,
            strong_model: 1.0,
            fallback: 0.1,
        })
        self.fallback_token_threshold = fallback_token_threshold
        self.models = [fast_model, strong_model, fallback]

    def cost_estimate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Linear cost model: cpt * (input + output). Returns 0.0 for an
        unmodeled name (safe default, never raises)."""
        cpt = float(self.cost_per_token.get(model, 0.0))
        in_t = max(0, int(input_tokens or 0))
        out_t = max(0, int(output_tokens or 0))
        return cpt * (in_t + out_t)

    def route(self, query_tokens: int, expected_output_tokens: int, budget: float) -> str:
        """Pick the best model affordable under ``budget``:
        - small/cheap requests go fast;
        - large expected output goes strong when affordable;
        - otherwise fallback."""
        budget = max(0.0, float(budget))
        total = max(0, int(query_tokens or 0)) + max(0, int(expected_output_tokens or 0))
        if total <= 0:
            return self.fast_model
        if self.cost_estimate(self.strong_model, total, 0) <= budget:
            return self.strong_model
        if self.cost_estimate(self.fast_model, total, 0) <= budget:
            return self.fast_model
        return self.fallback

    def best_model_for_budget(self, knowledge_costs: Dict[str, float], budget: float) -> str:
        """Greedy knapsack-lite: choose the model that maximizes value-per-cost
        subject to ``budget``. ``knowledge_costs`` maps model name -> fixed cost.
        Falls back to ``fallback`` when nothing fits."""
        budget = max(0.0, float(budget))
        best = None
        best_val = -1.0
        for m in self.models:
            fixed = float(knowledge_costs.get(m, 0.0))
            if fixed <= budget:
                v = self.value.get(m, 0.0) / (fixed if fixed > 0 else 1e-12)
                if v > best_val:
                    best_val = v
                    best = m
        return best if best is not None else self.fallback

    def reset(self) -> None:
        """No mutable runtime state beyond config; nothing to reset."""

    def stats(self) -> Dict[str, object]:
        return {
            "models": list(self.models),
            "cost_per_token": dict(self.cost_per_token),
            "value": dict(self.value),
        }


# ---------------------------------------------------------------------------
# 4. Difficulty routing  (CALM #45 / LayerSkip #50)
# ---------------------------------------------------------------------------

# Known hard keywords that strongly suggest a strong model is worthwhile.
_HARD_KEYWORDS = (
    "proof", "prove", "theorem", "lemma", "derive", "derivation", "optimize",
    "complexity", "time complexity", "space complexity", "deadlock", "race condition",
    "distributed", "consensus", "concurrency", "gradient", "convergence", "big-o",
)


def _rare_word_density(text: str) -> float:
    """Fraction of tokens (alphabetic, len>4) that are not in a common-word set."""
    words = [w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w]
    if not words:
        return 0.0
    long = [w for w in words if len(w) > 4]
    if not long:
        return 0.0
    common = {
        "about", "after", "again", "where", "which", "while", "their", "there",
        "these", "those", "would", "could", "should", "because", "between",
        "through", "during", "before", "another", "however", "therefore",
        "important", "whenever", "something", "nothing", "everything", "people",
    }
    rare = sum(1 for w in long if w not in common)
    return rare / len(long)


def difficulty_score(query: str) -> float:
    """Eq R4 — difficulty score in [0,1] combining length, rare-word density and
    known-hard keywords. Higher = harder.

    Provenance: CALM #45 (early-exit by difficulty), LayerSkip #50 (skip layers
    on easy inputs).
    """
    # (source: "CALM #45 / LayerSkip #50 difficulty routing")
    text = query or ""
    if not text.strip():
        return 0.0
    length = _clamp(estimate_tokens(text) / 300.0)          # longer -> harder
    rare = _clamp(_rare_word_density(text))                  # uncommon vocab
    hits = sum(1 for k in _HARD_KEYWORDS if k in text.lower())
    keyword = _clamp(hits / 3.0)                             # harder keywords
    return _clamp(0.35 * length + 0.30 * rare + 0.35 * keyword)


def should_route_to_strong(difficulty: float, threshold: float = 0.6) -> bool:
    """Eq R5 — route to a strong model iff difficulty >= threshold."""
    return float(difficulty) >= float(threshold)


# ---------------------------------------------------------------------------
# 5. Confidence-gated token budget  (#230))
# ---------------------------------------------------------------------------


def confidence_token_budget(confidence, m_min=20, m_max=256, theta=5.0, gate=0.45):
    """Eq R6 — clone of ``perf_math.eq8_confidence_to_max_tokens`` semantics:
    linear + power mapping of a confidence score to a token budget in
    [m_min, m_max]. Returns ``None`` below ``gate`` (hard gate).

    Provenance: #230 (request routing / budget).
    """
    # (source: "#230 confidence-gated token budget")
    c = float(confidence)
    if c < float(gate):
        return None
    denom = 1.0 - float(gate)
    normalized = (c - float(gate)) / denom if denom > 0 else 1.0
    scaled = normalized ** max(0.0, float(theta))
    budget = float(m_min) + scaled * (float(m_max) - float(m_min))
    return int(budget)


# ---------------------------------------------------------------------------
# 6. Zipf cache sizing
# ---------------------------------------------------------------------------


def zipf_popularity(rank: int, s: float = 1.0) -> float:
    """Eq R7 — Zipf popularity of the ``rank``-th most popular item:
    R(rank) = 1/rank^s. Guarded: rank >= 1, s >= 0."""
    rank = max(1, int(rank))
    s = max(0.0, float(s))
    if s == 0.0:
        return 1.0
    return 1.0 / (rank ** s)


def cache_hit_ratio(rank_kept: int, n_items: int, s: float = 1.0) -> float:
    """Eq R8 — fraction of total request mass covered by caching the top
    ``rank_kept`` most-popular of ``n_items`` items under a Zipf distribution.
    """
    n_items = max(1, int(n_items))
    rank_kept = int(rank_kept)
    if rank_kept <= 0:
        return 0.0
    rank_kept = min(rank_kept, n_items)
    total = sum(zipf_popularity(r, s) for r in range(1, n_items + 1))
    if total <= 0:
        return 0.0
    kept = sum(zipf_popularity(r, s) for r in range(1, rank_kept + 1))
    return _clamp(kept / total)


def zipf_capacity_hit(hit_rate_target: float, n_items: int, s: float = 1.0) -> int:
    """Eq R9 — number of most-popular items (by Zipf rank ~ C/rank^s) needed to
    cover a target cumulative hit fraction. Returns 0 for an unsatisfiable
    target, or ``n_items`` when everything is needed (safe default, never
    raises)."""
    n_items = max(1, int(n_items))
    hit_rate_target = _clamp(float(hit_rate_target), 0.0, 1.0)
    if hit_rate_target <= 0.0:
        return 0

    total = sum(zipf_popularity(r, s) for r in range(1, n_items + 1))
    if total <= 0:
        return 0

    cum = 0.0
    for r in range(1, n_items + 1):
        cum += zipf_popularity(r, s)
        if cum / total >= hit_rate_target:
            return r
    return n_items  # target unreachable with only n_items items


# ---------------------------------------------------------------------------
# 7. Latency budget / deadline partitioning  (#230 request routing)
# ---------------------------------------------------------------------------


def deadline_scheduler(deadline_ms: float, stages: Sequence[float]) -> List[float]:
    """Eq R10 — split a total deadline (ms) across stages proportional to a
    weight vector, returning per-stage budgets. Degenerate inputs (empty stages,
    zero total weight) yield a safe even or zero split (never raises)."""
    deadline_ms = max(0.0, float(deadline_ms))
    stages = [max(0.0, float(w)) for w in (stages or [])]
    if not stages:
        return []
    total = sum(stages)
    if total <= 0:
        share = deadline_ms / len(stages)
        return [share for _ in stages]
    return [deadline_ms * (w / total) for w in stages]


def feasible_under_deadline(est_stage_times: Sequence[float], deadline_ms: float) -> bool:
    """Eq R11 — True iff the sum of estimated per-stage latencies is within the
    total deadline. Empty estimates are treated as instant and therefore always
    feasible."""
    est_stage_times = [max(0.0, float(t)) for t in (est_stage_times or [])]
    if not est_stage_times:
        return True
    return sum(est_stage_times) <= max(0.0, float(deadline_ms))


# ---------------------------------------------------------------------------
# 8. Hysteresis / switch counter
# ---------------------------------------------------------------------------


def should_switch(new_score: float, old_score: float, hysteresis: float = 0.1) -> bool:
    """Eq R12 — only switch when the new score beats the old by at least the
    hysteresis margin, preventing thrashing on marginal differences."""
    return (float(new_score) - float(old_score)) >= float(hysteresis)


# ---------------------------------------------------------------------------
# 9. Resource-constrained consensus
# ---------------------------------------------------------------------------


def majority_consensus(votes: Sequence) -> Tuple[Optional[object], float]:
    """Eq R13 — most frequent option and its fraction of total votes. Returns
    (None, 0.0) on empty input (safe default, never raises)."""
    if not votes:
        return (None, 0.0)
    counts: Dict[object, int] = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    top = max(counts, key=counts.get)
    frac = counts[top] / len(votes)
    return (top, _clamp(frac))


def consensus_clearance(agreement_frac: float, threshold: float = 0.5) -> bool:
    """Eq R14 — True iff the agreement fraction reaches the required threshold."""
    return _clamp(float(agreement_frac)) >= float(threshold)


# ---------------------------------------------------------------------------
# 10. Load surcharge on routing score
# ---------------------------------------------------------------------------


def load_adjusted_score(base_score: float, load_ewma: float, penalty: float = 1.0) -> float:
    """Eq R15 — routing score reduced by a penalty times current EWMA load:
    ``base_score - penalty*load_ewma``."""
    return float(base_score) - float(penalty) * max(0.0, float(load_ewma))


# ---------------------------------------------------------------------------
# NOT-IMPLEMENTABLE (training / hardware-only router items)
# ---------------------------------------------------------------------------

# NOT_IMPLEMENTABLE: Requires a policy-gradient (REINFORCE/PPO) training loop,
# reward signals from production traffic and gradient updates across requests.
NOT_IMPLEMENTABLE_PG_ROUTING = (
    "learned-policy-gradient:needs-training-loop & per-request reward"
)

# NOT_IMPLEMENTABLE: Per-layer early-exit depends on layer-wise hidden-state
# entropy / confidence that only exists inside a running model's forward pass.
NOT_IMPLEMENTABLE_LAYER_EXIT = (
    "per-layer-early-exit:needs-model-internal-activations (CALM #45 / LayerSkip #50)"
)

# NOT_IMPLEMENTABLE: Speculative-decoding budget allocation requires a KV-cache
# budget and a draft/target model pair loading — both hardware/kernel-dependent.
NOT_IMPLEMENTABLE_SPEC_DECODE = (
    "speculative-decode-budget:needs-kv-cache-budget & draft+target model weights"
)

# NOT_IMPLEMENTABLE: Hardware-aware quantization/perf-aware layer offload needs
# per-device memory/FLOPs tables and kernel profiles unavailable in pure Python.
NOT_IMPLEMENTABLE_HW_ROUTING = (
    "hardware-aware-routing:needs-device-memory & FLOPs/kernel profiles"
)


__all__ = [
    "LoadBalancer", "select_idle_model", "BudgetAwareRouter",
    "difficulty_score", "should_route_to_strong",
    "confidence_token_budget", "zipf_popularity", "cache_hit_ratio",
    "zipf_capacity_hit", "deadline_scheduler", "feasible_under_deadline",
    "should_switch", "majority_consensus", "consensus_clearance",
    "load_adjusted_score",
    "NOT_IMPLEMENTABLE_PG_ROUTING", "NOT_IMPLEMENTABLE_LAYER_EXIT",
    "NOT_IMPLEMENTABLE_SPEC_DECODE", "NOT_IMPLEMENTABLE_HW_ROUTING",
]
