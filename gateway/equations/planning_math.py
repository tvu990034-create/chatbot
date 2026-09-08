"""
Planning / reasoning / verification optimization module.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Planning / reasoning / verification optimization family from the user's optimization catalog.

Everything here is pure stdlib and runs with no model: all inputs are scalar
statistics, per-step values, or already-sampled candidate answers.  The
model-weight-dependent steps (policy-value networks, learned value heads,
reward-model-driven self-refine) have no implementation in this runtime and are
declared as NOT_IMPLEMENTABLE_* constants with provenance at the bottom so that
every optimization from the spec has a home.

A later wiring agent calls these from the live path; nothing here performs IO,
network access or model loading.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional, Sequence, Tuple

# Shared primitives come only from gateway.opt_core per CONVENTIONS.md.  A
# guarded import keeps this module standalone-usable if opt_core is missing.
try:  # pragma: no cover - defensive
    from gateway.opt_core import (
        normalize_math_answer as _normalize_math_answer,
        vote_responses as _vote_responses,
    )
except Exception:  # pragma: no cover
    _normalize_math_answer = None
    _vote_responses = None


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(x):
        return hi if x > 0 else lo
    return max(lo, min(hi, x))


def _canon(vote: Any) -> str:
    """Canonical identity of an answer: math-normalized when possible."""
    if _normalize_math_answer is not None:
        key = _normalize_math_answer(str(vote))
        if key:
            return key
    s = str(vote).strip()
    return s.lower() if s else ""


def _inv_normal_cdf(p: float, lo: float = -6.0, hi: float = 6.0) -> float:
    """Inverse standard-normal CDF (Beasley-Springer-Moro, clamped edges).

    Pure stdlib replacement for scipy.stats.norm.ppf used only inside
    best-of-N math.  Returns sensible finite bounds at the tails, never NaN.
    """
    p = _clamp(p)
    if p <= 0.0:
        return lo
    if p >= 1.0:
        return hi
    if p == 0.5 or not math.isfinite(p):
        return 0.0
    t = math.sqrt(-2.0 * math.log(p if p < 0.5 else 1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x = t - (c0 + c1 * t + c2 * t * t) / (
        1.0 + d1 * t + d2 * t * t + d3 * t * t * t
    )
    return -x if p < 0.5 else x


# ---------------------------------------------------------------------------
# MCTS / UCT
# ---------------------------------------------------------------------------


def uct_score(win_frac: float, parent_visits: int, node_visits: int,
              c: float = 1.41) -> float:
    """EqP1 - UCT node-selection score.

    score = win_frac + c * sqrt(log(parent_visits) / (node_visits + 1)).
    Unvisited nodes return +inf so unexplored children are always chosen first.
    ("s tier.txt", "mcts-uct")
    """
    score = _clamp(win_frac)
    visits = max(0, int(node_visits))
    if visits == 0:
        return math.inf
    c = c if math.isfinite(c) and c > 0 else 1.41
    parent = max(1.0, float(parent_visits))
    if parent <= 1.0:
        return score
    exploration = c * math.sqrt(math.log(parent) / (visits + 1))
    return score + exploration


def simulation_budget(time_remaining_ms: float, per_sim_ms: float,
                      max_sims: int = 100) -> int:
    """EqP2 - number of MCTS simulations affordable before the deadline.

    floor(time_remaining / per_sim) capped at max_sims; 0 if no time left.
    Unknown (<=0 / non-finite) per-sim cost trusts the caller's cap.
    ("reasoning docs", "mcts-budget")
    """
    cap = max(0, int(max_sims))
    if cap == 0:
        return 0
    t = max(0.0, float(time_remaining_ms))
    if not math.isfinite(t):
        return cap
    p = float(per_sim_ms)
    if not math.isfinite(p) or p <= 0.0:
        return cap
    return min(cap, int(t // p))


# ---------------------------------------------------------------------------
# Self-consistency voting
# ---------------------------------------------------------------------------


def agreement_fraction(votes: Sequence[Any]) -> float:
    """EqP3 - fraction of votes belonging to the majority normalized answer.

    0.0 for an empty ballot; 1.0 when every vote is the same answer.
    Answers are deduped via ``gateway.opt_core.normalize_math_answer`` when
    available (so "\\boxed{7}" == "7" == "7.0").
    ("s tier.txt", "self-consistency")
    """
    votes = list(votes or [])
    n = len(votes)
    if n == 0:
        return 0.0
    counts = {}
    for v in votes:
        key = _canon(v)
        counts[key] = counts.get(key, 0) + 1
    return _clamp(max(counts.values()) / n)


def should_verify(agreement_frac: float, threshold: float = 0.6) -> bool:
    """EqP4 - confidence gate: low self-consistency sends the answer to a
    verification step instead of committing it.

    Verify when agreement < threshold (the "verify before trust" rule).
    ("reasoning docs", "verify-before-commit")
    """
    return _clamp(agreement_frac) < threshold


def cisc_style_confidence(responses: Sequence[Any]) -> float:
    """EqP5 - self-consistency confidence: fraction of the most-common
    normalized answer (mirror of ``gateway.opt_core.cisc_confidence``).
    ("s tier.txt", "self-consistency")
    """
    return agreement_fraction(responses)


def answer_consensus_check(answers: Sequence[Any],
                           threshold: float = 0.6) -> Tuple[str, float, bool]:
    """EqP6 - calibration of a candidate-answer set.

    Returns (best, agreement, is_confident) where ``best`` is the voted answer
    (boxed member preferred), agreement in [0,1] and is_confident True only
    above the threshold.  Empty input -> ('', 0.0, False), never raises.
    ("s tier.txt", "self-consistency-aggregation")
    """
    answers = list(answers or [])
    if not answers:
        return "", 0.0, False
    best = _vote_responses(answers) if _vote_responses is not None else str(answers[0])
    agreement = agreement_fraction(answers)
    return best, agreement, not should_verify(agreement, threshold)


# ---------------------------------------------------------------------------
# Best-of-N selection
# ---------------------------------------------------------------------------


def best_of_n_expected(mu: float, sigma: float, n: int) -> float:
    """EqP7 - expected quality of the best of n i.i.d. samples.

    E[max] ~= mu + sigma * Phi^-1((n - 0.375) / (n + 0.25)) via the normal
    order-statistic approximation.  Self-contained pure math (local inverse
    normal CDF); n=1 -> exactly mu; sigma<=0 -> mu; never NaN.
    ("a tier.txt", "best-of-n")
    """
    n = max(0, int(n))
    if n == 0:
        return float(mu)
    sigma = abs(float(sigma))
    if sigma == 0.0:
        return float(mu)
    p = (n - 0.375) / (n + 0.25)
    return float(mu) + sigma * _inv_normal_cdf(p)


def optimal_n(mu: float, sigma: float, cost_per_sample: float = 0.005,
              max_n: int = 8) -> int:
    """EqP8 - best sample count trading expected best quality vs sampling cost.

    Maximizes best_of_n_expected(mu, sigma, n) - n*cost_per_sample over
    [1, max_n].  Tiny sigma -> 1 (extra samples buy nothing); large sigma ->
    >1.  Invalid inputs collapse to the safe default of 1.
    ("a tier.txt", "best-of-n-optimal")
    """
    sigma = abs(float(sigma))
    cost = max(0.0, float(cost_per_sample)) if math.isfinite(cost_per_sample) else 0.0
    max_n = max(1, int(max_n))
    if sigma <= 0.0:
        return 1
    best_n = 1
    best_val = best_of_n_expected(mu, sigma, 1) - cost
    for n in range(2, max_n + 1):
        val = best_of_n_expected(mu, sigma, n) - n * cost
        if val > best_val:
            best_val, best_n = val, n
    return best_n


def best_candidate(candidates: Sequence[Any],
                   score_fn: Optional[Callable[[Any], float]] = None) -> Optional[Any]:
    """EqP9 - pick the argmax candidate under a scoring function.

    Ties resolve to the first candidate; empty input -> None.
    Default scorer treats each candidate's float value as its own score.
    ("reasoning docs", "best-of-n-select")
    """
    cands = list(candidates or [])
    if not cands:
        return None

    def _default_score(c: Any) -> float:
        try:
            return float(c)
        except (TypeError, ValueError):
            return 0.0

    scorer = score_fn if score_fn is not None else _default_score
    return max(cands, key=scorer)


# ---------------------------------------------------------------------------
# Verification scoring
# ---------------------------------------------------------------------------


def verification_score(candidate: Any, checks: Sequence[Any]) -> float:
    """EqP10 - weighted fraction of passed verification checks for a candidate.

    ``checks`` items are (passed: bool, weight: float) tuples; a bare bool is
    treated as weight 1.0.  Empty check list -> 0.0 (nothing verified yet),
    result is clamped to [0,1].  ``candidate`` is carried for provenance only.
    ("reasoning docs", "verification")
    """
    total_weight = 0.0
    passed_weight = 0.0
    for item in (checks or []):
        if isinstance(item, tuple):
            passed, weight = (item + (1.0,))[:2]
        else:
            passed, weight = item, 1.0
        weight = float(weight)
        if not math.isfinite(weight):
            weight = 1.0 if weight > 0 else 0.0
        weight = max(0.0, weight)
        total_weight += weight
        if passed:
            passed_weight += weight
    if total_weight <= 0.0:
        return 0.0
    return _clamp(passed_weight / total_weight)


def prob_pass(score: float, n_checks: int) -> float:
    """EqP11 - probability a candidate passes, sharpened by check count.

    p = 0.5 + (score - 0.5) * (1 + 0.25*log1p(n_checks)); more independent
    checks push a middling score toward its side of 0.5.  Clamped to [0,1].
    ("reasoning docs", "verification-probability")
    """
    s = _clamp(score)
    n = max(0, int(n_checks))
    scale = 1.0 + 0.25 * math.log1p(n)
    return _clamp(0.5 + (s - 0.5) * scale)


# ---------------------------------------------------------------------------
# Lookahead value / reward aggregation
# ---------------------------------------------------------------------------


def max_q(state_values: Sequence[float], discount: float = 0.99) -> float:
    """EqP12 - discounted lookahead Q of a rollout: sum discount^i * value_i.

    Iterative reward aggregation; discount in [0,1] decays later steps.
    Empty rollout -> 0.0.  Non-finite values are skipped (never NaN output).
    ("reasoning docs", "lookahead")
    """
    discount = _clamp(float(discount), 0.0, 1.0)
    if not state_values:
        return 0.0
    total = 0.0
    factor = 1.0
    for v in state_values:
        if math.isfinite(v):
            total += float(v) * factor
        factor *= discount
    return total


def mean_q(values: Sequence[float]) -> float:
    """EqP13 - mean per-step value (average reward) of a rollout.

    Arithmetic mean of finite entries; empty -> 0.0, never NaN.
    ("reasoning docs", "lookahead")
    """
    finite = [v for v in (values or []) if math.isfinite(v)]
    if not finite:
        return 0.0
    return sum(finite) / len(finite)


# ---------------------------------------------------------------------------
# Branching control
# ---------------------------------------------------------------------------


def expand_if_promising(uct: float, threshold: float = 1.5) -> bool:
    """EqP14 - expand a tree node only when its UCT score looks worthwhile.

    +inf (unexplored child) is always promising; -inf never.  Purely a
    scoring gate; callers keep tree state themselves.
    ("reasoning docs", "mcts-expand")
    """
    if uct == math.inf:
        return True
    if uct == -math.inf:
        return False
    if not math.isfinite(uct):
        return False
    return uct >= threshold


def depth_limit_knowledge(difficulty: float, max_depth: int = 8) -> int:
    """EqP15 - deeper search budget for harder queries.

    limit = round(max_depth * 0.5 * (1 + difficulty)); clamped difficulty to
    [0,1], result at least 1.  Hard queries (difficulty ~1) get the full
    budget, trivial ones half of it.
    ("reasoning docs", "search-budget")
    """
    difficulty = _clamp(difficulty)
    max_depth = max(1, int(max_depth))
    return max(1, int(round(max_depth * 0.5 * (1.0 + difficulty))))


# ---------------------------------------------------------------------------
# NOT-IMPLEMENTABLE: model-weight / training-only planning steps
# ---------------------------------------------------------------------------
# Each constant records the provenance filename and why pure Python cannot
# implement it here.  Callers guard with getattr(settings, "enable_...", False).

# NOT_IMPLEMENTABLE: needs a trained policy-and-value network (weights) plus
# batched GPU rollout; pure math cannot produce neural prior log-probs.
NOT_IMPLEMENTABLE_POLICY_VALUE_NETWORK = (
    "mcts-neural-policy-value:needs-trained-network-weights & gpu-rollout"
)  # ("s tier.txt", "mcts")

# NOT_IMPLEMENTABLE: a learned value head requires gradient-descent training on
# outcome labels; no training loop / weights exist in this runtime.
NOT_IMPLEMENTABLE_LEARNED_VALUE_HEAD = (
    "value-head:needs-trained-value-head-weights & training-loop"
)  # ("reasoning docs", "learned-value")

# NOT_IMPLEMENTABLE: reward-model-driven iterative self-refine is an RLHF-style
# optimization loop needing a trained reward model and rollouts.
NOT_IMPLEMENTABLE_SELF_REFINE_REWARD_MODEL = (
    "iterative-self-refine:needs-reward-model + rlhf-loop"
)  # ("Tài liệu (28)", "self-refine")


__all__ = [
    "uct_score",
    "simulation_budget",
    "agreement_fraction",
    "should_verify",
    "cisc_style_confidence",
    "answer_consensus_check",
    "best_of_n_expected",
    "optimal_n",
    "best_candidate",
    "verification_score",
    "prob_pass",
    "max_q",
    "mean_q",
    "expand_if_promising",
    "depth_limit_knowledge",
    "NOT_IMPLEMENTABLE_POLICY_VALUE_NETWORK",
    "NOT_IMPLEMENTABLE_LEARNED_VALUE_HEAD",
    "NOT_IMPLEMENTABLE_SELF_REFINE_REWARD_MODEL",
]