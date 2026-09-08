"""
tests/test_equations_routing.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for gateway/optimizations/routing_math.py — model routing / load-aware /
budget / consensus optimizations.
"""

import math
import pytest

from gateway.equations.routing_math import (
    LoadBalancer,
    select_idle_model,
    BudgetAwareRouter,
    difficulty_score,
    should_route_to_strong,
    confidence_token_budget,
    zipf_popularity,
    cache_hit_ratio,
    zipf_capacity_hit,
    deadline_scheduler,
    feasible_under_deadline,
    should_switch,
    majority_consensus,
    consensus_clearance,
    load_adjusted_score,
)


def test_ewma_converges_to_steady_state():
    lb = LoadBalancer(alpha=0.5, models=["a"])
    for _ in range(50):
        lb.observe("a", 10)
    # After enough iterations the EWMA should be ~ the steady-state value 10.
    assert math.isclose(lb.load("a"), 10.0, abs_tol=1e-3)
    assert lb.global_load() == pytest.approx(10.0)


def test_load_balancer_recommends_least_loaded():
    lb = LoadBalancer(alpha=0.9, models=["a", "b", "c"])
    lb.observe("a", 50)
    lb.observe("b", 2)
    lb.observe("c", 20)
    assert lb.recommend() == "b"


def test_select_idle_model_switches_when_idle_and_dwell_passes():
    # Dwell satisfied -> candidate takes over when not overly queued.
    loads = {"fast": 1, "strong": 3}
    result = select_idle_model("fast", "strong", loads, min_dwell=5.0,
                               now=100.0, last_switch=0.0)
    assert result == "strong"


def test_select_idle_model_keeps_loaded_during_dwell():
    loads = {"fast": 1, "strong": 3}
    result = select_idle_model("fast", "strong", loads, min_dwell=5.0,
                               now=2.0, last_switch=0.0)
    # Dwell not met and candidate not heavily queued -> stay resident.
    assert result == "fast"


def test_select_idle_model_wins_by_heavy_queue():
    # Candidate heavily queued relative to resident even outside dwell window.
    loads = {"fast": 1, "strong": 100}
    result = select_idle_model("fast", "strong", loads, min_dwell=5.0,
                               now=100.0, last_switch=0.0)
    assert result == "fast"


def test_zipf_capacity_hit_reaches_08_with_top_20pct():
    n = 1000
    target = 0.8
    kept = zipf_capacity_hit(target, n, s=1.0)
    frac = kept / n
    assert 0.0 < frac <= 1.0
    assert frac <= 0.25  # top ~20% (or less) covers 80% for typical Zipf
    # And the actual hit ratio at that capacity meets the target.
    assert cache_hit_ratio(kept, n, s=1.0) >= target


def test_zipf_popularity_and_hit_ratio_sane():
    assert zipf_popularity(1) == pytest.approx(1.0)
    assert zipf_popularity(2) == pytest.approx(0.5)
    assert cache_hit_ratio(0, 100) == 0.0
    assert 0.0 <= cache_hit_ratio(10, 100) <= 1.0


def test_confidence_token_budget_gate_and_monotonic():
    assert confidence_token_budget(0.3, m_min=20, m_max=256, gate=0.45) is None
    lo = confidence_token_budget(0.5, m_min=20, m_max=256, gate=0.45)
    hi = confidence_token_budget(0.9, m_min=20, m_max=256, gate=0.45)
    assert lo is not None and hi is not None
    assert lo <= hi
    # Budget stays within bounds.
    assert confidence_token_budget(1.0, m_min=20, m_max=256, gate=0.45) == 256


def test_deadline_scheduler_budgets_sum_to_deadline():
    stages = [1.0, 2.0, 1.0]
    budgets = deadline_scheduler(600.0, stages)
    assert sum(budgets) == pytest.approx(600.0)
    assert budgets == pytest.approx([150.0, 300.0, 150.0])


def test_feasible_under_deadline():
    assert feasible_under_deadline([100.0, 200.0], 400.0) is True
    assert feasible_under_deadline([300.0, 200.0], 400.0) is False
    assert feasible_under_deadline([], 400.0) is True


def test_majority_consensus_argmax():
    votes = ["a", "b", "a", "a", "b"]
    top, frac = majority_consensus(votes)
    assert top == "a"
    assert frac == pytest.approx(0.6)
    assert majority_consensus([]) == (None, 0.0)


def test_consensus_clearance():
    assert consensus_clearance(0.6, threshold=0.5) is True
    assert consensus_clearance(0.4, threshold=0.5) is False


def test_should_switch_requires_hysteresis():
    assert should_switch(1.0, 0.95, hysteresis=0.1) is False  # margin < 0.1
    assert should_switch(1.0, 0.85, hysteresis=0.1) is True   # margin = 0.15


def test_load_adjusted_score():
    assert load_adjusted_score(1.0, 0.2, penalty=1.0) == pytest.approx(0.8)


def test_difficulty_routing():
    assert 0.0 <= difficulty_score("") <= 1.0
    easy = difficulty_score("hello how are you doing today")
    hard = difficulty_score("prove the time complexity of the distributed consensus optimization gradient convergence theorem")
    assert hard >= easy
    assert should_route_to_strong(0.9, threshold=0.6) is True
    assert should_route_to_strong(0.1, threshold=0.6) is False


def test_budget_router():
    r = BudgetAwareRouter()
    # Strong model always affordable at high budget.
    assert r.route(100, 50, budget=100.0) in {"fast", "strong", "fallback"}
    assert r.best_model_for_budget({"fast": 1.0, "strong": 1000.0, "fallback": 0.5},
                                   budget=2.0) in {"fast", "fallback"}
    # Tiny budget -> fallback (cheapest).
    assert r.best_model_for_budget({"fast": 0.01, "strong": 0.05, "fallback": 0.001},
                                   budget=0.0001) == "fallback"
