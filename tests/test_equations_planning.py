"""
Tests for gateway/optimizations/planning_math.py - MCTS/UCT, self-consistency,
best-of-N, verification and lookahead optimization family.

Verifies mathematical properties (bounds, monotonicity, degenerate inputs),
not just "no exception".
"""

from __future__ import annotations

import math

from gateway.equations.planning_math import (
    agreement_fraction,
    answer_consensus_check,
    best_candidate,
    best_of_n_expected,
    cisc_style_confidence,
    depth_limit_knowledge,
    expand_if_promising,
    max_q,
    mean_q,
    optimal_n,
    prob_pass,
    should_verify,
    simulation_budget,
    uct_score,
    verification_score,
)


# ---------------------------------------------------------------------------
# UCT / MCTS budget
# ---------------------------------------------------------------------------

def test_uct_increases_with_win_frac():
    assert uct_score(0.9, 100, 10) > uct_score(0.5, 100, 10)


def test_uct_decreases_with_node_visits():
    # More visitations -> lower exploration bonus -> lower overall score.
    assert uct_score(0.9, 100, 10) > uct_score(0.9, 100, 100)


def test_uct_unexplored_node_always_chosen():
    # Unvisited child returns +inf so selection prefers it over any visited one.
    assert uct_score(0.0, 100, 0) == math.inf
    assert uct_score(0.0, 100, 0) > uct_score(1.0, 100, 100)


def test_uct_guard_parent_visits():
    # parent_visits <= 1 -> no exploration term (log(1)=0), just win_frac.
    assert abs(uct_score(0.5, 0, 5) - 0.5) < 1e-12
    assert abs(uct_score(0.5, 1, 5) - 0.5) < 1e-12


def test_simulation_budget_bounds():
    assert simulation_budget(1000, 10, 100) == 100
    assert simulation_budget(1000, 10, 5) == 5
    assert simulation_budget(100, 60, 100) == 1
    assert simulation_budget(0, 10, 8) == 0
    assert simulation_budget(-5, 10, 8) == 0
    assert simulation_budget(1000, 0, 8) == 8       # unknown per-sim cost
    assert simulation_budget(100, 10, 0) == 0       # zero cap


# ---------------------------------------------------------------------------
# Self-consistency
# ---------------------------------------------------------------------------

def test_agreement_fraction_range_and_identity():
    assert agreement_fraction([]) == 0.0
    assert agreement_fraction(["a", "a", "a"]) == 1.0
    assert agreement_fraction(["a"]) == 1.0
    assert abs(agreement_fraction(["42", "42", "43"]) - 2.0 / 3.0) < 1e-12


def test_agreement_fraction_normalizes_math_answers():
    # "42", "\\boxed{42}" and "42.0" are the same normalized answer.
    assert agreement_fraction(["\\boxed{42}", "42", "42.0"]) == 1.0


def test_should_verify_gate():
    assert should_verify(0.4, threshold=0.6) is True
    assert should_verify(0.8, threshold=0.6) is False
    assert should_verify(0.6, threshold=0.6) is False


def test_cisc_style_confidence():
    assert cisc_style_confidence([]) == 0.0
    assert abs(cisc_style_confidence(["x", "x", "y"]) - 2.0 / 3.0) < 1e-12
    assert cisc_style_confidence(["7", "\\boxed{7}"]) == 1.0


def test_answer_consensus_check():
    best, agreement, confident = answer_consensus_check(["\\boxed{7}", "7", "7"])
    assert "boxed" in best
    assert abs(agreement - 1.0) < 1e-12
    assert confident is True
    b2, a2, c2 = answer_consensus_check([])
    assert b2 == "" and a2 == 0.0 and c2 is False
    _b, _a, c3 = answer_consensus_check(["a", "b", "c"])
    assert c3 is False


# ---------------------------------------------------------------------------
# Best-of-N
# ---------------------------------------------------------------------------

def test_best_of_n_expected_floor_is_mu():
    assert abs(best_of_n_expected(0.5, 0.2, 1) - 0.5) < 1e-12
    assert abs(best_of_n_expected(0.5, 0.0, 5) - 0.5) < 1e-12
    assert best_of_n_expected(0.5, 0.2, 8) >= 0.5


def test_best_of_n_expected_nondecreasing_in_n():
    prev = best_of_n_expected(0.6, 0.2, 1)
    for n in range(2, 40):
        cur = best_of_n_expected(0.6, 0.2, n)
        assert cur >= prev
        assert math.isfinite(cur)
        prev = cur


def test_optimal_n_tiny_sigma_is_one():
    assert optimal_n(0.6, 0.001, cost_per_sample=0.005, max_n=8) == 1


def test_optimal_n_large_sigma_is_greater_than_one():
    assert optimal_n(0.6, 1.0, cost_per_sample=0.001, max_n=40) > 1


def test_optimal_n_free_samples_take_all():
    assert optimal_n(0.6, 0.2, cost_per_sample=0.0, max_n=8) == 8


def test_best_candidate_argmax():
    cands = [(1, 2), (3, 4), (2, 9)]
    assert best_candidate(cands, score_fn=lambda c: c[1]) == (2, 9)
    assert best_candidate([10, 4, 8]) == 10            # default float scorer
    assert best_candidate([], score_fn=lambda c: c) is None


def test_best_candidate_tie_first_wins():
    assert best_candidate([(1, 5), (2, 5)], score_fn=lambda c: c[1]) == (1, 5)


# ---------------------------------------------------------------------------
# Verification scoring
# ---------------------------------------------------------------------------

def test_verification_score_weighted():
    checks = [(True, 1.0), (False, 1.0), (True, 2.0)]
    assert abs(verification_score("cand", checks) - 0.75) < 1e-12
    assert verification_score("cand", []) == 0.0
    assert verification_score("cand", [True, False]) == 0.5   # bare bools


def test_prob_pass_clamped_and_monotonic():
    assert prob_pass(1.0, 5) == 1.0
    assert prob_pass(0.0, 5) == 0.0
    assert abs(prob_pass(0.7, 0) - 0.7) < 1e-12
    assert 0.0 <= prob_pass(0.7, 3) <= 1.0
    assert prob_pass(0.7, 8) > prob_pass(0.7, 1)


# ---------------------------------------------------------------------------
# Lookahead / branching
# ---------------------------------------------------------------------------

def test_max_q_discounted_rollout():
    assert max_q([]) == 0.0
    assert abs(max_q([1.0, 2.0, 3.0], discount=1.0) - 6.0) < 1e-12
    assert abs(max_q([1.0, 2.0, 3.0], discount=0.0) - 1.0) < 1e-12
    assert math.isfinite(max_q([1.0, -2.0, 3.0], discount=0.95))


def test_mean_q():
    assert mean_q([]) == 0.0
    assert abs(mean_q([2.0, 4.0, 6.0]) - 4.0) < 1e-12
    assert math.isfinite(mean_q([1.0, float("nan")])) == 1.0  # NaN skipped


def test_expand_if_promising():
    assert expand_if_promising(2.0, threshold=1.5) is True
    assert expand_if_promising(1.0, threshold=1.5) is False
    assert expand_if_promising(math.inf, threshold=1.5) is True


def test_depth_limit_knowledge():
    assert depth_limit_knowledge(1.0, max_depth=8) == 8
    assert depth_limit_knowledge(0.0, max_depth=8) == 4
    assert depth_limit_knowledge(1.5, max_depth=8) == 8      # clamped
    assert depth_limit_knowledge(0.8, 8) > depth_limit_knowledge(0.2, 8)
    assert depth_limit_knowledge(0.5, 4) >= 1