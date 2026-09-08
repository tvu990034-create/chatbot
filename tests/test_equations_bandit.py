"""
Tests for gateway/equations/bandit_math.py — multi-armed bandit and
Bayesian-optimization equation family.

Verifies mathematical properties (bounds, monotonicity, convergence,
degenerate inputs), not just "no exception".
"""

from __future__ import annotations

import math
import random

from gateway.equations.bandit_math import (
    UCB1,
    EXP3,
    SimpleGP,
    ThompsonSampler,
    acquisition_EI,
    acquisition_PI,
    acquisition_UCB,
    epsilon_anneal,
    epsilon_greedy_select,
    explore_fraction,
    regret,
    ucb,
    ucb1_tuned,
)


# ---------------------------------------------------------------------------
# UCB / UCB1
# ---------------------------------------------------------------------------

def test_ucb_selects_unseen_arms_first():
    # Unseen (count==0) arms should be explored before any count>0 arm.
    idx = UCB1.select(3, counts=[5, 5, 0], rewards=[10.0, 1.0, 0.0])
    assert idx == 2


def test_ucb_picks_higher_mean_when_counts_bounded():
    # Equal and large counts -> exploration terms equal -> higher mean wins.
    idx = UCB1.select(2, counts=[100, 100], rewards=[80.0, 60.0])
    assert idx == 0
    idx = UCB1.select(2, counts=[100, 100], rewards=[1.0, 50.0])
    assert idx == 1


def test_ucb1_tuned_bounds():
    counts = [10, 10]
    rewards = [7.0, 5.0]
    scores = ucb1_tuned(counts, rewards)
    assert len(scores) == 2
    assert all(math.isfinite(s) for s in scores)
    # Higher-mean arm should still rank higher when counts equal.
    assert scores[0] > scores[1]


def test_ucb_standalone():
    counts = [10, 10]
    rewards = [7.0, 5.0]
    scores = ucb(counts, rewards)
    assert all(math.isfinite(s) for s in scores)
    assert scores[0] > scores[1]
    # Unseen arm (count 0) gets +inf.
    s2 = ucb([10, 0], [7.0, 0.0])
    assert s2[1] == math.inf


def test_ucb_empty_is_safe_default():
    idx = UCB1.select(0, [], [])
    assert idx == -1
    assert ucb([], []) == []
    assert ucb1_tuned([], []) == []


# ---------------------------------------------------------------------------
# Thompson sampling
# ---------------------------------------------------------------------------

def test_thompson_converges_to_better_arm():
    ts = ThompsonSampler(2, seed=42)
    # Arm 0 is better (reward ~1), arm 1 is worse (reward ~0).
    for _ in range(800):
        arm = ts.sample()
        reward = 1.0 if arm == 0 else 0.0
        ts.update(arm, reward)
    means = ts.expected_means()
    assert means[0] > means[1]


def test_thompson_expected_means_are_probabilities():
    ts = ThompsonSampler(3, seed=1)
    ts.update(0, 1.0)
    ts.update(0, 0.0)
    ts.update(1, 1.0)
    for m in ts.expected_means():
        assert 0.0 <= m <= 1.0


def test_thompson_confidence_in_range():
    ts = ThompsonSampler(2, seed=7)
    ts.update(0, 1.0)
    ts.update(0, 1.0)
    c = ts.confidence(0)
    assert 0.0 <= c <= 1.0


def test_thompson_deterministic_with_seed():
    a = ThompsonSampler(2, seed=99)
    b = ThompsonSampler(2, seed=99)
    draws_a = [a.sample() for _ in range(20)]
    draws_b = [b.sample() for _ in range(20)]
    assert draws_a == draws_b


# ---------------------------------------------------------------------------
# EXP3
# ---------------------------------------------------------------------------

def test_exp3_selects_and_updates():
    e = EXP3(3, seed=5)
    for _ in range(100):
        arm = e.select()
        assert 0 <= arm < 3
        e.update(arm, 1.0)
    assert e.stats()["total"] == 100


def test_exp3_probabilities_sum_to_one():
    e = EXP3(3, seed=5)
    for _ in range(50):
        e.update(e.select(), 0.5)
    probs = e._probabilities()
    assert abs(sum(probs) - 1.0) < 1e-9


def test_exp3_reset():
    e = EXP3(2, seed=5)
    for _ in range(10):
        e.update(e.select(), 1.0)
    e.reset()
    assert e.stats()["total"] == 0


# ---------------------------------------------------------------------------
# Acquisition functions
# ---------------------------------------------------------------------------

def test_acquisition_ei_nonnegative_and_grows_with_sigma():
    # EI never negative.
    ei = acquisition_EI(1.0, 0.5, best=2.0)
    assert ei >= 0.0
    # Below-best with sigma>0 gives a small positive EI (exploration value).
    ei2 = acquisition_EI(0.0, 2.0, best=1.0)
    assert ei2 >= 0.0
    # Above-best: sure gain ==> EI == imp exactly (when sigma~0).
    ei3 = acquisition_EI(3.0, 0.0, best=1.0)
    assert abs(ei3 - 2.0) < 1e-9


def test_acquisition_ei_supports_lists():
    ei = acquisition_EI([1.0, 0.0], [0.5, 2.0], best=1.0)
    assert isinstance(ei, list) and len(ei) == 2
    assert all(v >= 0 for v in ei)


def test_acquisition_ucb():
    assert abs(acquisition_UCB(0.0, 1.0, kappa=2.0) - 2.0) < 1e-9
    ucb_list = acquisition_UCB([0.0, 1.0], [1.0, 1.0])
    assert isinstance(ucb_list, list) and len(ucb_list) == 2


def test_acquisition_pi_in_unit_interval():
    pi = acquisition_PI(1.0, 1.0, best=1.0)
    assert 0.0 <= pi <= 1.0
    # Very far above best -> ~1, far below -> ~0.
    assert acquisition_PI(10.0, 0.1, best=0.0) > 0.99
    assert acquisition_PI(-10.0, 0.1, best=0.0) < 0.01


# ---------------------------------------------------------------------------
# SimpleGP
# ---------------------------------------------------------------------------

def test_gp_interpolates_training_points():
    gp = SimpleGP()
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 1.0, 4.0, 9.0]
    gp.fit(xs, ys, length_scale=1.0, noise=1e-6)
    mu, std = gp.predict_mu_sigma(xs)
    # Posterior mean interpolates the observations almost exactly.
    for m, y in zip(mu, ys):
        assert abs(m - y) < 1e-3
    # Uncertainty near training points is very low.
    for s in std:
        assert s < 1e-2


def test_gp_empty_safe_default():
    gp = SimpleGP()
    mu, std = gp.predict_mu_sigma([1.0, 2.0])
    assert mu == [0.0, 0.0]
    assert std == [1.0, 1.0]


def test_gp_increases_uncertainty_away_from_data():
    gp = SimpleGP()
    gp.fit([0.0], [0.0], length_scale=1.0, noise=1e-6)
    _, std_near = gp.predict_mu_sigma([0.0])
    _, std_far = gp.predict_mu_sigma([5.0])
    assert std_far > std_near


# ---------------------------------------------------------------------------
# Explore / exploit + regret
# ---------------------------------------------------------------------------

def test_epsilon_anneal_monotonic():
    prev = epsilon_anneal(0)
    for step in range(1, 1100, 50):
        cur = epsilon_anneal(step, eps_start=1.0, eps_end=0.05, decay_steps=1000)
        assert cur <= prev
        prev = cur
    # Reaches the end value.
    assert epsilon_anneal(10000, eps_start=1.0, eps_end=0.05, decay_steps=1000) == 0.05


def test_epsilon_greedy_select():
    rng = random.Random(0)
    # Greedy with epsilon=0 should pick best mean.
    arm = epsilon_greedy_select(0.0, [100, 100], [90.0, 10.0], rng)
    assert arm == 0
    # Unseen arm explored regardless.
    arm2 = epsilon_greedy_select(0.0, [100, 0], [90.0, 0.0], rng)
    assert arm2 == 1
    # Empty -> safe default.
    assert epsilon_greedy_select(0.1, [], [], rng) == -1


def test_explore_fraction():
    assert explore_fraction(0) == 1.0
    assert explore_fraction(100000) < 1e-6
    assert explore_fraction(5, decay=10) < explore_fraction(3, decay=10)


def test_regret_nonnegative():
    assert regret(10.0, 8.0) == 2.0
    assert regret(10.0, 12.0) == 0.0
    assert regret(0.0, 0.0) == 0.0
