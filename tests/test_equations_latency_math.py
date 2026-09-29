"""
Tests for gateway/equations/latency_math.py — latency-peak measurement and
optimization family for the NON-MONOTONIC num_predict -> latency curve.

Verifies mathematical properties (bounds, monotonicity, peak detection,
degenerate inputs), not just "no exception".
"""

from __future__ import annotations

import math

from gateway.equations.latency_math import (
    LatencyScanner,
    jzs_bf10,
    latency,
    median_latency,
    parabola_vertex,
    peak_bracket,
    peak_budget,
    plan_budgets,
    refine_from_bracket,
    sample_size_needed,
    savings_test,
    suggest_next_budget,
    ucb_chart,
    welch_t,
)


# ---------------------------------------------------------------------------
# Latency dimension (Eq 1-8)
# ---------------------------------------------------------------------------

def test_latency_basic_dimensions():
    d = latency(0.0, 10.0, out_tokens=100, out_chars=500)
    assert d["tau_s"] == 10.0
    assert abs(d["tokens_per_s"] - 10.0) < 1e-9
    assert abs(d["chars_per_s"] - 50.0) < 1e-9


def test_latency_degenerate_no_nan():
    d = latency(5.0, 5.0)
    assert d["tau_s"] == 0.0
    assert d["tokens_per_s"] == 0.0
    assert d["chars_per_s"] == 0.0
    d2 = latency(10.0, 0.0, out_tokens=5)
    assert d2["tau_s"] == 0.0
    assert all(math.isfinite(v) for v in d2.values())


# ---------------------------------------------------------------------------
# Peak detection (Eq 9-20)
# ---------------------------------------------------------------------------

def test_peak_budget_finds_interior_max():
    # The qwen3-shaped case: the biggest latency is in the MIDDLE.
    med = {128: 60.0, 384: 240.0, 768: 90.0}
    assert peak_budget(med) == 384


def test_peak_budget_plateau_picks_cheapest():
    med = {128: 200.0, 384: 200.0, 768: 50.0}
    assert peak_budget(med, plateau_tol=5.0) == 128
    # With tol=0 both 128 and 384 tie at the max -> cheapest wins.
    assert peak_budget(med, plateau_tol=0.0) == 128


def test_peak_budget_empty_safe():
    assert peak_budget({}) == -1
    assert peak_budget({384: 0.0}) == 384


def test_peak_bracket_walks_descending_shoulders():
    obs = {128: [60.0], 256: [140.0], 384: [240.0], 640: [90.0], 1152: [45.0]}
    b_lo, b_peak, b_hi = peak_bracket(obs)
    assert b_peak == 384
    assert b_lo == 128
    # The peak at 384 descends through 640 (90) and 1152 (45), so the walk
    # brackets the full shoulder on both sides.
    assert b_hi == 1152


def test_peak_bracket_degenerate():
    assert peak_bracket({}) == (-1, -1, -1)
    assert peak_bracket({384: [1.0]}) == (384, 384, 384)


# ---------------------------------------------------------------------------
# Fitting (Eq 21-40)
# ---------------------------------------------------------------------------

def test_parabola_vertex_exact():
    # y = -(x-5)^2 + 7 through x = 3, 5, 7.
    x, y = parabola_vertex(3.0, 3.0, 5.0, 7.0, 7.0, 3.0)
    assert abs(x - 5.0) < 1e-6
    assert abs(y - 7.0) < 1e-6


def test_parabola_vertex_collinear_fallback():
    x, y = parabola_vertex(1.0, 2.0, 2.0, 4.0, 3.0, 6.0)
    assert x == 2.0
    assert y == 4.0


def test_refine_from_bracket_interior_peak():
    obs = {100: [10.0], 200: [40.0], 400: [70.0], 700: [25.0]}
    x, _y = refine_from_bracket(obs)
    # Parabola vertex between 200 and 700 lands near the 400 peak.
    assert 250 <= x <= 650


def test_refine_from_bracket_degenerate_safe():
    x, y = refine_from_bracket({})
    assert x == -1.0 and y == 0.0


# ---------------------------------------------------------------------------
# Acquisition (Eq 41-60)
# ---------------------------------------------------------------------------

def test_ucb_chart_unprobed_candidates_are_inf():
    obs = {384: [240.0, 260.0]}
    chart = ucb_chart(obs, candidates=[128, 384, 1152])
    assert chart[128] == math.inf
    assert chart[1152] == math.inf
    assert 0 < chart[384] < math.inf


def test_suggest_next_prefers_unprobed():
    obs = {384: [240.0]}
    assert suggest_next_budget(obs, [128, 384, 1152]) in (128, 1152)


def test_suggest_next_empty_safe():
    assert suggest_next_budget({}, []) == -1


# ---------------------------------------------------------------------------
# Validation (Eq 61-90)
# ---------------------------------------------------------------------------

def test_welch_t_identical_has_p_one():
    t, df, p = welch_t([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert abs(t) < 1e-9
    assert p == 1.0


def test_welch_t_separated_low_p():
    _t, _df, p = welch_t([1.0, 1.1], [10.0, 10.2])
    assert p < 0.05


def test_welch_t_empty_safe():
    t, df, p = welch_t([], [1.0, 2.0])
    assert t == 0.0 and df == 1.0 and p == 1.0
    t2, df2, p2 = welch_t([1.0], [1.0, 2.0])
    assert df2 > 0 and 0.0 <= p2 <= 1.0


def test_jzs_bf10_flat_and_separated():
    # Identical groups -> BF10 ~= 1 (H1 not favored).
    bf_flat = jzs_bf10([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert 0.1 <= bf_flat <= 10.0
    # Strongly separated groups -> BF10 strongly > 1 (evidence for H1).
    bf_sep = jzs_bf10([1.0, 1.1, 1.0], [10.0, 10.2, 10.1])
    assert bf_sep > 10.0
    assert bf_sep >= bf_flat


def test_jzs_bf10_degenerate_safe():
    assert jzs_bf10([], []) == 1.0
    assert 0.0 <= jzs_bf10([1.0], []) <= 1e6


def test_savings_test_verdicts():
    opt = [5.0, 6.0, 5.5]
    base = [300.0, 310.0, 290.0]
    res = savings_test(opt, base, delta_cut=10.0)
    assert res["verdict"] == "save"
    assert res["median_save_s"] > 200
    res_tie = savings_test(opt, [10.0, 12.0, 11.0], delta_cut=50.0)
    assert res_tie["verdict"] == "tie"
    assert savings_test([], [1.0])["verdict"] == "no-data"


# ---------------------------------------------------------------------------
# Sampling design (Eq 181-200)
# ---------------------------------------------------------------------------

def test_plan_budgets_straddles_range():
    plan = plan_budgets(96, 1536)
    assert plan[0] == 96 and plan[-1] == 1536
    assert plan == sorted(set(plan))
    # A denser card includes interior points crossing the qwen3 peak region.
    assert any(384 <= b <= 640 for b in plan)


def test_plan_budgets_degenerate():
    assert plan_budgets(384, 384) == [384, 384]
    assert plan_budgets(1000, 100, max_n=1) == [100, 1000]


def test_sample_size_needed_monotonic():
    big = sample_size_needed(1.0, 5.0)
    small_effect = sample_size_needed(0.1, 5.0)
    assert small_effect > big
    zero = sample_size_needed(0.0, 5.0)
    assert zero == 4096
    assert sample_size_needed(10.0, 1.0) >= 2


# ---------------------------------------------------------------------------
# LatencyScanner integrated driver
# ---------------------------------------------------------------------------

def test_scanner_finds_peak_on_nonmonotone_curve():
    # Single-peak curve (qwen3-shaped: interior max), NOT monotone.
    def measure(b):
        # peak = 240s at b=384, falls to ~100s on both sides.
        return 240.0 - 0.5 * (b - 384) ** 2 / 384.0

    scanner = LatencyScanner(lo=96, hi=1152)
    report = scanner.scan(measure)
    peak_b, peak_v = scanner.peak()
    assert peak_b == 384
    assert abs(peak_v - 240.0) < 1e-6
    assert report["n_obs"] >= 5
    assert 96 <= peak_b <= 1152


def test_scanner_add_none_is_ignored():
    s = LatencyScanner()
    s.add(384, None)
    s.add(384, 12.5)
    s.add(384, 13.0)
    assert s.stats()["n_obs"] == 2
    # Median of [12.5, 13.0].
    assert s.peak() == (384, 12.75)


def test_scanner_reset():
    s = LatencyScanner()
    s.add(384, 10.0)
    s.reset()
    assert s.stats()["n_budgets"] == 0


def test_scanner_ucb_when_all_probed():
    s = LatencyScanner(lo=96, hi=1152)
    s.add(96, 50.0)
    s.add(384, 200.0)
    s.add(1152, 80.0)
    # Highest UCB stays near the big-latency budget.
    sug = s.next_budget()
    assert sug in s.candidates()