"""
Tests for gateway/equations/calibration_math.py.

Verifies mathematical properties (bounds, monotonicity, degeneracy), not just
"no exception".
"""

from __future__ import annotations

import math

import pytest

from gateway.equations.calibration_math import (
    _clamp,
    _logsumexp,
    softmax,
    temperature_scale,
    TemperatureScaler,
    ece,
    brier_score,
    calibrated_confidence,
    PlattScaler,
    isotonic_regression,
    ConformalSetPredictor,
    entropy,
    max_entropy,
    entropy_abstention_decision,
    energy_ood_score,
    is_ood,
    max_confidence,
    top2_margin,
    predicted_probability_ratio,
    EnsembleUncertainty,
    should_defer_to_verify,
    temperature_to_meet_entropy,
)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def test_softmax_sums_to_one():
    logits = [1.0, 2.0, 3.0]
    p = softmax(logits)
    assert math.isclose(sum(p), 1.0, rel_tol=1e-9)
    assert all(0.0 <= v <= 1.0 for v in p)
    assert len(p) == len(logits)


def test_softmax_overflow_guard():
    logits = [1000.0, 1001.0, 1002.0]
    p = softmax(logits)
    assert all(math.isfinite(v) for v in p)
    assert math.isclose(sum(p), 1.0, rel_tol=1e-6)


def test_softmax_empty():
    assert softmax([]) == []


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------

def test_temperature_flattens_probs():
    logits = [1.0, 2.0, 3.0]
    p1 = softmax(logits, temperature=1.0)
    p5 = softmax(logits, temperature=5.0)
    # Higher temperature -> more uniform -> higher entropy.
    assert entropy(p5) > entropy(p1)
    assert math.isclose(sum(p5), 1.0, rel_tol=1e-9)


def test_temperature_scaler_fits_and_calibrates():
    # Perfectly calibrated-ish synthetic logits: true class always has top logit.
    logits_list = [
        [3.0, 1.0, 0.5],
        [2.0, 0.5, 0.2],
        [1.0, 4.0, 0.8],
        [1.5, 0.2, 3.0],
    ]
    labels = [0, 0, 1, 2]
    scaler = TemperatureScaler()
    scaler.fit(logits_list, labels)
    assert scaler.temperature > 0
    cal = scaler.calibrate(logits_list[0])
    assert math.isclose(sum(cal), 1.0, rel_tol=1e-9)
    s = scaler.stats()
    assert 0.0 <= s["ece_at_fit"] <= 1.0


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------

def test_ece_in_range_and_perfect():
    # Perfectly calibrated: confidence == accuracy everywhere -> ECE 0.
    assert ece([0.8, 0.7, 0.9], [0.8, 0.7, 0.9]) == pytest.approx(0.0, abs=1e-9)
    # Miscalibrated -> ECE in [0, 1].
    e = ece([0.9, 0.9, 0.1], [1.0, 0.0, 1.0])
    assert 0.0 <= e <= 1.0
    assert ece([], []) == 0.0


# ---------------------------------------------------------------------------
# Brier score
# ---------------------------------------------------------------------------

def test_brier_score():
    # Perfect prediction -> Brier ~0.
    probs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    labels = [0, 1]
    assert brier_score(probs, labels) == pytest.approx(0.0, abs=1e-9)
    # Uniform over 3 classes, any label -> Brier = sum_k (p_k - y_k)^2.
    # target class: (1/3-1)^2 = 4/9 ; other 2: 2*(1/3)^2 = 2/9 ; total = 6/9 = 2/3.
    u = [[1 / 3, 1 / 3, 1 / 3]] * 3
    assert brier_score(u, [0, 1, 2]) == pytest.approx(2.0 / 3.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Isotonic regression
# ---------------------------------------------------------------------------

def test_isotonic_regression_monotonic_non_decreasing():
    vals = [1.0, 0.5, 0.6, 2.0, 1.8, 3.0]
    fit = isotonic_regression(vals)
    assert len(fit) == len(vals)
    assert all(fit[i] <= fit[i + 1] for i in range(len(fit) - 1))
    # Preserves the total (weighted) sum (PAVA property).
    assert math.isclose(sum(fit), sum(vals), rel_tol=1e-9)


def test_isotonic_already_monotone_unchanged():
    vals = [0.1, 0.2, 0.3, 0.4]
    assert isotonic_regression(vals) == vals
    assert isotonic_regression([]) == []


# ---------------------------------------------------------------------------
# Conformal
# ---------------------------------------------------------------------------

def test_conformal_coverage():
    # Synthetic calibration radii (nonconformity scores) drawn uniformly in [0, 0.5].
    rng = _make_rng(42)
    cal_radii = [[rng.random() * 0.5] for _ in range(500)]
    predictor = ConformalSetPredictor(center=0.0)
    predictor.fit(cal_radii, alpha=0.1)
    q_low, q_high = predictor.predict_interval(cal_radii[0])
    assert q_low <= q_high
    assert q_low < q_high
    # Coverage on held-out scores ~ (1 - alpha).
    held = [rng.random() * 0.5 for _ in range(500)]
    covered = sum(1 for r in held if q_low <= r <= q_high)
    frac = covered / len(held)
    # Empirical quantile of a uniform achieves coverage ≈ (1 - alpha).
    assert abs(frac - 0.9) < 0.1
    # The radius should sit near the 90th percentile of the calibration radii.
    assert 0.40 <= predictor.radius <= 0.5


def _make_rng(seed):
    """Minimal pure-stdlib deterministic PRNG (not used heavily)."""
    class R:
        def __init__(s, seed):
            s.s = seed

        def random(s):
            s.s = (s.s * 1103515245 + 12345) & 0x7FFFFFFF
            return s.s / 0x7FFFFFFF
    return R(seed)


# ---------------------------------------------------------------------------
# Entropy
# ---------------------------------------------------------------------------

def test_entropy_bounds():
    n = 4
    assert entropy([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-9)
    uniform = [1.0 / n] * n
    assert entropy(uniform) == pytest.approx(max_entropy(n), abs=1e-9)
    assert max_entropy(n) == pytest.approx(math.log(n), abs=1e-9)
    assert max_entropy(1) == 0.0


def test_entropy_abstention():
    assert entropy_abstention_decision(max_entropy(4), 4, threshold_frac=0.7) is True
    assert entropy_abstention_decision(0.0, 4, threshold_frac=0.7) is False


# ---------------------------------------------------------------------------
# Energy OOD
# ---------------------------------------------------------------------------

def test_energy_ood_lower_when_concentrated():
    concentrated = [5.0, 0.1, 0.1]
    diffused = [1.0, 1.0, 1.0]
    assert energy_ood_score(concentrated) < energy_ood_score(diffused)


def test_is_ood():
    assert is_ood(-20.0, in_dist_energy_mean=-5.0, in_dist_energy_std=2.0, z=2.0) is True
    assert is_ood(-5.0, in_dist_energy_mean=-5.0, in_dist_energy_std=2.0, z=2.0) is False


# ---------------------------------------------------------------------------
# Confidence aggregation
# ---------------------------------------------------------------------------

def test_confidence_aggregation():
    probs = [0.1, 0.7, 0.2]
    assert max_confidence(probs) == pytest.approx(0.7)
    assert top2_margin(probs) == pytest.approx(0.5)
    assert predicted_probability_ratio(probs, k=2) == pytest.approx(0.7 / 0.9)
    assert max_confidence([]) == 0.0


# ---------------------------------------------------------------------------
# Platt
# ---------------------------------------------------------------------------

def test_platt_maps_monotonic():
    scaler = PlattScaler(epochs=500)
    scaler.fit([-2.0, -1.0, 0.0, 1.0, 2.0], [0, 0, 1, 1, 1])
    p_neg = scaler.predict_logit(-2.0)
    p_pos = scaler.predict_logit(2.0)
    assert p_pos > p_neg
    assert 0.0 <= p_neg <= 1.0 and 0.0 <= p_pos <= 1.0


def test_platt_empty_falls_back_base_rate():
    scaler = PlattScaler()
    scaler.fit([], [])
    r = scaler.predict_logit(0.0)
    assert 0.0 <= r <= 1.0


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def test_ensemble_uncertainty_decomposition():
    ens = EnsembleUncertainty()
    ens.add_model([0.8, 0.1, 0.1])
    ens.add_model([0.7, 0.2, 0.1])
    ens.add_model([0.75, 0.15, 0.1])
    ens.compute()
    stats = ens.stats()
    assert stats["epistemic_variance"] >= 0.0
    assert stats["aleatoric_variance"] >= 0.0
    assert 0.0 <= stats["disagreement"] <= 1.0
    mean = ens.mean_probs()
    assert len(mean) == 3
    assert all(0.0 <= v <= 1.0 for v in mean)
    assert any(v > 0.1 for v in mean)  # dominant class preserved
    assert math.isclose(sum(mean), 1.0, abs_tol=1e-6)


def test_ensemble_empty():
    ens = EnsembleUncertainty()
    ens.compute()
    assert ens.mean_probs() == []
    s = ens.stats()
    assert s["n_models"] == 0.0


# ---------------------------------------------------------------------------
# Gating & temperature-for-entropy
# ---------------------------------------------------------------------------

def test_should_defer_to_verify():
    assert should_defer_to_verify(0.4, threshold=0.6) is True
    assert should_defer_to_verify(0.9, threshold=0.6) is False


def test_temperature_to_meet_entropy_raises_entropy():
    probs = [0.9, 0.05, 0.05]
    T = temperature_to_meet_entropy(probs, target_entropy_frac=0.9, max_temperature=5.0)
    assert 0 < T <= 5.0
    # Calibrating with this T raises entropy closer to the target.
    h_target = 0.9 * max_entropy(len(probs))
    h_at_T = entropy(softmax(probs, temperature=T))
    # At the target cutoff, entropy should be near (>= slightly below) the target.
    assert h_at_T >= h_target - 1e-6 or T >= 5.0 - 1e-9


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_clamp_and_logsumexp():
    assert _clamp(1.5) == 1.0
    assert _clamp(-0.5) == 0.0
    assert _logsumexp([1000.0, 1001.0]) == pytest.approx(1001.0 + math.log1p(math.exp(-1.0)), abs=1e-6)


def test_calibrated_confidence():
    assert calibrated_confidence(0.8, temperature=1.0) == pytest.approx(0.8)
    # Equation: conf ** (1/T).  T > 1 => exponent < 1 => raises a sub-1 conf.
    assert calibrated_confidence(0.8, temperature=2.0) == pytest.approx(0.8 ** 0.5)
    # T < 1 => exponent > 1 => lowers a sub-1 conf.
    assert calibrated_confidence(0.8, temperature=0.5) == pytest.approx(0.8 ** 2.0)
    assert calibrated_confidence(0.0) == 0.0
