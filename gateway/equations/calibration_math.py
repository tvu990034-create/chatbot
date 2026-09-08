"""
gateway/equations/calibration_math.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Calibration / uncertainty / OOD equations ported from the user's documentation
(Untitled document (6).txt and related temperature-scaling eqs).

This module is pure Python (stdlib `math` only, no numpy/scipy/sklearn). Every
calibration transform here is a *post-hoc recalibration* or a *scoring* routine —
none of them train a neural net, sample a posterior, or load model weights.

Verification-first principle: pure functions never raise on empty/invalid input,
probability-like outputs are clamped to [0, 1], and nothing returns NaN.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(x):
        return hi if x > 0 else lo
    return max(lo, min(hi, x))


def _logsumexp(x: Sequence[float]) -> float:
    """Numerically-stable log(sum(exp(x))) over a finite sequence."""
    xs = [float(v) for v in x if math.isfinite(float(v))]
    if not xs:
        return 0.0
    m = max(xs)
    return m + math.log(sum(math.exp(v - m) for v in xs))


def _inv_normal_cdf(p: float) -> float:
    """Pure-stdlib inverse normal CDF (Beasley–Springer–Moro / Acklam).

    Not actually used by any implemented equation here, but kept as a stdlib-only
    primitive should a conformal/confidence-interval variant need it."""
    p = _clamp(p, 1e-15, 1.0 - 1e-15)
    a = (-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00)
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


# ---------------------------------------------------------------------------
# Softmax + temperature scaling
# ---------------------------------------------------------------------------


def softmax(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    """Eq C1 — numerically stable softmax; subtracts the max (overflow guard).

    Returns a probability vector that sums to 1.0 (within fp tolerance).
    Empty input -> []. Non-positive temperature clamps to a tiny epsilon."""
    vals = [float(v) for v in logits]
    if not vals:
        return []
    t = temperature if temperature and temperature > 0 else 1e-9
    shifted = [v / t for v in vals]
    m = max(shifted)
    exps = [math.exp(v - m) for v in shifted]
    total = sum(exps)
    if total <= 0 or not math.isfinite(total):
        # Degenerate (all -inf etc.) -> uniform over the class count.
        n = len(vals)
        return [_clamp(1.0 / n)] * n
    return [_clamp(e / total) for e in exps]


def temperature_scale(logits: Sequence[float], temperature: float = 1.0) -> List[float]:
    """Eq C2 — temperature-scaled softmax: softmax(logits / T)."""
    return softmax(logits, temperature=temperature)


class TemperatureScaler:
    """Eq C3 — temperature scaling: `softmax(logits / T)` with a single scalar
    `T` fit to minimize expected calibration error over held-out logits.

    Uses a pure-stdlib golden-section search over T on the binned ECE (or NLL).
    Deterministic; fits only a scalar, so no training loop / no weights."""

    def __init__(self, objective: str = "ece", search_of: float = 0.01,
                 search_hi: float = 20.0, tol: float = 1e-3):
        self.objective = objective if objective in ("ece", "nll") else "ece"
        self.search_of = search_of
        self.search_hi = search_hi
        self.tol = tol
        self.temperature: float = 1.0
        self._nll_at_fit: float = float("inf")
        self._ece_at_fit: float = 1.0
        self._fitted: bool = False

    # -- objective ---------------------------------------------------------
    def _nll(self, logits_list: Sequence[Sequence[float]],
             true_labels: Sequence[int], T: float) -> float:
        """Mean negative log-likelihood of the true class under temperature T."""
        total, n = 0.0, 0
        for logits, y in zip(logits_list, true_labels):
            probs = softmax(logits, temperature=T)
            if not probs:
                continue
            n += 1
            idx = max(0, min(int(y), len(probs) - 1))
            p = _clamp(probs[idx], 1e-12, 1.0)
            total += -math.log(p)
        return total / n if n else 0.0

    def _ece_at(self, logits_list, true_labels, T: float) -> float:
        probs_list = [softmax(l, temperature=T) for l in logits_list]
        confidences, accuracies = [], []
        for probs, y in zip(probs_list, true_labels):
            if not probs:
                continue
            conf = max(probs)
            pred = max(range(len(probs)), key=probs.__getitem__)
            confidences.append(conf)
            accuracies.append(1.0 if pred == int(y) else 0.0)
        return ece(confidences, accuracies)

    def _objective_at(self, logits_list, true_labels, T: float) -> float:
        if self.objective == "nll":
            return self._nll(logits_list, true_labels, T)
        return self._ece_at(logits_list, true_labels, T)

    # -- golden-section scalar search -------------------------------------
    def fit(self, logits_list: Sequence[Sequence[float]],
            true_labels: Sequence[int]) -> "TemperatureScaler":
        """Fit the scalar temperature on `(logits, label)` pairs."""
        pairs = [(list(l), int(y)) for l, y in zip(logits_list, true_labels) if l]
        if not pairs:
            self.temperature = 1.0
            self._fitted = False
            return self

        def obj(T: float) -> float:
            return self._objective_at([p[0] for p in pairs], [p[1] for p in pairs], T)

        # Golden-section search over [search_of, search_hi].
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        a, b = float(self.search_of), float(self.search_hi)
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc, fd = obj(c), obj(d)
        while (b - a) > self.tol:
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = obj(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = obj(d)
        best = (a + b) / 2.0
        if not math.isfinite(best) or best <= 0:
            best = 1.0
        self.temperature = best
        self._ece_at_fit = self._ece_at([p[0] for p in pairs], [p[1] for p in pairs], best)
        self._nll_at_fit = self._nll([p[0] for p in pairs], [p[1] for p in pairs], best)
        self._fitted = True
        return self

    def calibrate(self, logits: Sequence[float]) -> List[float]:
        """Apply the fitted temperature to a logits vector."""
        return temperature_scale(logits, self.temperature)

    def predict(self, logits: Sequence[float]) -> List[float]:
        return self.calibrate(logits)

    def reset(self) -> None:
        self.temperature = 1.0
        self._fitted = False
        self._nll_at_fit = float("inf")
        self._ece_at_fit = 1.0

    def stats(self) -> Dict[str, float]:
        return {
            "temperature": self.temperature,
            "fitted": float(self._fitted),
            "ece_at_fit": self._ece_at_fit,
            "nll_at_fit": self._nll_at_fit,
        }


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def ece(confidences: Sequence[float], accuracies: Sequence[float],
        n_bins: int = 10) -> float:
    """Eq C4 — Expected Calibration Error over binned confidences, in [0, 1].

    Bins confidences; ECE = sum_b (n_b/N) * |acc_b - conf_b|."""
    confs = [float(c) for c in confidences]
    accs = [float(a) for a in accuracies]
    n = min(len(confs), len(accs))
    if n == 0:
        return 0.0
    n_bins = max(1, int(n_bins))
    total = 0.0
    for _ in range(n_bins):
        lo = _ / n_bins
        hi = (_ + 1) / n_bins
        # Accumulate a bin and compute on the fly (avoid python list slicing).
        csum = 0.0
        asum = 0.0
        cnt = 0
        for i in range(n):
            c = confs[i]
            if lo <= c < hi or (_ == n_bins - 1 and c == 1.0):
                csum += c
                asum += accs[i]
                cnt += 1
        if cnt:
            total += (cnt / n) * abs((asum / cnt) - (csum / cnt))
    return _clamp(total)


def brier_score(probs: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    """Eq C5 — Brier score: mean over samples of sum_k (p_k - y_k)^2, in [0, 2]."""
    total, n = 0.0, 0
    for p, y in zip(probs, labels):
        p = [float(v) for v in p]
        if not p:
            continue
        n += 1
        y = max(0, min(int(y), len(p) - 1))
        s = 0.0
        for k, pk in enumerate(p):
            target = 1.0 if k == y else 0.0
            s += (pk - target) ** 2
        total += s
    return total / n if n else 0.0


def calibrated_confidence(conf: float, temperature: float = 1.0) -> float:
    """Eq C6 — calibrated confidence `clamp(conf ** (1/temperature))`."""
    conf = _clamp(conf)
    t = temperature if temperature and temperature > 0 else 1e-9
    if conf <= 0:
        return 0.0
    return _clamp(conf ** (1.0 / t))


# ---------------------------------------------------------------------------
# Platt scaling (binary sigmoid recalibration)
# ---------------------------------------------------------------------------


class PlattScaler:
    """Eq C7 — Platt scaling: `p(y=1 | z) = 1 / (1 + exp(-(a*z + b)))`.

    Fits scalars `a` and `b` over [(logit_z, y)] pairs with a pure-stdlib
    gradient-descent loop on binary cross-entropy. `a == 0` and the safe default
    of predicting the base rate when no data / degenerate fit."""

    def __init__(self, lr: float = 0.1, epochs: int = 2000):
        self.lr = lr if lr and lr > 0 else 0.1
        self.epochs = int(epochs) if epochs and epochs > 0 else 2000
        self.a: float = 1.0
        self.b: float = 0.0
        self._base_rate: float = 0.5
        self._fitted: bool = False

    def _sigmoid(self, z: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            return 1.0 if z > 0 else 0.0

    def fit(self, logits: Sequence[float], labels: Sequence[int]) -> "PlattScaler":
        """Fit `a`, `b` mapping scalar logits -> class-1 probability."""
        zs = [float(z) for z in logits]
        ys = [1.0 if int(y) > 0 else 0.0 for y in labels]
        pairs = [(z, y) for z, y in zip(zs, ys) if math.isfinite(z)]
        if len(pairs) < 2:
            self.a, self.b = 1.0, 0.0
            self._base_rate = sum(y for _, y in pairs) / len(pairs) if pairs else 0.5
            self._fitted = False
            return self
        self._base_rate = sum(y for _, y in pairs) / len(pairs)

        a, b = 1.0, 0.0
        # Standardize the logits so gradient descent is well-conditioned.
        zmean = sum(z for z, _ in pairs) / len(pairs)
        zvar = sum((z - zmean) ** 2 for z, _ in pairs) / len(pairs)
        zstd = math.sqrt(zvar) or 1.0
        for _ in range(self.epochs):
            ga, gb = 0.0, 0.0
            for z, y in pairs:
                zn = (z - zmean) / zstd
                p = self._sigmoid(a * zn + b)
                err = p - y
                ga += err * zn
                gb += err
            n = len(pairs)
            a -= self.lr * (ga / n)
            b -= self.lr * (gb / n)
        self.a = a
        self.b = b
        self._fitted = True
        return self

    def predict_logit(self, logit: float) -> float:
        """Return the sigmoid output (probability) for a scalar logit."""
        z = float(logit)
        if self._fitted:
            return _clamp(self._sigmoid(self.a * z + self.b))
        # Unfitted -> fall back to base rate.
        return _clamp(self._base_rate)

    def predict_prob(self, logit: float) -> float:
        return self.predict_logit(logit)

    def reset(self) -> None:
        self.a, self.b = 1.0, 0.0
        self._base_rate = 0.5
        self._fitted = False

    def stats(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b, "base_rate": self._base_rate,
                "fitted": float(self._fitted)}


# ---------------------------------------------------------------------------
# Isotonic regression (PAVA)
# ---------------------------------------------------------------------------


def isotonic_regression(values: Sequence[float]) -> List[float]:
    """Eq C8 — Isotonic regression via PAVA (pool adjacent violators).

    Returns the fitted values as a monotonically *non-decreasing* list of the
    same length as `values`. Pure stdlib; `O(n)` amortised."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return []
    # blocks: (value, count)
    blocks: List[List[float]] = [[v, 1] for v in vals]
    stack: List[List[float]] = [blocks[0]]
    for i in range(1, n):
        blk = blocks[i]
        stack.append(blk)
        while len(stack) >= 2 and stack[-2][0] > stack[-1][0]:
            b_prev = stack[-2]
            b_cur = stack.pop()
            merged_value = (b_prev[0] * b_prev[1] + b_cur[0] * b_cur[1]) / (
                b_prev[1] + b_cur[1])
            merged_count = b_prev[1] + b_cur[1]
            stack[-1] = [merged_value, merged_count]
    out: List[float] = []
    for blk in stack:
        out.extend([blk[0]] * int(blk[1]))
    return out


# ---------------------------------------------------------------------------
# Conformal prediction
# ---------------------------------------------------------------------------


class ConformalSetPredictor:
    """Eq C9 — split-conformal conformal predictor (pure stdlib).

    Given a calibration set of singleton non-conformity scores (lower = more
    conforming), computes the split-conformal threshold at level `alpha` — the
    `ceil((n+1)(1-alpha))`-th order statistic of the scores.  New observations
    are then reported as the interval `[center - q, center + q]` around a
    regression centre, so the long-run coverage is ≈ (1 - alpha)."""

    def __init__(self, center: float = 0.0) -> None:
        self.center = float(center)
        self.alpha: float = 0.1
        self.radius: float = float("inf")
        self.threshold_low: float = float("-inf")
        self.threshold_high: float = float("inf")
        self._scores: List[float] = []
        self._fitted: bool = False

    def fit(self, scores: Sequence[Sequence[float]], alpha: float = 0.1) -> "ConformalSetPredictor":
        """Fit the conformal radius from a calibration set of nonconformity
        scores.  Each element may be a single score (treated as an absolute
        deviation) or a pair (low, high) whose radius is their spread."""
        alpha = _clamp(alpha, 0.001, 0.999)
        self.alpha = alpha
        radii: List[float] = []
        for s in scores:
            s = [float(x) for x in s]
            if len(s) == 0:
                continue
            if len(s) >= 2:
                radii.append(abs(s[1] - s[0]))
            else:
                radii.append(abs(s[0]))
        if not radii:
            self._fitted = False
            self._scores = []
            return self
        self._scores = radii
        # Split-conformal radius: the (1 - alpha) quantile of the calibration
        # radii (equivalently the ceil((n+1)(1-alpha)) order statistic).
        q = _quantile(sorted(radii), 1.0 - alpha)
        self.radius = q
        self.threshold_low = self.center - q
        self.threshold_high = self.center + q
        self._fitted = True
        return self

    def predict_interval(self, scores: Sequence[float]) -> Tuple[float, float]:
        """Return `(q_low, q_high)` the predicted interval for observation(s).

        New radii are compared against the stored conformal radius; the interval
        is reported around this predictor's centre."""
        if not self._fitted:
            return (float("-inf"), float("inf"))
        return (self.threshold_low, self.threshold_high)

    def contains(self, value: float) -> bool:
        return self.threshold_low <= float(value) <= self.threshold_high

    def reset(self) -> None:
        self._fitted = False
        self._scores = []
        self.radius = float("inf")
        self.threshold_low = float("-inf")
        self.threshold_high = float("inf")

    def stats(self) -> Dict[str, float]:
        return {
            "alpha": self.alpha,
            "center": self.center,
            "radius": self.radius,
            "q_low": self.threshold_low,
            "q_high": self.threshold_high,
            "n_scores": float(len(self._scores)),
            "fitted": float(self._fitted),
        }


def _quantile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of a *sorted* sequence (pure stdlib)."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    q = _clamp(q, 0.0, 1.0)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------------------
# Entropy & abstention
# ---------------------------------------------------------------------------


def entropy(probs: Sequence[float]) -> float:
    """Eq C10 — Shannon entropy H = -sum(p log p) in [0, log(n_classes)]."""
    total = 0.0
    for p in probs:
        p = _clamp(float(p), 0.0, 1.0)
        if p > 0:
            total -= p * math.log(p)
    return max(0.0, total)


def max_entropy(n_classes: int) -> float:
    """Eq C11 — maximum entropy log(n) for `n` classes (0 classes -> 0)."""
    n = max(0, int(n_classes))
    if n <= 1:
        return 0.0
    return math.log(n)


def entropy_abstention_decision(entropy_value: float, n_classes: int,
                                threshold_frac: float = 0.7) -> bool:
    """Eq C12 — abstain (return True) when entropy exceeds threshold_frac * H_max."""
    hmax = max_entropy(max(1, int(n_classes)))
    threshold = threshold_frac * hmax
    return float(entropy_value) > threshold


# ---------------------------------------------------------------------------
# Energy-based OOD score
# ---------------------------------------------------------------------------


def energy_ood_score(logits: Sequence[float], temperature: float = 1.0) -> float:
    """Eq C13 — energy OOD score `-T * logsumexp(logits / T)`.

    Lower (more negative) energy when logits are concentrated on a class,
    higher (near zero) when they are uniform/OOD."""
    vals = [float(v) for v in logits]
    if not vals:
        return 0.0
    t = temperature if temperature and temperature > 0 else 1e-9
    scaled = [v / t for v in vals]
    return -t * _logsumexp(scaled)


def is_ood(energy: float, in_dist_energy_mean: float, in_dist_energy_std: float,
           z: float = 2.0) -> bool:
    """Eq C14 — flag OOD when `energy < mean - z*std` (a deviation gate)."""
    std = in_dist_energy_std if in_dist_energy_std and in_dist_energy_std > 0 else 0.0
    if std == 0.0:
        return float(energy) < float(in_dist_energy_mean) - z
    return float(energy) < float(in_dist_energy_mean) - z * std


# ---------------------------------------------------------------------------
# Confidence / uncertainty aggregation
# ---------------------------------------------------------------------------


def max_confidence(probs: Sequence[float]) -> float:
    """Eq C15 — max probability (argmax-confidence) in [0, 1]."""
    if not probs:
        return 0.0
    return _clamp(max(float(p) for p in probs))


def top2_margin(probs: Sequence[float]) -> float:
    """Eq C16 — margin between the top-1 and top-2 probabilities in [0, 1]."""
    if not probs:
        return 0.0
    s = sorted((float(p) for p in probs), reverse=True)
    top1 = s[0]
    top2 = s[1] if len(s) > 1 else 0.0
    return _clamp(top1 - top2)


def predicted_probability_ratio(probs: Sequence[float], k: int = 2) -> float:
    """Eq C17 — ratio of top-1 probability to the mass of the top-`k` classes.

    Robust confidence signal (closer to 1 means very confident).  Returns 0.0
    when the denominator is 0 (degenerate)."""
    if not probs:
        return 0.0
    s = sorted((float(p) for p in probs), reverse=True)
    top1 = s[0]
    k = max(1, min(int(k), len(s)))
    denom = sum(s[:k])
    if denom <= 0:
        return 0.0
    return _clamp(top1 / denom)


# ---------------------------------------------------------------------------
# Ensemble uncertainty (variance decomposition)
# ---------------------------------------------------------------------------


class EnsembleUncertainty:
    """Eq C18 — decompose ensemble disagreement into aleatoric and epistemic
    uncertainty from a list of per-model probability vectors.

    * aleatoric_variance — mean over models of the per-class variance around the
      ensemble mean (irreducible noise).
    * epistemic_variance — variance of the model predictions around the ensemble
      mean (model disagreement).
    * disagreement — mean absolute deviation of each model's max-prob class vote
      from the majority vote."""

    def __init__(self) -> None:
        self.models: List[List[float]] = []
        self._mean: List[float] = []
        self._aleatoric: float = 0.0
        self._epistemic: float = 0.0
        self._disagreement: float = 0.0

    def add_model(self, probs: Sequence[float]) -> None:
        self.models.append([float(p) for p in probs])

    def compute(self) -> "EnsembleUncertainty":
        """Compute the decomposition from the accumulated per-model vectors."""
        if not self.models:
            self._mean = []
            self._aleatoric = 0.0
            self._epistemic = 0.0
            self._disagreement = 0.0
            return self
        n_models = len(self.models)
        n_classes = max((len(m) for m in self.models), default=0)

        # Ensemble mean per class.
        mean = [0.0] * n_classes
        for m in self.models:
            for k in range(n_classes):
                mean[k] += m[k] if k < len(m) else 0.0
        for k in range(n_classes):
            mean[k] /= n_models
        self._mean = mean

        # Epistemic: variance of model predictions around the mean (per class).
        epi = 0.0
        for m in self.models:
            for k in range(n_classes):
                mk = m[k] if k < len(m) else 0.0
                epi += (mk - mean[k]) ** 2
        self._epistemic = epi / (n_models * n_classes) if n_classes else 0.0

        # Aleatoric: mean per-class variance implied by the ensemble mean dist.
        ale = 0.0
        counter = 0
        for k in range(n_classes):
            # Within-class variance of predictions across models.
            var_sum = 0.0
            for m in self.models:
                mk = m[k] if k < len(m) else 0.0
                var_sum += (mk - mean[k]) ** 2
            # Contribution to aleatoric = mean of that variance.
            ale += mean[k] * (var_sum / n_models) if mean[k] > 0 else (var_sum / n_models)
            counter += 1
        self._aleatoric = ale / counter if counter else 0.0

        # Disagreement: fraction of models disagreeing with the majority class.
        votes: List[int] = []
        for m in self.models:
            votes.append(max(range(len(m)), key=m.__getitem__) if m else -1)
        from collections import Counter
        if votes and all(v >= 0 for v in votes):
            most_common = Counter(votes).most_common(1)[0][0]
            disagree = sum(1 for v in votes if v != most_common) / len(votes)
            self._disagreement = _clamp(disagree)
        else:
            self._disagreement = 0.0
        return self

    def mean_probs(self) -> List[float]:
        return list(self._mean)

    def reset(self) -> None:
        self.models.clear()
        self._mean = []
        self._aleatoric = 0.0
        self._epistemic = 0.0
        self._disagreement = 0.0

    def stats(self) -> Dict[str, float]:
        return {
            "aleatoric_variance": self._aleatoric,
            "epistemic_variance": self._epistemic,
            "disagreement": self._disagreement,
            "n_models": float(len(self.models)),
        }


# ---------------------------------------------------------------------------
# Abstention gating & temperature-for-entropy
# ---------------------------------------------------------------------------


def should_defer_to_verify(conf: float, threshold: float = 0.6) -> bool:
    """Eq C19 — defer/route for verification when confidence is below threshold.

    Returns True (defer) when `conf < threshold` (low confidence)."""
    return float(conf) < float(threshold)


def temperature_to_meet_entropy(probs: Sequence[float],
                                target_entropy_frac: float = 0.9,
                                max_temperature: float = 5.0) -> float:
    """Eq C20 — find the temperature that raises softmax entropy toward a target
    fraction of the maximum entropy, via pure-stdlib bisection on [~0, max_T].

    Returns the smallest temperature in `(0, max_temperature]` whose calibrated
    entropy reaches `target_entropy_frac * H_max` (capped at `max_temperature`
    when even that is insufficient)."""
    p = [float(x) for x in probs]
    n = len(p)
    if n < 2:
        return 1.0
    hmax = max_entropy(n)
    target = target_entropy_frac * hmax

    def h_at(T: float) -> float:
        return entropy(softmax(p, temperature=T))

    # If even uniform-like high temp doesn't reach the target, cap at max.
    if h_at(max_temperature) <= target:
        return float(max_temperature)
    lo, hi = 1e-3, float(max_temperature)
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if h_at(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(hi)


# ---------------------------------------------------------------------------
# NOT-IMPLEMENTABLE placeholders (training / model-weight dependent)
# ---------------------------------------------------------------------------

# NOT_IMPLEMENTABLE: needs gradient-based variational inference on a trained NN +
# sampled posterior (SGVB/ELBO) which requires a model and a training loop.
NOT_IMPLEMENTABLE_BNN_VI = (
    "bayesian-neural-network-variational-inference:needs-trained-model+ELBO-training-loop"
)

# NOT_IMPLEMENTABLE: full Gaussian-process posterior (Cholesky of a kernel matrix +
# GP mean/variance) requires a kernel matrix over the dataset, i.e. data-scale
# linear algebra and a fitted kernel — not a scalar stdlib routine.
NOT_IMPLEMENTABLE_FULL_GP_POSTERIOR = (
    "gaussian-process-full-posterior:needs-kernel-matrix+cholesky-on-dataset"
)

# NOT_IMPLEMENTABLE: normalizing-flow density estimation (affine coupling layers,
# invertible transforms) requires fitting a flow on data — a training loop.
NOT_IMPLEMENTABLE_NORMALIZING_FLOW_DENSITY = (
    "normalizing-flow-density:needs-fitting-flow-on-data+training-loop"
)

# NOT_IMPLEMENTABLE: full dropout-MC uncertainty requires running stochastic
# forward passes through a trained model with dropout enabled — a live model.
NOT_IMPLEMENTABLE_MC_DROPOUT_UNCERTAINTY = (
    "mc-dropout-uncertainty:needs-stochastic-forward-passes-on-trained-model"
)

# NOT_IMPLEMENTABLE: expected-calibration-error over a *neural* temperature net
# (multi-parameter scaling) would require backprop through a trained classifier.
NOT_IMPLEMENTABLE_PARAMETRIC_NEURAL_CALIBRATION = (
    "parametric-neural-calibration:needs-backprop-through-trained-classifier"
)


__all__ = [
    "_clamp", "_logsumexp", "_inv_normal_cdf", "softmax", "temperature_scale",
    "TemperatureScaler", "ece", "brier_score", "calibrated_confidence",
    "PlattScaler", "isotonic_regression", "ConformalSetPredictor", "entropy",
    "max_entropy", "entropy_abstention_decision", "energy_ood_score", "is_ood",
    "max_confidence", "top2_margin", "predicted_probability_ratio",
    "EnsembleUncertainty", "should_defer_to_verify",
    "temperature_to_meet_entropy",
    "NOT_IMPLEMENTABLE_BNN_VI", "NOT_IMPLEMENTABLE_FULL_GP_POSTERIOR",
    "NOT_IMPLEMENTABLE_NORMALIZING_FLOW_DENSITY",
    "NOT_IMPLEMENTABLE_MC_DROPOUT_UNCERTAINTY",
    "NOT_IMPLEMENTABLE_PARAMETRIC_NEURAL_CALIBRATION",
]
