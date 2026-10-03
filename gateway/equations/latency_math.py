"""
Latency-peak measurement & optimization module (pure stdlib).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The num_predict -> latency curve for qwen3 on CPU is NON-MONOTONIC: budget 160
produced an 887-char ramble at 264s while budget 384 produced the bare number
at ~138s (measured on the same item).  A "reduced budgams is always faster /
a larger budget is always slower" monotone model therefore inverts the true
objective, so this module treats the curve as unknown and PEAK-CENTERED:

  * EqSet-L (Eq 1-20)  — measurement dimensions + bracketing scan.
  * Eq 21-40           — curve-fitting / vertex refinement of the peak.
  * Eq 41-60           — acquisition (UCB over candidate budgets).
  * Eq 61-80           — statistics & validation (Welch t, JZS Bayes factor).
  * Eq 181-200         — sampling design (plan_budgets, sample size).

Everything is pure stdlib and deterministic-able: an injected callable is the
only I/O, numeric integration is closed-form Simpson on a log grid, and every
probability is clamped to [0, 1]. No network, no models, no third-party libs.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Local primitives (pure stdlib)
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(x):
        return hi if x > 0 else lo
    return max(lo, min(hi, x))


def _mean(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs)) / len(xs)


def _variance(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return float(sum((x - mu) ** 2 for x in xs)) / (len(xs) - 1)


def _std(xs: Sequence[float]) -> float:
    return math.sqrt(max(0.0, _variance(xs)))


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _mad(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    m = _median(xs)
    return _median([abs(x - m) for x in xs])


# ---------------------------------------------------------------------------
# Normal CDF / inverse-CDF (Acklam rational approximation, pure stdlib)
# ---------------------------------------------------------------------------


def _normal_cdf(z: float) -> float:
    """Standard normal CDF via math.erf; clamped to [0,1]."""
    return _clamp(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def _inv_normal_cdf(p: float, lo: float = 1e-15, hi: float = 1.0 - 1e-15) -> float:
    """Acklam's inverse of the standard normal CDF (|err| < 1.15e-9)."""
    p = _clamp(p, lo, hi)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 0.97575
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


# ---------------------------------------------------------------------------
# Regularized incomplete beta (continued fraction, pure stdlib)
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float, max_iter: int = 300,
            epsilon: float = 3e-12) -> float:
    """Lentz's continued fraction for the incomplete beta; guarded for a=0."""
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b). Clamped to [0,1]."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log1p(-x))
    bt = math.exp(ln)
    if x < (a + 1.0) / (a + b + 2.0):
        return _clamp(bt * _betacf(a, b, x) / a)
    return _clamp(1.0 - bt * _betacf(b, a, 1.0 - x) / b)


def _t_cdf(t: float, df: float) -> float:
    """Student-t two-parameter CDF (pure stdlib).

    P(T <= t) = 1 - 0.5 * I_x(df/2, 1/2) for t >= 0, and 0.5 * I_x otherwise,
    with x = df / (df + t^2) -- the standard identity linking the t tail and
    the regularized incomplete beta.
    """
    df = max(0.5, float(df))
    x = df / (df + t * t)
    tail = 0.5 * _betai(df / 2.0, 0.5, _clamp(x))
    return _clamp(1.0 - tail if t >= 0.0 else tail)


def _two_tail_p(t: float, df: float) -> float:
    p = _t_cdf(abs(t), df)
    return _clamp(2.0 * (1.0 - p))


# ---------------------------------------------------------------------------
# EqSet-L (Eq 1-20) — measurement dimensions + bracketing scan
# ---------------------------------------------------------------------------


def latency(t_start: float, t_end: float, out_tokens: int = 0,
            out_chars: int = 0) -> dict:
    """Eq 1-8 — the measurement dimensions.

    tau = total wall-clock (s); throughput = out_tokens / tau and
    chars/s = out_chars / tau.  A zero-length or inverted interval is a safe
    zero tuple, never NaN.  tau is the decision variable: it is monotone in
    budget on *short* answers and NON-monotone on rambles, so callers must
    bracket, never assume monotonicity.
    """
    tau = max(0.0, float(t_end) - float(t_start))
    tok_s = float(out_tokens) / tau if tau > 0 else 0.0
    char_s = float(out_chars) / tau if tau > 0 else 0.0
    return {"tau_s": tau, "tokens_per_s": round(tok_s, 4),
            "chars_per_s": round(char_s, 4)}


def median_latency(obs: Dict[int, Sequence[float]]) -> Dict[int, float]:
    """Eq 9 — per-budget median latency. Empty sample -> 0.0."""
    return {b: _median(v) for b, v in obs.items()}


def peak_budget(medians: Dict[int, float], plateau_tol: float = 0.0) -> int:
    """Eq 10-12 — argmax budget of the median curve.

    The curve is peak-centered: pick the budget whose median latency is
    largest.  Within ``plateau_tol`` seconds of the max, the SMALLEST budget
    wins (cheapest way to hit the plateau).  Empty input returns -1.
    """
    if not medians:
        return -1
    peak = max(medians, key=lambda b: medians[b])
    top = medians[peak]
    candidates = [b for b, m in medians.items()
                  if m >= top - max(0.0, plateau_tol)]
    return min(candidates)


def peak_bracket(obs: Dict[int, Sequence[float]],
                 plateau_tol: float = 0.0) -> Tuple[int, int, int]:
    """Eq 13-20 — bracket (b_lo, b_peak, b_hi) around the dominant peak.

    Walks out from the argmax budget while the median descends, so the triple
    brackets a single mode.  With fewer than 2 distinct budgets the bracket
    collapses to (b, b, b).  Deterministic, never raises.
    """
    med = median_latency(obs)
    if not med:
        return (-1, -1, -1)
    budgets = sorted(med)
    i_peak = budgets.index(peak_budget(med, plateau_tol))
    b_peak = budgets[i_peak]
    b_lo, b_hi = b_peak, b_peak
    # walk left while strictly descending
    j = i_peak - 1
    while j >= 0:
        if med[budgets[j]] < med[budgets[j + 1]] - max(0.0, plateau_tol):
            b_lo = budgets[j]
            j -= 1
        else:
            break
    # walk right while strictly descending
    j = i_peak + 1
    while j < len(budgets):
        if med[budgets[j]] < med[budgets[j - 1]] - max(0.0, plateau_tol):
            b_hi = budgets[j]
            j += 1
        else:
            break
    return (b_lo, b_peak, b_hi)


# ---------------------------------------------------------------------------
# Eq 21-40 — curve fitting / vertex refinement
# ---------------------------------------------------------------------------


def parabola_vertex(x1: float, y1: float, x2: float, y2: float,
                    x3: float, y3: float) -> Tuple[float, float]:
    """Eq 21-25 — x of the parabola vertex through three points.

    Returns (x*, y*) of the fitted vertex.  If the denominator is ~0 the
    three points are collinear (no well-defined peak) and the middle point is
    returned along with its value.  y* is the vertex value read through the
    Lagrange interpolating parabola (no separate a/b/c coefficient algebra).
    NaN-safe.
    """
    denom = (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(denom) < 1e-12:
        return (float(x2), float(y2))
    x_star = 0.5 * ((x1 * x1 * (y2 - y3) + x2 * x2 * (y3 - y1)
                     + x3 * x3 * (y1 - y2)) / denom)

    def _lagrange(x: float) -> float:
        return (y1 * (x - x2) * (x - x3) / ((x1 - x2) * (x1 - x3))
                + y2 * (x - x1) * (x - x3) / ((x2 - x1) * (x2 - x3))
                + y3 * (x - x1) * (x - x2) / ((x3 - x1) * (x3 - x2)))

    return (float(x_star), float(_lagrange(x_star)))


def refine_from_bracket(obs: Dict[int, Sequence[float]]) -> Tuple[float, float]:
    """Eq 26-30 — refine the peak with a parabola through the bracket triple."""
    (b_lo, b_peak, b_hi) = peak_bracket(obs)
    if b_lo < 0 or b_lo == b_peak == b_hi:
        return (float(b_peak), 0.0)
    med = median_latency(obs)
    x, y = parabola_vertex(b_lo, med[b_lo], b_peak, med[b_peak],
                           b_hi, med[b_hi])
    return (x, y)


# ---------------------------------------------------------------------------
# Eq 41-60 — acquisition (UCB over candidate budgets)
# ---------------------------------------------------------------------------


def ucb_chart(obs: Dict[int, Sequence[float]], candidates: Sequence[int],
              c: float = 1.0, length_penalty: float = 0.0) -> Dict[int, float]:
    """Eq 41-50 — UCB score per candidate budget.

    score = median_b + c * MAD_b - length_penalty * log2(b).  Unprobed
    candidates get +inf so they are explored before any measured budget is
    re-picked.  Driven toward the latency PEAK (max), matching Eq 1.
    """
    med = median_latency(obs)
    out: Dict[int, float] = {}
    for b in candidates:
        b = int(b)
        if b not in med:
            out[b] = math.inf
            continue
        spread = _mad(obs[b])
        out[b] = med[b] + float(c) * spread - float(length_penalty) * math.log2(max(2, b))
    return out


def suggest_next_budget(obs: Dict[int, Sequence[float]],
                        candidates: Sequence[int], c: float = 1.0,
                        length_penalty: float = 0.0) -> int:
    """Eq 51-60 — the best next budget to probe (max-U CB, unseen first)."""
    if not candidates:
        return -1
    chart = ucb_chart(obs, candidates, c, length_penalty)
    return max(candidates, key=lambda b: chart[b])


# ---------------------------------------------------------------------------
# Eq 61-80 — statistics & validation
# ---------------------------------------------------------------------------


def welch_t(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    """Eq 61-70 — Welch's two-sample t. Returns (t, df, two-tail p)."""
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    if not xs or not ys:
        return (0.0, 1.0, 1.0)
    n1, n2 = len(xs), len(ys)
    v1, v2 = _variance(xs), _variance(ys)
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se <= 0.0:
        return (0.0, float(n1 + n2 - 2), 1.0) if _mean(xs) == _mean(ys) \
            else (math.copysign(float("inf"), _mean(xs) - _mean(ys)),
                  float(n1 + n2 - 2), 0.0)
    t = (_mean(xs) - _mean(ys)) / se
    num = (v1 / n1 + v2 / n2) ** 2
    denom = (v1 / n1) ** 2 / max(1, n1 - 1) + (v2 / n2) ** 2 / max(1, n2 - 1)
    df = num / denom if denom > 0 else float(n1 + n2 - 2)
    return (float(t), float(df), _two_tail_p(t, df))


def jzs_bf10(xs: Sequence[float], ys: Sequence[float],
             r: float = 0.7071, grid_n: int = 400) -> float:
    """Eq 71-80 — JZS Bayes factor BF10 for a two-sample comparison (scaled
    Cauchy prior of scale ``r`` over effect size).

    Implemented as the one-group equivariant integral over g with the
    inverse-gamma(1/2, r^2/2) prior (that is exactly the scaled-Cauchy on the
    effect).  Integration is composite Simpson on a log-equidistant g grid;
    BF10 = 1 flat, >1 toward H1 (distinct means), <1 toward H0.  Clamped to
    [0, 1e6] for numerical safety.
    """
    xs = [float(x) for x in xs]
    ys = [float(y) for y in ys]
    if not xs or not ys:
        return 1.0
    t, df, _p = welch_t(xs, ys)
    t, df = float(t), max(0.5, float(df))
    n = len(xs)

    denom = (1.0 + t * t / df) ** (-(df + 1.0) / 2.0)
    if denom <= 0.0:
        return 1.0

    g0, g1 = 1e-4, 1e4
    acc = 0.0
    for i in range(grid_n + 1):
        frac = i / grid_n
        g = g0 * (g1 / g0) ** frac
        w = 3.0 if (i % 2 == 0) else 6.0
        if i in (0, grid_n):
            w = 1.0
        prior_g = (r * r / 2.0) ** 0.5 / math.sqrt(math.pi) * (
            (g ** -1.5) * math.exp(-(r * r) / (2.0 * g)))
        integrand = ((1.0 + n * g) ** -0.5
                     * (1.0 + t * t / (df * (1.0 + n * g))) ** (-(df + 1.0) / 2.0)
                     * prior_g * g)  # dg = g * d(ln g)
        if math.isfinite(integrand):
            acc += w * integrand
    step = math.log(g1 / g0) / grid_n
    return _clamp((acc * step / 3.0) / denom, 0.0, 1e6)


def savings_test(xs_opt: Sequence[float], xs_base: Sequence[float],
                 delta_cut: float = 10.0, alpha: float = 0.05) -> dict:
    """Eq 81-90 — decide whether ``opt`` is a reliable latency SAVING vs base.

    Declares "save" only when the median saving exceeds ``delta_cut`` seconds
    AND is statistically significant at ``alpha``.  Reports the median, MAD,
    t/p, and the verdict.  Degenerate (empty) inputs verdict False.
    """
    xs_opt = [float(x) for x in xs_opt]
    xs_base = [float(x) for x in xs_base]
    if not xs_opt or not xs_base:
        return {"median_save_s": 0.0, "mad_s": 0.0, "t": 0.0, "df": 1.0,
                "p": 1.0, "significant": False, "verdict": "no-data"}
    med_opt, med_base = _median(xs_opt), _median(xs_base)
    save = med_base - med_opt
    t, df, p = welch_t(xs_base, xs_opt)
    significant = _clamp(p, 0.0, 1.0) <= max(0.0, float(alpha))
    verdict = "save" if (save > float(delta_cut) and significant) else \
        ("worse" if save < -float(delta_cut) and significant else "tie")
    return {"median_save_s": round(save, 3), "mad_s": round(_mad(xs_opt), 3),
            "t": round(t, 4), "df": round(df, 3), "p": round(_clamp(p), 6),
            "significant": significant, "verdict": verdict}


# ---------------------------------------------------------------------------
# Eq 181-200 — sampling design (budget cards + sample sizes)
# ---------------------------------------------------------------------------


def plan_budgets(lo: int, hi: int, mult: float = 2.0, max_n: int = 8,
                 base: Optional[int] = None) -> List[int]:
    """Eq 181-190 — a geometric budget card straddling ``[lo, hi]``.

    The 03-family ladder (128 -> 384 -> 1152, mult=3) was too sparse to
    bracket a qwen3 peak, so the default here is a denser 02-family ladder
    (mult=2).  Start at ``base`` (default lo) and multiply by ``mult``,
    keeping only ints inside the range (plus the range edges).  Never raises;
    returns at least {lo, hi} when max_n < 2.
    """
    lo, hi = int(lo), int(hi)
    if hi < lo:
        lo, hi = hi, lo
    mult = float(mult) if mult and mult > 1.0 else 2.0
    max_n = max(1, int(max_n))
    if max_n < 2 or hi <= lo:
        return [lo, hi]
    seed = lo if base is None else int(base)
    if not (lo <= seed <= hi):
        seed = lo
    out: List[int] = []
    b = float(seed)
    while b <= hi and len(out) < max_n - 1:
        out.append(int(round(b)))
        b *= mult
    out = [lo] + out + [hi]
    # de-dupe preserving order, clamp to range
    seen: List[int] = []
    for b in out:
        b = max(lo, min(hi, b))
        if b not in seen:
            seen.append(b)
    return sorted(seen)


def sample_size_needed(effect_s: float, sigma_s: float, alpha: float = 0.05,
                       power: float = 0.8, cap: int = 4096) -> int:
    """Eq 191-200 — per-group sample size for a two-sided Welch t-test.

    n = 2 * sigma^2 * (z_{1-a/2} + z_{power})^2 / effect^2, rounded UP, min 2.
    A zero effect returns ``cap`` (no number of samples can distinguish).
    """
    effect_s = abs(float(effect_s))
    sigma_s = abs(float(sigma_s))
    if effect_s <= 0.0:
        return max(2, int(cap))
    z = _inv_normal_cdf(1.0 - float(alpha) / 2.0) + _inv_normal_cdf(float(power))
    n = math.ceil(2.0 * sigma_s * sigma_s * z * z / (effect_s * effect_s))
    return max(2, min(int(cap), n))


def three_point_card(lo: int = 160, mid: int = 384,
                     hi: int = 768) -> List[int]:
    """Eq 181-190 — the minimal card that can see a local latency max.

    tau(b) for qwen3 on CPU is NON-monotonic (budget 160 rambled 887 chars
    at 264s while 384 answered bare at ~138s), so b* must come from a
    measurement that brackets the peak, never from a hardcoded budget.
    Three points are the minimum: the middle point slower than BOTH
    neighbors proves a local max.  Defaults anchor the two measured points
    (160 = known ramble, 384 = known clean) plus a high-side probe.
    Sorted, de-duplicated, never raises.
    """
    pts = sorted({int(lo), int(mid), int(hi)})
    return [b for b in pts if b > 0] or [int(mid)]


def pick_b_star(medians: Dict[int, float],
                ok_by_budget: Optional[Dict[int, bool]] = None,
                length_penalty: float = 0.0,
                lengths: Optional[Dict[int, float]] = None) -> dict:
    """Eq 181-200 — b* from a measured card: fastest CORRECT off-peak budget.

    ``medians`` maps budget -> median seconds; ``ok_by_budget`` marks which
    budgets answered correctly (a ramble that never converges counts as not
    ok, which is how the 160-budget ramble loses even before timing out).
    Among correct budgets the score is ``median + length_penalty * chars``,
    so a fast-but-rambling budget still loses to a clean answer; ties break
    toward the smaller budget.  With no correct budget the peak is returned
    (most likely to contain an answer at all).  Empty input -> b_star -1.
    """
    ok = ok_by_budget or {}
    med = {int(b): float(m) for b, m in (medians or {}).items()
           if math.isfinite(float(m))}
    peak = peak_budget(med)
    correct = [b for b in med if ok.get(b)]
    if not correct:
        return {"b_star": peak, "peak": peak, "n_correct": 0,
                "measured": sorted(med)}
    lens = lengths or {}
    pen = max(0.0, float(length_penalty))

    def _score(b: int) -> Tuple[float, int]:
        return (med[b] + pen * float(lens.get(b, 0.0)), b)

    best = min(correct, key=_score)
    return {"b_star": best, "peak": peak, "n_correct": len(correct),
            "measured": sorted(med)}


# ---------------------------------------------------------------------------
# Integrated driver — LatencyScanner (EqSet-L + Eq 21-40 + Eq 181-200)
# ---------------------------------------------------------------------------


class LatencyScanner:
    """Eq 181-200 integrated scanner.

    Holds per-budget latency observations, exposes the peak, its bracket and
    a UCB next-budget suggestion.  ``measure`` is an injected callable
    ``(budget: int) -> seconds: float`` (or None on failure) — the only I/O.
    Deterministic with a seeded/default callable; thread-safe by construction
    (no shared mutable globals).
    """

    def __init__(self, lo: int = 128, hi: int = 1152, budget_mult: float = 3.0,
                 max_n: int = 8, c: float = 1.0, length_penalty: float = 0.0,
                 plateau_tol: float = 0.0):
        self.lo = int(lo)
        self.hi = int(hi)
        self.budget_mult = float(budget_mult)
        self.max_n = int(max_n)
        self.c = float(c)
        self.length_penalty = float(length_penalty)
        self.plateau_tol = float(plateau_tol)
        self.obs: Dict[int, List[float]] = {}
        self.order: List[int] = []

    # -- state -------------------------------------------------------------
    def add(self, budget: int, seconds: Optional[float]) -> None:
        """Record one observation; None (failure) is ignored."""
        if seconds is None or not math.isfinite(float(seconds)):
            return
        budget = int(budget)
        self.obs.setdefault(budget, []).append(float(seconds))
        if budget not in self.order:
            self.order.append(budget)

    def reset(self) -> None:
        self.obs.clear()
        self.order.clear()

    def stats(self) -> dict:
        return {"n_budgets": len(self.obs),
                "n_obs": sum(len(v) for v in self.obs.values()),
                "budgets": list(self.order)}

    # -- queries -----------------------------------------------------------
    def candidates(self, max_n: Optional[int] = None) -> List[int]:
        return plan_budgets(self.lo, self.hi, self.budget_mult,
                            max_n or self.max_n)

    def median_chart(self) -> Dict[int, float]:
        return median_latency(self.obs)

    def peak(self) -> Tuple[int, float]:
        med = self.median_chart()
        b = peak_budget(med, self.plateau_tol)
        return (b, med.get(b, 0.0))

    def bracket(self) -> Tuple[int, int, int]:
        return peak_bracket(self.obs, self.plateau_tol)

    def refined_peak(self) -> Tuple[float, float]:
        return refine_from_bracket(self.obs)

    def next_budget(self) -> int:
        return suggest_next_budget(self.obs, self.candidates(), self.c,
                                   self.length_penalty)

    # -- scan driver -------------------------------------------------------
    def scan(self, measure: Callable[[int], Optional[float]],
             neglect_measured: bool = True) -> dict:
        """Eq 181-200 — probe the budget card then the refined peak.

        Also probes the parabola-refined vertex (rounded to an int within
        range) and the UCB suggestion if they were not already measured, so
        the scan converges on the true (possibly non-monotone) peak in O(k)
        calls.
        """
        votes = list(self.candidates())
        if neglect_measured:
            votes = [b for b in votes if b not in self.obs]
        for b in votes:
            t = measure(b)
            self.add(b, t)
        # refine the peak once
        x, _y = self.refined_peak()
        b_refine = int(round(x))
        if b_refine not in self.obs and self.lo <= b_refine <= self.hi:
            self.add(b_refine, measure(b_refine))
        # UCB acquisition
        b_ucb = self.next_budget()
        if b_ucb > 0 and b_ucb not in self.obs:
            self.add(b_ucb, measure(b_ucb))
        peak_b, peak_v = self.peak()
        return {"peak_budget": peak_b, "peak_latency_s": round(peak_v, 3),
                "bracket": list(self.bracket()), "n_obs": self.stats()["n_obs"],
                "median_chart": {str(b): round(m, 3)
                                 for b, m in sorted(self.median_chart().items())}}


__all__ = [
    # primitives
    "_clamp", "_mean", "_variance", "_std", "_median", "_mad",
    "_normal_cdf", "_inv_normal_cdf", "_betai", "_t_cdf", "_two_tail_p",
    # measurement (Eq 1-20)
    "latency", "median_latency", "peak_budget", "peak_bracket",
    # fitting (Eq 21-40)
    "parabola_vertex", "refine_from_bracket",
    # acquisition (Eq 41-60)
    "ucb_chart", "suggest_next_budget",
    # validation (Eq 61-80)
    "welch_t", "jzs_bf10", "savings_test",
    # sampling design (Eq 181-200)
    "plan_budgets", "sample_size_needed",
    "three_point_card", "pick_b_star",
    # integrated driver
    "LatencyScanner",
]