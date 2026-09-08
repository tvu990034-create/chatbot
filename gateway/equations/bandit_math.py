"""
Multi-armed bandit and Bayesian-optimization optimization module (pure stdlib).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-armed bandit and Bayesian-optimization optimizations, in pure stdlib.

This module implements the exploration/exploitation family from the user's catalog:
Thompson sampling (#217), UCB / Thompson temperature (#227), and the
EWMA/EXP3/UCB/EI/PI acquisition functions from the
a-tier / s-tier files.

Everything here is deterministic-able via an injected ``random.Random`` (or the
module-level ``random``) and never requires a training loop, GPU, or model
weights. Optimizations that genuinely require those are declared as clearly-marked
``NOT_IMPLEMENTABLE_*`` constants at the bottom.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Local probability primitives (pure stdlib — no scipy)
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(x):
        return hi if x > 0 else lo
    return max(lo, min(hi, x))


def _normal_pdf(x: float) -> float:
    """Standard normal PDF phi(x) = exp(-x^2/2)/sqrt(2*pi)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF Phi(x) via the erf-based approximation. In [0,1]."""
    return _clamp(0.5 * (1.0 + _erf(x / math.sqrt(2.0))))


def _erf(x: float) -> float:
    """Abramowitz & Stegun 7.1.26 rational approximation of erf; |err| <= 1.5e-7."""
    # For large |x|, erf saturates at +/-1 — avoid overflow on the t = 1/(1+p x) term.
    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    sign = 1.0 if x >= 0 else -1.0
    ax = abs(x)
    if ax > 6.0:
        return sign
    t = 1.0 / (1.0 + p * ax)
    poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
    return sign * (1.0 - poly * math.exp(-ax * ax))


def _beta_sample(alpha: float, beta: float, rng: random.Random) -> float:
    """A Beta(alpha, beta) draw via the ratio of Gamma variates (Johnk's method
    fallback to a normal-based approximation handles extreme shape params).

    Pure stdlib: exponential uniforms are the ``math.log`` of a standard uniform.
    Returns a float in [0, 1] (clamped).
    """
    if alpha <= 0 or beta <= 0:
        return 0.5
    alpha = float(alpha)
    beta = float(beta)
    # Large shape parameters: use the normal approximation of the Gamma,
    # then ratio of gammas -> the beta is approximately normal about the mean.
    # We still draw via gammas below for the common case (affordable + exact).
    def _gamma(shape: float) -> float:
        if shape < 1.0:
            # Marsaglia-Tsang requires a>=1; boost with an extra Exp(1).
            extra = -math.log(1.0 - rng.random())
            return _gamma(shape + 1.0) * extra ** (1.0 / shape)
        # Marsaglia & Tsang 2000 for a >= 1.
        d = shape - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            z = rng.gauss(0.0, 1.0)
            v = (1.0 + c * z) ** 3
            if v <= 0.0:
                continue
            u = rng.random()
            if u < 1.0 - 0.0331 * z ** 4:
                return d * v
            if math.log(u) < 0.5 * z * z + d * (1.0 - v + math.log(v)):
                return d * v

    # Handle huge shapes that would overflow the gamma RMSD algorithm number of
    # iterations (these correspond to effectively-deterministic means anyway).
    if alpha >= 1e4 or beta >= 1e4:
        mu = alpha / (alpha + beta)
        var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
        return _clamp(rng.gauss(mu, math.sqrt(var)))

    ga = _gamma(alpha)
    gb = _gamma(beta)
    denom = ga + gb
    if denom <= 0.0:
        return 0.5
    return _clamp(ga / denom)


# ---------------------------------------------------------------------------
# #217 — Thompson sampling (Beta-Bernoulli)
# ---------------------------------------------------------------------------


class ThompsonSampler:
    """Eq 217 — Beta-Bernoulli Thompson sampling.

    Each arm carries a Beta(a, b) posterior. ``sample()`` picks the arm whose
    Beta draw is largest (probability matching), ``update()`` applies the
    Bernoulli outcome as a conjugate update. Fully deterministic-able with a
    seeded ``random.Random``.

    ("src", "Tài liệu (15)")
    """

    def __init__(self, n_arms: int, seed: Optional[int] = None,
                 prior_a: float = 1.0, prior_b: float = 1.0):
        self.n_arms = max(1, int(n_arms))
        self.prior_a = float(prior_a)
        self.prior_b = float(prior_b)
        self.rng = random.Random(seed)
        self.alpha: List[float] = [self.prior_a] * self.n_arms
        self.beta: List[float] = [self.prior_b] * self.n_arms
        self._counts: List[int] = [0] * self.n_arms
        self._updates = 0

    def sample(self) -> int:
        """Return the arm index with the maximum Beta draw (probability matching)."""
        best_arm = 0
        best_draw = -1.0
        for i in range(self.n_arms):
            d = _beta_sample(self.alpha[i], self.beta[i], self.rng)
            if d > best_draw:
                best_draw = d
                best_arm = i
        return best_arm

    def update(self, arm: int, reward: float) -> None:
        """Conjugate Beta update: reward in [0,1] treated as Bernoulli success.
        ``reward`` is clipped to {0,1} convention (>=0.5 -> success)."""
        arm = int(arm)
        if not (0 <= arm < self.n_arms):
            return
        success = 1.0 if _clamp(reward) >= 0.5 else 0.0
        self.alpha[arm] += success
        self.beta[arm] += 1.0 - success
        self._counts[arm] += 1
        self._updates += 1

    def expected_means(self) -> List[float]:
        """Prior-mean (a / (a+b)) per arm, in [0,1]."""
        return [
            _clamp(self.alpha[i] / (self.alpha[i] + self.beta[i]))
            for i in range(self.n_arms)
        ]

    def confidence(self, arm: int) -> float:
        """Posterior std-dev of the mean (a proxy for confidence; lower = more
        confident), normalized to [0,1] as 1/(1+std)."""
        arm = int(arm)
        if not (0 <= arm < self.n_arms):
            return 0.0
        a = self.alpha[arm]
        b = self.beta[arm]
        n = a + b
        if n <= 0:
            return 0.5
        var = (a * b) / (n * n * (n + 1.0))
        std = math.sqrt(var)
        # Higher confidence (lower std) -> closer to 1.
        return _clamp(1.0 / (1.0 + std))

    def reset(self) -> None:
        self.alpha = [self.prior_a] * self.n_arms
        self.beta = [self.prior_b] * self.n_arms
        self._counts = [0] * self.n_arms
        self._updates = 0

    def stats(self) -> dict:
        return {
            "n_arms": self.n_arms,
            "updates": self._updates,
            "counts": list(self._counts),
            "means": self.expected_means(),
        }


# ---------------------------------------------------------------------------
# #227 — UCB1 / upper-confidence-bound family
# ---------------------------------------------------------------------------


class UCBArm:
    """Eq 227 (UCB1) — per-arm bookkeeping: cumulative reward, pull count, and
    the UCB score = mean + sqrt(2*ln(total) / count). Unseen arms score +inf so
    they are explored first.

    ("src", "Tài liệu (16)")
    """

    def __init__(self):
        self.reward_sum: float = 0.0
        self.count: int = 0

    def update(self, reward: float) -> None:
        self.reward_sum += float(reward)
        self.count += 1

    def mean(self) -> float:
        if self.count <= 0:
            return 0.0
        return self.reward_sum / self.count

    def score(self, total: int, c: float = 2.0) -> float:
        """UCB = mean + sqrt(c * ln(total) / count); +inf when count == 0."""
        if self.count <= 0:
            return math.inf
        if total <= 1:
            exploration = 0.0
        else:
            exploration = math.sqrt(c * math.log(max(1, total)) / self.count)
        return self.mean() + exploration


class UCB1:
    """Eq 227 UCB1 — a vectorized wrapper over ``UCBArm`` with ``select``/``update``.

    ``select(n_arms, counts, rewards)`` is also provided as a stateless helper:
    it returns the index of an unseen arm first, otherwise the max-UCB arm.

    ("src", "Tài liệu (17)")
    """

    def __init__(self, n_arms: int, c: float = 2.0):
        self.n_arms = max(1, int(n_arms))
        self.c = float(c)
        self.arms = [UCBArm() for _ in range(self.n_arms)]

    def select(self) -> int:
        return UCB1.select(self.n_arms,
                           self.counts(),
                           self.cumulative_rewards())

    def update(self, arm: int, reward: float) -> None:
        arm = int(arm)
        if 0 <= arm < self.n_arms:
            self.arms[arm].update(reward)

    def counts(self) -> List[int]:
        return [a.count for a in self.arms]

    def cumulative_rewards(self) -> List[float]:
        return [a.reward_sum for a in self.arms]

    def means(self) -> List[float]:
        return [a.mean() for a in self.arms]

    def reset(self) -> None:
        self.arms = [UCBArm() for _ in range(self.n_arms)]

    def stats(self) -> dict:
        return {
            "n_arms": self.n_arms,
            "c": self.c,
            "counts": self.counts(),
            "means": self.means(),
        }

    @staticmethod
    def select(n_arms: int, counts: Sequence[int],
               rewards: Sequence[float], c: float = 2.0) -> int:
        """Stateless UCB1 arm selection. Unseen (count==0) arms are picked first
        (round-robin by index); otherwise the max-UCB arm wins. Returns -1 when
        n_arms < 1 (safe default)."""
        n_arms = int(n_arms)
        if n_arms < 1:
            return -1
        counts = list(counts) + [0] * (n_arms - len(counts))
        rewards = list(rewards) + [0.0] * (n_arms - len(rewards))
        total = max(1, sum(counts))

        unseen = [i for i in range(n_arms) if counts[i] <= 0]
        if unseen:
            return unseen[0]

        best_i = 0
        best_v = -math.inf
        for i in range(n_arms):
            mean = rewards[i] / counts[i]
            exploration = math.sqrt(c * math.log(total) / counts[i])
            v = mean + exploration
            if v > best_v:
                best_v = v
                best_i = i
        return best_i


def ucb(counts: Sequence[int], rewards: Sequence[float],
        total: Optional[int] = None, c: float = 2.0) -> List[float]:
    """Eq 227 UCB with exploration parameter ``c``:
    score_i = mean_i + sqrt(c * ln(total) / count_i). Unseen arms get +inf.

    ("src", "Tài liệu (16)")
    """
    counts = list(counts)
    rewards = list(rewards)
    n = max(len(counts), len(rewards))
    counts += [0] * (n - len(counts))
    rewards += [0.0] * (n - len(rewards))
    if total is None:
        total = sum(counts)
    total = max(1, int(total))
    out: List[float] = []
    for i in range(n):
        if counts[i] <= 0:
            out.append(math.inf)
            continue
        mean = rewards[i] / counts[i]
        exploration = math.sqrt(c * math.log(total) / counts[i])
        out.append(mean + exploration)
    return out


def ucb1_tuned(counts: Sequence[int], rewards: Sequence[float],
               variances: Optional[Sequence[float]] = None) -> List[float]:
    """Eq 228 — UCB1-Tuned: mean + sqrt( ln(total)/count * min(1/4, V_i) ),
    where V_i = sample_variance + sqrt(2*ln(total)/count). If ``variances`` is
    not provided, sample variances are computed from reward values (assumed to
    be per-pull observations). Unseen arms score +inf.

    ("src", "Tài liệu (17)")
    """
    counts = list(counts)
    rewards = list(rewards)
    n = max(len(counts), len(rewards))
    counts += [0] * (n - len(counts))
    rewards += [0.0] * (n - len(rewards))
    total = max(1, sum(counts))

    if variances is not None:
        variances = list(variances) + [0.0] * (n - len(variances))
    else:
        # No variance given: per-arm sample variance needs per-pull history,
        # which we don't have — fall back to Bernoulli-ish variance mean(1-mean).
        variances = []
        for i in range(n):
            if counts[i] <= 0:
                variances.append(0.0)
            else:
                m = rewards[i] / counts[i]
                variances.append(max(0.0, m * (1.0 - m)) if 0 <= m <= 1 else 0.25)

    out: List[float] = []
    for i in range(n):
        if counts[i] <= 0:
            out.append(math.inf)
            continue
        ln = math.log(total)
        v = variances[i] + math.sqrt(2.0 * ln / counts[i])
        bound = math.sqrt(ln / counts[i] * min(0.25, v))
        out.append(rewards[i] / counts[i] + bound)
    return out


# ---------------------------------------------------------------------------
# Epsilon-greedy + annealing
# ---------------------------------------------------------------------------


def epsilon_greedy_select(epsilon: float, counts: Sequence[int],
                          rewards: Sequence[float],
                          rng: Optional[random.Random] = None) -> int:
    """Eq 229 — epsilon-greedy: with probability epsilon pick a random arm;
    otherwise pick the arm with the highest empirical mean. Unseen arms are
    explored regardless (count==0 -> sampled); returns -1 when no arms.

    ("src", "Tài liệu (18)")
    """
    rng = rng or random
    counts = list(counts)
    rewards = list(rewards)
    n = max(len(counts), len(rewards))
    if n < 1:
        return -1
    counts += [0] * (n - len(counts))
    rewards += [0.0] * (n - len(rewards))
    epsilon = _clamp(epsilon)

    unseen = [i for i in range(n) if counts[i] <= 0]
    if unseen:
        return rng.choice(unseen)

    if rng.random() < epsilon:
        return rng.randrange(n)

    means = [rewards[i] / counts[i] for i in range(n)]
    best = max(range(n), key=lambda i: means[i])
    return best


def epsilon_anneal(step: int, eps_start: float = 1.0,
                   eps_end: float = 0.05, decay_steps: int = 1000) -> float:
    """Eq 230 — epsilon annealing: linear decay from ``eps_start`` to
    ``eps_end`` over ``decay_steps``, holding at ``eps_end`` past the horizon.

    ("src", "Tài liệu (18)")
    """
    step = max(0, int(step))
    decay_steps = max(1, int(decay_steps))
    eps_start = _clamp(eps_start)
    eps_end = _clamp(eps_end)
    frac = min(1.0, step / decay_steps)
    if frac >= 1.0:
        return eps_end
    return _clamp(eps_start + (eps_end - eps_start) * frac)


# ---------------------------------------------------------------------------
# EXP3 — adversarial bandit
# ---------------------------------------------------------------------------


class EXP3:
    """Eq 231 — EXP3 adversarial bandit (Auer et al., 2002).

    Weights update with an importance-weighted estimator of the reward:
    ``x_hat = reward / probability_of_arm``; ``w_i *= exp(eta * x_hat)``.
    ``eta = sqrt(ln(n) / (n * T))`` where ``T`` is the total number of pulls.
    ``select()`` returns an arm sampled from the softmax over ``exp(eta * cum)``.

    ("src", "Tài liệu (19)")
    """

    def __init__(self, n_arms: int, seed: Optional[int] = None, eta: Optional[float] = None):
        self.n_arms = max(1, int(n_arms))
        self.rng = random.Random(seed)
        self.eta = eta
        self.weights: List[float] = [1.0] * self.n_arms
        self._counts: List[int] = [0] * self.n_arms
        self._total = 0

    def _probabilities(self) -> List[float]:
        if self.eta is None:
            t = max(1, self._total + 1)
            self.eta = math.sqrt(math.log(self.n_arms) / (self.n_arms * t))
        scores = [self.eta * w for w in self.weights]
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        total = sum(exps)
        if total <= 0:
            return [1.0 / self.n_arms] * self.n_arms
        return [e / total for e in exps]

    def select(self) -> int:
        probs = self._probabilities()
        r = self.rng.random()
        acc = 0.0
        for i, p in enumerate(probs):
            acc += p
            if r <= acc:
                return i
        return self.n_arms - 1

    def update(self, arm: int, reward: float) -> None:
        arm = int(arm)
        if not (0 <= arm < self.n_arms):
            return
        reward = max(0.0, min(1.0, float(reward)))
        probs = self._probabilities()
        p = probs[arm] if probs[arm] > 0 else 1.0 / self.n_arms
        x_hat = reward / p   # importance-weighted estimator
        self.weights[arm] *= math.exp(self.eta * x_hat) if self.eta else 1.0
        # Scale down to avoid overflow on grossly large counts.
        mx = max(self.weights)
        if mx > 1e300:
            self.weights = [w / mx for w in self.weights]
        self._counts[arm] += 1
        self._total += 1

    def reset(self) -> None:
        self.weights = [1.0] * self.n_arms
        self._counts = [0] * self.n_arms
        self._total = 0
        self.eta = None

    def stats(self) -> dict:
        return {
            "n_arms": self.n_arms,
            "total": self._total,
            "counts": list(self._counts),
            "weights": [w for w in self.weights],
        }


# ---------------------------------------------------------------------------
# Bayesian optimization acquisition functions (EI / UCB / PI)
# ---------------------------------------------------------------------------


def _as_seq(x) -> List[float]:
    if isinstance(x, (int, float)):
        return [float(x)]
    return [float(v) for v in x]


def acquisition_EI(mu, sigma, best, c: float = 0.0):
    """Eq 233 — Expected Improvement acquisition.

    EI(x) = (mu - best - c)*Phi(z) + sigma*phi(z),  z = (mu - best - c)/sigma.
    ``c`` is an exploration trade-off parameter. Returns a scalar if ``mu`` is a
    scalar, else a list (parallel to ``mu``). sigma==0 -> EI = max(0, mu - best - c).
    Never negative (clamped to 0).

    ("src", "a-tier acquisition-EI.txt")
    """
    scalar = isinstance(mu, (int, float)) and isinstance(sigma, (int, float))
    mu_s = _as_seq(mu)
    sig_s = _as_seq(sigma)
    n = max(len(mu_s), len(sig_s))
    out = []
    for i in range(n):
        m = mu_s[i] if i < len(mu_s) else mu_s[-1]
        s = sig_s[i] if i < len(sig_s) else sig_s[-1]
        imp = m - float(best) - float(c)
        if s <= 0.0:
            out.append(max(0.0, imp))
            continue
        z = imp / s
        ei = imp * _normal_cdf(z) + s * _normal_pdf(z)
        out.append(max(0.0, ei))
    return out[0] if scalar else out


def acquisition_UCB(mu, sigma, kappa: float = 2.0):
    """Eq 234 — GP-UCB acquisition: mu + kappa * sigma.

    ("src", "a-tier acquisition-UCB.txt")
    """
    scalar = isinstance(mu, (int, float)) and isinstance(sigma, (int, float))
    mu_s = _as_seq(mu)
    sig_s = _as_seq(sigma)
    n = max(len(mu_s), len(sig_s))
    out = []
    for i in range(n):
        m = mu_s[i] if i < len(mu_s) else mu_s[-1]
        s = sig_s[i] if i < len(sig_s) else sig_s[-1]
        out.append(m + float(kappa) * s)
    return out[0] if scalar else out


def acquisition_PI(mu, sigma, best):
    """Eq 235 — Probability of Improvement acquisition: Phi((mu - best)/sigma).

    Returns a probability in [0,1]. sigma==0 -> 1.0 if mu >= best else 0.0.

    ("src", "s-tier acquisition-PI.txt")
    """
    scalar = isinstance(mu, (int, float)) and isinstance(sigma, (int, float))
    mu_s = _as_seq(mu)
    sig_s = _as_seq(sigma)
    n = max(len(mu_s), len(sig_s))
    out = []
    for i in range(n):
        m = mu_s[i] if i < len(mu_s) else mu_s[-1]
        s = sig_s[i] if i < len(sig_s) else sig_s[-1]
        if s <= 0.0:
            out.append(1.0 if m >= float(best) else 0.0)
        else:
            out.append(_normal_cdf((m - float(best)) / s))
    return out[0] if scalar else out


# ---------------------------------------------------------------------------
# Simple GP posterior moments (squared-exponential kernel)
# ---------------------------------------------------------------------------


def _mat_inv(m: List[List[float]]) -> List[List[float]]:
    """Gaussian-elimination (row-reduction) matrix inverse. Pure stdlib.
    Returns the identity of the same shape on singular input (guarded with
    jitter by the caller)."""
    n = len(m)
    if n == 0:
        return []
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i, row in enumerate(m)]
    for col in range(n):
        # Partial pivot
        pivot = col
        best = abs(aug[col][col]) if col < n and aug[col][col] else 0.0
        for r in range(col + 1, n):
            v = abs(aug[r][col]) if aug[r][col] else 0.0
            if v > best:
                best = v
                pivot = r
        if best < 1e-12:
            # Singular (after jitter this is rare); return identity to be safe.
            return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= piv
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0.0:
                continue
            for j in range(2 * n):
                aug[r][j] -= factor * aug[col][j]
    return [row[n:] for row in aug]


class SimpleGP:
    """Eq 236..237 — exact GP regression with a squared-exponential kernel.

    ``fit(x, y, length_scale, noise)`` caches ``(K + sig_n^2 I)^{-1}`` once.
    ``predict_mu_sigma(xs)`` returns the posterior mean mu and std arrays at
    query points using kernel-matrix inversion (Gaussian elimination). Jitter
    (1e-9) is added to the diagonal to guard against singular kernels.

    This is the small-data exact GP — suitable for acquisition / exploration
    on modest observation counts, not large-scale hyperparameter learning.

    ("src", "s-tier GP-posterior.txt")
    """

    def __init__(self):
        self.x: List[float] = []
        self.y: List[float] = []
        self.length_scale: float = 1.0
        self.noise: float = 1e-6
        self._K_inv: Optional[List[List[float]]] = None

    def _kernel(self, a: float, b: float) -> float:
        d = a - b
        return math.exp(-0.5 * (d * d) / (self.length_scale * self.length_scale))

    def fit(self, x: Sequence[float], y: Sequence[float],
            length_scale: float = 1.0, noise: float = 1e-6) -> None:
        self.x = [float(v) for v in x]
        self.y = [float(v) for v in y]
        self.length_scale = float(length_scale) if length_scale else 1.0
        self.noise = float(noise) if noise else 1e-6
        n = len(self.x)
        if n == 0:
            self._K_inv = None
            return
        K = [[self._kernel(self.x[i], self.x[j]) for j in range(n)]
             for i in range(n)]
        # Jitter + observation noise on the diagonal.
        jitter = 1e-9
        for i in range(n):
            K[i][i] += self.noise + jitter
        self._K_inv = _mat_inv(K)

    def predict_mu_sigma(self, xs: Sequence[float]) -> Tuple[List[float], List[float]]:
        """Return (posterior_mean, posterior_std) at query points ``xs``."""
        n = len(self.x)
        if n == 0 or self._K_inv is None:
            xs_l = [float(v) for v in xs]
            return ([0.0] * len(xs_l), [1.0] * len(xs_l))

        K_inv = self._K_inv
        mu_out: List[float] = []
        std_out: List[float] = []
        for q in [float(v) for v in xs]:
            k = [self._kernel(q, self.x[i]) for i in range(n)]
            alpha = [sum(k[i] * K_inv[i][j] for i in range(n)) for j in range(n)]
            mu = sum(alpha[j] * self.y[j] for j in range(n))
            k_star = self._kernel(q, q) if q in self.x else 1.0
            var = max(0.0, k_star - sum(alpha[j] * k[j] for j in range(n)))
            mu_out.append(mu)
            std_out.append(math.sqrt(var))
        return mu_out, std_out

    def reset(self) -> None:
        self.x = []
        self.y = []
        self._K_inv = None

    def stats(self) -> dict:
        return {
            "n_points": len(self.x),
            "length_scale": self.length_scale,
            "noise": self.noise,
        }


# ---------------------------------------------------------------------------
# Explore / exploit fraction
# ---------------------------------------------------------------------------


def explore_fraction(step: int, decay: float = 100.0) -> float:
    """Eq 238 — explore/exploit fraction: exp(-step / decay), an exponential
    decay of the exploration budget over time. Returns a value in (0, 1].

    ("src", "Tài liệu (20)")
    """
    step = max(0, int(step))
    decay = float(decay) if decay and decay > 0 else 100.0
    return _clamp(math.exp(-step / decay))


# ---------------------------------------------------------------------------
# Evaluate bandit policy — regret
# ---------------------------------------------------------------------------


def regret(cumulative_best_reward: float, cumulative_actual_reward: float) -> float:
    """Eq 239 — positive regret: the shortfall of the actual cumulative reward
    versus the best arm's cumulative reward. max(0, best - actual).

    ("src", "Tài liệu (21)")
    """
    best = float(cumulative_best_reward)
    actual = float(cumulative_actual_reward)
    return max(0.0, best - actual)


# ---------------------------------------------------------------------------
# NOT-IMPLEMENTABLE constants (training / hardware / model-weight dependent)
# ---------------------------------------------------------------------------


# NOT_IMPLEMENTABLE: gradient-bandit with a neural-network value approximator —
# the value head must be trained with backprop on GPUs and holds tunable weights.
NOT_IMPLEMENTABLE_GRADIENT_BANDIT_NN = (
    "gradient-bandit:neural-value-head — requires NN training loop, gradient "
    "descent (torch/tf), and trained weights; infeasible in pure stdlib."
)

# NOT_IMPLEMENTABLE: full Gaussian-process regression over large datasets with
# learned hyperparameters — this needs iterative nonlinear optimization of the
# marginal-likelihood (scipy.optimize) and O(n^3) solve on big matrices.
NOT_IMPLEMENTABLE_GP_LEARNED_HYPERPARAMS = (
    "gp-learned-hyperparams:large-data — requires scipy.optimize for "
    "marginal-likelihood training and memory-heavy kernel matrices on big data; "
    "SimpleGP above is the small-data exact surrogate only."
)

# NOT_IMPLEMENTABLE: contextual bandit with a deep neural network policy trained
# by the hybrid proximal-policy objective — needs an RL training loop + GPU.
NOT_IMPLEMENTABLE_CONTEXTUAL_DEEP_BANDIT = (
    "contextual-deep-bandit — requires an RL/PPO training loop over an NN policy "
    "and GPU; the pure stdlib EXP3 / Thompson sampler here are context-free."
)

# NOT_IMPLEMENTABLE: Thompson sampling with sparse, high-dimensional deep features
# (e.g. Bayesian neural-network posteriors) — needs variational/training machinery.
NOT_IMPLEMENTABLE_THOMPSON_BAYESIAN_NN = (
    "thompson-bayesian-nn — requires variational inference over NN weight "
    "posteriors (training); not feasible in pure stdlib."
)


__all__ = [
    # primitives
    "_clamp", "_normal_cdf", "_normal_pdf", "_erf", "_beta_sample",
    # UCB / UCB1
    "UCBArm", "UCB1", "ucb", "ucb1_tuned",
    # explore/exploit
    "epsilon_greedy_select", "epsilon_anneal",
    # EXP3
    "EXP3",
    # Thompson
    "ThompsonSampler",
    # acquisition
    "acquisition_EI", "acquisition_UCB", "acquisition_PI",
    # GP
    "_mat_inv", "SimpleGP",
    # explore fraction / regret
    "explore_fraction", "regret",
    # NOT_IMPLEMENTABLE
    "NOT_IMPLEMENTABLE_GRADIENT_BANDIT_NN",
    "NOT_IMPLEMENTABLE_GP_LEARNED_HYPERPARAMS",
    "NOT_IMPLEMENTABLE_CONTEXTUAL_DEEP_BANDIT",
    "NOT_IMPLEMENTABLE_THOMPSON_BAYESIAN_NN",
]
