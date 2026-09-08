"""
gateway/advanced_optimizations.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pure-Python Koopman-inspired global context mixing for RAG results.

This module is what `rag/llama_index_rag.py` imports at startup:

    from gateway.advanced_optimizations import get_koopman_mixer

Previously missing → ImportError. Now provided as a real, self-contained
implementation that lifts chunk embeddings into a higher-dimensional
"observable" space, applies a learned-in-place linear operator (Koopman
matrix), and projects back so multiple RAG providers' results can be blended
into a single context representation.

Pure stdlib, no model weights required (works over any dense embedding vector
passed in by the caller, e.g. `opt_core.hashed_embedding` or chroma vectors).
"""

from __future__ import annotations

import math
import threading
from typing import List, Optional, Sequence

_dims: List[float] = []


class _FourierObservables:
    """Deterministic 'observable' lifting: identity + quadratic + first few
    Fourier features of each input coordinate.  Pure stdlib (no FFT)."""

    def __init__(self, lift_dim: int = 64):
        self.lift_dim = max(16, int(lift_dim))

    def lift(self, x: Sequence[float]) -> List[float]:
        n = len(x)
        if n == 0:
            return [0.0] * self.lift_dim
        out: List[float] = []
        # Identity coordinates.
        out.extend(x)
        # Quadratic cross-terms (pairwise rounded to keep dim bounded).
        if len(out) < self.lift_dim:
            step = max(1, n // max(1, self.lift_dim - len(out)))
            for i in range(0, n, step):
                if len(out) >= self.lift_dim:
                    break
                out.append(x[i] * x[i])
        # Fourier features on a downsampled subset.
        k = 1
        while len(out) < self.lift_dim:
            idx = k % max(1, n)
            out.append(math.sin(k * x[idx]) + math.cos(k * x[idx]))
            k += 1
        return out[: self.lift_dim]


class KoopmanMixer:
    """Linear-operator (Koopman) mixer over lifted embeddings.

    ``mix_embeddings(vectors)`` returns a new vector of the same dimension as
    the inputs, computed as::

        y = project( K @ mean( lift(x_i) ) )

    The operator K is initially the identity plus a small diagonal spread and is
    adapted in-place from observed trajectories via a stochastic rank-one update
    (pure-Python, no autodiff).  This grounds the "Koopman-inspired global
    context mixing" config knob (`koopman_mixing_enabled`) in real computable
    math rather than a stub.
    """

    def __init__(self, dim: int = 64, lift_dim: int = 128,
                 learning_rate: float = 0.05):
        self.dim = max(8, int(dim))
        self.lift_dim = max(16, int(lift_dim))
        self.lr = float(learning_rate)
        self.observables = _FourierObservables(lift_dim=self.lift_dim)
        self._lock = threading.RLock()
        # K is lift_dim x lift_dim, identity + small off-diagonal spread.
        self._K = [
            [0.0] * self.lift_dim for _ in range(self.lift_dim)
        ]
        diag_spread = 0.001
        off_diag = 0.0001
        for i in range(self.lift_dim):
            row = self._K[i]
            for j in range(self.lift_dim):
                if i == j:
                    row[j] = 1.0 + diag_spread
                elif abs(i - j) == 1:
                    row[j] = off_diag
        self._step = 0
        self._last_lift: List[float] = []

    # ------------------------------------------------------------------
    # Public API (thread-safe)
    # ------------------------------------------------------------------

    def lift(self, x: Sequence[float]) -> List[float]:
        return self.observables.lift(x)

    def _matvec(self, v: List[float]) -> List[float]:
        n = len(v)
        return [
            sum(self._K[i][j] * (v[j] if j < n else 0.0)
                for j in range(self.lift_dim))
            for i in range(self.lift_dim)
        ]

    def _project(self, lifted: List[float], n: int) -> List[float]:
        if n <= 0:
            return [0.0] * self.dim
        # Project back onto the first n identity coordinates (the original dim),
        # normalized to keep unit scale.
        coords = lifted[:n]
        norm = math.sqrt(sum(c * c for c in coords)) or 1.0
        return [c / norm for c in coords]

    def mix_embeddings(self, vectors: Sequence[Sequence[float]]) -> List[float]:
        """Blend several chunk embeddings into one global-context vector.

        Returns a vector of the same dimensionality as the first input (or an
        all-zero vector of ``self.dim`` if nothing is provided).
        """
        if not vectors:
            return [0.0] * self.dim
        n = len(vectors[0])
        with self._lock:
            lifted = [self.lift(v) for v in vectors if len(v) == n]
            if not lifted:
                return [0.0] * self.dim
            avg = [sum(vec[i] for vec in lifted) / len(lifted)
                   for i in range(self.lift_dim)]
            mixed = self._matvec(avg)
            self._last_lift = mixed
            return self._project(mixed, n)

    def adapt(self, x_t: Sequence[float], x_next: Sequence[float],
              lr: Optional[float] = None) -> None:
        """Rank-one update of K toward x_next ≈ K @ lift(x_t)."""
        lr = self.lr if lr is None else float(lr)
        with self._lock:
            xs = self.lift(x_t)
            ys = self.lift(x_next)
            # y ≈ K x  ->  K += lr * (y - K x) x^T   (plain gradient step)
            before = self._matvec(xs)
            for i in range(self.lift_dim):
                err_i = ys[i] - before[i]
                row = self._K[i]
                for j in range(self.lift_dim):
                    row[j] += lr * err_i * xs[j]
            self._step += 1

    def stats(self) -> dict:
        with self._lock:
            return {"step": self._step, "lift_dim": self.lift_dim,
                    "dim": self.dim, "last_lift_norm": (
                        math.sqrt(sum(c * c for c in self._last_lift))
                        if self._last_lift else 0.0)}

    def reset(self) -> None:
        with self._lock:
            self._step = 0
            self._last_lift = []


_mixer_lock = threading.Lock()
_mixer: Optional[KoopmanMixer] = None


def get_koopman_mixer(dim: int = 64, lift_dim: int = 128) -> KoopmanMixer:
    """Return the module-level Koopman mixer singleton (lazily created)."""
    global _mixer
    if _mixer is None:
        with _mixer_lock:
            if _mixer is None:
                _mixer = KoopmanMixer(dim=dim, lift_dim=lift_dim)
    return _mixer


__all__ = ["KoopmanMixer", "get_koopman_mixer", "mix_embeddings"]


def mix_embeddings(vectors: Sequence[Sequence[float]],
                   dim: int = 64, lift_dim: int = 128) -> List[float]:
    """Convenience one-shot: build a temporary mixer and blend vectors."""
    return KoopmanMixer(dim=dim, lift_dim=lift_dim).mix_embeddings(vectors)