"""
PageRank: local push (EQ-PPR-PUSH), stationary iteration (EQ-PAGERANK-STATIONARY),
prune (EQ-PAGERANK-PRUNE).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# EQ-PAGERANK-PRUNE — prune_score = rel_score * (PR^beta)
# ---------------------------------------------------------------------------
def prune_with_pagerank(
    candidates: List[Tuple[str, float]],
    pagerank: Dict[str, float],
    beta: float = 0.5,
    keep_k: int = 100,
    score_floor: float = 0.01,
) -> List[str]:
    """Prune candidate set using PageRank-boosted scores (§21st equation)."""
    if not candidates:
        return []
    pruned_scores = []
    for doc_id, rel_score in candidates:
        pr = pagerank.get(doc_id, 0.15)
        pruned_scores.append((doc_id, rel_score * (pr**beta)))
    pruned_scores.sort(key=lambda x: x[1], reverse=True)
    max_score = pruned_scores[0][1]
    above = [(d, s) for d, s in pruned_scores if s >= score_floor * max_score]
    return [d for d, _ in above[:keep_k]]


# ---------------------------------------------------------------------------
# EQ-PPR-PUSH — Personalized PageRank via local push (part_1.pdf)
# p[u] += alpha*r_u; share (1-alpha)*r_u/deg to neighbors
# ---------------------------------------------------------------------------
class PersonalizedPageRank:
    def __init__(
        self,
        edges: List[Tuple[int, int]],
        num_nodes: int,
        alpha: float = 0.85,
        epsilon: float = 1e-4,
        personalization: Dict[int, float] | None = None,
    ):
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.eps = epsilon
        self.out_edges: Dict[int, List[int]] = defaultdict(list)
        for u, v in edges:
            self.out_edges[u].append(v)
        if personalization:
            total = sum(personalization.values())
            self.e = np.zeros(num_nodes)
            for node, w in personalization.items():
                if 0 <= node < num_nodes:
                    self.e[node] = w / total
        else:
            self.e = np.ones(num_nodes) / num_nodes
        self.p = np.zeros(num_nodes)
        self.r = np.zeros(num_nodes)
        self._initialize_residuals()
        self._push_loop()

    def _initialize_residuals(self) -> None:
        self.r = self.e.copy()

    def _push_loop(self) -> None:
        for u in range(self.num_nodes):
            if self.r[u] > self.eps:
                self._push(u)

    def _push(self, u: int) -> None:
        r_u = self.r[u]
        if r_u <= self.eps:
            return
        deg = len(self.out_edges[u])
        self.p[u] += self.alpha * r_u
        if deg == 0:
            share = (1 - self.alpha) * r_u / self.num_nodes
            self.r += share
        else:
            share = (1 - self.alpha) * r_u / deg
            for v in self.out_edges[u]:
                self.r[v] += share
        self.r[u] = 0

    def get_scores(self) -> np.ndarray:
        return self.p.copy()


def stationary_pagerank(
    edges: List[Tuple[int, int]],
    num_nodes: int,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> np.ndarray:
    """EQ-PAGERANK-STATIONARY: pi = (1-alpha)/N + alpha * M * pi."""
    M = np.zeros((num_nodes, num_nodes))
    out_deg = np.zeros(num_nodes)
    for u, v in edges:
        M[v, u] += 1
        out_deg[u] += 1
    for u in range(num_nodes):
        if out_deg[u] > 0:
            M[:, u] /= out_deg[u]
        else:
            M[:, u] = 1.0 / num_nodes
    pi = np.ones(num_nodes) / num_nodes
    teleport = (1 - alpha) / num_nodes
    for _ in range(max_iter):
        new_pi = teleport + alpha * M @ pi
        if np.linalg.norm(new_pi - pi, 1) < tol:
            break
        pi = new_pi
    return pi
