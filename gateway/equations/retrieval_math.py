"""
optimization module.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Retrieval / RAG optimization family from the user's optimization catalog.

Covers MMR variants, score normalization for fusion, reciprocal rank fusion,
MI-gated retrieval, curiosity/novelty scoring, diversity-aware ranking,
budgeted token selection, and relevance gating.

Pure stdlib. With no live embedding model available on a query, callers may
use `hashed_embedding` / `cosine_similarity` from `gateway.opt_core`.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Shared helpers (pure stdlib, same style as cache_math.py)
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(x):
        return hi if x > 0 else lo
    return max(lo, min(hi, x))


def _cosine_sim(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; 0.0 on empty/mismatched (never NaN)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if denom == 0.0:
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)) / denom)


# ---------------------------------------------------------------------------
# Eq R1 -- Maximal Marginal Relevance (greedy indices)
# ---------------------------------------------------------------------------


def mmr(
    sim_doc: Sequence[float],
    sim_docs: Sequence[Sequence[float]],
    lambda_relevance: float = 0.7,
    top_k: int = 5,
) -> List[int]:
    """Eq R1 -- Maximal Marginal Relevance returning ranked indices.

    *sim_doc* is the query-to-document similarity vector (one score per
    candidate). *sim_docs* is the pairwise document-document similarity
    matrix (square, indexed by candidate).  At each step the candidate that
    maximises ``lambda * sim_doc[i] - (1-lambda) * max_j sim_docs[i][j]``
    over all not-yet-selected documents *j* is chosen.

    Returns a list of candidate indices in selection order.  Empty input
    returns ``[]``.
    """
    n = len(sim_doc)
    if n == 0:
        return []
    if len(sim_docs) != n:
        return list(range(min(n, top_k)))
    top_k = max(1, min(top_k, n))
    lam = _clamp(lambda_relevance, 0.0, 1.0)
    selected: List[int] = []
    candidates = set(range(n))

    for _ in range(top_k):
        best_idx = -1
        best_score = -math.inf
        for i in candidates:
            if not selected:
                penalty = 0.0
            else:
                penalty = max(sim_docs[i][j] for j in selected)
            score = lam * sim_doc[i] - (1.0 - lam) * penalty
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        candidates.discard(best_idx)
    return selected


# ---------------------------------------------------------------------------
# Eq R1b -- Incremental MMR score (single candidate)
# ---------------------------------------------------------------------------


def mmr_score(
    pairwise_sim: Sequence[float],
    relevance_scores: Sequence[float],
    lambda_relevance: float,
    selected: Sequence[int],
    candidate: int,
) -> float:
    """Eq R1b -- Incremental MMR score for a single candidate.

    ``pairwise_sim`` is the candidate's similarity row against all docs.
    ``relevance_scores`` is the query-doc similarity vector.
    Returns the scalar MMR score.  If nothing is selected the penalty is 0.
    """
    lam = _clamp(lambda_relevance, 0.0, 1.0)
    if not selected or candidate >= len(pairwise_sim):
        return lam * (relevance_scores[candidate] if candidate < len(relevance_scores) else 0.0)
    penalty = max(pairwise_sim[j] for j in selected if j < len(pairwise_sim))
    return lam * relevance_scores[candidate] - (1.0 - lam) * penalty


# ---------------------------------------------------------------------------
# Eq R2 -- Score normalization
# ---------------------------------------------------------------------------


def min_max_normalize(scores: Sequence[float]) -> List[float]:
    """Eq R2a -- Min-max normalization to [0, 1].  Empty -> [].
    All-equal -> uniform 0.5."""
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    if hi == lo:
        return [0.5] * len(scores)
    rng = hi - lo
    return [(s - lo) / rng for s in scores]


def z_score_normalize(scores: Sequence[float]) -> List[float]:
    """Eq R2b -- Z-score normalization.  Empty -> [].
    All-equal -> uniform 0.0.  Result clamped to [-3, 3] to avoid extreme tails."""
    n = len(scores)
    if n == 0:
        return []
    mu = sum(scores) / n
    var = sum((s - mu) ** 2 for s in scores) / n
    sigma = math.sqrt(var)
    if sigma == 0.0:
        return [0.0] * n
    return [_clamp((s - mu) / sigma, -3.0, 3.0) for s in scores]


def sigmoid_normalize(score: float, scale: float = 1.0) -> float:
    """Eq R2c -- Sigmoid normalization.  Maps any real to (0, 1).
    ``scale`` controls steepness (must be > 0)."""
    scale = scale if scale and scale > 0 else 1.0
    z = -scale * score
    try:
        return 1.0 / (1.0 + math.exp(z))
    except OverflowError:
        return 1.0 if z < 0 else 0.0


def normalize_scores(scores: Sequence[float], method: str = "minmax") -> List[float]:
    """Eq R2 -- Generic dispatcher for score normalization.

    *method*: "minmax" | "zscore" | "sigmoid".  Empty -> [].
    """
    if not scores:
        return []
    m = (method or "minmax").lower()
    if m == "zscore":
        return z_score_normalize(scores)
    if m == "sigmoid":
        mid = sum(scores) / len(scores) if scores else 0.0
        return [sigmoid_normalize(s - mid) for s in scores]
    return min_max_normalize(scores)


def cosine_to_similarity(raw_cosine: float) -> float:
    """Eq R2d -- SAFELY remap cosine in [-1, 1] to [0, 1] via ``(x+1)/2``.

    This is the assumption behind the live ``_eq8_gate`` in
    ``rag/llama_index_rag.py`` (line ~237).  Guards NaN/Inf.
    """
    if not math.isfinite(raw_cosine):
        return 0.5
    return (raw_cosine + 1.0) / 2.0


# ---------------------------------------------------------------------------
# Eq R3 -- Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def rrf_score(rank: int, k: int = 60) -> float:
    """Eq R3 -- Reciprocal rank score: ``1 / (k + rank)``.
    ``rank`` is 1-indexed."""
    k = max(1, k)
    if rank < 1:
        rank = 1
    return 1.0 / (k + rank)


def rrf(ranked_lists: Sequence[Sequence[int]], k: int = 60) -> List[Tuple[int, float]]:
    """Eq R3 -- Reciprocal Rank Fusion.

    Fuses multiple ranked lists (each a list of doc IDs in rank order)
    into a single ``(doc_id, fused_score)`` list sorted descending.

    Empty input returns ``[]``.
    """
    k = max(1, k)
    totals: Dict[int, float] = {}
    for rlist in ranked_lists:
        for rank_pos, doc_id in enumerate(rlist, start=1):
            totals[doc_id] = totals.get(doc_id, 0.0) + rrf_score(rank_pos, k)
    result = sorted(totals.items(), key=lambda t: (-t[1], t[0]))
    return [(doc_id, sc) for doc_id, sc in result]


# ---------------------------------------------------------------------------
# Eq R4 -- Weighted Reciprocal Rank
# ---------------------------------------------------------------------------


def weighted_reciprocal_rank(
    ranked_lists: Sequence[Sequence[int]],
    weights: Sequence[float],
) -> List[Tuple[int, float]]:
    """Eq R4 -- Weighted Reciprocal Rank Fusion.

    Each ranked list gets a weight (default 1.0 if fewer weights than lists).
    Returns ``(doc_id, weighted_score)`` sorted descending.
    """
    totals: Dict[int, float] = {}
    for i, rlist in enumerate(ranked_lists):
        w = weights[i] if i < len(weights) else 1.0
        w = max(0.0, w)
        for rank_pos, doc_id in enumerate(rlist, start=1):
            totals[doc_id] = totals.get(doc_id, 0.0) + w * rrf_score(rank_pos)
    result = sorted(totals.items(), key=lambda t: (-t[1], t[0]))
    return [(doc_id, sc) for doc_id, sc in result]


# ---------------------------------------------------------------------------
# Eq R5 -- Mutual-information-gated retrieval
# ---------------------------------------------------------------------------


def pointwise_mutual_information(p_xy: float, p_x: float, p_y: float) -> float:
    """Eq R5a -- Pointwise mutual information: log2(p_xy / (p_x * p_y)).

    Probabilities must be in (0, 1].  Returns 0.0 on invalid input.
    """
    if p_x <= 0.0 or p_y <= 0.0 or p_xy <= 0.0:
        return 0.0
    denom = p_x * p_y
    if denom <= 0.0:
        return 0.0
    ratio = p_xy / denom
    return math.log2(ratio)


def mi_gate(
    signal_entropy: float,
    noise_entropy: float,
    threshold: float = 0.5,
) -> bool:
    """Eq R5b -- MI-like signal-to-noise gate.

    ``signal_entropy`` is H(retrieved|query) (lower = more signal).
    ``noise_entropy`` is H(irrelevant|query) (higher = more noise).
    Returns True when the ratio passes the threshold, meaning signal
    dominates noise.

    Ratio = max(0, (threshold_alpha - signal) / max(noise, 1e-12))
    where we define alpha = 1.0 (information capacity).
    """
    _ = threshold  # used implicitly via normalization below
    noise = max(noise_entropy, 1e-12)
    ratio = max(0.0, (1.0 - signal_entropy) / noise)
    return ratio >= max(0.0, threshold)


# ---------------------------------------------------------------------------
# Eq R6 -- Curiosity / novelty score
# ---------------------------------------------------------------------------


def novelty_score(
    new_vec: Sequence[float],
    known_vecs: Sequence[Sequence[float]],
) -> float:
    """Eq R6a -- Novelty as max cosine distance to known vectors.

    ``1 - max_similarity``.  Returns 1.0 when no known vectors exist.
    """
    if not known_vecs or not new_vec:
        return 1.0
    max_sim = max(_cosine_sim(new_vec, kv) for kv in known_vecs)
    return 1.0 - _clamp(max_sim)


def curiosity_score(
    novelty: float,
    uncertainty: float,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """Eq R6b -- Curiosity = alpha * novelty + beta * uncertainty.

    Both inputs and the output are in [0, 1].
    """
    alpha = max(0.0, alpha)
    beta = max(0.0, beta)
    total = alpha + beta
    if total == 0.0:
        return 0.0
    return _clamp((alpha * _clamp(novelty) + beta * _clamp(uncertainty)) / total)


# ---------------------------------------------------------------------------
# Eq R7 -- Greedy diverse selection (diversity-aware ranking)
# ---------------------------------------------------------------------------


def greedy_diverse_select(
    scores: Sequence[float],
    sim_matrix: Sequence[Sequence[float]],
    top_k: int = 5,
    lambda_d: float = 0.7,
) -> List[int]:
    """Eq R7 -- Greedy diverse selection like MMR but as a pure function.

    At each step pick the candidate maximising
    ``lambda_d * score[i] - (1 - lambda_d) * max_r sim_matrix[i][r]``.

    Returns ordered list of selected indices.
    """
    n = len(scores)
    if n == 0:
        return []
    top_k = max(1, min(top_k, n))
    if len(sim_matrix) != n:
        return sorted(range(n), key=lambda i: -scores[i])[:top_k]
    lam = _clamp(lambda_d, 0.0, 1.0)
    selected: List[int] = []
    remaining = set(range(n))
    for _ in range(top_k):
        best_idx = -1
        best_val = -math.inf
        for i in remaining:
            penalty = max((sim_matrix[i][j] for j in selected), default=0.0)
            val = lam * scores[i] - (1.0 - lam) * penalty
            if val > best_val:
                best_val = val
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)
    return selected


# ---------------------------------------------------------------------------
# Eq R8 -- Cosine-distance embedding cache key
# ---------------------------------------------------------------------------


def chunk_cache_key(text: str) -> str:
    """Eq R8a -- Stable hash for a chunk of text.

    SHA-256 of the normalised (lowercased, whitespace-collapsed) text.
    """
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def embedding_similarity_from_embeddings(
    a: Sequence[float],
    b: Sequence[float],
) -> float:
    """Eq R8b -- Cheap cosine similarity between two pre-computed embeddings.
    Returns 0.0 on empty/mismatched (never NaN).
    """
    return _cosine_sim(a, b)


# ---------------------------------------------------------------------------
# Eq R9 -- Relevance hard gate
# ---------------------------------------------------------------------------


def relevance_gate(
    score: float,
    threshold: float = 0.5,
    strict: bool = False,
) -> bool:
    """Eq R9 -- Relevance hard-gate for chunk include/exclude.

    Returns ``True`` when ``score >= threshold`` (or ``>`` when strict).
    Guards NaN.
    """
    if not math.isfinite(score):
        return False
    if strict:
        return score > threshold
    return score >= threshold


# ---------------------------------------------------------------------------
# Eq R10 -- Query-reformulation score
# ---------------------------------------------------------------------------


def qf_similarity(original: str, reformulated: str) -> float:
    """Eq R10a -- Cosine/lexical hybrid similarity between original and
    reformulated query.  Uses feature-hashed character n-gram vectors for a
    cheap cosine component plus Jaccard lexical overlap, blended 50/50.
    Returns a score in [0, 1].
    """
    if not original or not reformulated:
        return 0.0
    dim = 128
    vec_a = _feature_hash(original, dim)
    vec_b = _feature_hash(reformulated, dim)
    cos_component = (_cosine_sim(vec_a, vec_b) + 1.0) / 2.0
    lex_component = _jaccard(original, reformulated)
    return _clamp(0.5 * cos_component + 0.5 * lex_component)


def _feature_hash(text: str, dim: int = 128) -> List[float]:
    """Small feature-hash helper for qf_similarity."""
    vec = [0.0] * dim
    normalized = re.sub(r"[^a-z0-9]", "", (text or "").lower())
    for n in (3, 4):
        if len(normalized) >= n:
            for i in range(len(normalized) - n + 1):
                gram = normalized[i : i + n]
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                idx = h % dim
                sign = 1.0 if (h >> 8) & 1 else -1.0
                vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and",
    "in", "for", "on", "with", "at", "by", "from", "or", "as",
})


def _jaccard(a: str, b: str) -> float:
    wa = {t for t in re.findall(r"[a-z0-9_]+", (a or "").lower()) if t not in _STOP_WORDS}
    wb = {t for t in re.findall(r"[a-z0-9_]+", (b or "").lower()) if t not in _STOP_WORDS}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def hybrid_relevance(lexical: float, semantic: float, alpha: float = 0.5) -> float:
    """Eq R10b -- Blended relevance: alpha*semantic + (1-alpha)*lexical.
    Output in [0, 1]."""
    a = _clamp(alpha)
    return _clamp(a * _clamp(semantic) + (1.0 - a) * _clamp(lexical))


# ---------------------------------------------------------------------------
# Eq R11 -- Top-k with stable tie-break
# ---------------------------------------------------------------------------


def topk_stable(scores: Sequence[float], k: int = 5) -> List[int]:
    """Eq R11 -- Top-k indices with deterministic tie-break (stable by index).

    When scores are equal, the earlier index wins.
    """
    if not scores or k <= 0:
        return []
    k = min(k, len(scores))
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda t: (-t[1], t[0]))
    return [idx for idx, _ in indexed[:k]]


# ---------------------------------------------------------------------------
# Eq R12 -- Budgeted selection (token budget)
# ---------------------------------------------------------------------------


def select_within_token_budget(
    chunk_scores: Sequence[float],
    chunk_tokens: Sequence[int],
    token_budget: int,
) -> List[int]:
    """Eq R12 -- Greedy budget-aware chunk selection.

    Picks chunks in descending score-per-token order until the token budget
    is exhausted.  Returns ordered list of chunk indices selected.
    Never exceeds *token_budget*.
    """
    if not chunk_scores or token_budget <= 0:
        return []
    n = min(len(chunk_scores), len(chunk_tokens))
    candidates = []
    for i in range(n):
        tokens = max(1, chunk_tokens[i])
        spt = chunk_scores[i] / tokens
        candidates.append((spt, -chunk_scores[i], i))
    candidates.sort(reverse=True)
    selected: List[int] = []
    remaining = token_budget
    for spt, _neg_sc, idx in candidates:
        cost = max(1, chunk_tokens[idx])
        if cost <= remaining:
            selected.append(idx)
            remaining -= cost
    selected.sort()
    return selected


# ---------------------------------------------------------------------------
# NOT-IMPLEMENTABLE constants
# ---------------------------------------------------------------------------

# Training/embedding-weight-dependent retrieval (neural rerankers, ColBERT,
# cross-encoder fine-tuning, learned-to-rank) cannot be implemented in pure
# Python without a live model and a training loop.

NOT_IMPLEMENTABLE_COLBERT_RERANK = (
    "colbert-reranker:requires-trained-ColBERT model weights, GPU "
    "tokenisation, and MaxSim scoring kernel"
)

NOT_IMPLEMENTABLE_CROSS_ENCODER = (
    "cross-encoder-finetuning:requires transformer model, training loop, "
    "and labelled query-document pairs"
)

NOT_IMPLEMENTABLE_LEARNED_TO_RANK = (
    "learned-to-rank:requires LambdaMART/similar learner trained on "
    "click-relevance dataset; no pure-stdlib equivalent"
)

NOT_IMPLEMENTABLE_NEURAL_MMR = (
    "neural-mmr:requires embedding model to compute pairwise doc "
    "similarities in semantic space; pure function above uses caller-provided "
    "similarity matrix instead"
)


# ---------------------------------------------------------------------------
# Module public API
# ---------------------------------------------------------------------------

__all__ = [
    "mmr",
    "mmr_score",
    "min_max_normalize",
    "z_score_normalize",
    "sigmoid_normalize",
    "normalize_scores",
    "cosine_to_similarity",
    "rrf_score",
    "rrf",
    "weighted_reciprocal_rank",
    "pointwise_mutual_information",
    "mi_gate",
    "novelty_score",
    "curiosity_score",
    "greedy_diverse_select",
    "chunk_cache_key",
    "embedding_similarity_from_embeddings",
    "relevance_gate",
    "qf_similarity",
    "hybrid_relevance",
    "topk_stable",
    "select_within_token_budget",
]
