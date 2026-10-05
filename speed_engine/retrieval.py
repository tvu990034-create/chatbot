"""
Retrieval: BM25 (EQ-BM25), query truncation (EQ-QUERY-TRUNC), dynamic index (EQ-DYNAMIC-INDEX),
dense cosine (EQ-DENSE-COSINE), cross-encoder rerank (EQ-CROSS-ENCODER), score gating (EQ-SCORE-GATE).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from speed_engine.pagerank import prune_with_pagerank
from speed_engine.prefilter import normalize


STOP_WORDS = frozenset(
    "a an the and or but in on at to for of with is are was were be been being "
    "have has had do does did will would could should may might can shall you your "
    "i me my we our he she it they them this that what which who whom how when where why".split()
)


@dataclass
class RetrievalResult:
    chunks: List[Tuple[str, str, float]]  # doc_id, text, score
    best_score: float


# ---------------------------------------------------------------------------
# EQ-QUERY-TRUNC — E(t) = IDF(t)*(1+ln(1+tf)); theta = (1-alpha)*E_max + alpha*E_avg
# ---------------------------------------------------------------------------
def truncate_query_essential(
    query: str,
    df_dict: Dict[str, int],
    n: int,
    alpha: float = 0.3,
) -> str:
    """Return essential keywords for BM25 leg (§22nd / §32nd)."""
    tokens = [t for t in query.lower().split() if any(c.isalpha() for c in t)]
    if not tokens:
        return query
    tf_q: Dict[str, int] = {}
    for t in tokens:
        tf_q[t] = tf_q.get(t, 0) + 1

    def idf(word: str) -> float:
        nt = max(df_dict.get(word, 0), 0)
        return math.log((n - nt + 0.5) / (nt + 0.5))

    e_scores = {t: idf(t) * (1.0 + math.log(1.0 + tf_q[t])) for t in tf_q}
    scores = list(e_scores.values())
    theta = (1.0 - alpha) * max(scores) + alpha * (sum(scores) / len(scores))
    kept = [t for t in tokens if e_scores.get(t, 0.0) > theta]
    return " ".join(kept) if kept else query


# ---------------------------------------------------------------------------
# EQ-DYNAMIC-INDEX — S(q)=1 iff content_words <= max_words
# ---------------------------------------------------------------------------
def is_short_query(query: str, max_words: int = 2) -> bool:
    words = query.strip().split()
    content = [w for w in words if w.lower() not in STOP_WORDS]
    return len(content) <= max_words


# ---------------------------------------------------------------------------
# EQ-BM25 — IDF and BM25Okapi scoring
# ---------------------------------------------------------------------------
class BM25Index:
    """
    score(D,Q) = sum IDF(qi) * f(qi,D)*(k1+1) / (f(qi,D) + k1*(1-b+b*|D|/avgdl))
    """

    def __init__(
        self,
        documents: List[Tuple[str, str]],
        k1: float = 1.2,
        b: float = 0.75,
    ):
        self.documents = documents  # (doc_id, text)
        self.k1 = k1
        self.b = b
        self._tokenized = [normalize(text).split() for _, text in documents]
        self._avgdl = sum(len(t) for t in self._tokenized) / max(len(self._tokenized), 1)
        self._df: Dict[str, int] = {}
        for tokens in self._tokenized:
            for w in set(tokens):
                self._df[w] = self._df.get(w, 0) + 1
        self.n = len(documents)

    def idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def score_doc(self, query_tokens: List[str], doc_idx: int) -> float:
        tokens = self._tokenized[doc_idx]
        dl = len(tokens)
        tf_map: Dict[str, int] = {}
        for w in tokens:
            tf_map[w] = tf_map.get(w, 0) + 1
        s = 0.0
        for q in query_tokens:
            if q not in tf_map:
                continue
            f = tf_map[q]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            s += self.idf(q) * f * (self.k1 + 1) / denom
        return s

    def top_k(self, query: str, k: int) -> List[Tuple[str, str, float]]:
        q_tokens = normalize(query).split()
        scored = []
        for i, (doc_id, text) in enumerate(self.documents):
            scored.append((doc_id, text, self.score_doc(q_tokens, i)))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]

    @property
    def df_dict(self) -> Dict[str, int]:
        return dict(self._df)


# ---------------------------------------------------------------------------
# EQ-DENSE-COSINE — sim(q,d) = q·d with L2-normalized vectors (bag-of-words hash)
# ---------------------------------------------------------------------------
class DenseIndex:
    """Fast in-memory cosine similarity via hashed bag-of-words embeddings."""

    def __init__(self, documents: List[Tuple[str, str]], dim: int = 128):
        self.documents = documents
        self.dim = dim
        self._vecs = [self._embed(text) for _, text in documents]

    def _embed(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for w in normalize(text).split():
            h = hash(w) % self.dim
            v[h] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def _dot(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def top_k(self, query: str, k: int) -> List[Tuple[str, str, float]]:
        qv = self._embed(query)
        scored = []
        for i, (doc_id, text) in enumerate(self.documents):
            scored.append((doc_id, text, self._dot(qv, self._vecs[i])))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]


# ---------------------------------------------------------------------------
# EQ-CROSS-ENCODER — pairwise relevance score (lightweight token overlap proxy)
# ---------------------------------------------------------------------------
def cross_encoder_score(query: str, document: str) -> float:
    q = set(normalize(query).split())
    d = set(normalize(document).split())
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


def rerank(
    query: str,
    candidates: List[Tuple[str, str, float]],
) -> List[Tuple[str, str, float]]:
    out = []
    for doc_id, text, _ in candidates:
        s = cross_encoder_score(query, text)
        out.append((doc_id, text, s))
    out.sort(key=lambda x: x[2], reverse=True)
    return out


# ---------------------------------------------------------------------------
# EQ-SCORE-GATE — relevant iff best_score >= threshold
# ---------------------------------------------------------------------------
def is_relevant(best_score: float, threshold: Optional[float]) -> bool:
    if threshold is None:
        return True
    return best_score >= threshold


class RetrievalPipeline:
    """
    BM25 → Dense → PageRank prune → Rerank (§16th dynamic switch, §24th gating input).
    """

    def __init__(
        self,
        documents: List[Tuple[str, str]],
        pagerank: Dict[str, float],
        k1: float = 1.2,
        b: float = 0.75,
        bm25_top_k: int = 100,
        dense_top_k: int = 50,
        use_dynamic_index_switch: bool = True,
        short_query_max_words: int = 2,
        bm25_min_results: int = 3,
        bm25_score_threshold: float = 0.0,
        use_query_truncation: bool = True,
        truncation_alpha: float = 0.3,
        use_pagerank_prune: bool = True,
        pagerank_beta: float = 0.5,
        pagerank_keep_k: int = 80,
        pagerank_score_floor: float = 0.01,
        chunk_limit: int = 1,
    ):
        self.bm25 = BM25Index(documents, k1=k1, b=b)
        self.dense = DenseIndex(documents)
        self.pagerank = pagerank
        self.bm25_top_k = bm25_top_k
        self.dense_top_k = dense_top_k
        self.use_dynamic_index_switch = use_dynamic_index_switch
        self.short_query_max_words = short_query_max_words
        self.bm25_min_results = bm25_min_results
        self.bm25_score_threshold = bm25_score_threshold
        self.use_query_truncation = use_query_truncation
        self.truncation_alpha = truncation_alpha
        self.use_pagerank_prune = use_pagerank_prune
        self.pagerank_beta = pagerank_beta
        self.pagerank_keep_k = pagerank_keep_k
        self.pagerank_score_floor = pagerank_score_floor
        self.chunk_limit = chunk_limit

    def retrieve(self, query: str) -> RetrievalResult:
        if self.use_dynamic_index_switch and is_short_query(query, self.short_query_max_words):
            chunks = self._short_path(query)
            if chunks is not None:
                return self._finalize(chunks)
        return self._finalize(self._full_path(query))

    def _bm25_query(self, query: str) -> str:
        if not self.use_query_truncation:
            return query
        return truncate_query_essential(
            query, self.bm25.df_dict, self.bm25.n, self.truncation_alpha
        )

    def _short_path(self, query: str) -> Optional[List[Tuple[str, str, float]]]:
        bq = self._bm25_query(query)
        bm25_results = self.bm25.top_k(bq, self.bm25_top_k)
        if len(bm25_results) < self.bm25_min_results:
            return None
        if bm25_results and bm25_results[0][2] < self.bm25_score_threshold:
            return None
        return bm25_results

    def _full_path(self, query: str) -> List[Tuple[str, str, float]]:
        bq = self._bm25_query(query)
        bm25_c = self.bm25.top_k(bq, self.bm25_top_k)
        dense_c = self.dense.top_k(query, self.dense_top_k)
        merged = self._merge(bm25_c, dense_c)
        if self.use_pagerank_prune and merged:
            ids = prune_with_pagerank(
                [(d, s) for d, _, s in merged],
                self.pagerank,
                self.pagerank_beta,
                self.pagerank_keep_k,
                self.pagerank_score_floor,
            )
            id_set = set(ids)
            merged = [c for c in merged if c[0] in id_set]
        return rerank(query, merged)

    def _merge(
        self,
        a: List[Tuple[str, str, float]],
        b: List[Tuple[str, str, float]],
    ) -> List[Tuple[str, str, float]]:
        seen: Set[str] = set()
        out = []
        for doc_id, text, score in a + b:
            if doc_id not in seen:
                seen.add(doc_id)
                out.append((doc_id, text, score))
        return out

    def _finalize(self, chunks: List[Tuple[str, str, float]]) -> RetrievalResult:
        limited = chunks[: self.chunk_limit]
        best = limited[0][2] if limited else 0.0
        return RetrievalResult(chunks=limited, best_score=best)
