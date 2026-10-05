"""
Cache layer: SimHash (§23), PreWarmed cascade, hit-rate tuner (§25), popularity eviction (Eq 48).
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from speed_engine.prefilter import normalize


# ---------------------------------------------------------------------------
# EQ-SIMHASH — 64-bit fingerprint via per-token MD5 voting
# ---------------------------------------------------------------------------
def simhash_64(text: str, hash_bits: int = 64) -> int:
    """Generate 64-bit SimHash (§23rd equation)."""
    tokens = text.split()
    v = [0] * hash_bits
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(hash_bits):
            if (h >> i) & 1:
                v[i] += 1
            else:
                v[i] -= 1
    fp = 0
    for i in range(hash_bits):
        if v[i] > 0:
            fp |= 1 << i
    return fp


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# EQ-SESSION-TTFT — session KV-cache TTFT prediction (§1st)
# ---------------------------------------------------------------------------
def session_cache_ttft_speedup(
    history_tokens: int,
    new_query_tokens: int,
    suffix_tokens: int = 5,
    prefill_time_per_token: float = 2.0,
    load_overhead_ms: float = 15.0,
    hit_rate: float = 1.0,
) -> dict:
    ttft_no = history_tokens * prefill_time_per_token
    ttft_cache = load_overhead_ms + (new_query_tokens + suffix_tokens) * prefill_time_per_token
    expected_ttft = hit_rate * ttft_cache + (1 - hit_rate) * ttft_no
    speedup = ttft_no / expected_ttft if expected_ttft > 0 else float("inf")
    reduction_pct = 100.0 * (1 - expected_ttft / ttft_no) if ttft_no > 0 else 0.0
    return {
        "ttft_no_cache_ms": round(ttft_no, 2),
        "ttft_cached_ms": round(ttft_cache, 2),
        "expected_ttft_ms": round(expected_ttft, 2),
        "speedup_factor": round(speedup, 2),
        "reduction_pct": round(reduction_pct, 1),
    }


# ---------------------------------------------------------------------------
# EQ-PREWARM-CACHE cascade: exact → SimHash → BM25 → miss
# EQ-PREWARM-HIT: observed_hit_rate vs target; tune threshold
# EQ-POPULARITY-EVICT (Eq 48): min access_count * recency
# ---------------------------------------------------------------------------
class PreWarmedCache:
    """Hybrid lookup exact → SimHash → BM25 (§23rd, §25th)."""

    def __init__(
        self,
        max_entries: int = 50_000,
        initial_threshold: int = 4,
        target_hit_rate: float = 0.95,
        bm25_score_threshold: float = 0.3,
    ):
        self.max_entries = max_entries
        self.threshold = initial_threshold
        self.target_hit_rate = target_hit_rate
        self.bm25_score_threshold = bm25_score_threshold
        self.exact: OrderedDict[str, dict] = OrderedDict()
        self.simhash_map: Dict[int, Tuple[str, str]] = {}
        self.access_counts: Dict[str, int] = {}
        self.hits = 0
        self.misses = 0
        self._corpus_keys: Optional[List[str]] = None

    def _simhash(self, text: str) -> int:
        return simhash_64(normalize(text))

    def put(self, raw_query: str, response: str) -> None:
        q = normalize(raw_query)
        while len(self.exact) >= self.max_entries:
            self._evict_one()
        self.exact[q] = {"response": response, "timestamp": time.time()}
        self.simhash_map[self._simhash(q)] = (q, response)
        self._corpus_keys = None

    def get(self, raw_query: str) -> Optional[str]:
        q = normalize(raw_query)
        # Stage 1: exact
        if q in self.exact:
            self.access_counts[q] = self.access_counts.get(q, 0) + 1
            self.exact.move_to_end(q)
            self.hits += 1
            return self.exact[q]["response"]
        # Stage 2: SimHash
        sh = self._simhash(q)
        for stored_sh, (_, resp) in self.simhash_map.items():
            if hamming_distance(sh, stored_sh) <= self.threshold:
                self.hits += 1
                return resp
        # Stage 3: BM25 proxy on cached queries
        if self.exact:
            if self._corpus_keys is None:
                self._corpus_keys = list(self.exact.keys())
            top = _bm25_top1(q, self._corpus_keys)
            if top and top[1] >= self.bm25_score_threshold:
                self.hits += 1
                return self.exact[top[0]]["response"]
        self.misses += 1
        self._update_threshold()
        return None

    def _evict_one(self) -> None:
        """EQ-POPULARITY-EVICT: minimize access_count * age."""
        if not self.exact:
            return
        now = time.time()
        worst = min(
            self.exact.keys(),
            key=lambda k: self.access_counts.get(k, 0) * (now - self.exact[k]["timestamp"]),
        )
        del self.exact[worst]
        sh = self._simhash(worst)
        self.simhash_map.pop(sh, None)

    def _update_threshold(self) -> None:
        """EQ-SIMHASH-THRESH (Eq 45 in §23): tune Hamming radius for target hit rate."""
        total = self.hits + self.misses
        if total < 20:
            return
        rate = self.hits / total
        if rate < self.target_hit_rate:
            self.threshold = min(self.threshold + 1, 16)
        elif rate > self.target_hit_rate + 0.02:
            self.threshold = max(self.threshold - 1, 1)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def _bm25_top1(query: str, corpus: List[str]) -> Optional[Tuple[str, float]]:
    """Lightweight BM25 top-1 over cache keys for fallback stage."""
    q_tokens = set(normalize(query).split())
    if not q_tokens:
        return None
    best_k, best_s = None, -1.0
    for key in corpus:
        k_tokens = set(key.split())
        if not k_tokens:
            continue
        score = len(q_tokens & k_tokens) / (len(q_tokens | k_tokens) + 1e-6)
        if score > best_s:
            best_s, best_k = score, key
    return (best_k, best_s) if best_k else None
