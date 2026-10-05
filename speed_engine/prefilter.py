"""
Pre-filter layer: Bloom (EQ-BLOOM-*), FAQ (EQ-FAQ), ZeroToken (EQ-ZERO-TOKEN).
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
from math import ceil, exp, log
from typing import Dict, Optional, Set


# ---------------------------------------------------------------------------
# EQ-NORMALIZE (Eq 44) — canonical text for exact match
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# EQ-ZERO-TOKEN — T_saved = N_trivial * (P + C); lookup is O(1)
# ---------------------------------------------------------------------------
class ZeroTokenResponder:
    """Returns canned reply for trivial inputs without LLM (§7th equation)."""

    def __init__(self, phrases: Dict[str, str]):
        self._phrases = {normalize(k): v for k, v in phrases.items()}

    def respond(self, user_input: str) -> Optional[str]:
        key = normalize(user_input)
        return self._phrases.get(key) if key else None


# ---------------------------------------------------------------------------
# EQ-FAQ — O(1) dict lookup after normalize()
# ---------------------------------------------------------------------------
class FAQDatabase:
    """Exact-match FAQ; bypasses retrieval and LLM (§6th equation)."""

    def __init__(self, records: Dict[str, str]):
        self._data = {normalize(q): a for q, a in records.items()}

    def get(self, question: str) -> Optional[str]:
        return self._data.get(normalize(question))


# ---------------------------------------------------------------------------
# EQ-BLOOM-FP, EQ-BLOOM-OPT-K, EQ-BLOOM-DOUBLE-HASH
# m = n * bits_per_element; k = ceil(m/n * ln(2))
# epsilon = (1 - exp(-k*n/m))^k; pos_i = (h1 + i*h2) % m
# ---------------------------------------------------------------------------
class BloomFilter:
    """Standard Bloom filter with optimal k and double hashing (part_1.pdf)."""

    def __init__(self, n: int, bits_per_element: int = 8):
        self.n = n
        self.m = n * bits_per_element
        self.k = max(1, int(ceil(self.m / max(n, 1) * log(2))))
        self.num_bytes = (self.m + 7) // 8
        self.bit_array = bytearray(self.num_bytes)

    def _get_hashes(self, item: str) -> tuple[int, int]:
        digest = hashlib.sha256(item.encode("utf-8")).digest()
        h1 = struct.unpack("<Q", digest[:8])[0]
        h2 = struct.unpack("<Q", digest[8:16])[0]
        return h1, h2

    def _bit_indices(self, h1: int, h2: int) -> list[int]:
        return [(h1 + i * h2) % self.m for i in range(self.k)]

    def add(self, item: str) -> None:
        h1, h2 = self._get_hashes(item)
        for pos in self._bit_indices(h1, h2):
            self.bit_array[pos // 8] |= 1 << (pos % 8)

    def contains(self, item: str) -> bool:
        h1, h2 = self._get_hashes(item)
        for pos in self._bit_indices(h1, h2):
            if not (self.bit_array[pos // 8] & (1 << (pos % 8))):
                return False
        return True

    def false_positive_rate(self) -> float:
        """p = (1 - e^{-k*n/m})^k"""
        return pow(1 - exp(-self.k * self.n / self.m), self.k)


# ---------------------------------------------------------------------------
# EQ-BLOOM-PRECISION — precision = P(known) / (P(known) + epsilon * (1-P(known)))
# ---------------------------------------------------------------------------
def bloom_precision(
    prob_known: float,
    bits_per_element: int = 8,
    k: Optional[int] = None,
) -> float:
    if not 0 < prob_known < 1:
        raise ValueError("prob_known must be in (0, 1)")
    k = k or max(1, int(round(bits_per_element * math.log(2))))
    n_over_m = 1.0 / bits_per_element
    epsilon = (1 - math.exp(-k * n_over_m)) ** k
    return prob_known / (prob_known + epsilon * (1 - prob_known))


def extract_entity(query: str, known_entities: Set[str] | None = None) -> str:
    """Prefer a token present in the known entity set; else no entity gate."""
    tokens = normalize(query).split()
    if known_entities:
        for t in tokens:
            if t in known_entities:
                return t
    return ""


class BloomPreFilter:
    """
    If entity extracted and not bf.contains(entity): skip retrieval instantly.
    Queries with no known entity token pass through (no false IDK).
    """

    def __init__(self, entities: Set[str], bits_per_element: int = 8):
        self._known = {e.lower() for e in entities}
        ents = list(self._known)
        self._bf = BloomFilter(n=max(len(ents), 1), bits_per_element=bits_per_element)
        for e in ents:
            self._bf.add(e)
        self._precision = bloom_precision(0.5, bits_per_element)

    def allows_retrieval(self, query: str) -> bool:
        entity = extract_entity(query, self._known)
        if not entity:
            return True
        return self._bf.contains(entity)

    @property
    def expected_precision(self) -> float:
        return self._precision
