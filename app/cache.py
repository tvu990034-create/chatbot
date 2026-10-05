"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

# cache.py
from __future__ import annotations
import hashlib, re, threading, time, math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
try:
    from phrase_normalizer import PhraseNormalizer
except ImportError:
    class PhraseNormalizer:
        def __init__(self, *args, **kwargs):
            pass
        def canonical(self, query):
            return query

class PreWarmedCache:
    def __init__(self, max_entries: int = 5000, initial_threshold: int = 4, normalizer=None):
        self.max_entries = max_entries
        self.hamming_threshold = initial_threshold
        self.lock = threading.RLock()
        self.exact: OrderedDict = OrderedDict()
        self.simhash_map: Dict[int, Tuple[str, str]] = {}
        self.access_counts: Dict[str, int] = {}
        self.hits = 0
        self.misses = 0
        self.hit_window = 100
        self.target_hit_rate = 0.9
        self.normalizer = normalizer          # can be None (no PhraseNormalizer)
        # Store embeddings alongside responses for parallel embedding optimization
        self.embeddings: Dict[str, any] = {}  # query_norm -> embedding vector

    _normalize_pattern = re.compile(r'[^\w\s]')
    _space_pattern = re.compile(r'\s+')
    
    @staticmethod
    def normalize(text: str) -> str:
        t = text.lower().strip()
        t = PreWarmedCache._normalize_pattern.sub('', t)
        t = PreWarmedCache._space_pattern.sub(' ', t).strip()
        return t

    @staticmethod
    def simhash_64(text: str) -> int:
        tokens = text.split()
        v = [0] * 64
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)
            # Unroll loop for better performance
            for i in range(0, 64, 8):
                mask = 0xFF << i
                chunk = (h >> i) & 0xFF
                for j in range(8):
                    if (chunk >> j) & 1:
                        v[i + j] += 1
                    else:
                        v[i + j] -= 1
        fp = 0
        for i in range(64):
            if v[i] > 0:
                fp |= (1 << i)
        return fp

    @staticmethod
    def hamming_distance(a: int, b: int) -> int:
        return (a ^ b).bit_count()

    def get(self, raw_query: str) -> Optional[str]:
        """Fast exact match only for maximum speed"""
        # Skip normalization for cache lookup to reduce overhead
        query_norm = self.normalize(raw_query)
        
        # Exact match only (fastest)
        if query_norm in self.exact:
            self.hits += 1
            return self.exact[query_norm][0]
        
        self.misses += 1
        return None
    
    def get_embedding(self, raw_query: str) -> Optional[any]:
        """Get cached embedding for a query"""
        query_norm = self.normalize(raw_query)
        with self.lock:
            return self.embeddings.get(query_norm)
    
    def put_embedding(self, raw_query: str, embedding: any):
        """Store embedding for a query"""
        query_norm = self.normalize(raw_query)
        with self.lock:
            self.embeddings[query_norm] = embedding

    def put(self, raw_query: str, response: str, embedding: any = None):
        """Add to cache with normalized key - optimized for speed"""
        if self.normalizer:
            raw_query = self.normalizer.canonical(raw_query)
        q = self.normalize(raw_query)
        with self.lock:
            self.exact[q] = (response, time.time(), 0)  # Skip simhash for speed
            # Store embedding if provided
            if embedding is not None:
                self.embeddings[q] = embedding
            # Remove oldest if at capacity
            if len(self.exact) > self.max_entries:
                oldest_key = next(iter(self.exact))
                self.exact.popitem(last=False)
                self.embeddings.pop(oldest_key, None)

    def clear(self):
        """Clear all cache data."""
        with self.lock:
            self.exact.clear()
            self.simhash_map.clear()
            self.access_counts.clear()
            self.embeddings.clear()
            self.hits = 0
            self.misses = 0