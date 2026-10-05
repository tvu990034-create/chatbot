"""
Cache Engine - Mathematical equations for caching
Extracted from PDF files - these are the ONLY equations that matter
"""

import math
import time
from typing import Dict, Any, Optional

class CacheEngine:
    """
    Implements ALL cache equations from PDF files
    These equations are the single source of truth for caching logic
    """
    
    def __init__(self):
        # Equation parameters from PDFs
        self.p_hit = 0.85  # 85% of FAQ queries are cached
        self.max_entries = 50
        self.ttl_ms = 86400000  # 24 hours
        self.subkey_size = 512
        self.similarity_threshold = 0.92
        self.disk_read_speed_bytes_per_sec = 2e9  # 2 GB/s typical NVMe
    
    # EQUATION: p_hit = 0.85
    def calc_p_hit(self) -> float:
        """p_hit = 0.85"""
        return self.p_hit
    
    # EQUATION: C(K) = 1 - (K0 / (K + K0))**alpha
    def calc_coverage(self, K: float, K0: float = 5.0, alpha: float = 1.5) -> float:
        """C(K) = 1 - (K0 / (K + K0))**alpha"""
        return 1 - (K0 / (K + K0)) ** alpha
    
    # EQUATION: K = K0 * ((1/(1-C))^(1/alpha) - 1)
    def calc_K_from_coverage(self, C: float, K0: float = 5.0, alpha: float = 1.5) -> float:
        """K = K0 * ((1/(1-C))^(1/alpha) - 1)"""
        return K0 * ((1.0 / (1.0 - C)) ** (1.0 / alpha) - 1)
    
    # EQUATION: factor = (1.0 / (1.0 - target)) ** (1.0 / alpha)
    def calc_factor(self, target: float, alpha: float = 1.5) -> float:
        """factor = (1.0 / (1.0 - target)) ** (1.0 / alpha)"""
        return (1.0 / (1.0 - target)) ** (1.0 / alpha)
    
    # EQUATION: if (normNew.length >= cachedKey.length + 2)
    def should_update_cache(self, new_length: int, cached_length: int) -> bool:
        """if (normNew.length >= cachedKey.length + 2)"""
        return new_length >= cached_length + 2
    
    # EQUATION: if (normNew.length === 0) return null
    def is_valid_input(self, length: int) -> bool:
        """if (normNew.length === 0) return null"""
        return length > 0
    
    # EQUATION: if distances[0][0] >= threshold
    def is_cache_hit(self, distance: float, threshold: float) -> bool:
        """if distances[0][0] >= threshold"""
        return distance >= threshold
    
    # EQUATION: specificity += 0.1
    def calc_specificity_increment(self) -> float:
        """specificity += 0.1"""
        return 0.1
    
    # EQUATION: extended_probs = example_probs + [0.85] * (L - len(example_probs))
    def calc_extended_probs(self, example_probs: list, L: int) -> list:
        """extended_probs = example_probs + [0.85] * (L - len(example_probs))"""
        return example_probs + [0.85] * (L - len(example_probs))
    
    # EQUATION: expected += prefix_prob * fail_prob * (i - 1)
    def calc_expected(self, prefix_prob: float, fail_prob: float, i: int) -> float:
        """expected += prefix_prob * fail_prob * (i - 1)"""
        return prefix_prob * fail_prob * (i - 1)
    
    # EQUATION: E = Σ_{i=1}^{∞} ((∏_{k=1}^{i-1} p_k) * (1 - p_i) * (i-1))
    def calc_E_correct(self, probs: list) -> float:
        """E = Σ_{i=1}^{∞} ((∏_{k=1}^{i-1} p_k) * (1 - p_i) * (i-1))"""
        E = 0.0
        for i, p in enumerate(probs, start=1):
            if i == 1:
                E += (1 - p) * 0
            else:
                prod = 1.0
                for k in range(1, i):
                    prod *= probs[k-1]
                E += prod * (1 - p) * (i - 1)
        return E
    
    # EQUATION: answerability(query: str, alpha: float = 0.5)
    def calc_answerability(self, query: str, alpha: float = 0.5) -> float:
        """answerability based on query length and complexity"""
        return min(len(query) / 100.0, 1.0) * alpha
    
    # EQUATION: cache hit rate calculation
    def calc_cache_hit_rate(self, hits: int, total: int) -> float:
        """cache_hit_rate = hits / total if total > 0 else 0.0"""
        return hits / total if total > 0 else 0.0
    
    # EQUATION: disk read time calculation
    def calc_disk_read_time_ms(self, bytes_to_read: int) -> float:
        """disk_read_time_ms = bytes_to_read / disk_read_speed_bytes_per_sec * 1000"""
        return bytes_to_read / self.disk_read_speed_bytes_per_sec * 1000
    
    # MASTER EQUATION: Cache performance calculation
    def calculate_cache_performance(self, query: str, cache_hits: int, cache_total: int) -> Dict[str, Any]:
        """
        MASTER EQUATION: Combines all cache equations
        Returns complete cache performance metrics
        """
        start_time = time.perf_counter()
        
        # Calculate cache hit rate
        hit_rate = self.calc_cache_hit_rate(cache_hits, cache_total)
        
        # Calculate answerability
        answerability = self.calc_answerability(query)
        
        # Calculate coverage for K=100
        coverage_100 = self.calc_coverage(100)
        
        # Calculate K needed for 80% coverage
        K_80 = self.calc_K_from_coverage(0.80)
        
        # Calculate factor for 90% target
        factor_90 = self.calc_factor(0.90)
        
        # Calculate specificity increment
        specificity = self.calc_specificity_increment()
        
        calc_time = (time.perf_counter() - start_time) * 1000
        
        return {
            "cache_hit_rate": hit_rate,
            "answerability": answerability,
            "coverage_100": coverage_100,
            "K_for_80_coverage": K_80,
            "factor_90": factor_90,
            "specificity_increment": specificity,
            "calculation_time_ms": calc_time,
            "p_hit": self.p_hit
        }


# Global cache engine instance
cache_engine = CacheEngine()
