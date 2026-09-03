"""
reasoning/reasoning_cache.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Intelligent caching system for reasoning results.

This module provides:
- Semantic similarity-based caching for reasoning results
- Adaptive cache sizing based on performance
- Cache warming for common problem patterns
- Result memoization for similar problems
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import heapq

logger = logging.getLogger(__name__)


class CacheEvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    ADAPTIVE = "adaptive"  # Adaptive based on hit rate


@dataclass
class CacheEntry:
    """A cache entry for reasoning results."""
    question: str
    task_type: str
    result: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    similarity_hash: str = ""
    
    def record_access(self):
        """Record an access to this cache entry."""
        self.access_count += 1
        self.last_access = time.time()
    
    def __lt__(self, other):
        """Comparison for sorting in priority queue."""
        return self.last_access < other.last_access


@dataclass
class CacheConfig:
    """Configuration for reasoning cache."""
    max_size: int = 1000
    eviction_policy: str = "adaptive"
    similarity_threshold: float = 0.85
    ttl_seconds: int = 3600  # 1 hour default
    enable_warming: bool = True
    warmup_examples: List[str] = field(default_factory=list)


class ReasoningCache:
    """
    Intelligent cache for reasoning results with semantic similarity.
    
    Features:
    - Semantic similarity-based lookup using question hashing
    - Adaptive eviction policies based on access patterns
    - Time-based expiration for stale results
    - Cache warming for common problem patterns
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self._lru_heap = []  # Min-heap for LRU eviction (last_access, key)
        self._heap_valid = True  # Track if heap needs rebuilding
        
        logger.info(
            f"ReasoningCache initialized with max_size={self.config.max_size}, "
            f"eviction_policy={self.config.eviction_policy}"
        )
    
    def _compute_similarity_hash(self, question: str, task_type: str) -> str:
        """
        Compute a similarity-preserving hash for the question.
        
        This normalizes the question by:
        1. Converting to lowercase
        2. Removing extra whitespace
        3. Extracting key terms (numbers, key phrases)
        """
        # Normalize question
        normalized = " ".join(question.lower().split())
        
        # Extract key terms (numbers, important words)
        import re
        numbers = re.findall(r'\d+', normalized)
        key_words = re.findall(r'\b\w{4,}\b', normalized)
        
        # Create a normalized representation
        normalized_repr = f"{task_type}:{'|'.join(sorted(numbers))}:{'|'.join(sorted(key_words))}"
        
        # Hash the normalized representation
        return hashlib.md5(normalized_repr.encode()).hexdigest()
    
    def _check_similarity(self, question: str, task_type: str, entry: CacheEntry) -> float:
        """
        Check similarity between question and cache entry.
        
        Returns a similarity score between 0 and 1.
        """
        if entry.task_type != task_type:
            return 0.0
        
        # Simple word overlap similarity
        question_words = set(question.lower().split())
        entry_words = set(entry.question.lower().split())
        
        if not question_words or not entry_words:
            return 0.0
        
        intersection = len(question_words & entry_words)
        union = len(question_words | entry_words)
        
        return intersection / union if union > 0 else 0.0
    
    def get(self, question: str, task_type: str = "math") -> Optional[Dict[str, Any]]:
        """
        Get cached reasoning result for a question.
        
        Args:
            question: The question to look up
            task_type: Type of reasoning task
        
        Returns:
            Cached result if found and valid, None otherwise
        """
        current_time = time.time()
        
        # First try exact hash match
        similarity_hash = self._compute_similarity_hash(question, task_type)
        
        if similarity_hash in self.cache:
            entry = self.cache[similarity_hash]
            
            # Check TTL
            if current_time - entry.timestamp > self.config.ttl_seconds:
                self._evict(similarity_hash)
                self.miss_count += 1
                return None
            
            # Record access
            entry.record_access()
            self.hit_count += 1
            
            # Mark heap as invalid since we changed access time
            self._heap_valid = False
            
            logger.debug(f"Cache hit for question: {question[:50]}...")
            return entry.result
        
        # Try similarity-based lookup
        for hash_key, entry in list(self.cache.items()):
            similarity = self._check_similarity(question, task_type, entry)
            
            if similarity >= self.config.similarity_threshold:
                # Check TTL
                if current_time - entry.timestamp > self.config.ttl_seconds:
                    self._evict(hash_key)
                    continue
                
                # Record access
                entry.record_access()
                self.hit_count += 1
                
                # Mark heap as invalid since we changed access time
                self._heap_valid = False
                
                logger.debug(f"Cache similarity hit ({similarity:.2f}) for question: {question[:50]}...")
                return entry.result
        
        self.miss_count += 1
        return None
    
    def put(self, question: str, task_type: str, result: Dict[str, Any]) -> None:
        """
        Cache a reasoning result.
        
        Args:
            question: The question
            task_type: Type of reasoning task
            result: The reasoning result to cache
        """
        # Check if cache is full
        if len(self.cache) >= self.config.max_size:
            self._evict_lru()
        
        # Create cache entry
        similarity_hash = self._compute_similarity_hash(question, task_type)
        entry = CacheEntry(
            question=question,
            task_type=task_type,
            result=result,
            similarity_hash=similarity_hash
        )
        
        self.cache[similarity_hash] = entry
        # Mark heap as invalid since we added a new entry
        self._heap_valid = False
        logger.debug(f"Cached result for question: {question[:50]}...")
    
    def _evict(self, hash_key: str) -> None:
        """Evict a specific entry from the cache."""
        if hash_key in self.cache:
            del self.cache[hash_key]
            self.eviction_count += 1
            # Mark heap as invalid since we removed an entry
            self._heap_valid = False
    
    def _evict_lru(self) -> None:
        """Evict the least recently used entry using heap for efficiency."""
        if not self.cache:
            return
        
        # Rebuild heap if invalid
        if not self._heap_valid:
            self._rebuild_heap()
        
        # Get LRU entry from heap
        if self._lru_heap:
            last_access, lru_key = heapq.heappop(self._lru_heap)
            # Verify the entry still exists and has the same access time
            if lru_key in self.cache and self.cache[lru_key].last_access == last_access:
                self._evict(lru_key)
                logger.debug(f"Evicted LRU entry: {lru_key}")
            else:
                # Heap was stale, retry
                self._evict_lru()
        else:
            # Fallback to linear search if heap is empty
            lru_key = min(self.cache.keys(), key=lambda k: self.cache[k].last_access)
            self._evict(lru_key)
            logger.debug(f"Evicted LRU entry (fallback): {lru_key}")
    
    def _rebuild_heap(self) -> None:
        """Rebuild the LRU heap from current cache state."""
        self._lru_heap = [(entry.last_access, key) for key, entry in self.cache.items()]
        heapq.heapify(self._lru_heap)
        self._heap_valid = True
    
    def _evict_adaptive(self) -> None:
        """Adaptive eviction based on hit rate and access patterns."""
        if not self.cache:
            return
        
        # Calculate current hit rate
        total_accesses = self.hit_count + self.miss_count
        if total_accesses == 0:
            hit_rate = 0.0
        else:
            hit_rate = self.hit_count / total_accesses
        
        # If hit rate is low, be more aggressive about eviction
        if hit_rate < 0.3:
            # Evict entries with low access count
            entries_to_evict = [
                k for k, v in self.cache.items()
                if v.access_count < 2
            ]
            for key in entries_to_evict[:len(self.cache) // 4]:  # Evict 25%
                self._evict(key)
            # Mark heap as invalid since we removed entries
            self._heap_valid = False
        else:
            # Use LRU
            self._evict_lru()
    
    def _evict_based_on_policy(self) -> None:
        """Evict entries based on configured policy."""
        if self.config.eviction_policy == "lru":
            self._evict_lru()
        elif self.config.eviction_policy == "adaptive":
            self._evict_adaptive()
        else:
            self._evict_lru()  # Default to LRU
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_accesses = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_accesses if total_accesses > 0 else 0.0
        
        return {
            "size": len(self.cache),
            "max_size": self.config.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "eviction_count": self.eviction_count,
            "hit_rate": hit_rate,
            "total_accesses": total_accesses,
        }
    
    def warm_up(self, examples: List[Tuple[str, str, Dict[str, Any]]]) -> None:
        """
        Warm up the cache with example problems and solutions.
        
        Args:
            examples: List of (question, task_type, result) tuples
        """
        if not self.config.enable_warming:
            return
        
        logger.info(f"Warming up cache with {len(examples)} examples")
        
        for question, task_type, result in examples:
            self.put(question, task_type, result)
        
        logger.info("Cache warm-up complete")


class CachedReasoningWrapper:
    """
    Wrapper for reasoning methods with automatic caching.
    
    This wrapper automatically caches reasoning results and provides
    intelligent cache lookups for similar problems.
    """
    
    def __init__(
        self,
        reasoning_method: Any,
        cache_config: Optional[CacheConfig] = None,
        enable_cache: bool = True,
    ):
        self.reasoning_method = reasoning_method
        self.cache = ReasoningCache(cache_config) if enable_cache else None
        self.enable_cache = enable_cache
        
        logger.info(
            f"CachedReasoningWrapper initialized for {reasoning_method.__class__.__name__}, "
            f"cache_enabled={enable_cache}"
        )
    
    def reason(
        self,
        question: str,
        model_fn: callable,
        task_type: str = "math",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform reasoning with caching.
        
        Args:
            question: Question to reason about
            model_fn: Function to call the model
            task_type: Type of reasoning task
            **kwargs: Additional arguments for reasoning method
        
        Returns:
            Reasoning result
        """
        # Check cache first
        if self.enable_cache and self.cache:
            cached_result = self.cache.get(question, task_type)
            if cached_result is not None:
                cached_result["cached"] = True
                return cached_result
        
        # Perform reasoning
        result = self.reasoning_method.reason(question, model_fn, task_type, **kwargs)
        result["cached"] = False
        
        # Cache the result
        if self.enable_cache and self.cache:
            self.cache.put(question, task_type, result)
        
        return result
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self.cache:
            return self.cache.get_stats()
        return {"cache_enabled": False}
    
    def clear_cache(self) -> None:
        """Clear the cache."""
        if self.cache:
            self.cache.clear()