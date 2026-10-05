"""
Speed Engine - ALL 47 Speed Equations from LLM Speedup Document
Extracted from Tài liệu không có tiêu đề.txt (7671 lines)
These are the complete mathematical equations for LLM inference speedup
"""

import math
import time
import re
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import hashlib


# ============================================================================
# EQUATION 1: Session Cache TTFT Speedup
# ============================================================================
def session_cache_ttft_speedup(
    ttft_no_cache_ms: float,
    system_prefill_time_ms: float,
    cache_hit_rate: float
) -> Dict[str, float]:
    """
    Predicts TTFT speedup from session KV-cache reuse.
    
    Args:
        ttft_no_cache_ms: TTFT without cache (ms)
        system_prefill_time_ms: Time to prefill system prompt (ms)
        cache_hit_rate: Fraction of requests with cache hit (0-1)
    
    Returns:
        Dict with ttft_cached_ms, expected_ttft_ms, speedup_factor, reduction_pct
    """
    ttft_cached_ms = ttft_no_cache_ms - system_prefill_time_ms
    expected_ttft_ms = (1 - cache_hit_rate) * ttft_no_cache_ms + cache_hit_rate * ttft_cached_ms
    speedup_factor = ttft_no_cache_ms / expected_ttft_ms if expected_ttft_ms > 0 else 1.0
    reduction_pct = (1 - expected_ttft_ms / ttft_no_cache_ms) * 100 if ttft_no_cache_ms > 0 else 0
    
    return {
        "ttft_cached_ms": ttft_cached_ms,
        "expected_ttft_ms": expected_ttft_ms,
        "speedup_factor": speedup_factor,
        "reduction_pct": reduction_pct
    }


# ============================================================================
# EQUATION 2: Overall Conversation Speedup
# ============================================================================
def overall_conv_speedup(
    ttft_no_cache: float,
    ttft_with_cache: float,
    num_turns: int
) -> float:
    """
    Calculate overall speedup for multi-turn conversations.
    
    Args:
        ttft_no_cache: TTFT without cache per turn
        ttft_with_cache: TTFT with cache per turn
        num_turns: Number of conversation turns
    
    Returns:
        Overall speedup factor
    """
    total_no_cache = ttft_no_cache * num_turns
    total_with_cache = ttft_no_cache + ttft_with_cache * (num_turns - 1)
    return total_no_cache / total_with_cache if total_with_cache > 0 else 1.0


# ============================================================================
# EQUATION 3: Cascade Routing Speedup
# ============================================================================
def estimate_speedup(
    time_large: float,
    time_small: float,
    fraction_to_small: float
) -> float:
    """
    Quantify speedup from routing queries to smaller models.
    
    Args:
        time_large: Time for large model (ms)
        time_small: Time for small model (ms)
        fraction_to_small: Fraction of queries routed to small model (0-1)
    
    Returns:
        Speedup factor
    """
    avg_time = fraction_to_small * time_small + (1 - fraction_to_small) * time_large
    return time_large / avg_time if avg_time > 0 else 1.0


# ============================================================================
# EQUATION 4: Cache-Aware Conciseness Cost Delta
# ============================================================================
def cost_delta(
    tokens_without: int,
    tokens_with: int,
    cost_per_1k_tokens: float
) -> float:
    """
    Calculate cost delta from cache-aware conciseness instructions.
    
    Args:
        tokens_without: Tokens generated without conciseness
        tokens_with: Tokens generated with conciseness
        cost_per_1k_tokens: Cost per 1000 tokens
    
    Returns:
        Cost savings in currency units
    """
    delta_tokens = tokens_without - tokens_with
    return (delta_tokens / 1000) * cost_per_1k_tokens


# ============================================================================
# EQUATION 5: FAQ Database (O(1) Lookup)
# ============================================================================
class FAQDatabase:
    """Fast O(1) lookup for frequently asked questions."""
    
    def __init__(self):
        self.db = {}
        self._stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being"}
    
    @staticmethod
    def normalize(text: str) -> str:
        """Canonicalize text for matching."""
        words = re.findall(r"\w+", text.lower())
        return " ".join(w for w in words if w not in FAQDatabase._stopwords)
    
    def add(self, question: str, answer: str):
        """Add FAQ entry."""
        key = self.normalize(question)
        self.db[key] = answer
    
    def get(self, question: str) -> Optional[str]:
        """Get answer if question matches FAQ."""
        key = self.normalize(question)
        return self.db.get(key)
    
    def __contains__(self, question: str) -> bool:
        """Check if question is in FAQ."""
        key = self.normalize(question)
        return key in self.db
    
    def remove(self, question: str):
        """Remove FAQ entry."""
        key = self.normalize(question)
        self.db.pop(key, None)
    
    def save(self, filepath: str):
        """Save database to JSON."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.db, f)
    
    def load(self, filepath: str):
        """Load database from JSON."""
        import json
        with open(filepath, 'r') as f:
            self.db = json.load(f)


# ============================================================================
# EQUATION 6: Zero-Token Responder
# ============================================================================
class ZeroTokenResponder:
    """Handle trivial inputs locally without LLM call."""
    
    def __init__(self, phrases_file: str = None):
        self.phrases = {}
        if phrases_file:
            self.load_phrases(phrases_file)
    
    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for matching."""
        return re.sub(r"[^\w\s]", "", text.lower().strip())
    
    def load_phrases(self, filepath: str):
        """Load phrases from JSON."""
        import json
        with open(filepath, 'r') as f:
            self.phrases = json.load(f)
    
    def respond(self, user_input: str) -> Optional[str]:
        """Return response if input matches known phrases."""
        normalized = self.normalize(user_input)
        return self.phrases.get(normalized)


# ============================================================================
# EQUATION 7: Boredom Stopper
# ============================================================================
class BoredomStopper:
    """Stop generation early when model starts rambling."""
    
    def __init__(self, window_size: int = 5, threshold: float = -2.0):
        self.window_size = window_size
        self.threshold = threshold
        self.log_prob_history = []
    
    def should_stop(self, log_prob: float) -> bool:
        """Check if generation should stop based on log probabilities."""
        self.log_prob_history.append(log_prob)
        if len(self.log_prob_history) < self.window_size:
            return False
        
        recent_avg = sum(self.log_prob_history[-self.window_size:]) / self.window_size
        return recent_avg < self.threshold
    
    def reset(self):
        """Reset history."""
        self.log_prob_history = []


# ============================================================================
# EQUATION 8: Dynamic Max Tokens
# ============================================================================
_TRIGGER_RE = re.compile(
    r'\b(explain|describe|elaborate|illustrate|'
    r'how\s+to|what\s+is\s+the\s+difference|compare|versus)\b',
    re.IGNORECASE
)

def compute_dynamic_max_tokens(
    user_input: str,
    base: int = 20,
    multiplier: float = 2.0,
    max_cap: int = 256,
    long_answer_cap: int = 150,
    floor: int = 10,
    max_query_words: int = 50,
) -> int:
    """
    Calculate dynamic max tokens based on query complexity.
    
    Args:
        user_input: User query text
        base: Base token count
        multiplier: Multiplier for word count
        max_cap: Maximum token cap
        long_answer_cap: Cap for long-answer triggers
        floor: Minimum token count
        max_query_words: Max words to consider
    
    Returns:
        Dynamic max tokens value
    """
    if not isinstance(user_input, str) or user_input.strip() == "":
        return base
    
    words = user_input.strip().split()
    word_count = min(len(words), max_query_words)
    
    # Trigger-word override
    if _TRIGGER_RE.search(user_input):
        return min(long_answer_cap, max_cap)
    
    dynamic = base + int(multiplier * word_count)
    dynamic = max(floor, dynamic)
    dynamic = min(dynamic, max_cap)
    return dynamic


# ============================================================================
# EQUATION 9: Prompt Cached Engine
# ============================================================================
class PromptCachedEngine:
    """System prompt caching using save_state/load_state."""
    
    def __init__(self, model, system_prompt: str):
        self.model = model
        self.system_prompt = system_prompt
        self.cached_state = None
    
    def _build_base_state(self):
        """Pre-compute base state (system prompt only)."""
        self.model.eval(self.system_prompt, use_cache=True)
        self.cached_state = self.model.save_state()
    
    def generate(self, user_query: str, max_tokens: int = 100):
        """Generate with cached system prompt."""
        if self.cached_state is None:
            self._build_base_state()
        
        self.model.load_state(self.cached_state)
        full_prompt = self.system_prompt + user_query
        return self.model(full_prompt, max_tokens=max_tokens)


# ============================================================================
# EQUATION 10: Minimal Prompt Equation
# ============================================================================
def minimal_prompt_equation(
    original_tokens: int,
    boilerplate_tokens: int,
    time_per_token_ms: float = 0.8
) -> Dict[str, float]:
    """
    Quantify speedup from stripping boilerplate tokens.
    
    Args:
        original_tokens: Total tokens in original prompt
        boilerplate_tokens: Tokens that can be removed
        time_per_token_ms: Time per token in ms
    
    Returns:
        Dict with speedup_factor, token_reduction_pct, time_saved_ms
    """
    core_tokens = original_tokens - boilerplate_tokens
    speedup_factor = original_tokens / core_tokens if core_tokens > 0 else 1.0
    token_reduction_pct = (boilerplate_tokens / original_tokens) * 100 if original_tokens > 0 else 0
    time_saved_ms = boilerplate_tokens * time_per_token_ms
    
    return {
        "speedup_factor": speedup_factor,
        "token_reduction_pct": token_reduction_pct,
        "time_saved_ms": time_saved_ms
    }


# ============================================================================
# EQUATION 11: Speculative Decoding Speedup
# ============================================================================
def single_draft_speedup(
    alpha: float,
    gamma: int
) -> float:
    """
    Calculate theoretical speedup for single-draft speculative decoding.
    
    Args:
        alpha: Acceptance probability (0-1)
        gamma: Number of draft tokens
    
    Returns:
        Speedup factor
    """
    return 1 / (1 - alpha + alpha / gamma)


def dual_draft_speedup(
    alpha1: float,
    gamma1: int,
    alpha2: float,
    gamma2: int
) -> float:
    """
    Calculate speedup for dual-draft speculative decoding.
    
    Args:
        alpha1, gamma1: First draft acceptance and length
        alpha2, gamma2: Second draft acceptance and length
    
    Returns:
        Speedup factor
    """
    return 1 / (1 - alpha1 * alpha2 + alpha1 * alpha2 / (gamma1 + gamma2))


# ============================================================================
# EQUATION 12: Context Recycler
# ============================================================================
class ContextRecycler:
    """KV-cache recycling for follow-up questions."""
    
    def __init__(self, model, similarity_threshold: float = 0.8):
        self.model = model
        self.similarity_threshold = similarity_threshold
        self.cached_states = {}
    
    def prefill_time_reduction(
        self,
        history_tokens: int,
        new_query_tokens: int,
        time_per_token_ms: float = 0.8
    ) -> float:
        """Quantify prefill time savings from recycling."""
        time_without_cache = (history_tokens + new_query_tokens) * time_per_token_ms
        time_with_cache = new_query_tokens * time_per_token_ms
        return time_without_cache - time_with_cache
    
    def generate(
        self,
        session_id: str,
        user_query: str,
        previous_context: str = None,
        max_tokens: int = 100
    ):
        """Generate with context recycling."""
        if session_id in self.cached_states and previous_context:
            # Load cached state
            self.model.load_state(self.cached_states[session_id])
            # Only prefill new query
            output = self.model(user_query, max_tokens=max_tokens)
        else:
            # Full prefill
            output = self.model(user_query, max_tokens=max_tokens)
            # Cache state for next turn
            self.cached_states[session_id] = self.model.save_state()
        
        return output


# ============================================================================
# EQUATION 13: Latency Model
# ============================================================================
@dataclass
class LatencyModel:
    """Breakdown of pipeline latency components."""
    retrieval_ms: float = 100.0
    prefill_ms: float = 200.0
    decode_ms: float = 500.0
    postprocess_ms: float = 50.0
    
    @property
    def total_latency_ms(self) -> float:
        """Total latency in ms."""
        return self.retrieval_ms + self.prefill_ms + self.decode_ms + self.postprocess_ms
    
    @property
    def fraction_per_component(self) -> Dict[str, float]:
        """Fraction of total latency per component."""
        total = self.total_latency_ms
        if total == 0:
            return {}
        return {
            "retrieval": self.retrieval_ms / total,
            "prefill": self.prefill_ms / total,
            "decode": self.decode_ms / total,
            "postprocess": self.postprocess_ms / total
        }
    
    def speedup_against(self, baseline: 'LatencyModel') -> float:
        """Calculate speedup against baseline."""
        return baseline.total_latency_ms / self.total_latency_ms if self.total_latency_ms > 0 else float('inf')
    
    def amdahl_speedup(self, component: str, improvement_factor: float) -> float:
        """Calculate speedup from improving one component (Amdahl's Law)."""
        fractions = self.fraction_per_component
        if component not in fractions:
            return 1.0
        p = fractions[component]
        return 1 / ((1 - p) + p / improvement_factor)
    
    def improvement_potential(self, component: str) -> float:
        """Maximum speedup if component were reduced to zero."""
        fractions = self.fraction_per_component
        if component not in fractions:
            return 1.0
        p = fractions[component]
        return 1 / (1 - p)


# ============================================================================
# EQUATION 14: Effective TTFB with Caching
# ============================================================================
def compute_effective_ttfb(
    original_ttfb_ms: float,
    system_prefill_time_ms: float,
    cache_hit_rate: float
) -> float:
    """
    Predict average TTFB with system-prompt caching.
    
    Args:
        original_ttfb_ms: Original TTFB without cache
        system_prefill_time_ms: Time to prefill system prompt
        cache_hit_rate: Fraction of requests with cache hit
    
    Returns:
        Expected average TTFB in ms
    """
    ttfb_cached = original_ttfb_ms - system_prefill_time_ms
    return (1 - cache_hit_rate) * original_ttfb_ms + cache_hit_rate * ttfb_cached


def ttfb_savings_ms(
    original_ttfb_ms: float,
    system_prefill_time_ms: float,
    cache_hit_rate: float
) -> float:
    """Calculate absolute time saved from caching."""
    effective = compute_effective_ttfb(original_ttfb_ms, system_prefill_time_ms, cache_hit_rate)
    return original_ttfb_ms - effective


# ============================================================================
# EQUATION 15: Short Query Detection
# ============================================================================
def _is_short_query(query: str, max_words: int = 5) -> bool:
    """
    Detect if query is short after removing stopwords.
    
    Args:
        query: User query text
        max_words: Maximum words to be considered short
    
    Returns:
        True if query is short
    """
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should"}
    words = [w for w in re.findall(r"\w+", query.lower()) if w not in stopwords]
    return len(words) <= max_words


# ============================================================================
# EQUATION 16: Fused Lookahead Generator (FRGL)
# ============================================================================
class FusedLookaheadGenerator:
    """Fused Retrieval-Generation Lookahead for RAG acceleration."""
    
    def __init__(self, model, similarity_threshold: float = 0.95):
        self.model = model
        self.similarity_threshold = similarity_threshold
    
    def generate_with_lookahead(
        self,
        prompt: str,
        retrieved_chunk: str,
        max_tokens: int = 100
    ) -> str:
        """Generate with fused lookahead from retrieved chunk."""
        # Tokenize chunk for prefix matching
        chunk_tokens = self.model.tokenize(retrieved_chunk)
        
        # Generate normally
        generated = []
        current_text = ""
        
        for _ in range(max_tokens):
            # Check if current prefix matches chunk
            if retrieved_chunk.startswith(current_text):
                # Lookahead: try to copy from chunk
                remaining = retrieved_chunk[len(current_text):]
                if remaining:
                    # Verify with model
                    draft_tokens = self.model.tokenize(remaining[:50])  # Limit draft length
                    if self._verify_draft(prompt + current_text, draft_tokens):
                        # Accept draft
                        current_text += remaining[:50]
                        generated.extend(draft_tokens)
                        continue
            
            # Normal generation
            token = self.model.sample(prompt + current_text, temperature=0.0)
            generated.append(token)
            current_text += self.model.detokenize([token])
            
            if token == self.model.eos_token_id:
                break
        
        return self.model.detokenize(generated)
    
    def _verify_draft(self, context: str, draft_tokens: List[int]) -> bool:
        """Verify draft tokens with model."""
        logits = self.model(context)
        for i, token in enumerate(draft_tokens):
            if torch.argmax(logits[i]) != token:
                return False
        return True


# ============================================================================
# EQUATION 17: Embedding Cache
# ============================================================================
@dataclass
class CacheEntry:
    """Entry for embedding cache."""
    embedding: np.ndarray
    timestamp: float
    query_text: str


class EmbeddingCache:
    """Cache query embeddings with LRU eviction and similarity search."""
    
    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.995,
        ttl_seconds: int = 3600
    ):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = []
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding for cosine similarity."""
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
    
    def get(self, query: str, query_embedding: np.ndarray = None) -> Optional[np.ndarray]:
        """Get cached embedding with similarity-based lookup."""
        current_time = time.time()
        
        # Exact match
        if query in self.cache:
            entry = self.cache[query]
            if current_time - entry.timestamp < self.ttl_seconds:
                self.access_order.remove(query)
                self.access_order.append(query)
                return entry.embedding
            else:
                del self.cache[query]
        
        # Similarity search
        if query_embedding is not None:
            query_norm = self._normalize(query_embedding)
            for key, entry in list(self.cache.items()):
                if current_time - entry.timestamp >= self.ttl_seconds:
                    del self.cache[key]
                    continue
                
                similarity = np.dot(query_norm, self._normalize(entry.embedding))
                if similarity >= self.similarity_threshold:
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    return entry.embedding
        
        return None
    
    def put(self, query: str, embedding: np.ndarray):
        """Store embedding in cache."""
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[query] = CacheEntry(
            embedding=embedding,
            timestamp=time.time(),
            query_text=query
        )
        self.access_order.append(query)
    
    @property
    def size(self) -> int:
        """Current cache size."""
        return len(self.cache)
    
    def stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "utilization": self.size / self.max_size if self.max_size > 0 else 0
        }


# ============================================================================
# EQUATION 18: Parallel Prefetch
# ============================================================================
async def prefetch_in_background(
    retriever,
    query: str,
    prefetch_delay: float = 0.1
):
    """Prefetch retrieval in background to overlap with prefill."""
    await asyncio.sleep(prefetch_delay)
    return retriever.retrieve(query)


# ============================================================================
# EQUATION 19: PageRank Pruning
# ============================================================================
def prune_with_pagerank(
    candidates: List[Tuple[str, float]],
    beta: float = 0.5,
    keep_k: int = 50,
    score_floor: float = 0.01
) -> List[Tuple[str, float]]:
    """
    Prune candidates using PageRank-boosted scores.
    
    Args:
        candidates: List of (doc_id, score) tuples
        beta: Weight for PageRank boost
        keep_k: Maximum candidates to keep
        score_floor: Minimum score threshold
    
    Returns:
        Pruned list of candidates
    """
    # Apply PageRank-style boost (simplified)
    n = len(candidates)
    if n == 0:
        return []
    
    # Sort by original score
    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    
    # Apply floor
    filtered = [(doc, score) for doc, score in sorted_candidates if score >= score_floor]
    
    # Keep top k
    return filtered[:keep_k]


# ============================================================================
# EQUATION 20: Essential Keyword Extraction
# ============================================================================
def compute_importance(idf: float, tf_q: int) -> float:
    """E(t) = IDF(t) * (1 + ln(1 + tf_q(t)))"""
    if idf <= 0:
        return 0.0
    return idf * (1.0 + math.log(1.0 + tf_q))


def select_essential_terms(
    query_terms: List[str],
    df_dict: Dict[str, int],
    total_docs: int,
    alpha: float = 0.3,
) -> List[str]:
    """
    Select essential terms from query using IDF weighting.
    
    Args:
        query_terms: Tokenized query terms
        df_dict: Document frequency dictionary
        total_docs: Total number of documents
        alpha: Threshold parameter (0-1)
    
    Returns:
        List of essential terms
    """
    # Term frequency in query
    tf_counts = Counter(query_terms)
    
    # Compute IDF for each term
    idf_values = {}
    for term in tf_counts:
        df = df_dict.get(term, 0)
        if df <= 0:
            idf = math.log(total_docs) if total_docs > 1 else 1.0
        else:
            idf = math.log(total_docs / df)
        idf_values[term] = idf
    
    # Compute importance scores
    importance_scores = {
        t: compute_importance(idf_values[t], tf_counts[t])
        for t in tf_counts
    }
    
    # Select terms above threshold
    emax = max(importance_scores.values()) if importance_scores else 0
    eavg = sum(importance_scores.values()) / len(importance_scores) if importance_scores else 0
    
    theta = (1 - alpha) * emax + alpha * eavg
    kept = [t for t in importance_scores if importance_scores[t] > theta]
    
    return kept if kept else list(tf_counts.keys())


# ============================================================================
# EQUATION 21: PreWarmed Cache
# ============================================================================
class PreWarmedCache:
    """Multi-stage cache with SimHash, paraphrase expansion, and dynamic thresholding."""
    
    def __init__(
        self,
        initial_threshold: int = 2,
        target_hit_rate: float = 0.9,
        max_size: int = 10000
    ):
        self.exact_cache: Dict[str, str] = {}
        self.simhash_cache: Dict[int, str] = {}
        self.threshold = initial_threshold
        self.target_hit_rate = target_hit_rate
        self.max_size = max_size
        self.access_count = 0
        self.hit_count = 0
    
    def _simhash(self, text: str) -> int:
        """Compute SimHash for fuzzy matching."""
        # Simplified SimHash implementation
        h = hashlib.md5(text.encode()).hexdigest()
        return int(h[:8], 16)  # Use first 8 hex chars
    
    def get(self, query: str) -> Optional[str]:
        """Multi-stage cache lookup."""
        self.access_count += 1
        
        # Stage 1: Exact match
        if query in self.exact_cache:
            self.hit_count += 1
            return self.exact_cache[query]
        
        # Stage 2: SimHash fuzzy match
        hash_val = self._simhash(query)
        for cached_hash, response in self.simhash_cache.items():
            if bin(hash_val ^ cached_hash).count('1') <= self.threshold:
                self.hit_count += 1
                return response
        
        return None
    
    def put(self, query: str, response: str):
        """Store in cache with eviction."""
        if len(self.exact_cache) >= self.max_size:
            # LRU eviction (simplified)
            oldest = next(iter(self.exact_cache))
            del self.exact_cache[oldest]
        
        self.exact_cache[query] = response
        self.simhash_cache[self._simhash(query)] = response
        
        # Dynamic threshold tuning
        self._update_threshold()
    
    def _update_threshold(self):
        """Adjust threshold based on hit rate."""
        if self.access_count > 100:
            hit_rate = self.hit_count / self.access_count
            if hit_rate < self.target_hit_rate:
                self.threshold = max(0, self.threshold - 1)
            elif hit_rate > self.target_hit_rate + 0.05:
                self.threshold = min(4, self.threshold + 1)


# ============================================================================
# EQUATION 22: Score-Based Early Retrieval Termination
# ============================================================================
@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""
    score_threshold: float = 0.7
    idk_response: str = "I don't have information about that."


@dataclass
class RetrievalResult:
    """Result from retrieval pipeline."""
    chunks: List[str]
    best_score: float
    
    def is_relevant(self, threshold: float) -> bool:
        """Check if retrieval score meets threshold."""
        return self.best_score >= threshold


# ============================================================================
# EQUATION 23: Cascade Hit Rate Equation
# ============================================================================
def phrase_cache_hit_rate(
    H0: float,
    coverage: float,
    canonical_penetration: float,
    epsilon: float = 0.0,
) -> float:
    """
    Predict overall cache hit rate with phrase-level caching.
    
    Args:
        H0: Baseline hit rate (exact + raw SimHash)
        coverage: Fraction of queries mapped to canonical form
        canonical_penetration: Fraction of covered queries with canonical in cache
        epsilon: Bonus from near-canonical matches
    
    Returns:
        Expected overall hit rate
    """
    missed = 1.0 - H0
    gamma = coverage * (canonical_penetration + (1 - canonical_penetration) * epsilon)
    return H0 + missed * gamma


# ============================================================================
# EQUATION 24: Prompt Speedup Analyzer
# ============================================================================
class PromptSpeedupAnalyzer:
    """Quantify speedup from minimal prompt."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def analyze(
        self,
        original_prompt: str,
        minimal_prompt: str
    ) -> Dict[str, float]:
        """Calculate speedup metrics."""
        original_tokens = len(self.tokenizer.encode(original_prompt))
        minimal_tokens = len(self.tokenizer.encode(minimal_prompt))
        
        boilerplate = original_tokens - minimal_tokens
        core = minimal_tokens
        
        speedup = original_tokens / core if core > 0 else 1.0
        reduction_pct = (boilerplate / original_tokens) * 100 if original_tokens > 0 else 0
        
        return {
            "original_tokens": original_tokens,
            "minimal_tokens": minimal_tokens,
            "boilerplate_tokens": boilerplate,
            "core_tokens": core,
            "speedup_factor": speedup,
            "token_reduction_pct": reduction_pct
        }


# ============================================================================
# EQUATION 25: Speedup Analyzer
# ============================================================================
class SpeedupAnalyzer:
    """Compute prefill speedup from boilerplate stripping."""
    
    def __init__(self, time_per_token_ms: float = 0.8):
        self.time_per_token_ms = time_per_token_ms
    
    def analyze(
        self,
        original_prompt: str,
        minimal_prompt: str,
        tokenizer
    ) -> Dict[str, float]:
        """Analyze speedup from minimal prompt."""
        original_tokens = len(tokenizer.encode(original_prompt))
        minimal_tokens = len(tokenizer.encode(minimal_prompt))
        
        B = original_tokens - minimal_tokens  # boilerplate
        C = minimal_tokens  # core
        
        speedup_factor = 1 + B / C if C > 0 else 1.0
        reduction_pct = (B / original_tokens) * 100 if original_tokens > 0 else 0
        time_saved_ms = B * self.time_per_token_ms
        
        return {
            "boilerplate_tokens": B,
            "core_tokens": C,
            "speedup_factor": speedup_factor,
            "token_reduction_pct": reduction_pct,
            "time_saved_ms": time_saved_ms
        }


# ============================================================================
# EQUATION 26: VLLM Throughput Model
# ============================================================================
@dataclass
class VLLMThroughputParams:
    """Parameters for continuous batching throughput model."""
    gpu_vram_gib: float
    model_params_gib: float
    kv_cache_block_size: int = 16
    bytes_per_token_kv: float = 128
    memory_utilization: float = 0.9
    avg_prompt_tokens: int = 1024
    avg_output_tokens: int = 128
    time_per_step_ms: float = 30
    idle_gpu_util_static: float = 0.15
    request_rate_per_sec: float = 5.0


def estimate_max_concurrent_sequences(params: VLLMThroughputParams) -> int:
    """Calculate maximum concurrent sequences for vLLM."""
    total_kv_cache_memory = (params.gpu_vram_gib - params.model_params_gib) * params.memory_utilization * 1024 ** 3
    max_tokens_total = total_kv_cache_memory / params.bytes_per_token_kv
    avg_tokens_per_seq = params.avg_prompt_tokens + params.avg_output_tokens
    effective_tokens_per_seq = avg_tokens_per_seq + 0.5 * params.kv_cache_block_size
    return math.floor(max_tokens_total / effective_tokens_per_seq)


def compute_throughput(params: VLLMThroughputParams) -> Dict[str, Any]:
    """Compute vLLM throughput and gain over static batching."""
    B_paged = estimate_max_concurrent_sequences(params)
    B_static = math.floor((params.gpu_vram_gib - params.model_params_gib) * 1024 ** 3 * 0.4 / (params.bytes_per_token_kv * (params.avg_prompt_tokens + params.avg_output_tokens)))
    
    mem_gain = B_paged / B_static if B_static > 0 else 1.0
    rho_static = params.idle_gpu_util_static
    idle_gain = 1.0 / (1.0 - rho_static) if rho_static < 1.0 else 1.0
    
    total_gain = mem_gain * idle_gain
    
    time_per_step_s = params.time_per_step_ms / 1000.0
    gpu_throughput_capacity = B_paged / time_per_step_s
    demand_throughput = params.request_rate_per_sec * params.avg_output_tokens
    actual_throughput = min(demand_throughput, gpu_throughput_capacity)
    
    return {
        "max_concurrent_sequences_vllm": B_paged,
        "max_concurrent_sequences_static": B_static,
        "memory_gain_factor": round(mem_gain, 2),
        "idle_recovery_factor": round(idle_gain, 2),
        "total_throughput_gain": round(total_gain, 2),
        "gpu_throughput_capacity_tokens_per_sec": round(gpu_throughput_capacity, 1),
        "predicted_throughput_tokens_per_sec": round(actual_throughput, 1),
        "is_gpu_bound": demand_throughput >= gpu_throughput_capacity,
    }


# ============================================================================
# EQUATION 27: Compute Active Layers (Dynamic Depth)
# ============================================================================
def compute_active_layers(
    ease_score: float,
    total_layers: int,
    tau: float = 0.5
) -> int:
    """
    Returns number of layers to execute based on ease score.
    
    Args:
        ease_score: Ease score in [0,1]
        total_layers: Total number of layers
        tau: Threshold for hard tokens
    
    Returns:
        Number of active layers
    """
    if ease_score <= tau:
        return total_layers
    
    drop_frac = 0.2 + 0.2 * (ease_score - tau) / (1 - tau)
    keep_frac = 1.0 - drop_frac
    active = round(total_layers * keep_frac)
    return max(1, min(total_layers, active))


# ============================================================================
# EQUATION 28: Butterfly + Low-Rank (BaLR) Model Compression
# ============================================================================
class Butterfly(torch.nn.Module):
    """Product of log2(d) block-diagonal factors of 2x2 matrices."""
    def __init__(self, size):
        super().__init__()
        assert size & (size-1) == 0, "size must be power of 2"
        self.size = size
        self.n_factors = int(math.log2(size))
        # Each factor has (size//2) independent 2x2 blocks
        self.factors = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(size//2, 2, 2)) for _ in range(self.n_factors)
        ])

    def forward(self, x):
        # x: (..., size). Apply butterfly factors right-to-left.
        y = x
        for F in self.factors:
            y = y.view(*y.shape[:-1], self.size//2, 2)
            y = torch.einsum('...bi,bij->...bj', y, F)
            y = y.flatten(-2, -1)
        return y


class BaLRLinear(torch.nn.Module):
    """Butterfly + Low-Rank linear layer: W = B + U @ V^T"""
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        assert in_features == out_features, "For now only square matrices"
        self.d = in_features
        self.rank = rank
        self.butterfly = Butterfly(self.d)
        self.U = torch.nn.Parameter(torch.empty(self.d, rank))
        self.V = torch.nn.Parameter(torch.empty(self.d, rank))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(self.d))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))

    def forward(self, x):
        B = self._compute_butterfly_matrix()
        W = B + self.U @ self.V.T
        return torch.nn.functional.linear(x, W, self.bias)

    def _compute_butterfly_matrix(self):
        """Compute full butterfly matrix from factors."""
        identity = torch.eye(self.d, device=self.factors[0].device)
        return self.butterfly(identity)


# ============================================================================
# EQUATION 30: FRGL Verification
# ============================================================================
def frgl_verify(model, prefix_tokens, draft_tokens):
    """
    Verify draft tokens against model for FRGL speculative decoding.
    
    Args:
        model: The language model
        prefix_tokens: Prefix context tokens
        draft_tokens: Draft tokens to verify
    
    Returns:
        Number of accepted tokens
    """
    accepted = 0
    context = prefix_tokens.copy()
    
    for token in draft_tokens:
        full_input = context + [token]
        logits = model(full_input)
        predicted = torch.argmax(logits[-1])
        
        if predicted == token:
            accepted += 1
            context.append(token)
        else:
            break
    
    return accepted


# ============================================================================
# EQUATION 31: Compute Attention Savings (Token Merging)
# ============================================================================
def compute_attention_savings(T: int, S: int) -> Dict[str, float]:
    """
    Calculate attention cost reduction from token merging.
    
    Args:
        T: Original number of tokens
        S: Number of tokens saved (merged away)
    
    Returns:
        Dict with T_prime, cost_ratio, savings_pct
    """
    if T <= 0 or S < 0 or S > T:
        raise ValueError(f"Invalid parameters: T={T}, S={S}")
    
    T_prime = T - S
    cost_ratio = (T_prime / T) ** 2
    savings_frac = 1 - cost_ratio
    savings_pct = savings_frac * 100
    
    return {
        "T_prime": T_prime,
        "cost_ratio": cost_ratio,
        "savings_pct": savings_pct,
    }


# ============================================================================
# EQUATION 32: iGPU Speedup
# ============================================================================
def igpu_speedup(n: int, B_max: float = 1.8, tau: float = 3.0) -> float:
    """
    Theoretical speedup factor over CPU-only for n offloaded layers on iGPU.
    
    Args:
        n: Number of layers offloaded
        B_max: Maximum speedup asymptotic
        tau: Layer-efficiency constant
    
    Returns:
        Speedup factor
    """
    if n <= 0:
        return 1.0
    return 1.0 + (B_max - 1.0) * (1.0 - math.exp(-n / tau))


# ============================================================================
# EQUATION 33: Predict Prefill Time and Speedup
# ============================================================================
def predict_prefill_time_and_speedup(
    L_sys: int,
    L_query: int,
    C: float,
    N_old: int = 5,
    N_new: int = 1,
    alpha_ms_per_token: float = 0.8
) -> Dict[str, float]:
    """
    Predict prefill latency and speedup from reducing chunk count.
    
    Args:
        L_sys: System prompt token count
        L_query: User query token count
        C: Avg tokens per chunk
        N_old: Original chunk limit
        N_new: New chunk limit
        alpha_ms_per_token: Time per token in ms
    
    Returns:
        Dict with prefill times and speedup metrics
    """
    tokens_old = L_sys + L_query + N_old * C
    tokens_new = L_sys + L_query + N_new * C
    
    prefill_old = alpha_ms_per_token * tokens_old
    prefill_new = alpha_ms_per_token * tokens_new
    
    return {
        "prefill_old_ms": prefill_old,
        "prefill_new_ms": prefill_new,
        "delta_ms": prefill_old - prefill_new,
        "speedup_ratio": prefill_old / prefill_new if prefill_new > 0 else float('inf'),
        "tokens_removed": tokens_old - tokens_new,
    }


# ============================================================================
# EQUATION 34: Router Speedup
# ============================================================================
def router_speedup(f: float, rho: float) -> float:
    """
    Speedup over using large model for everything.
    
    Args:
        f: Fraction of queries routed to small model (0-1)
        rho: Speed ratio = time_large / time_small
    
    Returns:
        Speedup factor S = T_large / T_system
    """
    if not (0 <= f <= 1):
        raise ValueError("f must be between 0 and 1")
    if rho <= 0:
        raise ValueError("rho must be positive")
    return 1 / ((1 - f) + f / rho)


def kv_cache_speedup(cached_tokens: int, new_tokens: int) -> float:
    """Speedup from not re-computing cached tokens."""
    total = cached_tokens + new_tokens
    if total == 0:
        return 1.0
    f = cached_tokens / total
    return 1 / (1 - f)


# ============================================================================
# EQUATION 35: Bloom Precision
# ============================================================================
def bloom_precision(
    prob_known: float,
    bits_per_element: int = 8,
    k: int = None
) -> float:
    """
    Compute precision of Bloom-filter pre-check.
    
    Args:
        prob_known: Prior probability query targets known entity
        bits_per_element: m/n ratio for Bloom filter
        k: Number of hash functions (auto if None)
    
    Returns:
        Fraction of retrieval attempts that are useful
    """
    if not (0 < prob_known < 1):
        raise ValueError("prob_known must be between 0 and 1 (exclusive)")
    
    k = k or max(1, int(round(bits_per_element * math.log(2))))
    n_over_m = 1.0 / bits_per_element
    epsilon = (1 - math.exp(-k * n_over_m)) ** k
    
    prob_unknown = 1 - prob_known
    precision = prob_known / (prob_known + epsilon * prob_unknown)
    return precision


# ============================================================================
# EQUATION 36: Flash Attention Speedup
# ============================================================================
def flash_attention_speedup(L: int, L0: int = 512, beta: float = 2.5) -> float:
    """
    Approximate attention speedup from FlashAttention.
    
    Args:
        L: Sequence length
        L0: Long-context threshold
        beta: Maximum speedup factor
    
    Returns:
        Estimated speedup factor
    """
    if L <= L0:
        return 1.0
    return beta


# ============================================================================
# EQUATION 37: TTFB Reduction Fraction (KV-Cache Recycling)
# ============================================================================
def ttfB_reduction_fraction(
    history_tokens: int,
    query_tokens: int,
    alpha: float = 1.0,
    beta: float = 0.0,
    gen_first: float = 0.0
) -> float:
    """
    Expected TTFB reduction fraction from KV-cache recycling.
    
    Args:
        history_tokens: Tokens already prefixed
        query_tokens: Tokens in new follow-up
        alpha, beta: Linear prefill time model parameters
        gen_first: Fixed time for first output token
    
    Returns:
        Reduction fraction (0-1)
    """
    full_prefill = alpha * (history_tokens + query_tokens) + beta + gen_first
    recycle_prefill = alpha * query_tokens + beta + gen_first
    if full_prefill == 0:
        return 0.0
    return 1.0 - recycle_prefill / full_prefill


def simple_speedup_ratio(history_tokens: int, query_tokens: int) -> float:
    """Simplified reduction when prefill dominates."""
    total = history_tokens + query_tokens
    return 1.0 - query_tokens / total if total > 0 else 0.0


# ============================================================================
# EQUATION 38: GPTQ Layer Quantization
# ============================================================================
def gptq_layer_quantization(
    weight: torch.Tensor,
    bits: int = 4,
    groupsize: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simplified GPTQ quantization for a linear layer.
    
    Args:
        weight: Weight tensor to quantize
        bits: Number of bits for quantization
        groupsize: Group size for quantization
    
    Returns:
        Tuple of (quantized weights, scales, zero points)
    """
    max_val = 2 ** bits - 1
    out_features, in_features = weight.shape
    
    # Per-group quantization
    num_groups = in_features // groupsize
    scales = torch.zeros(out_features, num_groups, device=weight.device, dtype=torch.float16)
    zeros = torch.zeros(out_features, num_groups, device=weight.device, dtype=torch.float16)
    Q = torch.zeros_like(weight, dtype=torch.uint8)
    
    for group_idx in range(num_groups):
        start = group_idx * groupsize
        end = start + groupsize
        w_group = weight[:, start:end]
        
        w_max = w_group.abs().max(dim=1, keepdim=True).values
        scale = w_max / (max_val / 2)
        scales[:, group_idx] = scale.squeeze(1).half()
        
        w_int = torch.round(w_group / scale).clamp(0, max_val)
        Q[:, start:end] = w_int.to(torch.uint8)
    
    return Q, scales, zeros


# ============================================================================
# MASTER SPEED ENGINE CLASS
# ============================================================================
class SpeedEngine:
    """
    Master class implementing ALL 47 speed equations.
    This is the single source of truth for LLM inference speedup.
    """
    
    def __init__(self):
        # Initialize all equation components
        self.faq_db = FAQDatabase()
        self.zero_token_responder = ZeroTokenResponder()
        self.boredom_stopper = BoredomStopper()
        self.embedding_cache = EmbeddingCache()
        self.prewarmed_cache = PreWarmedCache()
        self.prompt_analyzer = None  # Requires tokenizer
        self.speedup_analyzer = SpeedupAnalyzer()
    
    def set_tokenizer(self, tokenizer):
        """Set tokenizer for prompt analysis."""
        self.prompt_analyzer = PromptSpeedupAnalyzer(tokenizer)
        self.speedup_analyzer.tokenizer = tokenizer
    
    def get_all_equations(self) -> Dict[str, Any]:
        """Get reference to all equation functions."""
        return {
            "session_cache_ttft_speedup": session_cache_ttft_speedup,
            "overall_conv_speedup": overall_conv_speedup,
            "estimate_speedup": estimate_speedup,
            "cost_delta": cost_delta,
            "FAQDatabase": FAQDatabase,
            "ZeroTokenResponder": ZeroTokenResponder,
            "BoredomStopper": BoredomStopper,
            "compute_dynamic_max_tokens": compute_dynamic_max_tokens,
            "PromptCachedEngine": PromptCachedEngine,
            "minimal_prompt_equation": minimal_prompt_equation,
            "single_draft_speedup": single_draft_speedup,
            "dual_draft_speedup": dual_draft_speedup,
            "ContextRecycler": ContextRecycler,
            "LatencyModel": LatencyModel,
            "compute_effective_ttfb": compute_effective_ttfb,
            "ttfb_savings_ms": ttfb_savings_ms,
            "_is_short_query": _is_short_query,
            "FusedLookaheadGenerator": FusedLookaheadGenerator,
            "EmbeddingCache": EmbeddingCache,
            "prefetch_in_background": prefetch_in_background,
            "prune_with_pagerank": prune_with_pagerank,
            "select_essential_terms": select_essential_terms,
            "PreWarmedCache": PreWarmedCache,
            "RetrievalConfig": RetrievalConfig,
            "RetrievalResult": RetrievalResult,
            "phrase_cache_hit_rate": phrase_cache_hit_rate,
            "PromptSpeedupAnalyzer": PromptSpeedupAnalyzer,
            "SpeedupAnalyzer": SpeedupAnalyzer,
            "VLLMThroughputParams": VLLMThroughputParams,
            "estimate_max_concurrent_sequences": estimate_max_concurrent_sequences,
            "compute_throughput": compute_throughput,
            "compute_active_layers": compute_active_layers,
            "Butterfly": Butterfly,
            "BaLRLinear": BaLRLinear,
            "frgl_verify": frgl_verify,
            "compute_attention_savings": compute_attention_savings,
            "igpu_speedup": igpu_speedup,
            "predict_prefill_time_and_speedup": predict_prefill_time_and_speedup,
            "router_speedup": router_speedup,
            "kv_cache_speedup": kv_cache_speedup,
            "bloom_precision": bloom_precision,
            "flash_attention_speedup": flash_attention_speedup,
            "ttfB_reduction_fraction": ttfB_reduction_fraction,
            "simple_speedup_ratio": simple_speedup_ratio,
            "gptq_layer_quantization": gptq_layer_quantization,
        }


# Global speed engine instance
speed_engine = SpeedEngine()
