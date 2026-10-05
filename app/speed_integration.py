"""
Speed Integration Module - Implements all PDF equations for maximum speed
This module integrates the speed_engine optimization equations into the main chatbot
"""

import hashlib
import re
import time
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from collections import OrderedDict

# Import from speed_engine
try:
    from speed_engine.prefilter import normalize, ZeroTokenResponder, FAQDatabase, BloomPreFilter
    from speed_engine.cache import PreWarmedCache, simhash_64, session_cache_ttft_speedup
    from speed_engine.prompt import build_minimal_prompt, compute_dynamic_max_tokens, one_liner_speedup
    from speed_engine.generation import BoredomStopper, router_speedup, kv_cache_speedup
    from speed_engine.retrieval import RetrievalPipeline
    SPEED_ENGINE_AVAILABLE = True
except ImportError:
    SPEED_ENGINE_AVAILABLE = False
    print("Speed engine not available, using fallback implementations")


@dataclass
class SpeedOptimizationConfig:
    """Configuration for all speed optimization equations"""
    # EQ-ZERO-TOKEN
    use_zero_token: bool = True
    zero_token_phrases: Dict[str, str] = None
    
    # EQ-FAQ
    use_faq: bool = True
    faq_database: Dict[str, str] = None
    
    # EQ-BLOOM
    use_bloom: bool = True
    bloom_entities: List[str] = None
    bloom_bits_per_element: int = 8
    
    # EQ-SIMHASH & EQ-PREWARM-CACHE
    use_prewarm_cache: bool = True
    cache_max_entries: int = 50000
    simhash_threshold: int = 4
    cache_target_hit_rate: float = 0.95
    bm25_cache_score_threshold: float = 0.3
    
    # EQ-DYNAMIC-TOKENS
    use_dynamic_max_tokens: bool = True
    dynamic_tokens_base: int = 20
    dynamic_tokens_multiplier: float = 2.0
    dynamic_tokens_max_cap: int = 256
    dynamic_tokens_long_answer_cap: int = 150
    
    # EQ-MINIMAL-PROMPT
    use_minimal_prompt: bool = True
    one_liner_mode: bool = True
    
    # EQ-BOREDOM
    use_boredom_stopper: bool = True
    boredom_threshold: float = 0.15
    boredom_patience: int = 3
    boredom_min_tokens: int = 5
    
    # EQ-SESSION-TTFT
    use_session_cache: bool = True
    history_tokens: int = 100
    new_query_tokens: int = 20
    prefill_time_per_token: float = 2.0
    load_overhead_ms: float = 15.0
    
    # EQ-ROUTER
    use_model_routing: bool = False
    routing_fraction_simple: float = 0.7
    routing_speedup_ratio: float = 5.0


class SpeedOptimizationEngine:
    """
    Main engine that applies all PDF optimization equations
    for maximum chatbot speed
    """
    
    def __init__(self, config: SpeedOptimizationConfig):
        self.config = config
        self._init_components()
        
    def _init_components(self):
        """Initialize all optimization components"""
        
        # EQ-ZERO-TOKEN
        if self.config.use_zero_token and self.config.zero_token_phrases:
            if SPEED_ENGINE_AVAILABLE:
                self.zero_token = ZeroTokenResponder(self.config.zero_token_phrases)
            else:
                self.zero_token = self._fallback_zero_token()
        else:
            self.zero_token = None
            
        # EQ-FAQ
        if self.config.use_faq and self.config.faq_database:
            if SPEED_ENGINE_AVAILABLE:
                self.faq = FAQDatabase(self.config.faq_database)
            else:
                self.faq = self._fallback_faq()
        else:
            self.faq = None
            
        # EQ-BLOOM
        if self.config.use_bloom and self.config.bloom_entities:
            if SPEED_ENGINE_AVAILABLE:
                self.bloom = BloomPreFilter(
                    set(self.config.bloom_entities),
                    self.config.bloom_bits_per_element
                )
            else:
                self.bloom = None
        else:
            self.bloom = None
            
        # EQ-PREWARM-CACHE
        if self.config.use_prewarm_cache:
            if SPEED_ENGINE_AVAILABLE:
                self.cache = PreWarmedCache(
                    self.config.cache_max_entries,
                    self.config.simhash_threshold,
                    self.config.cache_target_hit_rate,
                    self.config.bm25_cache_score_threshold
                )
            else:
                self.cache = self._fallback_cache()
        else:
            self.cache = None
            
        # EQ-BOREDOM
        if self.config.use_boredom_stopper:
            self.boredom = BoredomStopper(
                self.config.boredom_threshold,
                self.config.boredom_patience,
                self.config.boredom_min_tokens
            )
        else:
            self.boredom = None
            
        # Session cache for EQ-SESSION-TTFT
        self.session_cache = {}
        self.session_cache_hits = 0
        self.session_cache_misses = 0
        
    def _fallback_zero_token(self):
        """Fallback implementation when speed_engine not available"""
        phrases = {k.lower(): v for k, v in (self.config.zero_token_phrases or {}).items()}
        return type('FallbackZeroToken', (), {
            'respond': lambda self, text: phrases.get(text.lower().strip())
        })()
        
    def _fallback_faq(self):
        """Fallback FAQ implementation"""
        faq_data = {k.lower(): v for k, v in (self.config.faq_database or {}).items()}
        return type('FallbackFAQ', (), {
            'get': lambda self, q: faq_data.get(q.lower().strip())
        })()
        
    def _fallback_cache(self):
        """Fallback cache implementation"""
        cache = OrderedDict()
        max_entries = self.config.cache_max_entries
        
        def get(key):
            return cache.get(key)
            
        def put(key, value):
            if len(cache) >= max_entries:
                cache.popitem(last=False)
            cache[key] = value
            
        return type('FallbackCache', (), {'get': get, 'put': put})()
    
    def try_fast_path(self, message: str) -> Optional[Tuple[str, str]]:
        """
        Try all fast paths in order of speed (fastest first)
        Returns (response, source) or None
        """
        # EQ-ZERO-TOKEN (fastest - O(1) dict lookup)
        if self.zero_token:
            response = self.zero_token.respond(message)
            if response:
                return response, "zero_token"
        
        # EQ-FAQ (second fastest - O(1) dict lookup)
        if self.faq:
            response = self.faq.get(message)
            if response:
                return response, "faq"
        
        # EQ-BLOOM (third - fast entity check)
        if self.bloom and not self.bloom.allows_retrieval(message):
            return "I don't have information about that entity.", "bloom_block"
        
        # EQ-PREWARM-CACHE (fourth - exact + simhash + BM25 cascade)
        if self.cache:
            cached = self.cache.get(message)
            if cached:
                return cached, "cache"
        
        # EQ-SESSION-TTFT (session KV cache speedup)
        session_key = self._get_session_key(message)
        if session_key in self.session_cache:
            self.session_cache_hits += 1
            return self.session_cache[session_key], "session_cache"
        self.session_cache_misses += 1
        
        return None
    
    def _get_session_key(self, message: str) -> str:
        """Generate session cache key using normalize"""
        if SPEED_ENGINE_AVAILABLE:
            return normalize(message)
        return message.lower().strip()
    
    def apply_prompt_optimizations(self, query: str, chunks: List[str]) -> str:
        """Apply EQ-MINIMAL-PROMPT and EQ-DYNAMIC-TOKENS"""
        
        # EQ-MINIMAL-PROMPT
        if self.config.use_minimal_prompt:
            if SPEED_ENGINE_AVAILABLE:
                prompt = build_minimal_prompt(
                    query, 
                    chunks, 
                    self.config.one_liner_mode
                )
            else:
                prompt = self._fallback_minimal_prompt(query, chunks)
        else:
            prompt = self._build_standard_prompt(query, chunks)
        
        # EQ-DYNAMIC-TOKENS
        if self.config.use_dynamic_max_tokens:
            if SPEED_ENGINE_AVAILABLE:
                max_tokens = compute_dynamic_max_tokens(
                    query,
                    self.config.dynamic_tokens_base,
                    self.config.dynamic_tokens_multiplier,
                    self.config.dynamic_tokens_max_cap,
                    self.config.dynamic_tokens_long_answer_cap
                )
            else:
                max_tokens = self._fallback_dynamic_tokens(query)
        else:
            max_tokens = 256
        
        return prompt, max_tokens
    
    def _fallback_minimal_prompt(self, query: str, chunks: List[str]) -> str:
        """Fallback minimal prompt"""
        if chunks:
            return f"{chunks[0]}\n\nQ: {query}\nA:"
        return f"Q: {query}\nA:"
    
    def _fallback_dynamic_tokens(self, query: str) -> int:
        """Fallback dynamic token calculation"""
        words = len(query.split())
        return min(20 + 2 * words, 256)
    
    def _build_standard_prompt(self, query: str, chunks: List[str]) -> str:
        """Standard prompt format"""
        if chunks:
            return f"context: {chunks[0]}\nuser: {query}\nassistant:"
        return f"user: {query}\nassistant:"
    
    def should_stop_generation(self, logprob: float) -> bool:
        """Apply EQ-BOREDOM stop condition"""
        if self.boredom:
            return self.boredom.should_stop(logprob)
        return False
    
    def calculate_session_cache_speedup(self) -> Dict:
        """Calculate EQ-SESSION-TTFT speedup metrics"""
        if SPEED_ENGINE_AVAILABLE:
            return session_cache_ttft_speedup(
                self.config.history_tokens,
                self.config.new_query_tokens,
                prefill_time_per_token=self.config.prefill_time_per_token,
                load_overhead_ms=self.config.load_overhead_ms
            )
        return {"speedup_factor": 1.0}
    
    def calculate_model_routing_speedup(self) -> float:
        """Calculate EQ-ROUTER speedup"""
        if self.config.use_model_routing and SPEED_ENGINE_AVAILABLE:
            return router_speedup(
                self.config.routing_fraction_simple,
                self.config.routing_speedup_ratio
            )
        return 1.0
    
    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        stats = {
            "session_cache_hits": self.session_cache_hits,
            "session_cache_misses": self.session_cache_misses,
            "session_cache_hit_rate": 0.0
        }
        
        total = self.session_cache_hits + self.session_cache_misses
        if total > 0:
            stats["session_cache_hit_rate"] = self.session_cache_hits / total
        
        if self.cache and SPEED_ENGINE_AVAILABLE:
            stats["prewarm_cache_hit_rate"] = self.cache.hit_rate
        
        return stats
    
    def optimize_response(self, response: str, query: str) -> str:
        """Apply response optimizations"""
        # Cache the response
        if self.cache:
            self.cache.put(query, response)
        
        # Store in session cache
        session_key = self._get_session_key(query)
        self.session_cache[session_key] = response
        
        return response


def create_max_speed_config() -> SpeedOptimizationConfig:
    """Create configuration with all speed optimizations maximized"""
    
    # Default zero-token phrases
    zero_token_phrases = {
        "hi": "Hello! How can I help you today?",
        "hello": "Hi there! How can I assist you?",
        "hey": "Hey! What can I help you with?",
        "bye": "Goodbye! Have a great day!",
        "thanks": "You're welcome!",
        "thank you": "You're welcome!",
        "ok": "Okay, got it!",
        "okay": "Okay, understood!",
    }
    
    # Default FAQ database
    faq_database = {
        "what is kv cache": "KV cache stores key-value pairs to avoid recomputing attention in transformer models.",
        "what is bm25": "BM25 is a ranking function used in information retrieval to estimate document relevance.",
        "what is attention": "Attention mechanisms allow neural networks to focus on different parts of input when producing output.",
        "what is machine learning": "Machine learning enables systems to learn from data without being explicitly programmed.",
    }
    
    # Default bloom entities
    bloom_entities = [
        "paris", "france", "london", "england", "berlin", "germany",
        "tokyo", "japan", "beijing", "china", "washington", "usa",
        "python", "javascript", "java", "c++", "rust", "go",
        "transformer", "attention", "neural network", "deep learning"
    ]
    
    return SpeedOptimizationConfig(
        use_zero_token=True,
        zero_token_phrases=zero_token_phrases,
        use_faq=True,
        faq_database=faq_database,
        use_bloom=True,
        bloom_entities=bloom_entities,
        bloom_bits_per_element=8,
        use_prewarm_cache=True,
        cache_max_entries=50000,
        simhash_threshold=4,
        cache_target_hit_rate=0.95,
        bm25_cache_score_threshold=0.3,
        use_dynamic_max_tokens=True,
        dynamic_tokens_base=20,
        dynamic_tokens_multiplier=2.0,
        dynamic_tokens_max_cap=256,
        dynamic_tokens_long_answer_cap=150,
        use_minimal_prompt=True,
        one_liner_mode=True,
        use_boredom_stopper=True,
        boredom_threshold=0.15,
        boredom_patience=3,
        boredom_min_tokens=5,
        use_session_cache=True,
        history_tokens=100,
        new_query_tokens=20,
        prefill_time_per_token=2.0,
        load_overhead_ms=15.0,
        use_model_routing=False,
        routing_fraction_simple=0.7,
        routing_speedup_ratio=5.0
    )