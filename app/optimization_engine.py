"""
Optimization Engine - Pure Performance Layer
Contains all optimization logic: cache, retrieval, zero-token, FAQ, embeddings.
This is the engine that makes the chatbot fast, not the chatbot itself.
"""

import os
import json
import asyncio
import logging
from typing import List, Dict, Optional, Any, AsyncGenerator
from dataclasses import dataclass

from .identity import ConversationState, deterministic_chat_identity_key
from .config import AdvancedConfig
from .retrieval import (
    PromptBuilder, RetrievalConfig, RetrievalPipeline,
    MinimalPromptBuilder, ZeroTokenResponder, FAQDatabase,
)
from .cache import PreWarmedCache
from .prompt import CloudPromptBuilder 
from .retrieval import _ensure_sentence_transformers, SENTENCE_TRANSFORMERS_AVAILABLE

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from the optimization engine."""
    cached_response: Optional[str] = None
    retrieval_sources: List[Dict] = None
    retrieval_latency_ms: Optional[float] = None
    prompt_builder: Optional[PromptBuilder] = None
    should_use_llm: bool = True
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.retrieval_sources is None:
            self.retrieval_sources = []
        if self.metrics is None:
            self.metrics = {}


class OptimizationEngine:
    """
    Pure optimization engine that handles all performance optimizations.
    This includes caching, retrieval, zero-token responses, FAQ, and embeddings.
    The chatbot uses this engine to get fast responses.
    """
    
    def __init__(
        self,
        documents: List[str],
        doc_ids: List[int],
        graph_edges: List[tuple] = None,
        system_prompt: str = "You are a helpful AI assistant.",
        adv_config: Optional[AdvancedConfig] = None
    ):
        """
        Initialize the optimization engine.
        
        Args:
            documents: List of documents for retrieval
            doc_ids: List of document IDs
            graph_edges: Graph edges for PageRank
            system_prompt: Default system prompt
            adv_config: AdvancedConfig instance (uses env vars if None)
        """
        if graph_edges is None:
            graph_edges = []
        
        self.adv = adv_config or AdvancedConfig.from_env()
        self.system_prompt = system_prompt
        
        # Retrieval pipeline
        self.retrieval = RetrievalPipeline(
            documents=documents,
            doc_ids=doc_ids,
            cfg=RetrievalConfig(
                embedding_model_name=os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
                cross_encoder_name=os.getenv("CROSS_ENCODER_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                use_pagerank_boost=self.adv.use_pagerank_boost,
                pagerank_boost_gamma=self.adv.pagerank_boost_gamma,
                use_pagerank_prune=self.adv.use_pagerank_prune,
                pagerank_prune_beta=self.adv.pagerank_prune_beta,
                pagerank_prune_keep_k=self.adv.pagerank_prune_keep_k,
                pagerank_prune_score_floor=self.adv.pagerank_prune_score_floor,
                use_dynamic_index_switch=self.adv.use_dynamic_index_switch,
                short_query_max_words=self.adv.short_query_max_words,
                use_essential_keywords=self.adv.use_essential_keywords,
                essential_keywords_alpha=self.adv.essential_keywords_alpha,
                bm25_top_k=100,
                dense_top_k=200,
                rerank_top_k=5,
                chunk_limit=self.adv.chunk_limit,
                score_threshold=self.adv.score_threshold,
                use_query_embedding_cache=os.getenv("USE_QUERY_EMBEDDING_CACHE", "false").lower() in ("true", "1", "yes"),
            ),
            graph_edges=graph_edges,
        )
        
        # Prompt builders
        self.prompt_builder_standard = PromptBuilder(system_prompt=system_prompt)
        self.prompt_builder_minimal = MinimalPromptBuilder()
        
        # Cache - use simple dict for exact hash-based caching
        self.cache = {}
        
        # Zero-token responder
        self.zero_token = ZeroTokenResponder() if self.adv.zero_token_enabled else None
        
        # FAQ database
        self.faq = FAQDatabase.from_json(self.adv.faq_db_path) if self.adv.faq_enabled and os.path.exists(self.adv.faq_db_path) else FAQDatabase()
        
        # Pre-warm FAQ cache
        if self.adv.faq_enabled and self.faq:
            for question, answer in self.faq._data.items():
                if isinstance(self.cache, dict):
                    import hashlib
                    cache_key = hashlib.sha256(f"{question}:42".encode()).hexdigest()
                    self.cache[cache_key] = answer
        
        # Conversation state (for identity keys)
        self.sessions: Dict[str, ConversationState] = {}
    
    def _get_state(self, session_id: str) -> ConversationState:
        """Get or create conversation state for identity key generation."""
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationState()
        return self.sessions[session_id]
    
    def check_cache(self, query: str, session_id: str, seed: int = 42) -> Optional[str]:
        """
        Check if query is in cache.
        
        Args:
            query: User query
            session_id: Session ID for identity key
            seed: Seed for deterministic identity key
            
        Returns:
            Cached response if found, None otherwise
        """
        # Use simple query-based cache key (without conversation state) for better cache hit rate
        # This allows repeated queries to be cached even in different conversation contexts
        import hashlib
        cache_key = hashlib.sha256(f"{query}:{seed}".encode()).hexdigest()
        cached = self.cache.get(cache_key) if isinstance(self.cache, dict) else None
        
        print(f"[CACHE DEBUG] check_cache: query='{query[:30]}...', seed={seed}, cache_key={cache_key[:16]}..., cached={'HIT' if cached else 'MISS'}")
        
        if cached:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return cached
        
        return None
    
    def check_faq(self, query: str) -> Optional[str]:
        """
        Check if query matches FAQ.
        
        Args:
            query: User query
            
        Returns:
            FAQ answer if found, None otherwise
        """
        if self.faq:
            answer = self.faq.get(query)
            if answer:
                logger.debug(f"FAQ hit for query: {query[:50]}...")
                return answer
        return None
    
    def check_zero_token(self, query: str) -> Optional[str]:
        """
        Check if query can be answered without LLM (zero-token).
        
        Args:
            query: User query
            
        Returns:
            Zero-token response if applicable, None otherwise
        """
        if self.zero_token:
            response = self.zero_token.respond(query)
            if response:
                logger.debug(f"Zero-token response for query: {query[:50]}...")
                return response
        return None
    
    def run_retrieval(self, query: str) -> tuple[List[Dict], float]:
        """
        Run retrieval pipeline for the query.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (sources, retrieval_latency_ms)
        """
        import time
        start_time = time.time()
        
        sources = self.retrieval.retrieve(query)
        
        latency_ms = (time.time() - start_time) * 1000
        logger.debug(f"Retrieval completed in {latency_ms:.2f}ms, found {len(sources)} sources")
        
        return sources, latency_ms
    
    def get_prompt_builder(self, use_minimal: bool = False) -> PromptBuilder:
        """
        Get the appropriate prompt builder.
        
        Args:
            use_minimal: Whether to use minimal prompt builder
            
        Returns:
            PromptBuilder instance
        """
        return self.prompt_builder_minimal if use_minimal else self.prompt_builder_standard
    
    def put_cache(self, query: str, response: str, session_id: str, seed: int = 42):
        """
        Put response in cache.
        
        Args:
            query: User query
            response: Generated response
            session_id: Session ID for identity key
            seed: Seed for deterministic identity key
        """
        # Use simple query-based cache key (without conversation state) to match check_cache
        import hashlib
        cache_key = hashlib.sha256(f"{query}:{seed}".encode()).hexdigest()
        if isinstance(self.cache, dict):
            self.cache[cache_key] = response
            print(f"[CACHE DEBUG] put_cache: query='{query[:30]}...', seed={seed}, cache_key={cache_key[:16]}..., response_len={len(response)}")
            print(f"[CACHE DEBUG] Cache size now: {len(self.cache)} entries")
    
    def predict_followup_queries(self, query: str) -> List[str]:
        """
        Predict likely follow-up queries for pre-computing embeddings.
        
        Args:
            query: Current query
            
        Returns:
            List of predicted follow-up queries
        """
        followups = []
        query_lower = query.lower()
        
        # Simple heuristics for common follow-up patterns
        if "what is" in query_lower or "define" in query_lower:
            topic = query.replace("what is", "").replace("define", "").strip()
            followups.extend([
                f"tell me more about {topic}",
                f"how does {topic} work",
                f"why is {topic} important"
            ])
        elif "how" in query_lower:
            topic = query.replace("how", "").strip()
            followups.extend([
                f"explain {topic} in detail",
                f"what are the benefits of {topic}",
                f"examples of {topic}"
            ])
        elif "why" in query_lower:
            topic = query.replace("why", "").strip()
            followups.extend([
                f"what causes {topic}",
                f"how to prevent {topic}",
                f"history of {topic}"
            ])
        else:
            # Generic follow-ups for any query
            followups.extend([
                f"tell me more about {query}",
                f"explain {query} in simple terms",
                f"what are the key points about {query}"
            ])
        
        return followups[:5]  # Limit to top 5 predictions
    
    async def precompute_embeddings_async(self, queries: List[str]):
        """
        Pre-compute embeddings for predicted follow-up queries in background.
        
        Args:
            queries: List of queries to pre-compute embeddings for
        """
        try:
            _ensure_sentence_transformers()
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                return
            
            # Get embedding model from retrieval pipeline
            embedding_model = self.retrieval.dense.model if hasattr(self.retrieval, 'dense') else None
            if embedding_model is None:
                return
            
            # Compute embeddings for all predicted queries
            embeddings = embedding_model.encode(queries, show_progress_bar=False)
            
            # Store in cache
            for query, embedding in zip(queries, embeddings):
                self.cache.put_embedding(query, embedding)
                
        except Exception as e:
            # Log but don't fail - this is a background optimization
            logger.debug(f"Embedding pre-computation skipped: {e}")
    
    async def optimize_query(
        self,
        query: str,
        session_id: str,
        enable_retrieval: bool = True,
        seed: int = 42
    ) -> OptimizationResult:
        print(f"[OPT ENGINE] optimize_query called: query='{query[:30]}...', session_id={session_id}, seed={seed}")
        """
        Run the full optimization pipeline for a query.
        
        Args:
            query: User query
            session_id: Session ID
            enable_retrieval: Whether to run retrieval
            seed: Seed for deterministic identity key
            
        Returns:
            OptimizationResult with cached response, sources, prompt builder, etc.
        """
        result = OptimizationResult()
        
        # 1. Check cache
        cached = self.check_cache(query, session_id, seed)
        if cached:
            result.cached_response = cached
            result.should_use_llm = False
            result.metrics['cache_hit'] = True
            return result
        
        # 2. Check FAQ
        faq_answer = self.check_faq(query)
        if faq_answer:
            result.cached_response = faq_answer
            result.should_use_llm = False
            result.metrics['faq_hit'] = True
            # Cache FAQ response
            self.put_cache(query, faq_answer, session_id, seed)
            return result
        
        # 3. Check zero-token
        zero_token_response = self.check_zero_token(query)
        if zero_token_response:
            result.cached_response = zero_token_response
            result.should_use_llm = False
            result.metrics['zero_token_hit'] = True
            # Cache zero-token response
            self.put_cache(query, zero_token_response, session_id, seed)
            return result
        
        # 4. Run retrieval if enabled
        if enable_retrieval:
            sources, latency_ms = self.run_retrieval(query)
            result.retrieval_sources = sources
            result.retrieval_latency_ms = latency_ms
            result.metrics['retrieval_latency_ms'] = latency_ms
            result.metrics['num_sources'] = len(sources)
        
        # 5. Determine prompt builder
        use_minimal = self.adv.minimal_prompt
        result.prompt_builder = self.get_prompt_builder(use_minimal)
        result.should_use_llm = True
        
        # 6. Predict follow-up queries for pre-computation (background)
        followups = self.predict_followup_queries(query)
        if followups:
            asyncio.create_task(self.precompute_embeddings_async(followups))
        
        return result
