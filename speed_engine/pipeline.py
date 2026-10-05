"""
End-to-end chat pipeline wiring equation modules in specification order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator, Iterator, Optional

from speed_engine.cache import PreWarmedCache
from speed_engine.config import SpeedConfig
from speed_engine.generation import BoredomStopper, SimulatedCloudEngine
from speed_engine.prefilter import BloomPreFilter, FAQDatabase, ZeroTokenResponder
from speed_engine.prompt import build_minimal_prompt, build_standard_prompt, compute_dynamic_max_tokens
from speed_engine.retrieval import RetrievalPipeline, is_relevant


class Source(str, Enum):
    ZERO_TOKEN = "zero_token"
    FAQ = "faq"
    BLOOM_BLOCK = "bloom_block"
    CACHE = "cache"
    SCORE_GATE = "score_gate"
    LLM = "llm"


@dataclass
class ChatOutcome:
    text: str
    source: Source
    best_score: float = 0.0


class SpeedPipeline:
    """
    a. Pre-filter (Bloom, FAQ, ZeroToken)
    b. Cache (PreWarmedCache / SimHash)
    c. Retrieval (BM25 → Dense → PageRank → Rerank)
    d. Score gating
    e. Prompt (MinimalPrompt, dynamic tokens)
    f. Generation (simulated cloud)
    g. Cache update
    """

    def __init__(self, config: SpeedConfig):
        self.cfg = config
        self.zero_token = ZeroTokenResponder(config.zero_token_phrases) if config.use_zero_token else None
        self.faq = FAQDatabase(config.faq) if config.use_faq else None
        self.bloom = (
            BloomPreFilter(set(config.bloom_entities), config.bloom_bits_per_element)
            if config.use_bloom and config.bloom_entities
            else None
        )
        self.cache = (
            PreWarmedCache(
                config.cache_max_entries,
                config.simhash_threshold,
                config.cache_target_hit_rate,
                config.bm25_cache_score_threshold,
            )
            if config.use_prewarm_cache
            else None
        )
        docs = list(config.documents) if config.documents else [
            ("doc1", "Paris is the capital of France. France is in Europe."),
            ("doc2", "BM25 is a bag-of-words retrieval function used in search engines."),
            ("doc3", "KV cache stores key-value tensors to speed up transformer inference."),
        ]
        self.retrieval = RetrievalPipeline(
            documents=docs,
            pagerank=config.pagerank or {d[0]: 1.0 for d in docs},
            k1=config.bm25_k1,
            b=config.bm25_b,
            bm25_top_k=config.bm25_top_k,
            dense_top_k=config.dense_top_k,
            use_dynamic_index_switch=config.use_dynamic_index_switch,
            short_query_max_words=config.short_query_max_words,
            bm25_min_results=config.bm25_min_results,
            bm25_score_threshold=config.bm25_score_threshold,
            use_query_truncation=config.use_query_truncation,
            truncation_alpha=config.truncation_alpha,
            use_pagerank_prune=config.use_pagerank_prune,
            pagerank_beta=config.pagerank_beta,
            pagerank_keep_k=config.pagerank_keep_k,
            pagerank_score_floor=config.pagerank_score_floor,
            chunk_limit=config.chunk_limit,
        )
        self.engine = SimulatedCloudEngine(
            base_latency_ms=config.simulated_cloud_ms,
            ms_per_token=config.simulated_token_ms,
            one_liner_mode=config.one_liner_mode,
        )
        self.engine.warm_keepalive()
        self.boredom = BoredomStopper(
            config.boredom_threshold,
            config.boredom_patience,
            config.boredom_min_tokens,
        )
        self._last_source = Source.LLM
        self._last_score = 0.0

    def try_fast_path(self, message: str) -> Optional[tuple[str, Source]]:
        """Return (text, source) for sub-millisecond paths without streaming overhead."""
        if self.zero_token:
            z = self.zero_token.respond(message)
            if z:
                return z, Source.ZERO_TOKEN
        if self.faq:
            faq_ans = self.faq.get(message)
            if faq_ans:
                return faq_ans, Source.FAQ
        if self.bloom and not self.bloom.allows_retrieval(message):
            return self.cfg.idk_response, Source.BLOOM_BLOCK
        if self.cache:
            cached = self.cache.get(message)
            if cached:
                return cached, Source.CACHE
        return None

    def resolve(self, message: str) -> ChatOutcome:
        text = "".join(self._stream_tokens(message))
        return ChatOutcome(text=text.strip(), source=self._last_source, best_score=self._last_score)

    def stream_events(self, message: str) -> Iterator[str]:
        for token in self._stream_tokens(message):
            yield f"data: {json.dumps({'type': 'token', 'value': token})}\n\n"
        yield f"data: {json.dumps({'type': 'done', 'source': self._last_source.value})}\n\n"

    async def stream_events_async(self, message: str) -> AsyncIterator[str]:
        for line in self.stream_events(message):
            yield line

    def _stream_tokens(self, message: str) -> Iterator[str]:
        if self.zero_token:
            z = self.zero_token.respond(message)
            if z:
                self._last_source = Source.ZERO_TOKEN
                self._last_score = 1.0
                yield z
                return

        if self.faq:
            faq_ans = self.faq.get(message)
            if faq_ans:
                self._last_source = Source.FAQ
                self._last_score = 1.0
                yield faq_ans
                return

        if self.bloom and not self.bloom.allows_retrieval(message):
            self._last_source = Source.BLOOM_BLOCK
            self._last_score = 0.0
            yield self.cfg.idk_response
            return

        if self.cache:
            cached = self.cache.get(message)
            if cached:
                self._last_source = Source.CACHE
                self._last_score = 1.0
                yield cached
                return

        result = self.retrieval.retrieve(message)
        self._last_score = result.best_score

        threshold = self.cfg.score_threshold if self.cfg.use_score_gating else None
        if not is_relevant(result.best_score, threshold):
            self._last_source = Source.SCORE_GATE
            yield self.cfg.idk_response
            return

        chunks = [c[1] for c in result.chunks]
        if self.cfg.minimal_prompt:
            prompt = build_minimal_prompt(message, chunks, self.cfg.one_liner_mode)
        else:
            prompt = build_standard_prompt(message, chunks)

        max_tokens = compute_dynamic_max_tokens(
            message,
            self.cfg.dynamic_tokens_base,
            self.cfg.dynamic_tokens_multiplier,
            self.cfg.dynamic_tokens_max_cap,
            self.cfg.dynamic_tokens_long_answer_cap,
        )

        self._last_source = Source.LLM
        response_parts: list[str] = []
        for tok in self.engine.generate_tokens(prompt, max_tokens):
            response_parts.append(tok + " ")
            yield tok + " "

        full = "".join(response_parts).strip()
        if self.cache and full:
            self.cache.put(message, full)
