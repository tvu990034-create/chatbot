"""RAG pipeline — knowledge base loading and retrieval."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

from speed_engine.prefilter import FAQDatabase, ZeroTokenResponder
from speed_engine.prompt import build_minimal_prompt, compute_dynamic_max_tokens
from speed_engine.retrieval import RetrievalPipeline

from local_chatbot.config import ChatbotConfig

# Add to sys.path once at module load (Bug #2 fix)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


def _load_documents(kb_path: str, force_reload: bool = False) -> List[Tuple[str, str]]:
    """Load documents from the knowledge base directory."""
    # Bug #11 fix: Allow cache invalidation
    cache_key = kb_path
    if force_reload and cache_key in RAGPipeline._document_cache:
        del RAGPipeline._document_cache[cache_key]

    documents: List[Tuple[str, str]] = []

    try:
        from app.data_ingestion import DataIngestion

        di = DataIngestion(knowledge_base_path=kb_path)
        texts, ids = di.load_all_documents()
        documents = list(zip(ids, texts))
    except Exception:
        pass

    if not documents:
        documents = _default_documents()

    return documents


def _default_documents() -> List[Tuple[str, str]]:
    return [
        ("doc1", "Paris is the capital of France."),
        ("doc2", "BM25 is a bag-of-words retrieval function used in search engines."),
        ("doc3", "KV cache stores key-value tensors from attention layers to speed up transformer inference."),
        ("doc4", "Quantization reduces model precision to lower bit widths for faster inference."),
        ("doc5", "Transformers use self-attention to process sequential data in parallel."),
        ("doc6", "Embeddings are dense vector representations that capture semantic meaning."),
        ("doc7", "Local inference runs AI models on your machine without cloud dependencies."),
    ]


class RAGPipeline:
    """Retrieval-augmented generation with fast-path shortcuts."""

    # Class-level document cache (Bug #4 fix)
    _document_cache: Dict[str, List[Tuple[str, str]]] = {}

    @classmethod
    def clear_document_cache(cls) -> None:
        """Clear the document cache to force reload (Bug #11 fix)."""
        cls._document_cache.clear()

    def __init__(self, cfg: ChatbotConfig):
        self.cfg = cfg
        # Check cache first (Bug #4 fix)
        cache_key = cfg.knowledge_base_path
        if cache_key in self._document_cache:
            documents = self._document_cache[cache_key]
        else:
            documents = _load_documents(cfg.knowledge_base_path, force_reload=False)
            self._document_cache[cache_key] = documents

        self.retrieval = RetrievalPipeline(
            documents=documents,
            pagerank={doc_id: 1.0 for doc_id, _ in documents},
            chunk_limit=cfg.chunk_limit,
            use_dynamic_index_switch=True,
            use_query_truncation=True,
            use_pagerank_prune=True,
        )

        # Query cache with LRU eviction (Bug #7 & #10 fix)
        from collections import OrderedDict
        self._retrieval_cache: OrderedDict[str, Tuple[str, int, list]] = OrderedDict()
        self._cache_max_entries = 1000

        self.faq = FAQDatabase({
            "What is BM25?": "BM25 is a bag-of-words retrieval function used in search engines.",
            "What is KV cache?": "KV cache stores computed attention key-value pairs to speed up generation.",
            "What is quantization?": "Quantization reduces model precision to lower bit widths for faster inference.",
            "What is the capital of France?": "The capital of France is Paris.",
        }) if cfg.use_faq else None

        self.zero_token = ZeroTokenResponder({
            "hi": "Hello! How can I help you today?",
            "hello": "Hi there! What would you like to know?",
            "hey": "Hey! What can I help you with?",
            "hey there": "Hello! How can I assist you?",
            "thanks": "You're welcome!",
            "thank you": "You're welcome!",
            "bye": "Goodbye! Have a great day.",
            "goodbye": "Goodbye! Have a great day.",
        }) if cfg.use_zero_token else None

    def fast_path(self, query: str) -> Tuple[str | None, str | None]:
        if self.zero_token:
            answer = self.zero_token.respond(query)
            if answer:
                return answer, "zero_token"
        if self.faq:
            answer = self.faq.get(query)
            if answer:
                return answer, "faq"
        return None, None

    def build_prompt(self, query: str) -> Tuple[str, int, list]:
        # Check cache (Bug #7 fix)
        if query in self._retrieval_cache:
            self._retrieval_cache.move_to_end(query)  # LRU
            return self._retrieval_cache[query]

        result = self.retrieval.retrieve(query)
        chunks = [text for _, text, _ in result.chunks]
        prompt = build_minimal_prompt(query, chunks, one_liner=self.cfg.one_liner_mode)
        max_tokens = compute_dynamic_max_tokens(query, max_cap=self.cfg.max_tokens)
        sources = [{"text": c[:200], "score": s} for _, c, s in result.chunks[:3]]

        # Cache result with LRU eviction (Bug #10 fix)
        self._retrieval_cache[query] = (prompt, max_tokens, sources)
        if len(self._retrieval_cache) > self._cache_max_entries:
            self._retrieval_cache.popitem(last=False)  # Remove oldest
        return prompt, max_tokens, sources
