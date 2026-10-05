"""RAG pipeline — knowledge base loading and retrieval."""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

from speed_engine.prefilter import FAQDatabase, ZeroTokenResponder
from speed_engine.prompt import build_minimal_prompt, compute_dynamic_max_tokens
from speed_engine.retrieval import RetrievalPipeline

from local_chatbot.config import ChatbotConfig


def _load_documents(kb_path: str) -> List[Tuple[str, str]]:
    """Load documents from the knowledge base directory."""
    documents: List[Tuple[str, str]] = []

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)

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

    def __init__(self, cfg: ChatbotConfig):
        self.cfg = cfg
        documents = _load_documents(cfg.knowledge_base_path)

        self.retrieval = RetrievalPipeline(
            documents=documents,
            pagerank={doc_id: 1.0 for doc_id, _ in documents},
            chunk_limit=cfg.chunk_limit,
            use_dynamic_index_switch=True,
            use_query_truncation=True,
            use_pagerank_prune=True,
        )

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
        result = self.retrieval.retrieve(query)
        chunks = [text for _, text, _ in result.chunks]
        prompt = build_minimal_prompt(query, chunks, one_liner=self.cfg.one_liner_mode)
        max_tokens = compute_dynamic_max_tokens(query, max_cap=self.cfg.max_tokens)
        sources = [{"text": c[:200], "score": s} for _, c, s in result.chunks[:3]]
        return prompt, max_tokens, sources
