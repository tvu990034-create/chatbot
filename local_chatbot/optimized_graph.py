"""Optimized LangGraph conversation workflow with full speed_engine integration."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from local_chatbot.config import ChatbotConfig
from local_chatbot.engine import LocalEngine
from speed_engine.config import SpeedConfig
from speed_engine.pipeline import SpeedPipeline, Source
from speed_engine.prefilter import FAQDatabase, ZeroTokenResponder

logger = logging.getLogger(__name__)


class ChatState(TypedDict, total=False):
    session_id: str
    query: str
    response: str
    source: str
    sources: List[Dict[str, Any]]
    latency_ms: float
    history: Annotated[List[Dict[str, str]], "append"]


@dataclass
class ChatResult:
    response: str
    source: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0


class ConversationMemory:
    """In-memory conversation history per session."""

    def __init__(self, max_turns: int = 20):
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self.max_turns = max_turns

    def add(self, session_id: str, role: str, content: str) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append({"role": role, "content": content})
        if len(self._sessions[session_id]) > self.max_turns * 2:
            self._sessions[session_id] = self._sessions[session_id][-(self.max_turns * 2):]

    def get(self, session_id: str) -> List[Dict[str, str]]:
        return self._sessions.get(session_id, [])

    def clear(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class OptimizedChatGraph:
    """
    Optimized conversation graph with full speed_engine integration:
    - Multi-layer caching (exact → SimHash → BM25)
    - Advanced retrieval (BM25 + dense + PageRank + reranking)
    - Smart fast paths (FAQ, zero-token, bloom filter)
    - Score gating for relevance
    - Dynamic token allocation
    """

    def __init__(self, cfg: Optional[ChatbotConfig] = None):
        self.cfg = cfg or ChatbotConfig()
        self.engine = LocalEngine(self.cfg)
        self.memory = ConversationMemory()
        self._speed_pipeline = self._build_speed_pipeline()
        self._graph = self._build_graph()

    def _load_documents(self) -> List[tuple]:
        """Load documents from knowledge base."""
        import os
        import sys
        from pathlib import Path

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, root)

        documents = []

        try:
            from app.data_ingestion import DataIngestion

            di = DataIngestion(knowledge_base_path=self.cfg.knowledge_base_path)
            texts, ids = di.load_all_documents()
            documents = list(zip(ids, texts))
        except Exception as e:
            logger.warning(f"Failed to load documents from {self.cfg.knowledge_base_path}: {e}")
            # Default documents
            documents = [
                ("doc1", "Paris is the capital of France."),
                ("doc2", "BM25 is a bag-of-words retrieval function used in search engines."),
                ("doc3", "KV cache stores key-value tensors to speed up transformer inference."),
                ("doc4", "Quantization reduces model precision to lower bit widths for faster inference."),
                ("doc5", "Transformers use self-attention to process sequential data in parallel."),
                ("doc6", "Embeddings are dense vector representations that capture semantic meaning."),
                ("doc7", "Local inference runs AI models on your machine without cloud dependencies."),
            ]

        return documents

    def _build_speed_pipeline(self) -> SpeedPipeline:
        """Build the optimized speed pipeline."""
        documents = self._load_documents()

        # FAQ database
        faq = {
            "What is BM25?": "BM25 is a bag-of-words retrieval function used in search engines.",
            "What is KV cache?": "KV cache stores computed attention key-value pairs to speed up generation.",
            "What is quantization?": "Quantization reduces model precision to lower bit widths for faster inference.",
            "What is the capital of France?": "The capital of France is Paris.",
        }

        # Zero-token phrases
        zero_token = {
            "hi": "Hello! How can I help you today?",
            "hello": "Hi there! What would you like to know?",
            "hey": "Hey! What can I help you with?",
            "hey there": "Hello! How can I assist you?",
            "thanks": "You're welcome!",
            "thank you": "You're welcome!",
            "bye": "Goodbye! Have a great day.",
            "goodbye": "Goodbye! Have a great day.",
        }

        # Build speed config
        speed_cfg = SpeedConfig(
            use_faq=self.cfg.use_faq,
            use_zero_token=self.cfg.use_zero_token,
            use_prewarm_cache=True,
            use_dynamic_index_switch=True,
            use_query_truncation=True,
            use_pagerank_prune=True,
            use_score_gating=False,  # Disabled for better coverage
            minimal_prompt=True,
            one_liner_mode=self.cfg.one_liner_mode,
            chunk_limit=self.cfg.chunk_limit,
            faq=faq,
            zero_token_phrases=zero_token,
            documents=tuple(documents),
        )

        # Create pipeline (will use simulated engine by default)
        pipeline = SpeedPipeline(speed_cfg)

        # Replace simulated engine with real local engine
        pipeline.engine = self._create_real_engine_wrapper()

        return pipeline

    def _create_real_engine_wrapper(self):
        """Create a wrapper that uses the real LocalEngine."""
        class RealEngineWrapper:
            def __init__(self, local_engine):
                self.local_engine = local_engine

            def generate_tokens(self, prompt: str, max_tokens: int):
                # Generate full response
                response = self.local_engine.generate(prompt, max_tokens=max_tokens)
                # Split into words for streaming compatibility
                words = response.split()
                for word in words:
                    yield word

        return RealEngineWrapper(self.engine)

    def _build_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph

            graph = StateGraph(ChatState)

            graph.add_node("fast_path", self._node_fast_path)
            graph.add_node("retrieve", self._node_retrieve)
            graph.add_node("generate", self._node_generate)

            graph.add_edge(START, "fast_path")

            def route_after_fast(state: ChatState) -> str:
                return END if state.get("response") else "retrieve"

            graph.add_conditional_edges("fast_path", route_after_fast)
            graph.add_edge("retrieve", "generate")
            graph.add_edge("generate", END)

            return graph.compile()
        except ImportError:
            return None

    def _node_fast_path(self, state: ChatState) -> ChatState:
        # Use speed pipeline's fast path
        fast_result = self._speed_pipeline.try_fast_path(state["query"])
        if fast_result:
            answer, source = fast_result
            return {**state, "response": answer, "source": source.value, "sources": []}
        return state

    def _node_retrieve(self, state: ChatState) -> ChatState:
        # Use speed pipeline's retrieval
        result = self._speed_pipeline.retrieval.retrieve(state["query"])
        chunks = [text for _, text, _ in result.chunks]
        # Store chunks for generation
        return {**state, "_chunks": chunks, "_best_score": result.best_score}

    def _node_generate(self, state: ChatState) -> ChatState:
        chunks = state.get("_chunks", [])
        best_score = state.get("_best_score", 0.0)

        # Score gating
        if self._speed_pipeline.cfg.use_score_gating:
            threshold = self._speed_pipeline.cfg.score_threshold
            if best_score < threshold:
                return {
                    **state,
                    "response": self._speed_pipeline.cfg.idk_response,
                    "source": "score_gate",
                    "sources": [],
                }

        # Build prompt
        from speed_engine.prompt import build_minimal_prompt, compute_dynamic_max_tokens

        prompt = build_minimal_prompt(state["query"], chunks, self.cfg.one_liner_mode)
        max_tokens = compute_dynamic_max_tokens(
            state["query"],
            self._speed_pipeline.cfg.dynamic_tokens_base,
            self._speed_pipeline.cfg.dynamic_tokens_multiplier,
            self._speed_pipeline.cfg.dynamic_tokens_max_cap,
            self._speed_pipeline.cfg.dynamic_tokens_long_answer_cap,
        )

        # Add conversation history
        history = self.memory.get(state.get("session_id", "default"))
        if history:
            context = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
            prompt = f"Previous conversation:\n{context}\n\n{prompt}"

        # Generate using real engine
        response = self.engine.generate(prompt, max_tokens=max_tokens)

        # Update cache
        if self._speed_pipeline.cache:
            self._speed_pipeline.cache.put(state["query"], response)

        return {**state, "response": response, "source": "llm"}

    def chat(self, query: str, session_id: str = "default") -> ChatResult:
        start = time.perf_counter()
        state: ChatState = {"session_id": session_id, "query": query, "history": []}

        if self._graph:
            result = self._graph.invoke(state)
        else:
            result = self._run_sequential(state)

        latency = (time.perf_counter() - start) * 1000
        response = result.get("response", "Sorry, I couldn't process that.")
        source = result.get("source", "unknown")
        sources = result.get("sources", [])

        self.memory.add(session_id, "user", query)
        self.memory.add(session_id, "assistant", response)

        return ChatResult(
            response=response,
            source=source,
            sources=sources,
            latency_ms=round(latency, 2),
        )

    def _run_sequential(self, state: ChatState) -> ChatState:
        state = self._node_fast_path(state)
        if state.get("response"):
            return state
        state = self._node_retrieve(state)
        return self._node_generate(state)

    def clear_session(self, session_id: str) -> None:
        self.memory.clear(session_id)
