"""LangGraph conversation workflow — retrieve then generate."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from local_chatbot.config import ChatbotConfig
from local_chatbot.engine import LocalEngine
from local_chatbot.rag import RAGPipeline

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

    def __init__(self, max_turns: int = 20, session_timeout_seconds: int = 3600):
        self._sessions: Dict[str, List[Dict[str, str]]] = {}
        self._last_access: Dict[str, float] = {}
        self.max_turns = max_turns
        self.session_timeout = session_timeout_seconds
        self._lock = threading.Lock()  # Bug #19 fix: Thread safety

    def add(self, session_id: str, role: str, content: str) -> None:
        import time
        with self._lock:  # Bug #19 fix: Thread safety
            self._cleanup_expired_sessions()
            if session_id not in self._sessions:
                self._sessions[session_id] = []
            self._sessions[session_id].append({"role": role, "content": content})
            self._last_access[session_id] = time.time()
            if len(self._sessions[session_id]) > self.max_turns * 2:
                self._sessions[session_id] = self._sessions[session_id][-(self.max_turns * 2):]

    def get(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:  # Bug #19 fix: Thread safety
            return self._sessions.get(session_id, [])

    def get_formatted_history(self, session_id: str) -> str:
        """Get formatted history string (Bug #12 fix)."""
        with self._lock:  # Bug #19 fix: Thread safety
            history = self._sessions.get(session_id, [])
            if not history:
                return ""
            return "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])

    def clear(self, session_id: str) -> None:
        with self._lock:  # Bug #19 fix: Thread safety
            self._sessions.pop(session_id, None)
            self._last_access.pop(session_id, None)

    def _cleanup_expired_sessions(self) -> None:
        """Remove sessions that haven't been accessed recently (Bug #15 fix)."""
        import time
        now = time.time()
        expired = [sid for sid, last in self._last_access.items() if now - last > self.session_timeout]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_access.pop(sid, None)


class LocalChatGraph:
    """
    Conversation graph inspired by LangGraph patterns:
    fast_path -> retrieve -> generate
    """

    def __init__(self, cfg: Optional[ChatbotConfig] = None):
        self.cfg = cfg or ChatbotConfig()
        self.engine = LocalEngine(self.cfg)
        self.rag = RAGPipeline(self.cfg)
        self.memory = ConversationMemory()
        self._graph = self._build_graph()

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
        answer, source = self.rag.fast_path(state["query"])
        if answer:
            return {**state, "response": answer, "source": source, "sources": []}
        return state

    def _node_retrieve(self, state: ChatState) -> ChatState:
        prompt, max_tokens, sources = self.rag.build_prompt(state["query"])
        return {**state, "_prompt": prompt, "_max_tokens": max_tokens, "sources": sources}

    def _node_generate(self, state: ChatState) -> ChatState:
        prompt = state.get("_prompt", state["query"])
        max_tokens = state.get("_max_tokens", self.cfg.max_tokens)
        # Use formatted history (Bug #12 fix)
        history = self.memory.get_formatted_history(state.get("session_id", "default"))
        if history:
            prompt = f"Previous conversation:\n{history}\n\n{prompt}"
        response = self.engine.generate(prompt, max_tokens=max_tokens)
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
