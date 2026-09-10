"""
agents/langgraph_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph stateful agent with a single canonical request path:

  1. Cache check (context-aware key via make_cache_identity)
  2. Quick-path bypass for greetings / simple arithmetic
  3. LangGraph: rag_node_async -> agent_node -> tools
  4. agent_node always uses resolve_generation_policy for temperature/max_tokens
  5. Results cached and metadata returned to caller
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from typing import Annotated, Any, Literal, Optional, TypedDict

from config import settings

logger = logging.getLogger(__name__)

# Optimization wiring is installed once per process (idempotent behind flags).
_wiring_installed = False
_wiring_lock = threading.Lock()


@dataclasses.dataclass
class ChatResult:
    """Structured response from achat/chat with full optimization metadata."""
    reply: str = ""
    cache_hit: bool = False
    model_used: str = ""
    rag_used: bool = False
    rag_chunks: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_used: bool = False
    generation_policy: str = ""


class AgentState(TypedDict):
    """State schema for the LangGraph agent (kept at module level so the lazy
    langchain/litellm type names resolve via module globals in LangGraph's
    runtime get_type_hints).  BaseMessage/add_messages are injected into
    module globals by build_agent() before the graph is constructed."""
    messages: "Annotated[list[Any], add_messages]"
    rag_context: str
    rag_fetch_time: float
    _req_ctx: "dict[str, Any]"
    rag_task: Any  # Eq2: pre-launched asyncio.Task returned by _rag_prefetch
    advanced_reasoning_used: bool
    reasoning_metadata: "dict[str, Any]"
    model_used: str
    generation_policy: str
    input_tokens: int
    output_tokens: int
    _tool_call_counts: "dict[str, int]"  # request-scoped repeated-call detector



def _truncate_tool_output(output: str) -> str:
    """BUG 58 FIX: Reduce huge tool outputs before they reach the model context.
    Uses structured truncation (JSON / code / logs) that preserves the most
    important head/tail sections."""
    if not output:
        return output

    from gateway.opt_core import structured_truncate

    max_chars = getattr(settings, 'agent_max_tool_output_chars', 5000)
    max_tokens = getattr(settings, 'agent_max_tool_output_tokens', 1000)
    return structured_truncate(output, max_chars=max_chars, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

def _get_langgraph():
    try:
        from langchain_core.messages import (
            AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
        )
        from langchain_core.tools import BaseTool
        from langgraph.graph import END, START, StateGraph
        from langgraph.graph.message import add_messages
        from langgraph.prebuilt import ToolNode
        return (
            AIMessage, BaseMessage, HumanMessage,
            SystemMessage, ToolMessage, BaseTool,
            END, START, StateGraph, add_messages, ToolNode,
        )
    except ImportError as exc:
        raise ImportError(
            "LangGraph not installed: pip install langgraph langchain-core"
        ) from exc


# ---------------------------------------------------------------------------
# Eq2 – Async RAG prefetch
# ---------------------------------------------------------------------------

async def _rag_prefetch(question: str) -> str:
    """
    Start RAG retrieval immediately and return a formatted context string.
    Called as an asyncio.Task to start retrieval early (no LLM overlap with current backend).

    Performance: uses retrieval-ONLY providers (embed + fetch top-k chunks,
    no LLM generation).  The main agent call does the synthesising, so a
    RAG request triggers exactly ONE LLM call instead of three (0 redundant
    RAG generations).
    """
    from config import RAGProvider
    provider = settings.rag_provider
    if provider == RAGProvider.NONE:
        return ""

    snippets: list[str] = []

    def _format_chunks(result: dict) -> str | None:
        chunks = result.get("chunks") or []
        if not chunks:
            return None
        sources = result.get("sources") or []
        lines = []
        for i, chunk in enumerate(chunks, start=1):
            src = sources[i - 1] if i - 1 < len(sources) else "unknown"
            text = chunk if isinstance(chunk, str) else str(chunk)
            lines.append(f"[chunk {i}] ({src})\n{text[:2000]}")
        return "\n\n".join(lines)

    async def _fetch_llama():
        try:
            from rag.llama_index_rag import get_rag
            result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: get_rag().retrieve(question)
            )
            body = _format_chunks(result) if isinstance(result, dict) else None
            if body:
                snippets.append(f"[LlamaIndex context]\n{body}")
        except Exception as exc:
            logger.warning("Eq2 LlamaIndex prefetch failed: %s", exc)

    async def _fetch_haystack():
        try:
            from rag.haystack_pipeline import get_haystack_rag
            result = await asyncio.get_running_loop().run_in_executor(
                None, lambda: get_haystack_rag().retrieve(question)
            )
            body = _format_chunks(result) if isinstance(result, dict) else None
            if body:
                snippets.append(f"[Haystack context]\n{body}")
        except Exception as exc:
            logger.warning("Eq2 Haystack prefetch failed: %s", exc)

    tasks = []
    if provider in (RAGProvider.LLAMA_INDEX, RAGProvider.BOTH):
        tasks.append(_fetch_llama())
    if provider in (RAGProvider.HAYSTACK, RAGProvider.BOTH):
        tasks.append(_fetch_haystack())

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    return "\n\n".join(snippets)


# ---------------------------------------------------------------------------
# Eq1 – smart memory window
# ---------------------------------------------------------------------------

def _trim_messages_eq1(messages: list, token_budget: int = 8192) -> list:
    """
    Token-based context trimming (preserves system prompt and recent conversation).

    Priority order:
      1. System messages (always kept)
      2. Current user message (always kept)
      3. Most recent conversation turns (newest first)
      4. Older conversation (added until budget exhausted)
    """
    if not messages:
        return messages

    from gateway.opt_core import estimate_tokens

    system_msgs = [m for m in messages if getattr(m, "type", "") == "system"]
    convo_msgs  = [m for m in messages if getattr(m, "type", "") != "system"]

    if not convo_msgs:
        return messages

    # Budget consumed by system messages
    system_tokens = sum(estimate_tokens(m.content) for m in system_msgs)
    remaining = token_budget - system_tokens

    if remaining <= 0:
        return system_msgs + convo_msgs[-1:]

    # Always keep the last message (current user message)
    last_msg = convo_msgs[-1]
    remaining -= estimate_tokens(last_msg.content)

    if len(convo_msgs) <= 1:
        return system_msgs + convo_msgs

    # Fill from newest backward (excluding the last message already accounted for)
    older = convo_msgs[:-1]
    kept: list = []
    for msg in reversed(older):
        cost = estimate_tokens(msg.content)
        if remaining - cost < 0:
            break
        kept.append(msg)
        remaining -= cost

    kept.reverse()
    result = system_msgs + kept + [last_msg]

    logger.debug(
        "Eq1 token trim: %d msgs → %d msgs  (budget=%d, kept≈%d tokens)",
        len(convo_msgs), len(kept) + 1,
        token_budget, token_budget - remaining,
    )
    return result


# ---------------------------------------------------------------------------
# Agent graph builder
# ---------------------------------------------------------------------------

def build_agent(extra_tools: list | None = None):
    (
        AIMessage, BaseMessage, HumanMessage,
        SystemMessage, ToolMessage, BaseTool,
        END, START, StateGraph, add_messages, ToolNode,
    ) = _get_langgraph()

    # Make the lazy-imported names resolvable by LangGraph's runtime
    # get_type_hints() on the module-level AgentState TypedDict (it evaluates
    # annotations against module globals, not function locals).
    from langchain_core.messages import (
        AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
    )
    globals()["AIMessage"] = AIMessage
    globals()["BaseMessage"] = BaseMessage
    globals()["HumanMessage"] = HumanMessage
    globals()["SystemMessage"] = SystemMessage
    globals()["ToolMessage"] = ToolMessage
    globals()["add_messages"] = add_messages

    from tools.aider_tool import get_code_tools

    tools = get_code_tools() + (extra_tools or [])
    _have_tools = bool(tools)

    # ---- Nodes ----

    async def agent_node(state: AgentState) -> dict[str, Any]:
        """Trim history, always apply generation policy, call LLM (async)."""
        import litellm
        from gateway.opt_core import (
            resolve_generation_policy, estimate_tokens,
            get_router_state, select_model, ollama_model_id,
            detect_code_intent, check_model_available,
        )

        msgs     = list(state["messages"])
        rag_ctx  = state.get("rag_context", "")
        t_rag    = state.get("rag_fetch_time", 0.0)
        req_ctx  = state.get("_req_ctx", {})

        # --- Canonical context building: system -> current -> retrieved ->
        #     recent history -> old history, checked against the model context
        #     limit.  This is the SINGLE prompt optimizer on the live path.
        from gateway.opt_core import build_context_messages
        system_content = req_ctx.get("system_prompt") or settings.agent_system_prompt
        built = build_context_messages(
            msgs,
            system_prompt=system_content,
            rag_context=rag_ctx,
            context_limit=getattr(settings, "context_window_size", 8192),
            budget=getattr(settings, "token_budget", 4096),
            compression_ratio=0.6,
            always_keep_n_history=6,
        )
        formatted_msgs = built.messages
        query_text = formatted_msgs[-1]["content"] if formatted_msgs else ""
        if built.compressed:
            logger.info("Compressed context: %s", built.reasoning)
        is_math = any(w in query_text.lower() for w in (
            "calculate", "solve", "equation", "+", "-", "*", "/",
        ))
        is_coding = detect_code_intent(query_text)
        is_complex = len(query_text) > 100
        needs_reasoning = any(w in query_text.lower() for w in (
            "why", "how", "explain", "reason",
        ))

        # NOTE: use a types.SimpleNamespace instead of a nested class — a class
        # body cannot see the enclosing-function locals (is_math = is_math fails).
        from types import SimpleNamespace
        _Analysis = SimpleNamespace(
            is_math=is_math, is_coding=is_coding, is_complex=is_complex,
            needs_reasoning=needs_reasoning,
            expected_response_length=(
                "short" if len(query_text) < 30 else "medium"),
            query_text=query_text,
        )

        # --- Request params ---
        req_model = req_ctx.get("model")
        req_temp = req_ctx.get("temperature")
        req_max = req_ctx.get("max_tokens")
        use_router = req_ctx.get("use_router", True)
        speed_mode = req_ctx.get("speed_mode", False)

        # --- Model selection (always applied) ---
        base_model = req_model or settings.default_model
        model_to_use = ollama_model_id(base_model)

        # Code intent routing: prefer code-specialized model if available
        code_model = None
        if is_coding and not req_model:
            code_candidates = ["ollama/deepseek-coder:1.3b", "ollama/codellama:7b"]
            # Check availability in parallel to avoid sequential network calls
            availability = await asyncio.gather(*[
                asyncio.to_thread(check_model_available, cm)
                for cm in code_candidates
            ])
            for cm, available in zip(code_candidates, availability):
                if available:
                    code_model = cm
                    break

        # If router is enabled, let RouterState decide
        if use_router and not req_model:
            router_state = get_router_state()
            model_to_use = ollama_model_id(
                select_model(
                    requested=settings.default_model,
                    routed=None,
                    code_model=code_model,
                    state=router_state,
                    expected_output_tokens=req_max or settings.litellm_max_tokens,
                )
            )
        elif code_model:
            model_to_use = code_model

        # Verify model availability; fall back to default if not found
        if not await asyncio.to_thread(check_model_available, model_to_use):
            logger.warning("Model %s not available, falling back to %s",
                          model_to_use, settings.default_model)
            model_to_use = ollama_model_id(settings.default_model)

        # --- Speed mode overrides ---
        policy = None  # only set when not in speed mode
        if speed_mode:
            # Speed mode: reduce work, use faster settings
            if is_math:
                # Math still needs precision
                speed_temp = 0.1
            else:
                speed_temp = min(req_temp or 0.3, 0.3)
            reasoning_intent = is_math or is_coding or needs_reasoning
            from gateway.opt_core import speed_mode_max_tokens
            speed_max = speed_mode_max_tokens(req_max, is_reasoning=reasoning_intent)
            final_temperature = speed_temp
            final_max_tokens = speed_max
        else:
            # --- Generation policy (always computed) ---
            base_params = {}
            if req_temp is not None:
                base_params["temperature"] = req_temp
            if req_max is not None:
                base_params["max_tokens"] = req_max

            policy = resolve_generation_policy(
                _Analysis,
                base=base_params,
                configured_max_tokens=req_max or settings.litellm_max_tokens,
                model_name=model_to_use,
            )

            # User-provided values override the policy
            final_temperature = req_temp if req_temp is not None else policy.temperature
            final_max_tokens = req_max if req_max is not None else policy.max_tokens

        # Record inflight for load-aware routing
        router_state = get_router_state()
        router_state.record_start(model_to_use)
        try:
            from gateway.opt_core import adaptive_generation_timeout
            kwargs = {
                "model": model_to_use,
                "messages": formatted_msgs,
                "temperature": final_temperature,
                "max_tokens": final_max_tokens,
                "timeout": adaptive_generation_timeout(
                    final_max_tokens,
                    getattr(settings, "generation_timeout", 15),
                ),
                "api_base": settings.litellm_api_base or "http://localhost:11434",
            }

            # Pass top_p and top_k from generation policy (unless speed mode)
            if not speed_mode and policy is not None:
                kwargs["top_p"] = policy.top_p
                kwargs["top_k"] = policy.top_k

            t_llm_start = time.perf_counter()
            response_obj = await litellm.acompletion(**kwargs)
            t_llm = time.perf_counter() - t_llm_start

            from langchain_core.messages import AIMessage
            response = AIMessage(content=response_obj.choices[0].message.content)
            actual_model = response_obj.model if hasattr(response_obj, 'model') else model_to_use
            usage = getattr(response_obj, "usage", None)
            out_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        finally:
            router_state.record_end(model_to_use)

        if t_rag > 0:
            logger.debug("RAG+LLM: rag=%.2fs llm=%.2fs", t_rag, t_llm)

        policy_reason = "speed_mode" if speed_mode else (
            policy.reason if policy is not None else ""
        )

        return {
            "messages": [response],
            "advanced_reasoning_used": False,
            "reasoning_metadata": {},
            "model_used": actual_model,
            "generation_policy": policy_reason,
            "input_tokens": built.input_tokens,
            "output_tokens": out_tokens,
        }

    async def rag_node_async(state: AgentState) -> dict[str, Any]:
        """RAG retrieval — runs exactly once per request."""
        req_ctx = state.get("_req_ctx", {})
        use_rag = req_ctx.get("use_rag", True)
        if not use_rag:
            return {"rag_context": "", "rag_fetch_time": 0.0}

        msgs       = state["messages"]
        last_human = next(
            (m for m in reversed(msgs) if m.type == "human"), None
        )
        if last_human is None:
            return {"rag_context": "", "rag_fetch_time": 0.0}

        # RAG skip fast path: simple queries don't need retrieval
        from gateway.opt_core import is_safe_quick_path
        
        # Additional simple query patterns for CLI efficiency
        def is_very_simple_query(query: str) -> bool:
            """Skip RAG for very simple conversational queries."""
            import re
            simple_patterns = [
                r'^hello', r'^hi', r'^hey', r'^thanks', r'^thank you',
                r'^what is your name', r'^who are you', r'^how are you',
                r'^good morning', r'^good afternoon', r'^good evening',
                r'^yes$', r'^no$', r'^ok$', r'^okay$'
            ]
            query_lower = query.lower().strip()
            return any(re.match(p, query_lower) for p in simple_patterns)
        
        if is_safe_quick_path(last_human.content) or is_very_simple_query(last_human.content):
            logger.debug("RAG skip: query is simple/conversational")
            return {"rag_context": "", "rag_fetch_time": 0.0}

        t0  = time.perf_counter()
        prefetch_task = state.get("rag_task")
        if prefetch_task is not None:
            # Eq2: retrieval was launched before the agent graph so it overlapped
            # with request setup; wait on it now.
            try:
                ctx = await prefetch_task
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("RAG prefetch task failed, using empty context", exc_info=True)
                ctx = ""
        else:
            ctx = await _rag_prefetch(last_human.content)
        t_fetch = time.perf_counter() - t0

        # RAG deduplication and relevance filtering
        if ctx:
            from gateway.opt_core import (
                dedupe_context_parts, estimate_tokens,
                hashed_embedding, cosine_similarity,
            )
            parts = [p.strip() for p in ctx.split("\n\n") if p.strip()]
            parts = dedupe_context_parts(parts)
            # Token budget: don't let RAG exceed the configured budget
            rag_budget = settings.rag_token_budget
            kept, used = [], 0
            for p in parts:
                cost = estimate_tokens(p)
                if used + cost > rag_budget:
                    break
                kept.append(p)
                used += cost
            ctx = "\n\n".join(kept)

            # BUG 9 FIX: RAG confidence gate — if the retrieved context has no
            # measurable lexical overlap with the question at all, it is almost
            # certainly off-topic; drop it rather than polluting the prompt.
            # The threshold is deliberately very conservative (0.05) so that only
            # pathological retrieval results are rejected.
            if kept:
                try:
                    q_emb = hashed_embedding(last_human.content)
                    scores = [cosine_similarity(q_emb, hashed_embedding(p)) for p in kept]
                    if max(scores) < 0.05:
                        logger.info(
                            "RAG relevance too low (%.3f) for %d chunks; dropping context",
                            max(scores), len(kept),
                        )
                        ctx = ""
                except Exception:
                    pass

        logger.debug("RAG done in %.2fs, ctx_len=%d", t_fetch, len(ctx))
        return {
            "rag_context":    ctx,
            "rag_fetch_time": t_fetch,
        }

    _MAX_TOOL_ROUNDS = getattr(settings, "agent_max_tool_calls", 10)

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if not (hasattr(last, "tool_calls") and last.tool_calls and _have_tools):
            return "__end__"

        # Per-request tool iteration limit: count completed tool rounds
        # (each ToolMessage represents one completed tool call)
        tool_rounds = sum(
            1 for m in state["messages"]
            if getattr(m, "type", "") == "tool"
        )
        if tool_rounds >= _MAX_TOOL_ROUNDS:
            logger.warning(
                "Max tool calls (%d) reached for request, stopping agent loop.",
                _MAX_TOOL_ROUNDS,
            )
            return "__end__"

        return "tools"
    # BUG 58 FIX: Custom tool node wrapper with output truncation
    # MAX_REPEATED_SAME_CALL: after this many identical tool+args calls,
    # short-circuit with an error instead of re-executing (request-scoped).
    _MAX_REPEATED_SAME_CALL = 3

    class TruncatedToolNode:
        """Tool node wrapper that truncates output and detects repeated calls.

        * Truncates tool outputs to prevent context explosion.
        * Counts identical tool+args calls per request; after
          ``_MAX_REPEATED_SAME_CALL`` repeats the same call is short-circuited
          with an error message instead of re-executing.
        * Uses ``asyncio.to_thread`` to avoid blocking the event loop.
        """
        def __init__(self, tools, base_tool_node):
            self.tools = tools
            self.base_tool_node = base_tool_node
        
        async def __call__(self, state: AgentState):
            # --- request-scoped repeated-call detector ---
            call_counts: dict = state.get("_tool_call_counts") or {}
            # Ensure the dict is mutable (TypedDict may give us a fresh copy)
            if "_tool_call_counts" not in state:
                state["_tool_call_counts"] = call_counts

            # Inspect the AIMessage that triggered this tool step — it carries
            # the tool_calls list with IDs, names, and args.
            last_ai = None
            for m in reversed(state.get("messages", [])):
                if getattr(m, "type", "") == "ai":
                    last_ai = m
                    break

            skip_ids: set = set()
            if last_ai and getattr(last_ai, "tool_calls", None):
                for tc in last_ai.tool_calls:
                    tc_id = tc.get("id", "")
                    tc_name = tc.get("name", "")
                    tc_args_str = str(tc.get("args", ""))
                    key = f"{tc_name}::{tc_args_str}"
                    count = call_counts.get(key, 0) + 1
                    call_counts[key] = count
                    if count > _MAX_REPEATED_SAME_CALL:
                        skip_ids.add(tc_id)

            # --- execute tools (skipping repeated ones) ---
            result = await asyncio.to_thread(self.base_tool_node.invoke, state)

            # Replace skipped tool results with a clear error message
            if skip_ids and "messages" in result:
                from langchain_core.messages import ToolMessage
                new_msgs = []
                for msg in result["messages"]:
                    if (getattr(msg, "type", "") == "tool"
                            and getattr(msg, "tool_call_id", "") in skip_ids):
                        new_msgs.append(ToolMessage(
                            content=(
                                f"Tool '{getattr(msg, 'name', '?')}' has been called "
                                f"with the same arguments {self._max_repeated}+ times "
                                f"this request. Re-calling is blocked — try a "
                                f"different approach or answer from what you already "
                                f"know."
                            ),
                            tool_call_id=msg.tool_call_id,
                            name=getattr(msg, "name", "tool"),
                        ))
                    else:
                        new_msgs.append(msg)
                result["messages"] = new_msgs

            # Truncate tool outputs in the result
            if "messages" in result:
                for msg in result["messages"]:
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        msg.content = _truncate_tool_output(msg.content)
            
            return result

        # Expose the constant so tests can reference it
        _max_repeated = _MAX_REPEATED_SAME_CALL

    # ---- Graph ----
    tool_node = ToolNode(tools) if tools else None
    # BUG 58 FIX: Wrap tool node with truncation
    if tool_node:
        tool_node = TruncatedToolNode(tools, tool_node)

    graph = StateGraph(AgentState)
    graph.add_node("rag",   rag_node_async)
    graph.add_node("agent", agent_node)

    graph.add_edge(START, "rag")
    graph.add_edge("rag",  "agent")

    if tool_node:
        graph.add_node("tools", tool_node)
        graph.add_edge("tools", "agent")
        graph.add_conditional_edges("agent", should_continue)
    else:
        # No tools configured: agent always ends after generating a reply.
        graph.add_edge("agent", END)

    compiled = graph.compile()
    logger.info("LangGraph agent compiled | tools=%d", len(tools))
    return compiled


# ---------------------------------------------------------------------------
# Singleton + convenience functions
# ---------------------------------------------------------------------------

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


def chat(user_message: str, history: list[dict] | None = None, *,
         model: str | None = None, temperature: float | None = None,
         max_tokens: int | None = None, system_prompt: str | None = None,
         use_rag: bool = True, use_router: bool = True,
         use_cache: bool = True,
         speed_mode: bool = False) -> str:
    """Synchronous high-level chat. Propagates all request params to the agent."""

    _req_ctx = {
        "model": model, "temperature": temperature, "max_tokens": max_tokens,
        "system_prompt": system_prompt, "use_rag": use_rag,
        "use_router": use_router, "use_cache": use_cache,
        "speed_mode": speed_mode,
    }

    # --- Cache check (context-aware key, scoped to history) guard: sync path
    # still goes through the agent graph (no quick-path shortcut here). ---
    if use_cache:
        try:
            from gateway.simple_cache import get_cache
            cache_key = make_cache_identity(
                user_message,
                model=model or settings.default_model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt or "",
                rag_enabled=use_rag,
                messages=list(history or []),
            )
            cached = get_cache().get(
                user_message, context={"history": history or []}, _key=cache_key)
            if cached is not None:
                logger.debug("Cache hit (sync): %.40s…", user_message)
                return cached
        except Exception:
            pass

    # Quick-path: trivial queries answered instantly without the agent graph.
    from gateway.opt_core import is_safe_quick_path, quick_arithmetic, quick_word_problem, quick_word_arithmetic
    if is_safe_quick_path(user_message):
        arithmetic = quick_arithmetic(user_message) or quick_word_arithmetic(user_message) or quick_word_problem(user_message)
        if arithmetic is not None:
            return arithmetic
        if not history:
            _greeting_reply = {
                "hello": "Hello! How can I help you?",
                "hi": "Hi! How can I help you?",
                "hey": "Hey! How can I help you?",
                "thanks": "You're welcome!",
                "thank you": "You're welcome!",
                "ok": "OK!",
                "okay": "OK!",
            }
            reply = _greeting_reply.get(
                user_message.strip().lower(), "How can I help you?")
            return reply

    # --- Full agent graph (lazy: skip langgraph import/build for quick paths) ---
    (_, BaseMessage, HumanMessage, _, _, _, _, _, _, _, _) = _get_langgraph()
    from langchain_core.messages import AIMessage

    agent = get_agent()
    msgs: list[BaseMessage] = []
    for turn in (history or []):
        role, content = turn.get("role", "user"), turn.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    msgs.append(HumanMessage(content=user_message))

    invoke_kwargs = {
        "messages": msgs, "rag_context": "", "rag_fetch_time": 0.0,
        "_req_ctx": _req_ctx, "_tool_call_counts": {},
    }

    result = agent.invoke(
        invoke_kwargs,
        config={"recursion_limit": settings.agent_recursion_limit},
    )
    reply = getattr(result["messages"][-1], "content", "")

    if use_cache and reply:
        try:
            from gateway.simple_cache import get_cache
            from gateway.opt_core import make_cache_identity
            cache_key = make_cache_identity(
                user_message,
                model=model or settings.default_model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt or "",
                rag_enabled=use_rag,
                messages=list(history or []),
            )
            get_cache().set(
                query=user_message, response=reply,
                context={"history": history or []}, _key=cache_key)
        except Exception:
            pass

    return reply


def _ensure_wiring() -> None:
    """Install optimization wiring (semantic cache, EWMA router) exactly once per
    process instead of re-importing and re-checking on every request."""
    global _wiring_installed
    if _wiring_installed:
        return
    with _wiring_lock:
        if _wiring_installed:
            return
        try:
            from gateway.equation_wiring import (
                install_semantic_cache, install_ewma_router,
            )
            from gateway.simple_cache import get_cache
            from gateway.opt_core import get_router_state
            install_semantic_cache(get_cache())
            install_ewma_router(get_router_state())
        except Exception:
            logger.warning("optimization wiring install failed", exc_info=True)
        finally:
            _wiring_installed = True


async def achat(user_message: str, history: list[dict] | None = None, *,
                model: str | None = None, temperature: float | None = None,
                max_tokens: int | None = None, system_prompt: str | None = None,
                use_rag: bool = True, use_router: bool = True,
                use_cache: bool = True,
                speed_mode: bool = False) -> ChatResult:
    """
    Async high-level chat.  Single canonical path:

      1. Cache check (context-aware key)
      2. Quick-path for simple queries (no agent graph)
      3. LangGraph: rag_node_async -> agent_node -> tools
      4. Cache store + metadata
    """
    from gateway.opt_core import (
        make_cache_identity, is_safe_quick_path,
        quick_arithmetic, quick_word_problem, quick_word_arithmetic, estimate_tokens,
    )

    # Install optimization wiring once (semantic cache, EWMA router, Koopman RAG).
    _ensure_wiring()

    _req_ctx = {
        "model": model, "temperature": temperature, "max_tokens": max_tokens,
        "system_prompt": system_prompt, "use_rag": use_rag,
        "use_router": use_router, "use_cache": use_cache,
        "speed_mode": speed_mode,
    }

    # --- 1. Cache check (context-aware key, scoped to conversation history) ---
    if use_cache:
        try:
            from gateway.simple_cache import get_cache
            cache_key = make_cache_identity(
                user_message,
                model=model or settings.default_model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt or "",
                rag_enabled=use_rag,
                messages=list(history or []),
            )
            cached = get_cache().get(
                user_message, context={"history": history or []}, _key=cache_key)
            if cached is not None:
                logger.debug("Cache hit: %.40s…", user_message)
                return ChatResult(
                    reply=cached, cache_hit=True,
                    model_used=model or settings.default_model,
                )
        except Exception:
            pass

    # --- 2. Quick-path: simple queries bypass the agent graph entirely.
    # Arithmetic is stateless and safe regardless of history; greetings/acks are
    # only answered directly when there is no conversation to contradict. ---
    if is_safe_quick_path(user_message):
        # Try arithmetic first (context-free)
        arithmetic = quick_arithmetic(user_message) or quick_word_arithmetic(user_message) or quick_word_problem(user_message)
        if arithmetic is not None:
            logger.debug("Quick path: arithmetic -> %s", arithmetic)
            return ChatResult(
                reply=arithmetic,
                model_used=model or settings.default_model,
                generation_policy="arithmetic",
            )
        # Greeting / acknowledgment — only when no prior conversation context
        if not history:
            _greeting_reply = {
                "hello": "Hello! How can I help you?",
                "hi": "Hi! How can I help you?",
                "hey": "Hey! How can I help you?",
                "thanks": "You're welcome!",
                "thank you": "You're welcome!",
                "ok": "OK!",
                "okay": "OK!",
            }
            reply = _greeting_reply.get(user_message.strip().lower(), "How can I help you?")
            return ChatResult(
                reply=reply,
                model_used=model or settings.default_model,
                generation_policy="greeting",
            )

    # --- 3. Full agent graph ---
    (_, BaseMessage, HumanMessage, _, _, _, _, _, _, _, _) = _get_langgraph()
    from langchain_core.messages import AIMessage
    agent = get_agent()
    msgs: list[BaseMessage] = []
    for turn in (history or []):
        role, content = turn.get("role", "user"), turn.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    msgs.append(HumanMessage(content=user_message))

    total_input_tokens = sum(estimate_tokens(m.content) for m in msgs)

    # Eq2: launch RAG retrieval BEFORE entering the agent graph so it overlaps
    # with graph setup / routing rather than serialising inside rag_node.
    rag_task = None
    if use_rag and not is_safe_quick_path(user_message):
        rag_task = asyncio.create_task(_rag_prefetch(user_message))

    try:
        result = await agent.ainvoke(
            {"messages": msgs, "rag_context": "", "rag_fetch_time": 0.0,
             "_req_ctx": _req_ctx, "_tool_call_counts": {}, "rag_task": rag_task},
            config={"recursion_limit": settings.agent_recursion_limit},
        )
    except BaseException:
        if rag_task is not None and not rag_task.done():
            rag_task.cancel()
        raise

    reply = getattr(result["messages"][-1], "content", "")
    rag_used = bool(result.get("rag_context"))
    reasoning_used = result.get("advanced_reasoning_used", False)
    policy_reason = result.get("generation_policy", "")
    actual_model = result.get("model_used", model or settings.default_model)
    # Prefer the token count actually sent to the model (after trimming/compression).
    actual_input_tokens = result.get("input_tokens") or total_input_tokens
    actual_output_tokens = result.get("output_tokens") or 0

    # --- 4. Cache store ---
    if use_cache and reply:
        try:
            from gateway.simple_cache import get_cache
            cache_key = make_cache_identity(
                user_message,
                model=model or settings.default_model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt or "",
                rag_enabled=use_rag,
                messages=list(history or []),
            )
            get_cache().set(
                query=user_message, response=reply,
                context={"history": history or []}, _key=cache_key)
        except Exception:
            pass

    return ChatResult(
        reply=reply, cache_hit=False, model_used=actual_model,
        rag_used=rag_used, input_tokens=actual_input_tokens,
        output_tokens=actual_output_tokens,
        reasoning_used=reasoning_used, generation_policy=policy_reason,
    )


async def achat_stream(user_message: str, history: list[dict] | None = None, *,
                       model: str | None = None, temperature: float | None = None,
                       max_tokens: int | None = None, system_prompt: str | None = None,
                       use_rag: bool = True, use_router: bool = True,
                       use_cache: bool = True, speed_mode: bool = False,
                       api_base: str | None = None):
    """Async streaming chat with full agent parity.

    Applies every optimization the non-streaming agent does:
      1. context-aware cache check (yields cached reply in one chunk)
      2. quick-path (arithmetic / greeting) — no LLM call
      3. Eq2 async RAG prefetch (overlaps retrieval with setup)
      4. Eq1 context building + trimming/compression and generation policy
      5. router + model-availability fallback
      6. streaming `acompletion` delta-by-delta, then caches the full reply.
    Unlike the graph tool loop (single LLM call), tools are not streamed.
    """
    import litellm
    from langchain_core.messages import AIMessage, HumanMessage
    from gateway.opt_core import (
        make_cache_identity, is_safe_quick_path, quick_arithmetic, quick_word_problem,
        quick_word_arithmetic,
        estimate_tokens, build_context_messages, resolve_generation_policy,
        get_router_state, select_model, ollama_model_id,
        detect_code_intent, check_model_available, dedupe_context_parts,
    )

    _ensure_wiring()

    # --- 1. Cache check ---
    if use_cache:
        try:
            from gateway.simple_cache import get_cache
            cache_key = make_cache_identity(
                user_message,
                model=model or settings.default_model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt or "",
                rag_enabled=use_rag,
                messages=list(history or []),
            )
            cached = get_cache().get(
                user_message, context={"history": history or []}, _key=cache_key)
            if cached is not None:
                logger.debug("Stream cache hit: %.40s…", user_message)
                yield cached
                return
        except Exception:
            pass

    # --- 2. Quick-path ---
    if is_safe_quick_path(user_message):
        arithmetic = quick_arithmetic(user_message) or quick_word_arithmetic(user_message) or quick_word_problem(user_message)
        if arithmetic is not None:
            yield arithmetic
            return
        if not history:
            _greeting_reply = {
                "hello": "Hello! How can I help you?",
                "hi": "Hi! How can I help you?",
                "hey": "Hey! How can I help you?",
                "thanks": "You're welcome!",
                "thank you": "You're welcome!",
                "ok": "OK!",
                "okay": "OK!",
            }
            yield _greeting_reply.get(
                user_message.strip().lower(), "How can I help you?")
            return

    # --- 3. Eq2: launch RAG retrieval before graph/decision work ---
    rag_task = None
    if use_rag and not is_safe_quick_path(user_message):
        rag_task = asyncio.create_task(_rag_prefetch(user_message))

    msgs: list = []
    for turn in (history or []):
        role, content = turn.get("role", "user"), turn.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    msgs.append(HumanMessage(content=user_message))

    total_input_tokens = sum(estimate_tokens(m.content) for m in msgs)

    # --- 4. Retrieve context ---
    rag_ctx = ""
    if rag_task is not None:
        try:
            rag_ctx = await rag_task
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("RAG prefetch failed (stream), continuing", exc_info=True)
    if rag_ctx:
        parts = [p.strip() for p in rag_ctx.split("\n\n") if p.strip()]
        parts = dedupe_context_parts(parts)
        rag_budget = settings.rag_token_budget
        kept, used = [], 0
        for p in parts:
            cost = estimate_tokens(p)
            if used + cost > rag_budget:
                break
            kept.append(p)
            used += cost
        rag_ctx = "\n\n".join(kept)

    # --- 5. Equal to agent_node decisioning (Eq1 trim, analysis, policy) ---
    system_content = system_prompt or settings.agent_system_prompt
    built = build_context_messages(
        msgs,
        system_prompt=system_content,
        rag_context=rag_ctx,
        context_limit=getattr(settings, "context_window_size", 8192),
        budget=getattr(settings, "token_budget", 4096),
        compression_ratio=0.6,
        always_keep_n_history=6,
    )
    formatted_msgs = built.messages
    query_text = formatted_msgs[-1]["content"] if formatted_msgs else ""
    if built.compressed:
        logger.info("Compressed context: %s", built.reasoning)

    from types import SimpleNamespace
    _Analysis = SimpleNamespace(
        is_math=any(w in query_text.lower() for w in (
            "calculate", "solve", "equation", "+", "-", "*", "/")),
        is_coding=detect_code_intent(query_text),
        is_complex=len(query_text) > 100,
        needs_reasoning=any(w in query_text.lower() for w in (
            "why", "how", "explain", "reason")),
        expected_response_length=(
            "short" if len(query_text) < 30 else "medium"),
        query_text=query_text,
    )

    req_model = model
    req_temp = temperature
    req_max = max_tokens
    base_model = req_model or settings.default_model
    model_to_use = ollama_model_id(base_model)

    code_model = None
    if _Analysis.is_coding and not req_model:
        code_candidates = ["ollama/deepseek-coder:1.3b", "ollama/codellama:7b"]
        availability = await asyncio.gather(*[
            asyncio.to_thread(check_model_available, cm)
            for cm in code_candidates
        ])
        for cm, available in zip(code_candidates, availability):
            if available:
                code_model = cm
                break

    if use_router and not req_model:
        router_state = get_router_state()
        model_to_use = ollama_model_id(
            select_model(
                requested=settings.default_model,
                routed=None,
                code_model=code_model,
                state=router_state,
                expected_output_tokens=req_max or settings.litellm_max_tokens,
            )
        )
    elif code_model:
        model_to_use = code_model

    if not await asyncio.to_thread(check_model_available, model_to_use):
        logger.warning("Model %s not available, falling back to %s",
                       model_to_use, settings.default_model)
        model_to_use = ollama_model_id(settings.default_model)

    policy = None
    if speed_mode:
        speed_temp = 0.1 if _Analysis.is_math else min(req_temp or 0.3, 0.3)
        reasoning_intent = _Analysis.is_math or _Analysis.is_coding or _Analysis.needs_reasoning
        from gateway.opt_core import speed_mode_max_tokens
        speed_max = speed_mode_max_tokens(req_max, is_reasoning=reasoning_intent)
        final_temperature = speed_temp
        final_max_tokens = speed_max
    else:
        base_params = {}
        if req_temp is not None:
            base_params["temperature"] = req_temp
        if req_max is not None:
            base_params["max_tokens"] = req_max
        policy = resolve_generation_policy(
            _Analysis,
            base=base_params,
            configured_max_tokens=req_max or settings.litellm_max_tokens,
            model_name=model_to_use,
        )
        final_temperature = req_temp if req_temp is not None else policy.temperature
        final_max_tokens = req_max if req_max is not None else policy.max_tokens

    router_state = get_router_state()
    router_state.record_start(model_to_use)
    try:
        from gateway.opt_core import adaptive_generation_timeout
        kwargs = {
            "model": model_to_use,
            "messages": formatted_msgs,
            "temperature": final_temperature,
            "max_tokens": final_max_tokens,
            "timeout": adaptive_generation_timeout(
                final_max_tokens,
                getattr(settings, "generation_timeout", 15),
            ),
            "api_base": settings.litellm_api_base or "http://localhost:11434",
            "stream": True,
        }
        if api_base:
            kwargs["api_base"] = api_base
        if not speed_mode and policy is not None:
            kwargs["top_p"] = policy.top_p
            kwargs["top_k"] = policy.top_k

        response = await litellm.acompletion(**kwargs)
        full = ""
        async for part in response:
            if not getattr(part, "choices", None):
                continue
            delta = getattr(part.choices[0], "delta", None)
            content = getattr(delta, "content", None)
            if content:
                full += content
                yield content
    finally:
        router_state.record_end(model_to_use)

    # --- 6. Cache store ---
    if use_cache and full:
        try:
            from gateway.simple_cache import get_cache
            cache_key = make_cache_identity(
                user_message,
                model=model or settings.default_model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt or "",
                rag_enabled=use_rag,
                messages=list(history or []),
            )
            get_cache().set(
                query=user_message, response=full,
                context={"history": history or []}, _key=cache_key)
        except Exception:
            pass
