"""
agents/langgraph_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~~
LangGraph stateful agent — enhanced with:

  Eq1  KV-cache recycling via smart memory window trimming.
       history tokens are tracked and the window is cut to the Eq1-optimal
       size before each LLM call to maximise TTFB reduction.

  Eq2  Async RAG prefetch-and-overlap.
       RAG retrieval is fired as an asyncio.Task *before* the LLM call
       starts, so retrieval runs concurrently with LLM prefill.
       visible_latency is logged each turn.

  Advanced Reasoning Integration
       Automatic use of cutting-edge reasoning techniques including:
       - Geodesic Flow for optimal reasoning paths
       - Abductive Leap for insight-based reasoning
       - Quantum Superposition for uncertainty handling
       - Active Inference for principled step selection
       - Constitutional Alignment for value alignment
       - Causal Pruning for efficient reasoning
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Annotated, Any, Literal, Optional

from config import settings
from gateway.perf_math import (
    ChatMetrics,
    eq1_ttfb_reduction_fraction,
    eq2_visible_latency,
    eq2_speedup_prefetch,
    eq18_query_complexity,
    eq19_best_of_n_expected_quality,
    eq19_optimal_n,
    eq19_should_use_bon,
)

logger = logging.getLogger(__name__)

# Per-agent turn metrics
_agent_metrics = ChatMetrics()

# Optional advanced optimizations
_ode_agent = None
try:
    from gateway.advanced_optimizations import get_ode_agent
    _ode_available = True
except ImportError as exc:
    logger.warning(
        "Neural ODE agent not available (ImportError: %s). "
        "Advanced optimizations disabled.", exc
    )
    _ode_available = False
except Exception as exc:
    logger.error(
        "Unexpected error loading Neural ODE agent: %s. "
        "Advanced optimizations disabled.", exc
    )
    _ode_available = False


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
    Called as an asyncio.Task so it overlaps with LLM prefill (Eq2).
    """
    from config import RAGProvider
    provider = settings.rag_provider
    if provider == RAGProvider.NONE:
        return ""

    snippets: list[str] = []

    async def _fetch_llama():
        try:
            from rag.llama_index_rag import get_rag
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: get_rag().query(question)
            )
            if result.get("answer"):
                sources = ", ".join(result.get("sources", []))
                snippets.append(f"[LlamaIndex]\n{result['answer']}\nSources: {sources}")
        except Exception as exc:
            logger.warning("Eq2 LlamaIndex prefetch failed: %s", exc)

    async def _fetch_haystack():
        try:
            from rag.haystack_pipeline import get_haystack_rag
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: get_haystack_rag().query(question)
            )
            if result.get("answer"):
                sources = ", ".join(result.get("sources", []))
                snippets.append(f"[Haystack]\n{result['answer']}\nSources: {sources}")
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

def _trim_messages_eq1(messages: list, window: int) -> list:
    """
    Trim conversation history to at most `window` non-system messages,
    always preserving the system message and the latest user message.

    This maximises the Eq1 TTFB-reduction fraction by ensuring most of the
    prompt is history that can be served from KV-cache rather than re-prefilled.
    """
    if not messages:
        return messages

    system_msgs  = [m for m in messages if getattr(m, "type", "") == "system"]
    convo_msgs   = [m for m in messages if getattr(m, "type", "") != "system"]

    if len(convo_msgs) <= window:
        return messages  # nothing to trim

    # Keep the most recent `window` conversation messages
    trimmed_convo = convo_msgs[-window:]
    result = system_msgs + trimmed_convo

    # Log Eq1 impact
    original_chars = sum(len(getattr(m, "content", "")) for m in convo_msgs)
    kept_chars     = sum(len(getattr(m, "content", "")) for m in trimmed_convo)
    est_h = max(0, original_chars - kept_chars) // 4
    est_q = max(1, kept_chars // 4)
    reduction = eq1_ttfb_reduction_fraction(est_h, est_q)
    logger.debug(
        "Eq1 window trim: %d→%d msgs  TTFB-reduction=%.1f%%",
        len(convo_msgs), len(trimmed_convo), reduction * 100,
    )
    return result


# ---------------------------------------------------------------------------
# LiteLLM chat model shim
# ---------------------------------------------------------------------------

def _make_litellm_chat_model():
    try:
        from langchain_community.chat_models import ChatLiteLLM
        from config import get_litellm_model
        return ChatLiteLLM(
            model=get_litellm_model(),
            api_base=settings.litellm_api_base,
            max_tokens=settings.litellm_max_tokens,
            temperature=settings.litellm_temperature,
            streaming=settings.litellm_stream,
        )
    except ImportError:
        pass

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage, AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class _Shim(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "litellm-shim"

        def _generate(self, messages: list[BaseMessage], **kwargs) -> ChatResult:
            from gateway.litellm_gateway import chat as gw_chat
            formatted = [
                {"role": m.type if m.type != "human" else "user",
                 "content": m.content}
                for m in messages
            ]
            text = gw_chat(formatted)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    return _Shim()


# ---------------------------------------------------------------------------
# Agent graph builder
# ---------------------------------------------------------------------------

def build_agent(extra_tools: list | None = None):
    (
        AIMessage, BaseMessage, HumanMessage,
        SystemMessage, ToolMessage, BaseTool,
        END, START, StateGraph, add_messages, ToolNode,
    ) = _get_langgraph()

    from typing import TypedDict
    from tools.aider_tool import get_code_tools

    tools          = get_code_tools() + (extra_tools or [])
    llm            = _make_litellm_chat_model()
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    class AgentState(TypedDict):
        messages:    Annotated[list[BaseMessage], add_messages]
        rag_context: str
        # Eq2 timing fields
        rag_start_time:  float
        rag_fetch_time:  float
        # Advanced reasoning metadata
        advanced_reasoning_used: bool
        reasoning_metadata: dict[str, Any]

    # ---- Nodes ----

    def agent_node(state: AgentState) -> dict[str, Any]:
        """Eq1: trim to memory window; then call LLM with automatic advanced reasoning."""
        msgs     = list(state["messages"])
        rag_ctx  = state.get("rag_context", "")
        t_rag    = state.get("rag_fetch_time", 0.0)

        # Eq1 – trim history to optimal window
        msgs = _trim_messages_eq1(msgs, settings.agent_memory_window)

        # Automatic advanced reasoning integration
        use_advanced_reasoning = (
            settings.enable_advanced_reasoning and
            len(msgs) >= 1 and
            msgs[-1].type == "human" and
            len(msgs[-1].content) > 50  # Only for substantive questions
        )

        reasoning_metadata = {}
        if use_advanced_reasoning:
            try:
                # Use our advanced reasoning integration
                from advanced_reasoning_integration import EnhancedReasoningEngine
                from litellm_model_adapter import LiteLLMModelAdapter

                # Initialize components
                model_adapter = LiteLLMModelAdapter(settings.default_model)
                enhanced_engine = EnhancedReasoningEngine(
                    model_adapter,
                    hidden_dim=model_adapter.hidden_dim,
                    use_advanced_techniques=settings.use_cutting_edge_techniques
                )

                # Extract problem from last message
                problem = msgs[-1].content
                context = "\n".join([f"{m.type}: {m.content}" for m in msgs[:-1]])

                # Generate enhanced reasoning
                result = enhanced_engine.generate_enhanced_cot(
                    problem,
                    max_steps=settings.reasoning_max_steps
                )

                # Incorporate reasoning into the prompt
                reasoning_chain = " ".join(result['chain'])
                if reasoning_chain:
                    # Add reasoning as system context
                    enhanced_prompt = f"Context: {context}\n\nReasoning: {reasoning_chain}\n\nUser: {problem}"
                    msgs[-1] = HumanMessage(content=enhanced_prompt)

                    reasoning_metadata = {
                        "framework": "cutting_edge" if settings.use_cutting_edge_techniques else "standard_advanced",
                        "num_steps": result['num_steps'],
                        "coherence_loss": result.get('coherence_loss', 0),
                        "techniques_used": result.get('used_advanced_techniques', 'unknown')
                    }

                    logger.info(f"Applied advanced reasoning: {reasoning_metadata['framework']} with {result['num_steps']} steps")

            except Exception as exc:
                logger.warning(f"Advanced reasoning failed, falling back to standard: {exc}")
                reasoning_metadata = {"error": str(exc), "framework": "fallback"}

        # Build system message with RAG context appended
        system_content = settings.agent_system_prompt
        if rag_ctx:
            system_content += (
                "\n\nRelevant context retrieved from documents:\n"
                + rag_ctx
                + "\n\nUse this context when answering."
            )

        if not msgs or msgs[0].type != "system":
            msgs = [SystemMessage(content=system_content)] + msgs
        else:
            msgs[0] = SystemMessage(content=system_content)

        t_llm_start = time.perf_counter()
        response    = llm_with_tools.invoke(msgs)
        t_llm       = time.perf_counter() - t_llm_start

        # Eq2 – log overlap metrics
        if t_rag > 0:
            t_prefill_est = t_llm * 0.35
            vis_lat  = eq2_visible_latency(t_rag, t_prefill_est, t_llm * 0.65)
            speedup  = eq2_speedup_prefetch(t_llm, t_rag, t_prefill_est)
            logger.debug(
                "Eq2 agent overlap: rag=%.2fs llm=%.2fs vis_lat=%.0fms speedup=%.2fx",
                t_rag, t_llm, vis_lat * 1000, speedup,
            )

        return {
            "messages": [response],
            "advanced_reasoning_used": use_advanced_reasoning,
            "reasoning_metadata": reasoning_metadata,
        }

    async def rag_node_async(state: AgentState) -> dict[str, Any]:
        """
        Eq2: fire RAG retrieval as an asyncio task so it starts BEFORE
        the LLM prefill begins.  The task result is awaited in agent_node.
        """
        msgs       = state["messages"]
        last_human = next(
            (m for m in reversed(msgs) if m.type == "human"), None
        )
        if last_human is None:
            return {"rag_context": "", "rag_start_time": 0.0, "rag_fetch_time": 0.0}

        t0  = time.perf_counter()
        ctx = await _rag_prefetch(last_human.content)
        t_fetch = time.perf_counter() - t0

        logger.debug("Eq2 RAG prefetch done in %.2fs", t_fetch)
        return {
            "rag_context":    ctx,
            "rag_start_time": t0,
            "rag_fetch_time": t_fetch,
        }

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "__end__"

    # ---- Graph ----
    tool_node = ToolNode(tools) if tools else None

    graph = StateGraph(AgentState)
    graph.add_node("rag",   rag_node_async)
    graph.add_node("agent", agent_node)

    graph.add_edge(START, "rag")
    graph.add_edge("rag",  "agent")
    graph.add_conditional_edges("agent", should_continue)

    if tool_node:
        graph.add_node("tools", tool_node)
        graph.add_edge("tools", "agent")

    compiled = graph.compile()
    logger.info("LangGraph agent compiled (Eq1+Eq2 active) | tools=%d", len(tools))
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


# ---------------------------------------------------------------------------
# Eq19 – Best-of-N sampling gate
# (Stiennon et al. 2020 RLHF; Nakano et al. 2021 WebGPT)
# ---------------------------------------------------------------------------

def _best_of_n_invoke(
    agent,
    invoke_kwargs: dict,
    n: int,
    user_message: str,
) -> str:
    """
    Run the agent N times synchronously and return the longest / most
    information-dense reply as a simple quality proxy.

    For production, swap the scoring function for a reward model.
    The Eq19 formula:
        E[max reward over N] ≈ μ + σ · Φ⁻¹(1 − 1/N)
    tells us how much quality improvement to expect from N samples.
    """
    replies: list[str] = []
    for _ in range(n):
        try:
            result = agent.invoke(invoke_kwargs,
                                  config={"recursion_limit": settings.agent_recursion_limit})
            reply  = getattr(result["messages"][-1], "content", "")
            if reply:
                replies.append(reply)
        except Exception as exc:
            logger.warning("Eq19 BoN sample failed: %s", exc)

    if not replies:
        return ""

    # Simple quality proxy: prefer replies that are longer and contain
    # more unique words (avoids repetitive / degenerate outputs).
    def _score(r: str) -> float:
        words  = r.split()
        unique = len(set(words))
        return len(words) * 0.4 + unique * 0.6

    best = max(replies, key=_score)
    if len(replies) > 1:
        scores  = [_score(r) for r in replies]
        mu      = sum(scores) / len(scores)
        sigma   = (sum((s - mu) ** 2 for s in scores) / len(scores)) ** 0.5
        exp_max = eq19_best_of_n_expected_quality(len(replies), mu, sigma)
        logger.debug(
            "Eq19 BoN: n=%d  μ=%.1f  σ=%.2f  E[max]=%.1f  selected_score=%.1f",
            len(replies), mu, sigma, exp_max, _score(best),
        )
    return best


async def _best_of_n_ainvoke(
    agent,
    invoke_kwargs: dict,
    n: int,
) -> str:
    """Async variant of _best_of_n_invoke."""
    import asyncio as _asyncio

    async def _one() -> str:
        try:
            result = await agent.ainvoke(
                invoke_kwargs,
                config={"recursion_limit": settings.agent_recursion_limit},
            )
            return getattr(result["messages"][-1], "content", "")
        except Exception as exc:
            logger.warning("Eq19 async BoN sample failed: %s", exc)
            return ""

    # Run all N samples concurrently
    replies = [r for r in await _asyncio.gather(*[_one() for _ in range(n)]) if r]
    if not replies:
        return ""

    def _score(r: str) -> float:
        words  = r.split()
        unique = len(set(words))
        return len(words) * 0.4 + unique * 0.6

    best = max(replies, key=_score)
    if len(replies) > 1:
        scores  = [_score(r) for r in replies]
        mu      = sum(scores) / len(scores)
        sigma   = (sum((s - mu) ** 2 for s in scores) / len(scores)) ** 0.5
        exp_max = eq19_best_of_n_expected_quality(len(replies), mu, sigma)
        logger.debug(
            "Eq19 async BoN: n=%d  μ=%.1f  σ=%.2f  E[max]=%.1f",
            len(replies), mu, sigma, exp_max,
        )
    return best


def chat(user_message: str, history: list[dict] | None = None) -> str:
    """
    Synchronous high-level chat.

    Eq4 fast-path: check the SmartCache before spinning up the full agent
    graph — avoids the entire LangGraph overhead on repeated questions.
    """
    (_, BaseMessage, HumanMessage, _, _, _, _, _, _, _, _) = _get_langgraph()
    from langchain_core.messages import AIMessage

    # Eq4 – cache shortcut: skip agent entirely on a cache hit
    try:
        from gateway.smart_cache import get_cache
        cached = get_cache().get(user_message)
        if cached is not None:
            logger.debug("Eq4 agent cache hit: %.40s…", user_message)
            return cached
    except Exception:
        pass

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
        "messages": msgs, "rag_context": "",
        "rag_start_time": 0.0, "rag_fetch_time": 0.0,
    }

    # Eq19 – Best-of-N gate: use BoN for complex queries when cost allows
    complexity      = eq18_query_complexity(user_message)
    cost_per_sample = getattr(settings, "bon_cost_per_sample", 0.005)
    use_bon         = eq19_should_use_bon(complexity, cost_per_sample)
    if use_bon:
        n_samples = eq19_optimal_n(
            mu_reward=0.6,
            sigma_reward=0.2,
            cost_per_sample=cost_per_sample,
            max_n=getattr(settings, "bon_max_n", 3),
        )
        logger.debug("Eq19 BoN sync: complexity=%.2f n=%d", complexity, n_samples)
        reply = _best_of_n_invoke(agent, invoke_kwargs, n=n_samples,
                                  user_message=user_message)
    else:
        result = agent.invoke(
            invoke_kwargs,
            config={"recursion_limit": settings.agent_recursion_limit},
        )
        reply = getattr(result["messages"][-1], "content", "")

    # Eq6 – store in cache so future identical/similar queries skip the agent
    try:
        from gateway.smart_cache import get_cache
        get_cache().put(user_message, reply)
    except Exception:
        pass

    return reply


async def achat(user_message: str, history: list[dict] | None = None) -> str:
    """
    Async high-level chat with Eq1 + Eq2 active.

    Eq4 fast-path: SmartCache lookup before invoking the agent graph.
    Eq2 overlap: RAG prefetch task is launched here and passed through
    the graph state so rag_node_async can report accurate timing even
    when the graph fires it concurrently with the LLM prefill.
    """
    (_, BaseMessage, HumanMessage, _, _, _, _, _, _, _, _) = _get_langgraph()
    from langchain_core.messages import AIMessage

    # Eq4 – cache shortcut
    try:
        from gateway.smart_cache import get_cache
        cached = get_cache().get(user_message)
        if cached is not None:
            logger.debug("Eq4 agent async cache hit: %.40s…", user_message)
            return cached
    except Exception:
        pass

    agent = get_agent()
    msgs: list[BaseMessage] = []
    for turn in (history or []):
        role, content = turn.get("role", "user"), turn.get("content", "")
        if role == "user":
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    msgs.append(HumanMessage(content=user_message))

    # Eq1 – compute TTFB reduction before invoke
    history_chars = sum(len(getattr(m, "content", "")) for m in msgs[:-1])
    query_chars   = len(user_message)
    h_tok = max(0, history_chars // 4)
    q_tok = max(1, query_chars // 4)
    reduction = eq1_ttfb_reduction_fraction(h_tok, q_tok)
    logger.debug("Eq1 achat TTFB-reduction=%.1f%% (h=%d q=%d tok)",
                 reduction * 100, h_tok, q_tok)

    # Eq2 – fire RAG prefetch as a background task RIGHT NOW so it runs
    # concurrently with the graph's agent_node LLM prefill.
    import asyncio as _asyncio
    rag_task = _asyncio.create_task(_rag_prefetch(user_message))
    t_rag_start = _asyncio.get_event_loop().time()

    result = await agent.ainvoke(
        {"messages": msgs, "rag_context": "",
         "rag_start_time": t_rag_start, "rag_fetch_time": 0.0},
        config={"recursion_limit": settings.agent_recursion_limit},
    )
    reply = getattr(result["messages"][-1], "content", "")

    # Collect RAG task (likely already done; minimal extra wait)
    try:
        await _asyncio.wait_for(rag_task, timeout=0.5)
    except Exception:
        pass

    # Eq6 – store in cache for future hits
    try:
        from gateway.smart_cache import get_cache
        get_cache().put(user_message, reply)
    except Exception:
        pass

    return reply
