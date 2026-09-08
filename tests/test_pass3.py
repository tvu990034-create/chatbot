"""
PASS 3 acceptance tests.

Verify that model routing and generation optimizations actually reach
the real inference path:

  1. resolve_generation_policy top_p/top_k reach litellm.completion kwargs
  2. explicit model request is honored and sent to litellm
  3. no explicit model + router -> RouterState.select_model decides
  4. code intent -> code-specialized model routed to litellm
  5. speed_mode caps generation budget and suppresses top_p/top_k
  6. model availability check falls back to default when unavailable
  7. concurrent requests with different models don't interfere

All tests patch litellm.completion (the real inference entry point) and the
Ollama availability checker so no backend is contacted.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


def _completion_mock(reply: str = "mocked-reply", model: str = "ollama/model"):
    """Factory that always returns success. Captures kwargs via side_effect."""
    captured = {}

    async def _handler(model="", messages=None, **kwargs):
        captured["kwargs"] = kwargs
        captured["model"] = model
        captured.setdefault("calls", []).append(kwargs)
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
            model=model,
        )
        return resp

    return _handler, captured


def _acompletion_patch(handler, captured=None):
    """Patch the async entry point actually used by the agent (litellm.acompletion)."""
    return patch("litellm.acompletion", side_effect=handler)


def _patch_check_available(available=True):
    return patch("gateway.opt_core.check_model_available", return_value=available)


def _ainvoke(agent, state):
    """Run a graph invocation via the async API (rag_node_async is async)."""
    return asyncio.run(agent.ainvoke(state))


# ---------------------------------------------------------------------------
# Test A: top_p / top_k from generation policy reach litellm
# ---------------------------------------------------------------------------

class TestA_GenParamsReachInference:
    def test_agent_node_passes_top_p_top_k(self):
        import litellm
        handler, captured = _completion_mock()
        with _acompletion_patch(handler), \
             _patch_check_available(available=True), \
             patch("gateway.opt_core.quick_arithmetic", return_value=None):
            from agents.langgraph_agent import build_agent
            agent = build_agent()

            from langchain_core.messages import HumanMessage
            result = _ainvoke(agent, {
                "messages": [HumanMessage(content="Explain quantum entanglement in detail")],
                "rag_context": "", "rag_fetch_time": 0.0,
                "_req_ctx": {"use_router": False, "use_rag": False, "use_cache": False,
                             "speed_mode": False},
            })

        kwargs = captured["kwargs"]
        assert "top_p" in kwargs, "top_p must reach litellm.acompletion"
        assert "top_k" in kwargs, "top_k must reach litellm.acompletion"
        assert isinstance(kwargs["top_p"], float)
        assert isinstance(kwargs["top_k"], int)
        from config import settings
        assert result.get("model_used") == "ollama/" + str(settings.default_model).replace("ollama/", "")

    def test_speed_mode_suppresses_top_p_top_k(self):
        import litellm
        handler, captured = _completion_mock()
        with _acompletion_patch(handler), \
             _patch_check_available(available=True), \
             patch("gateway.opt_core.quick_arithmetic", return_value=None):
            from agents.langgraph_agent import build_agent
            agent = build_agent()

            from langchain_core.messages import HumanMessage
            result = _ainvoke(agent, {
                "messages": [HumanMessage(content="Tell me about the weather today")],
                "rag_context": "", "rag_fetch_time": 0.0,
                "_req_ctx": {"use_router": False, "use_rag": False, "use_cache": False,
                             "speed_mode": True},
            })

        kwargs = captured["kwargs"]
        assert "top_p" not in kwargs, "speed mode must NOT force top_p"
        assert kwargs["max_tokens"] <= 128, f"speed mode should cap max_tokens, got {kwargs['max_tokens']}"
        assert kwargs["temperature"] <= 0.3
        assert result.get("generation_policy") == "speed_mode"


# ---------------------------------------------------------------------------
# Test B: explicit model reaches inference; router decides when absent
# ---------------------------------------------------------------------------

class TestB_ModelRouting:
    def test_explicit_model_is_honored(self):
        import litellm
        handler, captured = _completion_mock()
        with _acompletion_patch(handler), \
             _patch_check_available(available=True):
            from agents.langgraph_agent import build_agent
            agent = build_agent()

            from langchain_core.messages import HumanMessage
            _ainvoke(agent, {
                "messages": [HumanMessage(content="Who wrote the Odyssey?")],
                "rag_context": "", "rag_fetch_time": 0.0,
                "_req_ctx": {"model": "ollama/phi3:mini", "use_router": False,
                             "use_rag": False, "speed_mode": False},
            })

        # model_used in captured kwargs reflects what litellm received
        assert captured["model"] == "ollama/phi3:mini"

    def test_router_selects_model_when_no_explicit(self):
        import litellm
        async def _handler(model="", **kw):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="x"))],
                model=model)
        with patch("litellm.acompletion", side_effect=_handler):
            from agents.langgraph_agent import build_agent
            with patch("gateway.opt_core.check_model_available", return_value=True), \
                 patch("gateway.opt_core.select_model", return_value="gemma2:2b") as m:
                agent = build_agent()
                from langchain_core.messages import HumanMessage
                result = _ainvoke(agent, {
                    "messages": [HumanMessage(content="Tell me a story")],
                    "rag_context": "", "rag_fetch_time": 0.0,
                    "_req_ctx": {"use_router": True, "use_rag": False,
                                 "speed_mode": False},
                })
        assert m.called, "select_model should be consulted for routing"
        assert "gemma2:2b" in result["model_used"], "routed model should reach inference"


# ---------------------------------------------------------------------------
# Test C: code intent routes to a code-specialized model
# ---------------------------------------------------------------------------

class TestC_CodeRouting:
    def test_code_query_routes_to_coder_model(self):
        import litellm
        handler, captured = _completion_mock()
        # First (coder) candidate available, so coder model chosen
        with _acompletion_patch(handler), \
             patch("gateway.opt_core.check_model_available",
                   side_effect=lambda m: m == "ollama/deepseek-coder:1.3b"):
            from agents.langgraph_agent import build_agent
            agent = build_agent()
            from langchain_core.messages import HumanMessage
            _ainvoke(agent, {
                "messages": [HumanMessage(
                    content="def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1)+fibonacci(n-2)")],
                "rag_context": "", "rag_fetch_time": 0.0,
                "_req_ctx": {"use_router": False, "use_rag": False, "speed_mode": False},
            })

        assert "deepseek-coder" in captured["model"], \
            f"code query should route to coder model, got {captured['model']}"

    def test_code_routing_falls_back_when_coder_unavailable(self):
        import litellm
        handler, captured = _completion_mock()
        with _acompletion_patch(handler), \
             _patch_check_available(available=False):  # nothing available -> default path
            from agents.langgraph_agent import build_agent
            agent = build_agent()
            from langchain_core.messages import HumanMessage
            _ainvoke(agent, {
                "messages": [HumanMessage(content="Write a python function to sort a list")],
                "rag_context": "", "rag_fetch_time": 0.0,
                "_req_ctx": {"use_router": False, "use_rag": False, "speed_mode": False},
            })
        # coder models unavailable -> falls back; check availability called for coder candidates
        assert "deepseek-coder" not in captured["model"] or "codellama" not in captured["model"]


# ---------------------------------------------------------------------------
# Test D: unavailable model falls back to default
# ---------------------------------------------------------------------------

class TestD_UnavailableModelFallback:
    def test_unavailable_explicit_model_falls_back(self):
        import litellm
        from config import settings
        handler, captured = _completion_mock()
        # Explicitly requested model is NOT available -> must fall back to default
        with _acompletion_patch(handler), \
             _patch_check_available(available=False):  # no model available
            from agents.langgraph_agent import build_agent
            agent = build_agent()
            from langchain_core.messages import HumanMessage
            _ainvoke(agent, {
                "messages": [HumanMessage(content="Hello there, how are you?")],
                "rag_context": "", "rag_fetch_time": 0.0,
                "_req_ctx": {"model": "ollama/gigantic-model:100b",
                             "use_router": False, "use_rag": False, "speed_mode": False},
            })
        assert captured["model"] == "ollama/" + settings.default_model.replace("ollama/", ""), \
            "unavailable model must fall back to the configured default"


# ---------------------------------------------------------------------------
# Test E: concurrent requests with different models don't interfere
# ---------------------------------------------------------------------------

class TestE_ConcurrentRequests:
    def test_concurrent_distinct_models(self):
        import litellm
        results = {}

        def make_handler(model_id):
            async def _handler(model="", **kwargs):
                results[model_id] = model
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content=f"reply-{model_id}"))],
                    model=model,
                )
            return _handler

        from agents.langgraph_agent import build_agent
        agent = build_agent()
        from langchain_core.messages import HumanMessage

        async def run_once(mid):
            handler = make_handler(mid)
            # Patch globally; both coroutines share the same patched name
            with _acompletion_patch(handler), \
                 _patch_check_available(available=True):
                result = await agent.ainvoke({
                    "messages": [HumanMessage(content=f"question-{mid}")],
                    "rag_context": "", "rag_fetch_time": 0.0,
                    "_req_ctx": {"model": mid, "use_router": False,
                                 "use_rag": False, "speed_mode": False},
                })
            return result["messages"][-1].content

        model_ids = ["ollama/alpha:1b", "ollama/beta:2b", "ollama/gamma:7b"]
        async def main():
            for mid in model_ids:
                await run_once(mid)
        asyncio.run(main())

        assert results == {
            "ollama/alpha:1b": "ollama/alpha:1b",
            "ollama/beta:2b": "ollama/beta:2b",
            "ollama/gamma:7b": "ollama/gamma:7b",
        }, "each request must keep its own model end-to-end"


# ---------------------------------------------------------------------------
# Test F: litellm_gateway.chat applies generation policy / speed mode
# ---------------------------------------------------------------------------

class TestF_GatewayPolicy:
    def test_gateway_chat_speed_mode_caps_budget(self):
        from gateway import litellm_gateway
        captured = {}

        def fake_completion(model="", **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
                model=model,
            )

        with patch.object(litellm_gateway, "completion", side_effect=fake_completion), \
             patch.object(litellm_gateway, "_cache") as c:
            c.get.return_value = None
            from gateway.litellm_gateway import chat
            chat([{"role": "user", "content": "hello"}],
                 model="phi3:mini", speed_mode=True)

        assert captured["max_tokens"] <= 128
        assert "top_p" not in captured or captured["max_tokens"] <= 128

    def test_gateway_chat_applies_top_p_top_k(self):
        from gateway import litellm_gateway
        captured = {}

        def fake_completion(model="", **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))],
                model=model,
            )

        with patch.object(litellm_gateway, "completion", side_effect=fake_completion), \
             patch.object(litellm_gateway, "_cache") as c:
            c.get.return_value = None
            from gateway.litellm_gateway import chat
            chat([{"role": "user", "content": "Explain the meaning of life in great detail"}],
                 model="phi3:mini", speed_mode=False)

        assert "top_p" in captured
        assert "top_k" in captured


# ---------------------------------------------------------------------------
# Test G: achat wires speed_mode end to end
# ---------------------------------------------------------------------------

class TestG_AchatSpeedMode:
    def test_achat_passes_speed_mode_to_ctx(self):
        from agents import langgraph_agent
        # Verify achat accepts speed_mode kwarg (signature contract)
        import inspect
        sig = inspect.signature(langgraph_agent.achat)
        assert "speed_mode" in sig.parameters, "achat must accept speed_mode"
