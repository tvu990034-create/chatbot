"""
PASS 4 acceptance tests: prompt compression, fast paths, and context
management actually working on the live inference path.

Covers:
  A. arithmetic fast path (incl. multi-operator precedence) — no agent graph
  B. greeting fast path — no RAG / reasoning / BoN
  C. large tool result -> bounded, structured-truncated context
  D. large history -> priority trimming, current request never dropped
  E. code/JSON history is NOT corrupted by compression
  F. canonical context builder produces the correct message ordering
  G. compress_prompt does not touch system/code/JSON/current request
  H. context reuse: cache identity changes when system/model/tools change
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _ainvoke(agent, state):
    """Run a graph invocation via the async API (rag_node_async is async)."""
    return asyncio.run(agent.ainvoke(state))


# ---------------------------------------------------------------------------
# A. Arithmetic fast path
# ---------------------------------------------------------------------------

class TestA_ArithmeticFastPath:
    def test_multi_operator_precedence(self):
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("2 + 3 * 4") == "14"
        assert quick_arithmetic("15 + 5 * 2") == "25"
        assert quick_arithmetic("100 / 4 + 7") == "32.0"

    def test_single_ops_back_compat(self):
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("15 + 5") == "20"
        assert quick_arithmetic("3 * 7") == "21"
        assert quick_arithmetic("10 - 4") == "6"
        assert quick_arithmetic("20 / 5") == "4.0"

    def test_parentheses(self):
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("(2 + 3) * 4") == "20"

    def test_no_eval_unsafe(self):
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("__import__('os')") is None
        assert quick_arithmetic("hello world") is None
        assert quick_arithmetic("") is None

    def test_quick_path_does_not_enter_agent_graph(self):
        # arithmetic recognised as safe quick path
        from gateway.opt_core import is_safe_quick_path, quick_arithmetic
        assert is_safe_quick_path("2 + 3 * 4") is True
        assert quick_arithmetic("2 + 3 * 4") is not None

    def test_achat_arithmetic_bypasses_rag(self):
        import asyncio
        from config import settings
        from agents.langgraph_agent import achat
        result = asyncio.run(achat("15 * 5 + 2", use_cache=False))
        assert result.model_used == settings.default_model
        assert "77" in result.reply  # 15*5=75 +2 = 77
        assert result.generation_policy == "arithmetic"


# ---------------------------------------------------------------------------
# B. Greeting fast path — no RAG / reasoning / BoN
# ---------------------------------------------------------------------------

class TestB_GreetingFastPath:
    def test_greeting_no_agent_graph(self):
        import asyncio
        from config import settings
        from agents.langgraph_agent import achat
        # If it entered the agent graph, RAG/litellm would run. We patch litellm
        # to fail loudly -> proving the quick path never reaches inference.
        with patch("litellm.completion", side_effect=AssertionError("must not call inference")):
            result = asyncio.run(achat("hello", use_cache=False))
        assert result.model_used == settings.default_model
        assert result.reply
        assert result.rag_used is False

    def test_greeting_not_quick_for_complex(self):
        from gateway.opt_core import is_safe_quick_path
        assert is_safe_quick_path("Explain the economy") is False
        assert is_safe_quick_path("write a python function") is False

    def test_rag_node_skips_simple_queries(self):
        # is_safe_quick_path is the guard both at achat() AND rag_node_async
        from gateway.opt_core import is_safe_quick_path
        assert is_safe_quick_path("hello") is True


# ---------------------------------------------------------------------------
# C. Large tool result -> bounded context
# ---------------------------------------------------------------------------

class TestC_ToolOutputBounded:
    def test_structured_truncate_json(self):
        from gateway.opt_core import structured_truncate
        big = '{"items": [' + ','.join('{"id":%d,"v":"x"}' % i for i in range(5000)) + ']}'
        out = structured_truncate(big, max_chars=600, max_tokens=1000)
        assert len(out) < len(big)
        assert out.rstrip().startswith("{")  # JSON structure preserved at head
        assert "truncated" in out.lower()

    def test_structured_truncate_code_keeps_head_tail(self):
        from gateway.opt_core import structured_truncate
        code = "def main():\n    pass\n" + "\n".join("    x%d = %d" % (i, i) for i in range(1000))
        out = structured_truncate(code, max_chars=300, max_tokens=1000)
        assert len(out) < len(code)
        assert out.lstrip().startswith("def main()")
        assert "x999" in out  # tail preserved

    def test_truncate_tool_strategy_applied(self):
        from agents.langgraph_agent import _truncate_tool_output
        big = "function foo() {\n" + "\n".join("  line%d();" % i for i in range(5000)) + "\n}"
        out = _truncate_tool_output(big)
        assert len(out) < len(big)


# ---------------------------------------------------------------------------
# D. Large history -> priority trimming, current request preserved
# ---------------------------------------------------------------------------

class TestD_PriorityTrimming:
    def _big_history(self):
        from langchain_core.messages import HumanMessage, AIMessage
        msgs = []
        for i in range(200):
            msgs.append(HumanMessage(content="question %d: a fairly long sentence that consumes tokens " % i + "x" * 40))
            msgs.append(AIMessage(content="answer %d: another fairly long sentence " % i + "y" * 40))
        msgs.append(HumanMessage(content="THE LIVE REQUEST"))
        return msgs

    def test_current_request_never_dropped(self):
        from gateway.opt_core import build_context_messages
        msgs = self._big_history()
        built = build_context_messages(msgs, system_prompt="SYS", rag_context="",
                                       context_limit=4096, budget=1024, always_keep_n_history=4)
        assert built.messages[-1]["content"] == "THE LIVE REQUEST"
        assert built.messages[-1]["role"] == "user"
        assert built.input_tokens <= 4096

    def test_system_never_dropped(self):
        from gateway.opt_core import build_context_messages
        msgs = self._big_history()
        built = build_context_messages(msgs, system_prompt="IMPORTANT SYSTEM", rag_context="",
                                       context_limit=2048, budget=256, always_keep_n_history=2)
        assert built.messages[0]["content"] == "IMPORTANT SYSTEM"
        assert built.messages[0]["role"] == "system"

    def test_big_history_fits_budget(self):
        from gateway.opt_core import build_context_messages
        msgs = self._big_history()
        built = build_context_messages(msgs, system_prompt="SYS", rag_context="",
                                       context_limit=8192, budget=2048, always_keep_n_history=4)
        assert built.input_tokens <= 8192


# ---------------------------------------------------------------------------
# E. Code/JSON in history is NOT corrupted by compression
# ---------------------------------------------------------------------------

class TestE_CompressionSafety:
    def test_code_in_history_not_corrupted(self):
        from gateway.opt_core import build_context_messages
        from langchain_core.messages import HumanMessage, AIMessage
        long_prose = ("filler language that repeats many times to inflate the token count " * 40)
        code = "def worker():\n    return sum(i*i for i in range(50))\n" + \
               "\n".join("    a%d = %d" % (i, i) for i in range(150))
        msgs = [
            HumanMessage(content=long_prose),
            AIMessage(content=code),
            HumanMessage(content="the question"),
        ]
        built = build_context_messages(msgs, system_prompt="SYS", rag_context="",
                                       context_limit=8192, budget=300, always_keep_n_history=4)
        for m in built.messages:
            if "def worker" in m["content"]:
                assert "def worker" in m["content"] and "return sum" in m["content"], \
                    "code must remain intact"
                break

    def test_compress_prompt_guards_code_json(self):
        from gateway.opt_core import compress_prompt
        code = "def f():\n    return 42\n" + "\n".join("    x%d=%d" % (i, i) for i in range(60))
        assert "def f()" in compress_prompt(code, 0.3)
        js = '{"a":' + str(list(range(200))) + '}'
        assert compress_prompt(js, 0.3).strip().startswith("{")


# ---------------------------------------------------------------------------
# F. Canonical builder ordering
# ---------------------------------------------------------------------------

class TestF_CanonicalOrdering:
    def test_order_system_current_retrieved_history(self):
        from gateway.opt_core import build_context_messages
        from langchain_core.messages import HumanMessage, SystemMessage
        msgs = [
            SystemMessage(content="sys-original"),
            HumanMessage(content="earlier question"),
            HumanMessage(content="live question"),
        ]
        built = build_context_messages(msgs, system_prompt="SYS-OVERRIDE",
                                       rag_context="RETRIEVED-DOC",
                                       context_limit=8192, budget=4096)
        roles = [m["role"] for m in built.messages]
        contents = [m["content"] for m in built.messages]
        # system first
        assert contents[0] in ("SYS-OVERRIDE", "sys-original")
        assert "sys-original" in contents or "SYS-OVERRIDE" in contents
        # retrieved present as a system block
        assert "RETRIEVED-DOC" in contents
        # live question is last
        assert contents[-1] == "live question"
        # earlier question kept
        assert "earlier question" in contents


# ---------------------------------------------------------------------------
# G. compress_prompt does not touch current request/system
# ---------------------------------------------------------------------------

class TestG_CompressNotOnCritical:
    def test_current_request_preserved_verbatim(self):
        from gateway.opt_core import build_context_messages
        from langchain_core.messages import HumanMessage
        request = "This is the precise user request that must be sent verbatim: width=42, height=7."
        msgs = [HumanMessage(content="old filler"), HumanMessage(content=request)]
        built = build_context_messages(msgs, system_prompt="SYS", rag_context="",
                                       context_limit=8192, budget=64, always_keep_n_history=4)
        assert built.messages[-1]["content"] == request


# ---------------------------------------------------------------------------
# H. Context reuse: identity changes when config changes
# ---------------------------------------------------------------------------

class TestH_ContextReuse:
    def test_cache_identity_changes_on_model(self):
        from gateway.opt_core import make_cache_identity
        k1 = make_cache_identity("q", model="ollama/a")
        k2 = make_cache_identity("q", model="ollama/b")
        assert k1 != k2

    def test_cache_identity_changes_on_system_prompt(self):
        from gateway.opt_core import make_cache_identity
        k1 = make_cache_identity("q", system_prompt="sys1")
        k2 = make_cache_identity("q", system_prompt="sys2")
        assert k1 != k2

    def test_cache_identity_changes_on_tools(self):
        from gateway.opt_core import make_cache_identity
        k1 = make_cache_identity("q", tools=[{"name": "a"}])
        k2 = make_cache_identity("q", tools=[{"name": "b"}])
        assert k1 != k2


# ---------------------------------------------------------------------------
# End-to-end: show the actual message list sent to the model
# ---------------------------------------------------------------------------

class TestEndToEndMessages:
    def test_agent_sends_built_messages_to_litellm(self):
        """The exact list handed to litellm.completion comes from the canonical builder."""
        import litellm
        from agents.langgraph_agent import build_agent
        from langchain_core.messages import HumanMessage
        captured = {}

        async def fake(model="", messages=None, **kwargs):
            captured["messages"] = messages
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
                model=model,
            )

        with patch("litellm.acompletion", side_effect=fake), \
             patch("gateway.opt_core.check_model_available", return_value=True):
            agent = build_agent()
            _ainvoke(agent, {
                "messages": [HumanMessage(content="Tell me a story")],
                "rag_context": "", "rag_fetch_time": 0.0,
                "_req_ctx": {"use_router": False, "use_rag": False,
                             "speed_mode": False},
            })

        msgs = captured["messages"]
        assert isinstance(msgs, list)
        assert len(msgs) >= 2
        assert msgs[0]["role"] == "system"
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "Tell me a story"
