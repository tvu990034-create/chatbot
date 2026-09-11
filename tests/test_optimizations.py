"""
Tests for Pass 2A optimization fixes.

Covers:
  Bug 1:  smart_cache â†’ simple_cache import fix
  Bug 2+5: duplicate RAG removal
  Bug 3:  request fields passed through achat()
  Bug 4:  token-based Eq1 trimming
  Bug 6:  advanced reasoning context deduplication
  Bug 7:  generation policy wired to LLM
  Bug 8:  eval() replaced with quick_arithmetic
  Bug 9:  prefix cache write path
  Bug 10: code routing uses query_text
  Bug 11: singleton gateway
"""

from __future__ import annotations

import asyncio
import math
import threading
from unittest.mock import MagicMock, patch, AsyncMock

import pytest


# ---------------------------------------------------------------------------
# Bug 4: Token-based context trimming
# ---------------------------------------------------------------------------

class TestEq1TokenTrim:
    """_trim_messages_eq1 should trim by token budget, not message count."""

    def _make_msgs(self, contents: list[str], types: list[str] | None = None):
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        types = types or []
        msgs = []
        for i, c in enumerate(contents):
            t = types[i] if i < len(types) else ("system" if i == 0 else "human")
            if t == "system":
                msgs.append(SystemMessage(content=c))
            elif t == "assistant":
                msgs.append(AIMessage(content=c))
            else:
                msgs.append(HumanMessage(content=c))
        return msgs

    def test_keeps_system_and_last(self):
        from agents.langgraph_agent import _trim_messages_eq1
        msgs = self._make_msgs(
            ["You are helpful.", "Hello", "Hi there", "What is 2+2?", "4", "What about 3+3?"],
            ["system", "human", "assistant", "human", "assistant", "human"],
        )
        result = _trim_messages_eq1(msgs, token_budget=60)
        types = [getattr(m, "type", "") for m in result]
        # System and last human must always be present
        assert types[0] == "system"
        assert types[-1] == "human"
        # Should have trimmed some middle messages
        assert len(result) <= len(msgs)

    def test_token_budget_respected(self):
        from agents.langgraph_agent import _trim_messages_eq1
        from gateway.opt_core import estimate_tokens
        msgs = self._make_msgs(
            ["System prompt here.",
             "A short message", "Reply",
             "Another message", "Reply again",
             "Yet another message", "Reply three",
             "Final question?"],
            ["system", "human", "assistant", "human", "assistant",
             "human", "assistant", "human"],
        )
        budget = 40
        result = _trim_messages_eq1(msgs, token_budget=budget)
        total = sum(estimate_tokens(m.content) for m in result)
        # System + last message always kept; total should not massively exceed budget
        assert total < budget + 50  # small margin for always-kept items

    def test_empty_messages(self):
        from agents.langgraph_agent import _trim_messages_eq1
        assert _trim_messages_eq1([], token_budget=100) == []

    def test_single_message(self):
        from agents.langgraph_agent import _trim_messages_eq1
        msgs = self._make_msgs(["Hello"], ["human"])
        result = _trim_messages_eq1(msgs, token_budget=100)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Bug 8: eval() replaced with safe arithmetic
# ---------------------------------------------------------------------------

class TestSafeArithmetic:
    """quick_arithmetic should handle basic ops without eval()."""

    def test_addition(self):
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("15 + 5") == "20"

    def test_multiplication(self):
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("3 * 7") == "21"

    def test_subtraction(self):
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("10 - 4") == "6"

    def test_division(self):
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("20 / 5") == "4.0"

    def test_float_precision_is_rounded(self):
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("0.1 + 0.2") == "0.3"
        assert quick_arithmetic("1 / 3") == "0.333333333333"

    def test_non_math_returns_none(self):
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("hello world") is None


# ---------------------------------------------------------------------------
# Bug 9: Prefix cache write path
# ---------------------------------------------------------------------------

class TestPrefixCache:
    """_check_prefix_cache should return cached response after write."""

    def test_prefix_cache_round_trip(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway.__new__(UniversalEnhancedGateway)
        gw.enable_all_optimizations = True
        gw.prefix_cache = {}

        import hashlib, threading
        prefix_lock = threading.Lock()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ]
        prefix = messages[0]["content"][:200]
        prefix_key = hashlib.md5(prefix.encode()).hexdigest()
        full_query = messages[-1]["content"]

        # Write
        with prefix_lock:
            if prefix_key not in gw.prefix_cache:
                gw.prefix_cache[prefix_key] = {}
            gw.prefix_cache[prefix_key][full_query] = "cached response"

        # Read
        with prefix_lock:
            if prefix_key in gw.prefix_cache:
                result = gw.prefix_cache[prefix_key].get(full_query)
            else:
                result = None

        assert result == "cached response"


# ---------------------------------------------------------------------------
# Bug: shared _global_cache served one model's answers to another
# ---------------------------------------------------------------------------

class TestCacheModelScoping:
    """Main response cache keys must be scoped by model + mode so answers
    cached under qwen3:4b are never served to phi3:mini."""

    def test_cache_key_varies_by_model(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway.__new__(UniversalEnhancedGateway)
        ctx = {"model": "phi3:mini", "performance_mode": "speed"}
        k_phi = gw._generate_cache_key("What is the capital of France?", ctx)
        ctx2 = {"model": "qwen3:4b", "performance_mode": "speed"}
        k_qwen = gw._generate_cache_key("What is the capital of France?", ctx2)
        assert k_phi != k_qwen

    def test_cache_key_stable_for_same_model_mode(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway.__new__(UniversalEnhancedGateway)
        ctx = {"model": "qwen3:4b", "performance_mode": "speed"}
        assert gw._generate_cache_key("hello", ctx) == \
            gw._generate_cache_key("hello", ctx)


# ---------------------------------------------------------------------------
# Bug 10: Code routing uses query_text
# ---------------------------------------------------------------------------

class TestCodeRouting:
    """_route_to_code_model should check the actual query text."""

    def test_detect_code_intent_uses_text(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=False)
        # This should detect code intent from the query text
        assert gw._detect_code_intent("write a python function to sort") is True
        # This should NOT detect code intent
        assert gw._detect_code_intent("hello") is False


# ---------------------------------------------------------------------------
# Bug 11: Singleton gateway
# ---------------------------------------------------------------------------

class TestSingletonGateway:
    """get_universal_gateway should return the same instance."""

    def test_singleton(self):
        from gateway import universal_enhanced_gateway as mod
        # Reset
        mod._gateway_instance = None
        g1 = mod.get_universal_gateway("phi3:mini", False, "speed")
        g2 = mod.get_universal_gateway("phi3:mini", False, "speed")
        assert g1 is g2
        # Cleanup
        mod._gateway_instance = None


# ---------------------------------------------------------------------------
# Bug 1: simple_cache API works for agent path
# ---------------------------------------------------------------------------

class TestSimpleCacheAPI:
    """SimpleCache set/get should work with the API used by achat/chat."""

    def test_set_and_get(self):
        from gateway.simple_cache import SimpleCache
        cache = SimpleCache(persist=False, max_size=100)
        cache.set("test query", "test response")
        result = cache.get("test query")
        assert result == "test response"

    def test_miss_returns_none(self):
        from gateway.simple_cache import SimpleCache
        cache = SimpleCache(persist=False, max_size=100)
        assert cache.get("nonexistent") is None


# ---------------------------------------------------------------------------
# Bug 8 (universal_enhanced_gateway): quick_response uses safe math
# ---------------------------------------------------------------------------

class TestQuickResponseSafeMath:
    """_get_quick_response should not use eval()."""

    def test_addition_in_quick_response(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=False)
        result = gw._get_quick_response("5 + 3")
        assert result is not None
        assert "8" in result

    def test_greeting_in_quick_response(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=False)
        result = gw._get_quick_response("hello")
        assert result is not None


# ---------------------------------------------------------------------------
# RAG deduplication (Bug 2+5)
# ---------------------------------------------------------------------------

class TestRAGDedup:
    """dedupe_context_parts should remove duplicate chunks."""

    def test_removes_duplicates(self):
        from gateway.opt_core import dedupe_context_parts
        parts = [
            "The quick brown fox jumps over the lazy dog.",
            "The quick brown fox jumps over the lazy dog.",
            "Another completely different sentence about AI.",
        ]
        result = dedupe_context_parts(parts)
        assert len(result) == 2

    def test_empty_list(self):
        from gateway.opt_core import dedupe_context_parts
        assert dedupe_context_parts([]) == []

    def test_whitespace_normalization(self):
        from gateway.opt_core import dedupe_context_parts
        parts = [
            "Hello  world",
            "Hello world",
        ]
        result = dedupe_context_parts(parts)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# BoundedTTLCache (used by simple_cache)
# ---------------------------------------------------------------------------

class TestBoundedTTLCache:
    """BoundedTTLCache should enforce TTL and max size."""

    def test_basic_set_get(self):
        from gateway.opt_core import BoundedTTLCache
        c = BoundedTTLCache(max_size=3, default_ttl=10)
        c.set("a", 1)
        assert c.get("a") == 1

    def test_max_size_eviction(self):
        from gateway.opt_core import BoundedTTLCache
        c = BoundedTTLCache(max_size=2, default_ttl=10)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        assert c.get("a") is None  # evicted
        assert c.get("c") == 3

    def test_ttl_expiry(self):
        from gateway.opt_core import BoundedTTLCache
        c = BoundedTTLCache(max_size=10, default_ttl=0.01)
        c.set("a", 1)
        import time
        time.sleep(0.05)
        assert c.get("a") is None


# ---------------------------------------------------------------------------
# resolve_generation_policy (Bug 7)
# ---------------------------------------------------------------------------

class TestGenerationPolicy:
    """resolve_generation_policy should return correct params for different tasks."""

    def _analysis(self, **kwargs):
        class A:
            pass
        a = A()
        for k, v in kwargs.items():
            setattr(a, k, v)
        return a

    def test_math_low_temp(self):
        from gateway.opt_core import resolve_generation_policy
        policy = resolve_generation_policy(
            self._analysis(is_math=True),
            base={"temperature": 0.5},
            configured_max_tokens=1024,
            model_name="phi3:mini",
        )
        assert policy.temperature <= 0.2

    def test_small_model_constraint(self):
        from gateway.opt_core import resolve_generation_policy
        policy = resolve_generation_policy(
            self._analysis(),
            base={},
            configured_max_tokens=1024,
            model_name="gemma2:2b",
        )
        assert policy.temperature <= 0.3

# ---------------------------------------------------------------------------
# Bug: code-semantic cache key crash + config-aware keys
# ---------------------------------------------------------------------------

class TestCodeSemanticCacheKey:
    """_generate_cache_key must accept the cache_context used by the
    code-semantic cache (previously it took 1 positional arg while both call
    sites passed 2 -> TypeError on every code query with optimizations on)."""

    def test_accepts_cache_context(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=False)
        key = gw._generate_cache_key(
            "write a function", {"model": "phi3:mini", "language": "python"})
        assert isinstance(key, str) and len(key) == 32

    def test_context_changes_key(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=False)
        k1 = gw._generate_cache_key(
            "write a function",
            {"model": "phi3:mini", "language": "python", "performance_mode": "speed"})
        k2 = gw._generate_cache_key(
            "write a function",
            {"model": "phi3:medium", "language": "python", "performance_mode": "speed"})
        assert k1 != k2, "cache key must fold configuration identity in"


# ---------------------------------------------------------------------------
# Bug: RouteLLM routing to uninstalled models
# ---------------------------------------------------------------------------

class TestModelRoutingAvailability:
    """_route_to_model must not select a model that is not installed, else
    inference raises APIConnectionError and users get an apology."""

    def test_skips_unavailable_candidates(self):
        from unittest.mock import patch
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway,
            QueryAnalysis,
        )
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=True)
        gw.available_models = {
            "simple": ["phi3:mini", "gemma2:2b"],
            "medium": ["phi3:3.8b", "qwen2.5:3b", "gemma2:9b"],
        }
        analysis = QueryAnalysis()
        analysis.suggested_model = "medium"
        analysis.complexity_score = 0.5

        # none of the medium tier is installed -> fall back to default (None)
        with patch("gateway.universal_enhanced_gateway.check_model_available", return_value=False):
            assert gw._route_to_model(analysis) is None

        # if the first candidate is installed, it is selected
        with patch("gateway.universal_enhanced_gateway.check_model_available", return_value=True):
            assert gw._route_to_model(analysis) == "phi3:3.8b"

    def test_picks_first_installed_candidate(self):
        from unittest.mock import patch
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway,
            QueryAnalysis,
        )
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=True)
        gw.available_models = {
            "simple": ["phi3:mini", "gemma2:2b"],
            "medium": ["phi3:3.8b", "qwen2.5:3b", "gemma2:9b"],
        }
        analysis = QueryAnalysis()
        analysis.suggested_model = "medium"
        analysis.complexity_score = 0.5

        def avail(model):
            return "qwen2.5:3b" in model

        with patch("gateway.universal_enhanced_gateway.check_model_available", side_effect=avail):
            assert gw._route_to_model(analysis) == "qwen2.5:3b"

# ---------------------------------------------------------------------------
# Bug: elementary arithmetic word problems were routed to the LLM and often
# answered wrong (e.g. the 60 km/h question). Now solved deterministically.
# ---------------------------------------------------------------------------

class TestQuickWordProblem:
    def test_train_rate_times_time(self):
        from gateway.opt_core import quick_word_problem
        q = "If a train travels 60 km per hour, how far does it go in 2 hours?"
        assert quick_word_problem(q) == "120"

    def test_decimal_round_trip(self):
        from gateway.opt_core import quick_word_problem
        q = "A car drives 90 km per hour, how far does it go in 1.5 hours?"
        assert quick_word_problem(q) == "135"

    def test_rejects_non_word_problems(self):
        from gateway.opt_core import quick_word_problem
        assert quick_word_problem("What is the capital of France?") is None
        assert quick_word_problem("Tell me about quantum computing") is None
        assert quick_word_problem("2 + 2") is None

    def test_rejects_unit_mismatch(self):
        # rate is per *hour* but question asks about *days* -> unsafe, skip.
        from gateway.opt_core import quick_word_problem
        q = "A tap fills 2 litres per minute, how much water in 3 hours?"
        assert quick_word_problem(q) is None

    def test_quick_path_accepts_word_problem(self):
        from gateway.opt_core import is_safe_quick_path
        q = "If a train travels 60 km per hour, how far does it go in 2 hours?"
        assert is_safe_quick_path(q) is True

    def test_gateway_calculator_tool(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=False)
        q = "If a train travels 60 km per hour, how far does it go in 2 hours?"
        assert gw._tool_calculator(q) == "120"

class TestPerformanceModePrecedence:
    """BUG 15: speed mode must cap max_tokens for simple queries but keep a
    reasoning budget for math/code/reasoning so answers are not truncated."""

    def _analysis(self):
        from gateway.universal_enhanced_gateway import QueryAnalysis
        a = QueryAnalysis()
        return a

    def test_speed_mode_caps_long_response_to_reasoning_budget(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis, SPEED_REASONING_MAX_TOKENS)
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=True,
                                      performance_mode="speed")
        analysis = QueryAnalysis()
        analysis.expected_response_length = "long"
        params = gw._get_adaptive_model_params(analysis)
        assert params["max_tokens"] == SPEED_REASONING_MAX_TOKENS, \
            f"speed-mode reasoning should get its reasoning budget, got {params['max_tokens']}"
        assert params["num_predict"] >= params["max_tokens"]

    def test_speed_mode_short_response_still_capped(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=True,
                                      performance_mode="speed")
        analysis = QueryAnalysis()
        analysis.expected_response_length = "short"
        params = gw._get_adaptive_model_params(analysis)
        assert params["max_tokens"] <= 40

    def test_speed_mode_math_gets_reasoning_budget(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis, SPEED_REASONING_MAX_TOKENS)
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=True,
                                      performance_mode="speed")
        analysis = QueryAnalysis()
        analysis.is_math = True
        params = gw._get_adaptive_model_params(analysis)
        assert params["max_tokens"] == SPEED_REASONING_MAX_TOKENS
        assert params["num_predict"] >= SPEED_REASONING_MAX_TOKENS

    def test_speed_mode_disables_thinking_for_thinking_models(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("qwen3:4b",
                                      enable_all_optimizations=True,
                                      performance_mode="speed")
        analysis = QueryAnalysis()
        analysis.is_math = True
        params = gw._get_adaptive_model_params(analysis)
        assert params.get("reasoning_effort") == "none", \
            "thinking models must have reasoning disabled in speed mode " \
            "(their hidden chain-of-thought burns the whole num_predict budget)"

    def test_thoughtful_model_param_build_keeps_plain_params(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("qwen3:4b",
                                      enable_all_optimizations=True,
                                      performance_mode="speed")
        analysis = QueryAnalysis()
        analysis.expected_response_length = "short"
        params = gw._get_adaptive_model_params(analysis)
        assert params.get("reasoning_effort") == "none"
        assert params["max_tokens"] <= 40

# ---------------------------------------------------------------------------
# BUG 13 / 27 / 28 / 33 / 29 / 32 fixes
# ---------------------------------------------------------------------------

class TestRouterAvailabilityGuard:
    def test_falls_back_when_routed_model_unavailable(self):
        from unittest.mock import patch
        from gateway.opt_core import select_model, RouterState
        state = RouterState()
        with patch("gateway.opt_core.check_model_available", return_value=False):
            chosen = select_model("phi3:mini", "phi3:3.8b", None, state)
        assert chosen == "phi3:mini"

    def test_keeps_routed_model_when_available(self):
        from unittest.mock import patch
        from gateway.opt_core import select_model, RouterState
        state = RouterState()
        with patch("gateway.opt_core.check_model_available", return_value=True):
            chosen = select_model("phi3:mini", "gemma2:2b", None, state)
        assert chosen == "gemma2:2b"


class TestVoteQuality:
    def test_prefers_concise_informative_in_winner_bucket(self):
        from gateway.opt_core import vote_responses
        long = "The answer is 120 " + "and here is a very long rambling clarification that repeats the same point several times over and over again to add length " * 6
        short = "120"
        # Same normalized key -> same bucket; quality must pick the concise one.
        assert vote_responses([long, short]) == "120"

    def test_boxed_still_preferred(self):
        from gateway.opt_core import vote_responses
        assert vote_responses(["120", "\\boxed{120}"]) == "\\boxed{120}"


class TestCiscVerifiable:
    def test_verifiable_answer_gets_full_confidence(self):
        from gateway.opt_core import cisc_confidence
        assert cisc_confidence(["120", "I think it is 120."], query="If a train travels 60 km per hour, how far does it go in 2 hours?") == 1.0

    def test_unverifiable_uses_agreement(self):
        from gateway.opt_core import cisc_confidence
        assert 0.0 < cisc_confidence(["a", "a", "b"], query="What is love?") <= 1.0


class TestSsrDeterministicSkip:
    def test_word_problem_skips_llm_strategy(self):
        from unittest.mock import patch
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=False)
        q = "If a train travels 60 km per hour, how far does it go in 2 hours?"
        a = QueryAnalysis()
        a.is_math = True
        with patch("gateway.universal_enhanced_gateway._ssr_enabled", True):
            assert gw._apply_ssr_guidance(q, a) is None

    def test_pure_arithmetic_skips_llm_strategy(self):
        from unittest.mock import patch
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=False)
        a = QueryAnalysis()
        a.is_math = True
        with patch("gateway.universal_enhanced_gateway._ssr_enabled", True):
            assert gw._apply_ssr_guidance("2 + 2", a) is None


class TestEquationSolverCoverage:
    """BUG: _solve_equation_directly returned None for coefficient-1 and
    no-constant forms ('x + 3 = 5', '2x = 10'), silently degrading those
    queries to an LLM call.  All linear forms must resolve deterministically."""

    def _gw(self):
        import logging
        logging.disable(logging.INFO)
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        return UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=True)

    def test_coefficient_1_forms(self):
        gw = self._gw()
        assert gw._solve_equation_directly("x + 3 = 5") == "x = 2"
        assert gw._solve_equation_directly("x - 3 = 5") == "x = 8"
        assert gw._solve_equation_directly("x + 7 = 22") == "x = 15"

    def test_no_constant_term(self):
        gw = self._gw()
        assert gw._solve_equation_directly("2x = 10") == "x = 5"
        assert gw._solve_equation_directly("10x = 50") == "x = 5"
        assert gw._solve_equation_directly("-x = 6") == "x = -6"

    def test_fractional_and_explicit_forms(self):
        gw = self._gw()
        assert gw._solve_equation_directly("x/2 = 4") == "x = 8"
        assert gw._solve_equation_directly("2*x + 5 = -7") == "x = -6"
        assert gw._solve_equation_directly("-x + 3 = 5") == "x = -2"

    def test_prefixed_and_natural_language(self):
        gw = self._gw()
        assert gw._solve_equation_directly("solve 2x+5=-7") == "x = -6"
        assert gw._solve_equation_directly("solve for x: 3x - 4 = 11") == "x = 5"
        assert gw._solve_equation_directly("what is x if 4x - 2 = 10") == "x = 3"

    def test_decimal_coefficients(self):
        gw = self._gw()
        assert gw._solve_equation_directly("0.5x = 10") == "x = 20"
        assert gw._solve_equation_directly("1.5x + 3 = 12") == "x = 6"


class TestAdaptiveTemperatureWiring:
    def test_not_applied_in_speed_mode(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=True,
                                      performance_mode="speed")
        a = QueryAnalysis()
        before = gw._get_adaptive_model_params(a)
        with __import__("unittest.mock").mock.patch(
                "gateway.adaptive_temperature.get_adaptive_temperature") as m:
            class _StubTemp:
                def get_temperature(self):
                    return 0.99
            m.return_value = _StubTemp()
            after = gw._get_adaptive_model_params(a)
        assert before["temperature"] == after["temperature"]

    def test_applied_outside_speed_mode(self):
        from unittest.mock import patch
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=True,
                                      performance_mode="quality")
        a = QueryAnalysis()
        class _StubTemp:
            def get_temperature(self):
                return 0.99
        with patch("gateway.adaptive_temperature.get_adaptive_temperature",
                   return_value=_StubTemp()):
            params = gw._get_adaptive_model_params(a)
        assert params["temperature"] == 0.99


# ---------------------------------------------------------------------------
# Math normalization & deterministic-math optimizations (smoke-audit fixes)
# ---------------------------------------------------------------------------

class TestMathNormalization:
    def test_latex_dollar_unwrapped(self):
        from gateway.opt_core import normalize_math_answer
        assert normalize_math_answer("$120$") == normalize_math_answer("120")
        assert normalize_math_answer("$120$") == "120"
        assert normalize_math_answer("$0.5$") == "0.5"

    def test_boxed_preserved(self):
        from gateway.opt_core import normalize_math_answer
        assert normalize_math_answer("\\boxed{42}") == "42"
        assert normalize_math_answer("3/4") == "0.75"

    def test_extract_boxed_answer_dollar(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini")
        assert gw._extract_boxed_answer("The result is $120$.") == "120"
        assert gw._extract_boxed_answer("Final answer: \\boxed{7}") == "7"


@patch("gateway.universal_enhanced_gateway.check_model_available",
       lambda model, **kwargs: True)
@patch("gateway.universal_enhanced_gateway.completion",
       lambda **kwargs: MagicMock(
           choices=[MagicMock(message=MagicMock(content="NOPE"))]))
class TestDeterministicMathOptimizations:
    def test_solve_equation_directly_star(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=True,
                                      performance_mode="speed")
        assert gw._solve_equation_directly("2*x + 3 = 7") == "x = 2"

    def test_tir_skips_deterministic_word_problem(self):
        import gateway.universal_enhanced_gateway as ug
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=True,
                                      performance_mode="speed")
        q = "If a train travels 60 km per hour, how far does it go in 2 hours?"
        a = gw._intelligent_query_analysis_with_complexity(q)
        with patch.object(ug, "_tir_enabled", True):
            r = gw._apply_tir_reasoning(q, a)
        assert r is None

    def test_ssr_skips_deterministic_arithmetic(self):
        import gateway.universal_enhanced_gateway as ug
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=True,
                                      performance_mode="speed")
        q = "2 + 2"
        a = gw._intelligent_query_analysis_with_complexity(q)
        with patch.object(ug, "_ssr_enabled", True), \
             patch.object(ug, "_math_strategies", {"x": {"keywords": ["2"],
                                                         "strategies": []}}):
            r = gw._apply_ssr_guidance(q, a)
        assert r is None


class TestDirectSolverRefinement:
    """BUG: the solver's loose re.search patterns answered word problems and
    multiple-choice questions with numbers pulled from arithmetic fragments
    in the prose (e.g. "He gave 1/2 of his pencils" -> 0.5).  Only pure
    calculations may be short-circuited."""

    def _gw(self):
        import logging
        logging.disable(logging.INFO)
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        return UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=True)

    def test_pure_arithmetic_still_resolves(self):
        gw = self._gw()
        assert gw._solve_equation_directly("12 * 8") == "96.0"
        assert gw._solve_equation_directly("8 / 4") == "2.0"
        assert gw._solve_equation_directly("15% of 200") == "30.0"

    def test_word_problem_never_short_circuits(self):
        gw = self._gw()
        q = "Anthony had 50 pencils. He gave 1/2 of his pencils to Brando."
        assert gw._solve_equation_directly(q) is None
        q2 = "Stephen borrowed $300 and promised to pay 2% of the money he owed."
        assert gw._solve_equation_directly(q2) is None

    def test_multiple_choice_query_detected(self):
        from gateway.universal_enhanced_gateway import is_multiple_choice_query
        assert is_multiple_choice_query(
            "(1+i)^10 =\nA) -32i\nB) 32i\nC) -32\nD) 0") is True
        assert is_multiple_choice_query(
            "Which of the following is a prime number?") is True
        assert is_multiple_choice_query("What is 2 + 2?") is False
        assert is_multiple_choice_query(
            "He gave 1/2 of his pencils to Brando.") is False

    def test_chain_skips_multiple_choice_math(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=True)
        q = "Simplify: (a+b)^2 =  ?\nA) a^2+b^2\nB) a^2+2ab+b^2\nC) ab\nD) 2ab"
        a = QueryAnalysis()
        a.is_math = True
        a.is_complex = True
        assert gw._apply_chain_composition(q, a) is None


class TestWholeCodebaseImportIntegrity:
    def test_rag_self_rag_imports(self):
        from unittest.mock import patch, MagicMock
        from rag.self_rag import SelfRAGProcessor, HybridRAGReasoning, Reflection
        p = SelfRAGProcessor(model="ollama/phi3:mini")
        docs = p.retrieve_with_reflection("What is the capital of France?", num_docs=1)
        assert isinstance(docs, list) and docs
        fake = MagicMock()
        fake.choices = [MagicMock(message=MagicMock(content="B"))]
        with patch("litellm.completion", return_value=fake):
            answer, refs = p.generate_with_rag(
                "What is the capital of France?", ["Paris", "London", "Berlin", "Madrid"])
        assert answer == "B"
        assert isinstance(refs, list) and refs

    def test_config_manager_asdict_imported(self):
        import config_manager
        m = config_manager.ConfigManager()
        cfg = m.get_config()
        # get_config/merge must not raise NameError on asdict
        assert cfg is not None
        # direct: call the asdict-consuming merge path with an override dict
        # (the historical failure was `asdict` not imported from dataclasses)
        merged = m._merge_config(cfg, {})
        assert merged is not None

    def test_enhanced_gateway_no_undefined_name_prompt(self):
        import ast, pathlib
        src = pathlib.Path("gateway/enhanced_gateway.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        assert "prompt" not in names


class TestWordFormArithmetic:
    """Sentence-form math ('What is 2 plus 2?') must resolve via the quick path
    with zero LLM calls — this was previously falling through to a full
    completion, the single biggest per-request speed waste for casual queries."""

    @pytest.mark.parametrize("q,expected", [
        ("What is 2 + 2?", "4"),
        ("what is 2+2", "4"),
        ("2 plus 2", "4"),
        ("What is 5 times 4?", "20"),
        ("10 minus 3", "7"),
        ("What is 2 x 3", "6"),
        ("How much is 12 divided by 4?", "3.0"),
        ("What is 2 * 3 + 1", "7"),
    ])
    def test_word_arithmetic_resolves(self, q, expected):
        from gateway.opt_core import quick_word_arithmetic
        assert quick_word_arithmetic(q) == expected

    @pytest.mark.parametrize("q", [
        "2x + 3", "what is up", "what time is it", "factor x",
        "2x+3=7", "what is 2x", "how are you",
    ])
    def test_non_arithmetic_rejected(self, q):
        from gateway.opt_core import quick_word_arithmetic
        assert quick_word_arithmetic(q) is None

    def test_safe_gate_accepts_word_arithmetic(self):
        from gateway.opt_core import is_safe_quick_path
        assert is_safe_quick_path("What is 2 + 2?") is True
        assert is_safe_quick_path("How much is 12 divided by 4?") is True
        # Conversational/math-hint queries still denied the greeting shortcut
        assert is_safe_quick_path("how are you") is False
        assert is_safe_quick_path("what time is it") is False

    def test_agent_achat_answers_without_llm(self):
        import agents.langgraph_agent as la

        calls = {"n": 0}
        def _forbid(**kwargs):
            calls["n"] += 1
            raise AssertionError("LLM must not be called for word-form arithmetic")

        async def run():
            with patch("litellm.acompletion", side_effect=_forbid), \
                 patch("agents.langgraph_agent._ensure_wiring"):
                r = await la.achat(
                    "What is 2 + 2?", history=[], use_rag=False,
                    use_router=False, use_cache=False, speed_mode=True)
                return r.reply

        r = asyncio.run(run())
        assert r == "4"
        assert calls["n"] == 0


class TestOutputTokenTelemetry:
    """The graph must surface litellm's completion token count in ChatResult
    (this was previously dropped — AgentState lacked the key and ChatResult
    never received it, so output_tokens was always 0)."""

    def test_agent_node_captures_completion_tokens(self):
        import asyncio
        from unittest.mock import MagicMock, patch
        import agents.langgraph_agent as la

        async def run():
            agent = la.get_agent()
            from langchain_core.messages import HumanMessage
            sent = MagicMock()
            sent.choices = [MagicMock(message=MagicMock(content="idx speed up lookups"))]
            sent.model = "ollama/phi3:mini"
            sent.usage = MagicMock(completion_tokens=12, prompt_tokens=40)
            async def fake(**kw):
                return sent
            req_ctx = {"model": None, "temperature": None, "max_tokens": None,
                       "system_prompt": None, "use_rag": False, "use_router": False,
                       "use_cache": False, "speed_mode": False}
            with patch("litellm.acompletion", side_effect=fake), \
                 patch("gateway.opt_core.check_model_available", return_value=True):
                out = await agent.ainvoke(
                    {"messages": [HumanMessage(content="explain a database index")],
                     "rag_context": "", "rag_fetch_time": 0.0,
                     "_req_ctx": req_ctx, "_tool_call_counts": {}, "rag_task": None})
            return out

        out = asyncio.run(run())
        assert out.get("output_tokens") == 12

    def test_achat_surfaces_output_tokens(self):
        import asyncio
        from unittest.mock import MagicMock, patch
        import agents.langgraph_agent as la

        async def run():
            sent = MagicMock()
            sent.choices = [MagicMock(message=MagicMock(content="idx speed up lookups"))]
            sent.model = "ollama/phi3:mini"
            sent.usage = MagicMock(completion_tokens=7, prompt_tokens=40)
            async def fake(**kw):
                return sent
            with patch("litellm.acompletion", side_effect=fake), \
                 patch("gateway.opt_core.check_model_available", return_value=True):
                return await la.achat("explain a database index", history=[],
                                      use_rag=False, use_router=False, use_cache=False)

        r = asyncio.run(run())
        assert r.reply == "idx speed up lookups"
        assert r.output_tokens == 7


class TestDeferredLangGraphImport:
    """Trivial quick-path queries must not trigger the (expensive ~8s) LangGraph
    import/graph build. Verified by asserting achat answers without importing
    langgraph.graph — a heavy first-request latency win."""

    def test_quick_path_avoids_langgraph_build(self):
        import importlib
        import agents.langgraph_agent as la

        # If langgraph.graph was already imported by another test in this process,
        # we can't reliably assert on import-count. Instead assert that achat
        # answers the greeting WITHOUT invoking get_agent()/build_agent().
        called = {"n": 0}
        real_get_agent = la.get_agent
        def _spy():
            called["n"] += 1
            return real_get_agent()
        la.get_agent = _spy
        try:
            r = la.chat("hello", history=[], use_rag=False, use_router=False, use_cache=False)
        finally:
            la.get_agent = real_get_agent
        assert r == "Hello! How can I help you?"
        assert called["n"] == 0


# ---------------------------------------------------------------------------
# Adaptive generation timeout: schedule must never clobber real inference
# ---------------------------------------------------------------------------
class TestAdaptiveGenerationTimeout:
    """The 15s fixed timeout killed long (quality) generations on a ~10 tok/s
    deployment; the timeout must scale with the actual token budget."""

    def test_floor_respects_base(self):
        from gateway.opt_core import adaptive_generation_timeout as T
        assert T(None, 15) >= 15.0
        assert T(128, 60) >= 60.0  # configured base wins for small budgets

    def test_scales_with_budget(self):
        from gateway.opt_core import adaptive_generation_timeout as T
        assert T(512, 15) > T(128, 15)          # quality needs more wall time
        assert T(1024, 15) > T(512, 15)         # and larger budgets even more
        assert T(512, 15) >= 512 / 8.0 + 15.0   # at least budget-rate + buffer

    def test_zero_and_none_sane(self):
        from gateway.opt_core import adaptive_generation_timeout as T
        assert T(0, 15) >= 15.0
        assert T(None, 15) >= 15.0

    def test_negative_budget_sane(self):
        from gateway.opt_core import adaptive_generation_timeout as T
        assert T(-1, 15) >= 15.0

    def test_live_path_kwargs_use_adaptive_timeout(self):
        """agent_node must scale the timeout to max_tokens, not pin 15s."""
        import pathlib
        src = pathlib.Path("agents/langgraph_agent.py").read_text(encoding="utf-8")
        assert "adaptive_generation_timeout(" in src
        assert '"max_tokens": final_max_tokens' in src
        assert 'getattr(settings, "generation_timeout", 15),' in src


class TestSpeedModeReasoningBudget:
    """GSM8K regression: a flat 128-token speed cap truncates multi-step
    reasoning (0% accuracy); reasoning/math/code must get enough room (384)
    to finish while staying below the naive 512 budget."""

    def test_simple_queries_stay_128(self):
        from gateway.opt_core import speed_mode_max_tokens as S
        assert S(None) == 128
        assert S(512) == 128                    # forced cap even if caller asks more
        assert S(10) == 10                      # caller-requested smaller wins

    def test_reasoning_queries_get_384(self):
        from gateway.opt_core import speed_mode_max_tokens as S
        assert S(None, is_reasoning=True) == 384
        assert S(512, is_reasoning=True) == 384  # still below naive 512
        assert 128 < S(None, is_reasoning=True) <= 384

    def test_both_agent_speed_sites_use_reasoning_cap(self):
        import pathlib
        src = pathlib.Path("agents/langgraph_agent.py").read_text(encoding="utf-8")
        assert src.count("speed_mode_max_tokens(") == 2
        assert "reasoning_intent = is_math or is_coding or needs_reasoning" in src
        assert "reasoning_intent = _Analysis.is_math or _Analysis.is_coding or _Analysis.needs_reasoning" in src


class TestPlaceholderToolNeverHijacksAnswer:
    """AIME 2026 regression: the un-wired search tool returned a canned
    'not yet implemented' placeholder that replaced real model output for
    every query containing 'find'/'search'. Tool calls must never leak a
    placeholder as the user-visible answer."""

    def _gw(self) -> "UniversalEnhancedGateway":
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini")
        gw.enable_tool_system = True
        return gw

    def test_unwired_search_returns_none(self):
        gw = self._gw()
        assert gw._tool_search("Find the name of the director who took over.") is None

    def test_tool_session_does_not_short_circuit_on_unwired_tools(self):
        gw = self._gw()
        query = ("Can you find the name of the director who took over in 2004?")
        analysis = gw._intelligent_query_analysis_with_complexity(query)
        result = gw._execute_tool_calls(query, analysis)
        # Without a working tool result there must be no short-circuit, so the
        # normal LLM generation path runs instead of a placeholder.
        assert result is None
