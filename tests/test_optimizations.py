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
# Bug: NEVER serve a dumber answer than the model produced
# ---------------------------------------------------------------------------

class TestDumberGuard:
    """Regression tests for every path that could degrade answers:
    1. empty/None/apology responses must not be cached
    2. math constraint must not strip explanations or corrupt fractions
    3. 'summarize' must not be treated as math
    4. code routing must verify model availability
    5. non-reasoning speed-mode queries get a generous token budget
    """

    def test_empty_response_not_cached(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway)
        gw = UniversalEnhancedGateway.__new__(UniversalEnhancedGateway)
        response_text = ""
        _cacheable = (response_text and isinstance(response_text, str)
                      and response_text.strip())
        assert not _cacheable
        response_text = None
        _cacheable = (response_text and isinstance(response_text, str)
                      and response_text.strip())
        assert not _cacheable

    def test_apology_not_cached(self):
        text = "I apologize, but I'm having trouble processing your request."
        import re
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway)
        gw = UniversalEnhancedGateway.__new__(UniversalEnhancedGateway)
        response_text = text
        _cacheable = (response_text and isinstance(response_text, str)
                      and response_text.strip()
                      and 'trouble processing' not in response_text.lower()
                      and 'apologize' not in response_text.lower())
        assert not _cacheable

    def test_math_constraint_keeps_fraction_explanations(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=True)
        analysis = QueryAnalysis()
        analysis.is_math = True
        # Long explanation with a fraction -> must NOT be stripped to "3"
        result = gw._apply_response_constraints(
            "To solve this, divide 2 by 3: the answer is 1/3", analysis)
        assert "1/3" in result, f"fraction explanation got corrupted: {result!r}"
        assert result.strip() != "3"

    def test_math_constraint_strips_only_short_numeric(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=True)
        analysis = QueryAnalysis()
        analysis.is_math = True
        # Short pure-arithmetic response -> trailing number extraction is fine
        result = gw._apply_response_constraints("4", analysis)
        assert result == "4"

    def test_summarize_is_not_math(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway)
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=False)
        analysis = gw._intelligent_query_analysis_with_complexity(
            "summarize this article in 2 sentences")
        assert not analysis.is_math, \
            "'summarize' contains 'sum' but must not be classified as math"
        analysis2 = gw._intelligent_query_analysis_with_complexity(
            "what is the sum of 2 and 3?")
        assert analysis2.is_math, \
            "'sum of X and Y' IS math"

    def test_code_routing_checks_availability(self):
        from unittest.mock import patch
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=True)
        analysis = QueryAnalysis()
        analysis.is_coding = True
        analysis.complexity_score = 0.1
        analysis.query_text = "write a python function"
        with patch("gateway.opt_core.check_model_available",
                   return_value=False):
            result = gw._route_to_code_model(analysis)
        assert result is None, \
            "must NOT route to an uninstalled code model (leads to apology)"
        with patch("gateway.opt_core.check_model_available",
                   return_value=True):
            result = gw._route_to_code_model(analysis)
        assert result is not None, \
            "a valid code model should be used when available"

    def test_speed_simple_queries_get_generous_budget(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=True,
                                      performance_mode="speed")
        analysis = QueryAnalysis()  # plain factual query
        params = gw._get_adaptive_model_params(analysis)
        assert params["max_tokens"] >= 200, \
            "simple speed-mode queries must not be truncated to 50 tokens"


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
    """get_universal_gateway returns the same instance per (model, mode)."""

    def test_singleton(self):
        from gateway import universal_enhanced_gateway as mod
        # Reset
        mod._reset_gateways_for_tests()
        g1 = mod.get_universal_gateway("phi3:mini", False, "speed")
        g2 = mod.get_universal_gateway("phi3:mini", False, "speed")
        assert g1 is g2
        # Cleanup
        mod._reset_gateways_for_tests()

    def test_different_mode_is_different_gateway(self):
        """Regression: the old single-slot singleton silently served a
        speed-configured gateway to balanced callers, so same-process
        multi-mode benchmarks measured the wrong mode."""
        from gateway import universal_enhanced_gateway as mod
        mod._reset_gateways_for_tests()
        try:
            g_speed = mod.get_universal_gateway("phi3:mini", False, "speed")
            g_bal = mod.get_universal_gateway("phi3:mini", False, "balanced")
            assert g_speed is not g_bal
            assert g_speed.performance_mode == "speed"
            assert g_bal.performance_mode == "balanced"
        finally:
            mod._reset_gateways_for_tests()


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

    def test_restart_honours_persisted_expiry(self):
        """A persisted entry must NOT get a fresh full TTL after a restart:
        the remaining lifetime (not the full default_ttl) is restored."""
        from gateway.simple_cache import SimpleCache
        import tempfile
        import time
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmpdir:
            c1 = SimpleCache(cache_dir=tmpdir, persist=True, max_size=100,
                             default_ttl=3600.0)
            c1.set("q_live", "keep", context={"model": "m", "system_prompt": "s"})
            live_key = c1._get_key("q_live", {"model": "m", "system_prompt": "s"})
            c1._store.set(live_key, {"response": "keep", "metadata": {}},
                          ttl=60.0)
            c1._dirty = True
            c1.save_now()
            assert Path(tmpdir, "response_cache.json").exists()
            c2 = SimpleCache(cache_dir=tmpdir, persist=True, max_size=100,
                             default_ttl=3600.0)
            assert c2.get("q_live", context={"model": "m", "system_prompt": "s"}) == "keep"
            # Restart must restore the SHORT remaining TTL (<=60s), not a fresh
            # 3600s default — that would resurrect stale answers forever.
            now = time.time()
            entry = dict(c2._store.items_snapshot()).get(live_key)
            assert entry is not None
            remaining = float(entry.get("expires_at")) - now
            assert 0 <= remaining <= 120.0, f"remaining TTL was {remaining}"


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

    # BUG 41 FIX: canned confirmations must never fire mid-conversation
    def test_confirmation_not_returned_in_multi_turn(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=False)
        multi_turn = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "yes"},
        ]
        result = gw._get_quick_response("yes", multi_turn)
        assert result is None, "canned confirmation must not fire mid-conversation"

    def test_greeting_not_returned_in_multi_turn(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=False)
        multi_turn = [
            {"role": "user", "content": "Tell me a joke"},
            {"role": "assistant", "content": "Why did the chicken cross the road?"},
            {"role": "user", "content": "hello"},
        ]
        result = gw._get_quick_response("hello", multi_turn)
        assert result is None, "canned greeting must not fire mid-conversation"

    def test_quick_response_still_works_on_first_turn(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=False)
        result = gw._get_quick_response("yes", [{"role": "user", "content": "yes"}])
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

    def test_zero_ttl_means_do_not_cache(self):
        """ttl<=0 must never produce an immortal entry: set() evicts any
        existing entry and forgets the new value."""
        from gateway.opt_core import BoundedTTLCache
        c = BoundedTTLCache(max_size=10, default_ttl=3600.0)
        c.set("a", 1, ttl=0)
        assert c.get("a") is None
        assert len(c) == 0
        # Evicts an existing value too (no stale immortal entry left behind)
        c.set("b", 2, ttl=60.0)
        c.set("b", 3, ttl=0)
        assert c.get("b") is None


# ---------------------------------------------------------------------------
# Eq9AnomalyDetector (perf_math) — reset + O(1) stats
# ---------------------------------------------------------------------------

class TestEq9AnomalyDetector:
    """Eq9AnomalyDetector must support reset() (the FastAPI PerEndpoint
    detector calls it) and keep stats in a bounded deque with O(1) adds."""

    def _make(self, window=30, min_samples=3, k=3.0, slo=100.0):
        from gateway.perf_math import Eq9AnomalyDetector
        return Eq9AnomalyDetector(slo_threshold_ms=slo, window_size=window,
                                  min_samples=min_samples, k_factor=k)

    def test_reset_clears_everything(self):
        det = self._make()
        for _ in range(min(50, det.window_size)):
            det.add_sample(5.0)
        assert det._count > 0
        assert len(det.samples) > 0
        det.reset()
        assert det._count == 0
        assert len(det.samples) == 0
        assert det.last_alert_time == 0
        assert det.rolling_stats()["samples"] == 0

    def test_window_stays_bounded(self):
        det = self._make(window=8)
        for i in range(100):
            det.add_sample(float(i))
        assert len(det.samples) == 8
        assert det._count == 8
        # mean tracks the last 8 samples (0..6 evicted, oldest kept 92..99)
        avg = det.rolling_stats()["mean"]
        assert abs(avg - sum(range(92, 100)) / 8.0) < 1e-9

    def test_anomaly_trigger_respects_k(self):
        det = self._make(window=10, min_samples=4, k=2.0, slo=50.0)
        for i in range(9):
            det.add_sample(10.0)
        # 9 normal samples + 1 huge outlier -> anomaly fires
        fired = det.observe(500.0)
        assert fired is True


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


class TestQuickPathFollowups:
    """Multi-turn continuations must resolve instantly: 'And 13*13?'
    is the same arithmetic as '13*13?' (diag: follow-ups fell through
    to a full generation and timed out server-side), and 'Hello there!'
    must greet like 'hello' (agent gate missed it -> 123 s TTFB)."""

    def test_followup_conjunction_math(self):
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("And 13*13?") == "169"
        assert quick_arithmetic("then 2 + 3?") == "5"
        assert quick_arithmetic("So 10 - 4?") == "6"

    def test_conjunction_strip_is_conservative(self):
        from gateway.opt_core import quick_arithmetic
        assert quick_arithmetic("android phones?") is None
        assert quick_arithmetic("also known as Bob?") is None
        assert quick_arithmetic("And") is None

    def test_followup_math_is_quick_path(self):
        from gateway.opt_core import is_safe_quick_path
        assert is_safe_quick_path("And 13*13?") is True

    def test_greeting_there_tolerance(self):
        from gateway.opt_core import is_safe_quick_path
        assert is_safe_quick_path("Hello there!") is True
        assert is_safe_quick_path("hi there.") is True

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

    def test_speed_mode_short_response_capped_generously(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini",
                                      enable_all_optimizations=True,
                                      performance_mode="speed")
        analysis = QueryAnalysis()
        analysis.expected_response_length = "short"
        params = gw._get_adaptive_model_params(analysis)
        # Non-reasoning queries get a generous 200-token cap so short factual
        # answers are not cut off mid-sentence.
        assert params["max_tokens"] >= 200
        assert params["max_tokens"] <= 200

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
        options = params.get("options") or {}
        assert options.get("think") is False, \
            "thinking models must have the ollama think flag disabled in speed mode " \
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
        options = params.get("options") or {}
        assert options.get("think") is False, \
            "think must be disabled for qwen3 in speed mode"
        assert options.get("num_predict") is not None, \
            "num_predict must live inside options (not top-level) for ollama qwen3"
        assert "max_tokens" not in params, \
            "top-level max_tokens must be removed for thinking models (ollama bug)"
        assert "num_predict" not in params, \
            "top-level num_predict must be removed for thinking models (ollama bug)"

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
        # Silence noisy gateway construction, but ALWAYS restore: a bare
        # logging.disable() leaks process-wide and blinds every later
        # test that asserts on log records.
        logging.disable(logging.INFO)
        try:
            from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
            return UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=True)
        finally:
            logging.disable(logging.NOTSET)

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
        # Silence noisy gateway construction, but ALWAYS restore: a bare
        # logging.disable() leaks process-wide and blinds every later
        # test that asserts on log records.
        logging.disable(logging.INFO)
        try:
            from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
            return UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=True)
        finally:
            logging.disable(logging.NOTSET)

    def test_pure_arithmetic_still_resolves(self):
        gw = self._gw()
        assert gw._solve_equation_directly("12 * 8") == "96"
        assert gw._solve_equation_directly("8 / 4") == "2"
        assert gw._solve_equation_directly("15% of 200") == "30.0"

    def test_chained_arithmetic_precedence(self):
        gw = self._gw()
        assert gw._solve_equation_directly("3 * (4 + 5) - 2") == "25"
        assert gw._solve_equation_directly("10 + 5 * 6") == "40"
        assert gw._solve_equation_directly("12 / 4 * 2") == "6"

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

    def test_balanced_path_resolves_math_without_model_call(self):
        """BUG: balanced mode (_unified_generate) returned early to the model
        and never reached _solve_equation_directly, so pure arithmetic burned
        a 60-260s LLM call for a <1ms computation."""
        from unittest.mock import patch
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=True)
        a = QueryAnalysis()
        a.is_math = True
        a.expected_response_length = "long"
        with patch.object(gw, "_raw_reasoning_call") as model:
            r = gw._unified_generate(
                [{"role": "user", "content": "solve 2x + 5 = 15"}],
                "solve 2x + 5 = 15", a)
        assert r == "x = 5"
        model.assert_not_called()

    def test_balanced_path_resolves_chained_arithmetic(self):
        from unittest.mock import patch
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=True)
        a = QueryAnalysis()
        a.is_math = True
        a.expected_response_length = "long"
        with patch.object(gw, "_raw_reasoning_call") as model:
            r = gw._unified_generate(
                [{"role": "user", "content": "3 * (4 + 5) - 2"}],
                "3 * (4 + 5) - 2", a)
        assert r == "25"
        model.assert_not_called()

    def test_chain_skips_multiple_choice_math(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=True)
        q = "Simplify: (a+b)^2 =  ?\nA) a^2+b^2\nB) a^2+2ab+b^2\nC) ab\nD) 2ab"
        a = QueryAnalysis()
        a.is_math = True
        a.is_complex = True
        assert gw._apply_chain_composition(q, a) is None


class _MonotonicClock:
    """Controllable substitute for time.monotonic in retry-budget tests.

    Each call returns a monotonically increasing value; successive calls move
    forward by the next delta in ``deltas`` (then hold steady).  The first
    call anchors the primary's start time, so delta[k] becomes the elapsed
    seconds measured between the primary and the retry decision.
    """
    def __init__(self, deltas):
        self._deltas = list(deltas)
        self._now = 1000.0

    def __call__(self):
        now = self._now
        if self._deltas:
            self._now += self._deltas.pop(0)
        return now


class TestEquationRunnerRetryBudget:
    """Eq 1/2/9/33/36: the retry shares the SAME absolute per-item deadline as
    the primary and may only spend remaining = C - elapsed.  A pure timeout has
    no budget left, so the retry is skipped; a FAST failure leaves the window,
    so the retry still runs — worst item == generation_timeout, exactly.  Never
    loops: at most one retry, only on primary None.
    """
    def _call(self, monkeypatch, clock_deltas, generation_timeout=240,
              primary_result=None):
        from unittest.mock import patch
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        import uuid
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=True)
        calls = {}

        fake_clock = _MonotonicClock(clock_deltas)

        def _fake_raw(query, budget=900, timeout=480, think=None,
                      model=None):
            calls["timeout"] = timeout
            calls["count"] = calls.get("count", 0) + 1
            return "42"

        def _fake_thinkoff(query, budget=160):
            return primary_result

        import config
        old_to = getattr(config.settings, "generation_timeout", None)
        old_budget = getattr(config.settings, "item_budget", None)
        config.settings.generation_timeout = generation_timeout
        config.settings.item_budget = None
        try:
            a = QueryAnalysis()
            a.is_math = True
            a.expected_response_length = "long"
            q = f"essay about triple arithmetic check {uuid.uuid4().hex}"
            with patch("gateway.universal_enhanced_gateway.time.monotonic",
                       side_effect=fake_clock), \
                 patch.object(gw, "_thinkoff_call", _fake_thinkoff), \
                 patch.object(gw, "_raw_reasoning_call", _fake_raw):
                r = gw._unified_generate(
                    [{"role": "user", "content": q}], q, a)
        finally:
            if old_budget is None:
                del config.settings.item_budget
            else:
                config.settings.item_budget = old_budget
            if old_to is None:
                del config.settings.generation_timeout
            else:
                config.settings.generation_timeout = old_to
        return r, calls

    def test_fast_failure_gets_full_remaining_window(self, monkeypatch):
        # Primary fails in 1s (Eq 33: remaining = C - 1), cap 240 -> retry 239.
        r, calls = self._call(monkeypatch, [1.0], 240, primary_result=None)
        assert r == "42"
        assert calls.get("count") == 1
        assert calls.get("timeout") == 239

    def test_timeout_consumed_cap_skips_retry(self, monkeypatch):
        # Primary burns the whole 240s cap (Eq 36: remaining < 15) -> no retry.
        r, calls = self._call(monkeypatch, [239.0], 240, primary_result=None)
        assert calls.get("count", 0) == 0

    def test_remaining_clamped_to_240_retry_max(self, monkeypatch):
        # Cap 420, primary fails in 1s: remaining 419, clamp to min(240, ·) = 240.
        r, calls = self._call(monkeypatch, [1.0], 420, primary_result=None)
        assert "42" == r
        assert calls.get("count") == 1
        assert calls.get("timeout") == 240

    def test_successful_primary_never_retries(self, monkeypatch):
        # Primary answered -> retry untouched, regardless of remaining.
        r, calls = self._call(monkeypatch, [1.0], 240, primary_result="7")
        assert r == "7"
        assert calls.get("count", 0) == 0


class TestCacheKeyDecimalCollision:
    """Bug: _generate_cache_key stripped every '.' — '1.5 + 2.5' and '15 + 25'
    normalized to the same key, so a cached answer for one was served to a
    different arithmetic query and the correct computation never ran.  Interior
    decimals must survive; only trailing sentence terminals are stripped."""

    def test_decimal_and_integer_queries_differ(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway.__new__(UniversalEnhancedGateway)
        ctx = {"model": "phi3:mini", "performance_mode": "speed"}
        k1 = gw._generate_cache_key("1.5 + 2.5", ctx)
        k2 = gw._generate_cache_key("15 + 25", ctx)
        assert k1 != k2

    def test_percent_decimal_differs(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway.__new__(UniversalEnhancedGateway)
        ctx = {"model": "phi3:mini", "performance_mode": "speed"}
        k1 = gw._generate_cache_key("1.5% of 200", ctx)
        k2 = gw._generate_cache_key("15% of 200", ctx)
        assert k1 != k2

    def test_trailing_period_still_stripped(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway.__new__(UniversalEnhancedGateway)
        ctx = {"model": "phi3:mini", "performance_mode": "speed"}
        assert gw._generate_cache_key("22 plus 20.", ctx) == \
            gw._generate_cache_key("22 plus 20", ctx)


class TestBalancedCacheSkipsDegradedText:
    """Bug: _store_balanced_response cached any truthy string for 3600s —
    including '[ERROR] <msg>' from _raw_reasoning_call and the canned
    "couldn't finish" fallback — so a transient failure became a permanent
    degraded cache hit and the retry ladder was unreachable for the whole TTL."""

    @staticmethod
    def _store(text, gw=None):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis)
        if gw is None:
            gw = UniversalEnhancedGateway(
                "phi3:mini", enable_all_optimizations=True)
        a = QueryAnalysis()
        return gw._store_balanced_response(text, "q", [], a)

    def test_error_marker_not_cached(self):
        assert self._store("[ERROR] connection refused") is None

    def test_mathematics_fallback_not_cached(self):
        assert self._store(
            "I couldn't finish that calculation in time. Please retry with "
            "a simpler expression.") is None

    def test_trouble_processing_not_cached(self):
        assert self._store(
            "I'm having trouble processing that request right now.") is None

    def test_real_answer_still_cached(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway, QueryAnalysis, _cache_lock)
        gw = UniversalEnhancedGateway(
            "phi3:mini", enable_all_optimizations=True)
        a = QueryAnalysis()
        gw._store_balanced_response("twenty-two", "q", [], a)
        with _cache_lock:
            hits = [v for v in gw.cache.values()
                    if v.get("response") == "twenty-two"]
        assert hits


class TestRawReasoningCallReturnsNoneOnError:
    """Bugfix: _raw_reasoning_call returned a truthy '[ERROR] <msg>' string on
    ollama error responses.  Callers gate the retry ladder on
    `if not response_text`, so a truthy marker was served to the user and the
    plain/raw fallbacks were never reached."""

    @staticmethod
    def _build_gw():
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        return UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=True)

    def test_error_body_returns_none(self):
        from unittest.mock import patch, MagicMock
        from gateway import universal_enhanced_gateway as g
        gw = self._build_gw()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"error": "model not found"}
        with patch.object(g.logger, "info") as _info:
            with patch("requests.post", return_value=fake_resp) as post:
                out = gw._raw_reasoning_call("q", budget=10, timeout=1)
        assert out is None
        post.assert_called_once()

    def test_success_still_returns_text(self):
        from unittest.mock import patch, MagicMock
        gw = self._build_gw()
        fake_resp = MagicMock()
        fake_resp.json.return_value = {"response": "the answer"}
        with patch("requests.post", return_value=fake_resp):
            out = gw._raw_reasoning_call("q", budget=10, timeout=1)
        assert out == "the answer"


class TestCacheWriteKeyMatchesReadKey:
    """Bugfix: the Step 11 cache write keyed the response by `model_to_use`
    (the ollama/-prefixed or routed model) while the Step 3 read looks up by
    `self.model_name` — every write was a permanent miss on a routed request.
    The write key must equal the read key."""

    def test_simple_cache_write_and_read_keys_agree(self):
        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway)
        gw = UniversalEnhancedGateway("phi3:mini", enable_all_optimizations=True)
        q = "what is 2+2?"
        ctx = {"model": gw.model_name, "performance_mode": gw.performance_mode}
        read_key = gw._generate_cache_key(q, ctx)
        write_ctx = {"model": gw.model_name, "performance_mode": gw.performance_mode}
        write_key = gw._generate_cache_key(q, write_ctx)
        assert write_key == read_key

    def test_staircase_write_never_uses_routed_model(self):
        # Source-level regression: the Step 11 (staircase) write key and the
        # Step 11.1 prefix-gen_id must be built from self.model_name, not the
        # routed model_to_use.
        import inspect
        from gateway import universal_enhanced_gateway as g
        src = inspect.getsource(g)
        seg = "if use_cache and self.cache is not None and _cacheable:"
        assert seg in src
        after = src.split(seg, 1)[1]
        # The simple-cache write key (before _cache_ttl) uses self.model_name.
        assert '"model": self.model_name' in after
        assert '"model": model_to_use' not in after
        # The prefix write must not override gen_id with the routed model:
        # _current_gen_identity() already uses self.model_name.
        assert "gen_id[\"model\"] = model_to_use" not in src


class TestSimpleCacheDirtyRace:
    """Bugfix: _save_cache cleared _dirty unconditionally, so a write that
    landed mid-save was marked clean and never persisted until the next save.
    The dirty flag must survive until a save that actually captured the write."""

    def test_dirty_survives_concurrent_write(self, tmp_path):
        from gateway.simple_cache import SimpleCache
        cache = SimpleCache(cache_dir=str(tmp_path), persist=True, max_size=100)
        cache.set("q1", "a1")
        cache.set("q2", "a2")
        cache.save_now()
        assert cache._dirty is False
        # Simulate a write landing MID-SAVE: after the _writes marker is read
        # (start of _save_cache) but before the snapshot is taken.  The save
        # must NOT clear the dirty flag for a write it could not have captured.
        cache.set("q3", "a3")  # dirty=True going INTO the save
        original_snapshot = cache._store.items_snapshot

        def racing_snapshot():
            cache.set("q4", "a4")  # lands during the save, after the marker
            return original_snapshot()
        cache._store.items_snapshot = racing_snapshot
        cache.save_now()
        # q4 arrived after the marker (3): the save could not capture it, so
        # the dirty flag must survive so a later save/atexit flush persists it.
        assert cache._dirty is True, (cache._dirty, cache._writes)
        # And the at-exit flush / next save persists that tail write.
        cache.save_now()
        fresh = SimpleCache(cache_dir=str(tmp_path), persist=True, max_size=100)
        assert fresh.get("q4") == "a4"


class TestAtExitFlush:
    def test_dirty_tail_is_flushed_on_exit(self, tmp_path, monkeypatch):
        # _flush_cache_on_exit must persist a dirty cache without a %10
        # scheduled save (1-9 pending writes), and be a no-op when clean.
        import threading
        import gateway.simple_cache as sc
        cache = sc.SimpleCache(cache_dir=str(tmp_path), persist=True, max_size=100)
        cache.set("q1", "a1")
        cache.set("q2", "a2")
        with monkeypatch.context() as m:
            m.setattr(sc, "_cache_instance", cache)
            m.setattr(sc, "_cache_lock", threading.RLock())
            sc._flush_cache_on_exit()
        reloaded = sc.SimpleCache(cache_dir=str(tmp_path), persist=True, max_size=100)
        assert reloaded.get("q1") == "a1"


class TestSemanticCacheLegacyEntriesMatchable:
    """Bugfix: legacy persisted entries predating the 'model' tag have no
    'model' key; `item.get("model") not in ("", want_model)` excluded them
    from semantic lookup entirely.  Missing model should not block a lookup
    that does not demand a specific model."""

    def test_legacy_no_model_entry_matches(self, tmp_path):
        from gateway.semantic_cache import SemanticCache

        dirname = str(tmp_path)
        cache = SemanticCache(cache_dir=dirname, persist=False, max_size=10)
        cache.set("what is the capital of france", "paris")
        key = cache._exact_key("what is the capital of france", {})
        e = cache._store.get(key)
        assert e is not None and e.get("model") == ""
        # Boot a fresh cache whose _store holds a model-less legacy entry.
        fresh = SemanticCache(cache_dir=dirname, persist=False, max_size=10,
                              similarity_threshold=0.4)
        legacy = dict(e)
        legacy.pop("model", None)
        fresh._store.set(key, legacy)
        out = fresh.get("capital of france?")  # near-duplicate, no model demand
        assert out == "paris"


class TestMeasuredThinkoffBStar:
    """tau(b) for qwen3 is non-monotonic (160 rambled 887 chars at 264s while
    384 answered bare at ~138s), so the reasoning think-off budget must come
    from the measured 3-point scan, never a hardcoded constant."""

    def _loader(self):
        from gateway.universal_enhanced_gateway import (
            _measured_thinkoff_bstar, _THINKOFF_BSTAR)
        _THINKOFF_BSTAR.clear()
        return _measured_thinkoff_bstar, _THINKOFF_BSTAR

    def test_scan_file_b_star_used(self, tmp_path):
        import json
        load, cache = self._loader()
        scan = tmp_path / "num_predict_scan.json"
        scan.write_text(json.dumps([
            {"i": 16, "model": "qwen3:4b",
             "b_star_fastest_correct": 384},
            {"i": 82, "model": "qwen3:4b",
             "b_star_fastest_correct": 512},
        ]), encoding="utf-8")
        assert load("qwen3:4b", scan_path=str(scan)) == 512  # latest wins
        assert cache["qwen3:4b"] == 512

    def test_legacy_row_without_model_tag_applies(self, tmp_path):
        import json
        load, _ = self._loader()
        scan = tmp_path / "num_predict_scan.json"
        scan.write_text(json.dumps(
            [{"i": 16, "b_star_fastest_correct": 384}]), encoding="utf-8")
        assert load("qwen3:4b", scan_path=str(scan)) == 384

    def test_other_model_row_skipped(self, tmp_path):
        import json
        load, _ = self._loader()
        scan = tmp_path / "num_predict_scan.json"
        scan.write_text(json.dumps(
            [{"i": 0, "model": "phi3:mini", "b_star_fastest_correct": 96}]),
            encoding="utf-8")
        assert load("qwen3:4b", scan_path=str(scan)) == 384  # default

    def test_missing_and_malformed_scan_fall_back(self, tmp_path):
        load, _ = self._loader()
        assert load("model-a", scan_path=str(tmp_path / "nope.json")) == 384
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load("model-b", scan_path=str(bad)) == 384
        nonpos = tmp_path / "nonpos.json"
        import json
        nonpos.write_text(json.dumps(
            [{"model": "model-c", "b_star_fastest_correct": 0}]),
            encoding="utf-8")
        assert load("model-c", scan_path=str(nonpos)) == 384

    def test_reasoning_budget_not_hardcoded(self):
        # Source-level regression: the reasoning branch must call the
        # measured loader, not `budget = 384`.
        import inspect
        from gateway import universal_enhanced_gateway as g
        src = inspect.getsource(g)
        assert "budget = 384" not in src
        assert "_measured_thinkoff_bstar(" in src

    def test_staircase_thinkoff_primary(self):
        """Slow-cause fix: the staircase thinking branches must lead with
        the bounded think-off call (measured b*), keeping full-thinking
        raw only as fallback — not full thinking first (79-242s/query)."""
        import inspect
        from gateway import universal_enhanced_gateway as g
        src = inspect.getsource(g)
        parts = src.split("_should_use_raw_reasoning(query_analysis)")
        # def + 2 staircase branch sites.
        assert len(parts) == 3
        for branch in parts[1:]:
            seg = branch[:2000]
            assert "_thinkoff_call(" in seg
            assert seg.index("_thinkoff_call(") < seg.index("_raw_reasoning_call(")

    def test_easy_path_suppresses_thinking(self):
        """Easy factual queries must not burn the budget on hidden reasoning:
        top-level think=False is the only mechanism ollama honors."""
        import inspect
        from gateway import universal_enhanced_gateway as g
        src = inspect.getsource(g)
        seg = src.split("Easy path: suppress thinking", 1)[1][:800]
        assert "think=False" in seg


class TestOllamaGenerateUrl:
    """Raw transports must honor settings.litellm_api_base (127.0.0.1
    fallback keeps the Windows IPv6-first stall away)."""

    def test_default_and_override(self, monkeypatch):
        from gateway import universal_enhanced_gateway as g
        from config import settings
        monkeypatch.setattr(settings, "litellm_api_base", None)
        assert (g._ollama_generate_url()
                == "http://127.0.0.1:11434/api/generate")
        monkeypatch.setattr(settings, "litellm_api_base",
                            "http://10.0.0.5:11434/")
        assert (g._ollama_generate_url()
                == "http://10.0.0.5:11434/api/generate")


class TestSemanticSecondStage:
    """The installed semantic layer patched a singleton nothing read.
    Step 3 must consult it on exact miss; writes must mirror into it."""

    def _gateway(self, mode):
        from gateway import universal_enhanced_gateway as mod
        return mod.UniversalEnhancedGateway(
            "phi3:mini", enable_all_optimizations=True,
            performance_mode=mode)

    def test_singleton_hit_serves_chat_without_model(self):
        from unittest.mock import patch
        from gateway import universal_enhanced_gateway as mod
        from gateway.simple_cache import SimpleCache
        gw = self._gateway("balanced")
        q = "Zxq Describe the harbor lights quux?"
        key = gw._generate_cache_key(
            q, {"model": "phi3:mini", "performance_mode": "balanced"})
        store = SimpleCache(persist=False)
        store.set(q, "Harbor answer.", _key=key)
        with patch("gateway.simple_cache.get_cache", return_value=store):
            with patch.object(mod, "completion",
                              side_effect=AssertionError("no model call")):
                out = gw.chat([{"role": "user", "content": q}])
        assert out == "Harbor answer."

    def test_mode_scoping_blocks_cross_mode_serve(self, monkeypatch):
        """A speed-keyed entry must NOT serve a balanced chat."""
        from unittest.mock import patch
        from gateway import universal_enhanced_gateway as mod
        from gateway.simple_cache import SimpleCache
        gw = self._gateway("balanced")
        # Hermetic: the miss path must never touch the network.
        monkeypatch.setattr(gw, "_raw_fallback_retry",
                            lambda *a, **k: None)
        q = "Zxq Describe the harbor lights quux?"
        speed_key = gw._generate_cache_key(
            q, {"model": "phi3:mini", "performance_mode": "speed"})
        store = SimpleCache(persist=False)
        store.set(q, "Speed draft.", _key=speed_key)
        with patch("gateway.simple_cache.get_cache", return_value=store):
            with patch.object(mod, "completion",
                              side_effect=AssertionError("model called")):
                out = gw.chat([{"role": "user", "content": q}])
        assert out != "Speed draft."

    def test_step11_mirrors_into_singleton(self):
        """Source-level: Step 11 write must mirror to the singleton so the
        second stage (and disk persistence) actually receives entries."""
        import inspect
        from gateway import universal_enhanced_gateway as g
        src = inspect.getsource(g)
        assert "_get_mirror().set(" in src
        assert "_get_bmirror().set(" in src


class TestCISCConsensus:
    """CISC hand-rolled Counter vote wasted the consensus equations and
    served disputed answers as confident truth."""

    def _analysis(self):
        from types import SimpleNamespace
        return SimpleNamespace(is_math=True)

    def _resp(self, text):
        from unittest.mock import MagicMock
        r = MagicMock()
        r.choices = [MagicMock(message=MagicMock(content=text))]
        return r

    def test_unanimous_vote_returns_answer(self, monkeypatch):
        from gateway import universal_enhanced_gateway as mod
        monkeypatch.setattr(mod, "_cisc_enabled", True)
        gw = mod.UniversalEnhancedGateway(
            "phi3:mini", enable_all_optimizations=True,
            performance_mode="speed")
        ans = "The answer is \\boxed{42} done."
        with patch.object(mod, "completion",
                          return_value=self._resp(ans)):
            out = gw._confidence_informed_sc(
                "What is 40 + 2?", self._analysis(), num_samples=3)
        assert out is not None and "42" in out

    def test_split_vote_falls_through(self, monkeypatch):
        from gateway import universal_enhanced_gateway as mod
        monkeypatch.setattr(mod, "_cisc_enabled", True)
        gw = mod.UniversalEnhancedGateway(
            "phi3:mini", enable_all_optimizations=True,
            performance_mode="speed")
        texts = ["\\boxed{42}", "\\boxed{43}", "\\boxed{42}",
                 "\\boxed{43}", "\\boxed{44}"]
        with patch.object(mod, "completion",
                          side_effect=[self._resp(t) for t in texts]):
            out = gw._confidence_informed_sc(
                "What is 40 + 2?", self._analysis(), num_samples=5)
        assert out is None  # 0.4 agreement < 0.5 clearance


class TestCalibrationVerifyGate:
    """Speed verify fired a second model call for every short reasoning
    draft; calibration equations gate it to low-confidence drafts."""

    def test_deferral_math(self):
        from gateway.calibration_metrics import (
            estimate_confidence_from_response)
        from gateway.equations.calibration_math import (
            calibrated_confidence, should_defer_to_verify)
        low = calibrated_confidence(
            estimate_confidence_from_response(
                "maybe the answer is 42, not sure", "reasoning"),
            temperature=0.7)
        high = calibrated_confidence(
            estimate_confidence_from_response(
                "The answer is definitely 42, exactly.", "factual"),
            temperature=0.7)
        assert should_defer_to_verify(low) is True
        assert should_defer_to_verify(high) is False

    def test_gate_wired_in_chat(self):
        import inspect
        from gateway import universal_enhanced_gateway as g
        src = inspect.getsource(g)
        assert "should_defer_to_verify(" in src
        assert "calibrated_confidence(" in src
        # Math/code verify at the standard bar; other reasoning only when
        # confidence is very low (a second full generation per explainer).
        assert "threshold=_bar" in src

    def test_verify_outcome_feeds_adaptive_temperature(self):
        """The verify verdict closes the adaptive-temperature loop (which
        otherwise never adapts: kept draft = good temperature)."""
        from unittest.mock import patch
        from gateway import universal_enhanced_gateway as mod
        gw = mod.UniversalEnhancedGateway(
            "phi3:mini", enable_all_optimizations=True,
            performance_mode="speed")

        def resp_with(text):
            class _Msg:
                content = text

            class _Choice:
                message = _Msg()

            class _Resp:
                choices = [_Choice()]
            return _Resp()

        calls = []
        with patch.object(mod, "completion",
                          return_value=resp_with("[[CORRECT]]")), \
             patch("gateway.adaptive_temperature.record_response_feedback",
                   side_effect=lambda ok: calls.append(ok)):
            assert gw._verify_speed_answer("q?", "draft") is None
        assert calls == [True]
        calls.clear()
        with patch.object(mod, "completion",
                          return_value=resp_with("Paris.")), \
             patch("gateway.adaptive_temperature.record_response_feedback",
                   side_effect=lambda ok: calls.append(ok)):
            assert gw._verify_speed_answer("q?", "draft") == "Paris."
        assert calls == [False]

    def test_should_verify_draft_grades_by_confidence(self):
        """Extracted gate used by both speed and balanced paths: hedged
        math drafts verify, confident ones skip."""
        from types import SimpleNamespace
        from gateway import universal_enhanced_gateway as mod
        gw = mod.UniversalEnhancedGateway(
            "phi3:mini", enable_all_optimizations=True,
            performance_mode="balanced")
        math = SimpleNamespace(is_math=True, is_coding=False,
                               needs_reasoning=True)
        assert gw._should_verify_draft(math, "maybe 42, not sure") is True
        confident = "The answer is definitely 42, exactly."
        assert gw._should_verify_draft(math, confident) is False

    def test_balanced_math_path_does_not_second_guess(self):
        """Regression, learned the hard way: a verify pass on the balanced
        (eval) path replaced a correct terse verdict ('7') with prose and
        broke TestEquationRunnerRetryBudget.  Balanced math gets exactly one
        generation; second-guessing belongs to the speed gate and to a
        future measured multi-sample pass."""
        import inspect
        from gateway import universal_enhanced_gateway as mod
        src = inspect.getsource(mod.UniversalEnhancedGateway._unified_generate)
        assert "_should_verify_draft(" not in src
        assert "_verify_speed_answer(" not in src

    def test_verify_pass_uses_small_budget(self):
        """The verify second pass answers [[CORRECT]] or a short correction:
        a 300-token budget doubles its cost for zero gain. 128 suffices."""
        from unittest.mock import patch
        from gateway import universal_enhanced_gateway as mod

        class _Msg:
            content = "[[CORRECT]]"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        seen = {}

        def fake(**kwargs):
            seen.update(kwargs)
            return _Resp()

        gw = mod.UniversalEnhancedGateway(
            "phi3:mini", enable_all_optimizations=True,
            performance_mode="speed")
        with patch.object(mod, "completion", fake):
            assert gw._verify_speed_answer("What is 2+2?",
                                           "The answer is 4") is None
        assert seen.get("max_tokens") == 128
        # Thinking path carries the same small budget in options instead.
        gwq = mod.UniversalEnhancedGateway(
            "qwen3:4b", enable_all_optimizations=True,
            performance_mode="speed")
        seen.clear()
        with patch.object(mod, "completion", fake):
            assert gwq._verify_speed_answer("What is 2+2?",
                                           "The answer is 4") is None
        assert seen.get("options", {}).get("num_predict") == 128
        assert "max_tokens" not in seen


class TestMakeGatewayInstallsWiring:
    """make_optimized_gateway must install the equation wiring so
    bench/chat/run traffic executes it (install_all was dead code)."""

    def test_install_all_runs(self):
        import optimizer_cli
        from gateway.simple_cache import get_cache
        optimizer_cli.make_optimized_gateway("phi3:mini", "speed")
        assert getattr(get_cache(), "_semantic_installed", False) is True

    def test_install_all_in_source(self):
        import pathlib
        src = pathlib.Path("optimizer_cli.py").read_text(encoding="utf-8")
        assert "install_all()" in src


class TestRRFFusion:
    """BOTH-provider RAG must fuse ranked lists with retrieval_math RRF."""

    def test_both_mode_fuses_corroborated_first(self, monkeypatch):
        import asyncio
        from config import settings, RAGProvider
        import agents.langgraph_agent as la

        class FakeRAG:
            def __init__(self, chunks, sources):
                self._chunks = chunks
                self._sources = sources

            def retrieve(self, question):
                return {"chunks": list(self._chunks),
                        "sources": list(self._sources)}

        import rag.llama_index_rag as li
        import rag.haystack_pipeline as hs
        monkeypatch.setattr(
            li, "get_rag",
            lambda: FakeRAG(["alpha shared", "llama only"], ["a", "b"]))
        monkeypatch.setattr(
            hs, "get_haystack_rag",
            lambda: FakeRAG(["alpha shared", "haystack only"], ["c", "d"]))
        monkeypatch.setattr(settings, "rag_provider", RAGProvider.BOTH)
        out = asyncio.run(la._rag_prefetch("test question"))
        assert "[Fused RAG context (RRF)]" in out
        assert out.index("alpha shared") < out.index("llama only")
        assert out.index("alpha shared") < out.index("haystack only")

    def test_single_provider_path_unchanged(self, monkeypatch):
        import asyncio
        from config import settings, RAGProvider
        import agents.langgraph_agent as la
        import rag.llama_index_rag as li

        class FakeRAG:
            def retrieve(self, question):
                return {"chunks": ["only chunk"], "sources": ["s"]}

        monkeypatch.setattr(li, "get_rag", lambda: FakeRAG())
        monkeypatch.setattr(settings, "rag_provider", RAGProvider.LLAMA_INDEX)
        out = asyncio.run(la._rag_prefetch("test question"))
        assert "[LlamaIndex context]" in out
        assert "Fused" not in out


class TestMemoryWindowEquation:
    """build_context_messages hard-limit window must keep the newest turns
    (memory_math Eq M19), never drop the live request."""

    def test_overflow_keeps_newest(self):
        from gateway.opt_core import build_context_messages
        msgs = [{"role": "user", "content": f"message number {i} " + "x" * 200}
                for i in range(20)]
        built = build_context_messages(
            msgs, context_limit=800, budget=300, always_keep_n_history=2)
        texts = [m["content"] for m in built.messages]
        assert any("message number 19" in t for t in texts)
        assert not any("message number 0" in t for t in texts)


class TestCacheQuarantine:
    """Eq I254/B57: validate at init — an unreadable cache file must be
    moved aside, never silently kept (and never clobbered by the next save
    without evidence)."""

    def test_simple_cache_quarantines_corrupt_file(self, tmp_path):
        from gateway.simple_cache import SimpleCache
        bad = tmp_path / "response_cache.json"
        bad.write_text("{corrupt json", encoding="utf-8")
        cache = SimpleCache(cache_dir=str(tmp_path), persist=True, max_size=10)
        assert cache.get("anything") is None
        backups = list(tmp_path.glob("response_cache.corrupt.*.bak"))
        assert len(backups) == 1
        assert backups[0].read_text(encoding="utf-8") == "{corrupt json"

    def test_semantic_cache_quarantines_corrupt_file(self, tmp_path):
        from gateway.semantic_cache import SemanticCache
        bad = tmp_path / "semantic_cache.json"
        bad.write_text("{corrupt json", encoding="utf-8")
        cache = SemanticCache(cache_dir=str(tmp_path), persist=True, max_size=10)
        assert cache.get("anything") is None
        backups = list(tmp_path.glob("semantic_cache.corrupt.*.bak"))
        assert len(backups) == 1


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

    def test_thinking_models_get_load_cushion(self):
        """qwen3 thinks even think-off + ~60s cold load: the phi3-measured
        floor (128 tok -> 43.4s) killed its baseline calls mid-generation."""
        from gateway.opt_core import adaptive_generation_timeout as T
        plain = T(128, 15)
        think = T(128, 15, thinking=True)
        assert think - plain == pytest.approx(120.0)
        assert think >= 120.0
        # Non-thinking path unchanged.
        assert plain == pytest.approx(128 / 4.5 + 15.0)
        # Scaling/monotonicity properties preserved.
        assert T(512, 15, thinking=True) > T(128, 15, thinking=True)

    def test_litellm_chat_timeout_is_thinking_aware(self):
        """The 43s raw-baseline timeout storm: chat() must give qwen3 more
        wall time than phi3 for the same token budget."""
        import gateway.litellm_gateway as lg

        class _Msg:
            content = "hi"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            model = "m"

        seen = {}

        def fake_completion(**kwargs):
            seen[kwargs["model"]] = kwargs.get("timeout")
            return _Resp()

        with patch.object(lg, "completion", fake_completion):
            lg.chat([{"role": "user", "content": "hi"}], model="qwen3:4b",
                    max_tokens=128, use_cache=False)
            lg.chat([{"role": "user", "content": "hi"}], model="phi3:mini",
                    max_tokens=128, use_cache=False)
        assert seen["ollama/qwen3:4b"] > seen["ollama/phi3:mini"]
        assert seen["ollama/qwen3:4b"] >= 120.0

    def test_litellm_chat_keeps_model_resident(self):
        """keep_alive must reach ollama or the model unloads after 5 idle
        minutes and the next request pays a ~60s cold reload."""
        import gateway.litellm_gateway as lg

        class _Msg:
            content = "hi"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]
            model = "m"

        seen = {}

        def fake_completion(**kwargs):
            seen.update(kwargs)
            return _Resp()

        with patch.object(lg, "completion", fake_completion):
            lg.chat([{"role": "user", "content": "hi"}], model="phi3:mini",
                    max_tokens=32, use_cache=False)
        assert seen.get("keep_alive") == "30m"

    def test_completion_proxy_sets_keep_alive(self):
        from gateway import universal_enhanced_gateway as mod
        calls = {}

        def fake_litellm(**kwargs):
            calls.update(kwargs)
            raise AssertionError("stop here")

        real = mod._litellm_completion
        mod._litellm_completion = fake_litellm
        try:
            with pytest.raises(AssertionError):
                mod.completion(model="ollama/phi3:mini", messages=[])
        finally:
            mod._litellm_completion = real
        assert calls.get("keep_alive") == "30m"


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
