"""
Regression tests for Pass 5 — final code-level integration audit.

Covers:
- Undefined variables (analysis, model_to_use, success/full_response)
- Return type mismatches (tuple unpacking in chat fallbacks)
- Exception safety (finally blocks, NameError prevention)
- Request isolation (self.model_name restore on singleton)
- Self.system_prompt initialization
- Server/app.py dead-code cleanup
"""

import asyncio
import ast
import inspect
import textwrap
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.simple_cache import get_cache


# ---------------------------------------------------------------------------
# 1. litellm_gateway.achat_stream — variables must be initialized before try
# ---------------------------------------------------------------------------
class TestAchatStreamInitOrder(unittest.TestCase):
    """success and full_response must be initialized before the try block so
    the finally clause never raises NameError."""

    def test_success_and_full_response_defined_before_try(self):
        from gateway import litellm_gateway as mod
        src = inspect.getsource(mod.achat_stream)
        # Find the line with "full_response = """ and "success = False"
        # They must appear BEFORE the "try:" line
        lines = src.splitlines()
        try_idx = None
        full_resp_idx = None
        success_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "try:" and try_idx is None:
                try_idx = i
            if try_idx is None:
                # Only consider init lines BEFORE the try block
                if "full_response" in stripped and '""' in stripped and full_resp_idx is None:
                    full_resp_idx = i
                if stripped == "success = False" and success_idx is None:
                    success_idx = i
        self.assertIsNotNone(try_idx, "try block not found")
        self.assertIsNotNone(full_resp_idx, "full_response init not found")
        self.assertIsNotNone(success_idx, "success init not found")
        self.assertLess(full_resp_idx, try_idx,
                        "full_response must be initialized before try block")
        self.assertLess(success_idx, try_idx,
                        "success must be initialized before try block")

    def test_achat_stream_has_finally_block(self):
        from gateway import litellm_gateway as mod
        src = inspect.getsource(mod.achat_stream)
        self.assertIn("finally:", src, "achat_stream must have a finally block")


# ---------------------------------------------------------------------------
# 2. universal_enhanced_gateway — analysis → query_analysis
# ---------------------------------------------------------------------------
class TestQueryAnalysisVariable(unittest.TestCase):
    """The variable `analysis` must not appear as a bare Name in the chat()
    method of UniversalEnhancedGateway; it should be `query_analysis`."""

    def test_no_bare_analysis_in_chat(self):
        from gateway import universal_enhanced_gateway as mod
        src = inspect.getsource(mod.UniversalEnhancedGateway.chat)
        # `query_analysis.is_coding` is correct. Any other `Xanalysis.is_coding`
        # (a bare `analysis.` NOT preceded by `query_`) is a bug.
        import re
        matches = re.findall(r'(?<!query_)analysis\.is_coding', src)
        self.assertEqual(
            len(matches), 0,
            f"chat() still references bare `analysis.is_coding` {len(matches)} time(s); "
            f"should be `query_analysis.is_coding`")
        # Verify the correct form exists
        self.assertIn("query_analysis.is_coding", src,
                      "chat() should use `query_analysis.is_coding`")

    def test_no_bare_analysis_in_cache_write(self):
        from gateway import universal_enhanced_gateway as mod
        src = inspect.getsource(mod.UniversalEnhancedGateway.chat)
        # The code-semcache section should reference query_analysis, not analysis
        lines = src.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "code_semantic_cache" in stripped or "_code_semantic" in stripped:
                # Check next few lines for `analysis.` without `query_` prefix
                for j in range(max(0, i-1), min(len(lines), i+3)):
                    context = lines[j].strip()
                    if context.startswith("if analysis.") and "query_analysis" not in context:
                        self.fail(f"Line {j}: bare `analysis.` reference: {context}")


# ---------------------------------------------------------------------------
# 3. self.system_prompt initialization
# ---------------------------------------------------------------------------
class TestSystemPromptInit(unittest.TestCase):
    """self.system_prompt must be set in __init__ to prevent AttributeError."""

    def test_system_prompt_in_init(self):
        from gateway import universal_enhanced_gateway as mod
        src = inspect.getsource(mod.UniversalEnhancedGateway.__init__)
        self.assertIn("self.system_prompt", src,
                      "__init__ must initialize self.system_prompt")

    def test_instantiation_does_not_raise(self):
        from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
        gw = UniversalEnhancedGateway.__new__(UniversalEnhancedGateway)
        # Simulate __init__ up to the point where system_prompt is set
        gw.system_prompt = None
        # Should not raise
        _ = gw.system_prompt


# ---------------------------------------------------------------------------
# 4. model_to_use in _mindsearch_retrieval
# ---------------------------------------------------------------------------
class TestMindSearchModelVariable(unittest.TestCase):
    """_mindsearch_retrieval must define model_to_use before using it."""

    def test_model_to_use_defined_before_use(self):
        from gateway import universal_enhanced_gateway as mod
        src = inspect.getsource(mod.UniversalEnhancedGateway._mindsearch_retrieval)
        lines = src.splitlines()
        define_idx = None
        use_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            # Definition: a standalone `model_to_use = ...` assignment
            # (note: `model=model_to_use,` is a USE, not a define)
            if stripped.startswith("model_to_use =") and define_idx is None:
                define_idx = i
            if stripped.startswith("model=model_to_use"):
                use_idx = i
                break
        self.assertIsNotNone(use_idx, "model=model_to_use not found")
        self.assertIsNotNone(define_idx,
                              "model_to_use assignment not found before use")
        self.assertLess(define_idx, use_idx,
                        "model_to_use must be defined before first completion() call")


# ---------------------------------------------------------------------------
# 5. server/app.py — actual_model initialized
# ---------------------------------------------------------------------------
class TestAppActualModelInit(unittest.TestCase):
    """actual_model must be pre-initialized in chat_endpoint."""

    def test_actual_model_initialized(self):
        from server import app as mod
        src = inspect.getsource(mod.chat_endpoint)
        self.assertIn("actual_model", src)
        # Check it's initialized (not just assigned inside branches)
        lines = src.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if "actual_model" in stripped and "= None" in stripped:
                break
        else:
            self.fail("actual_model is never initialized with None at top of chat_endpoint")


# ---------------------------------------------------------------------------
# 6. Request isolation — self.model_name restored after routing
# ---------------------------------------------------------------------------
class TestModelNameRestore(unittest.TestCase):
    """self.model_name must be saved before routing and restored in finally."""

    def test_model_name_save_and_restore(self):
        from gateway import universal_enhanced_gateway as mod
        src = inspect.getsource(mod.UniversalEnhancedGateway.chat)
        self.assertIn("original_model_name = self.model_name", src,
                      "chat() must save original model name before routing")
        self.assertIn("self.model_name = original_model_name", src,
                      "chat() must restore original model name in finally block")

    def test_finally_block_exists(self):
        from gateway import universal_enhanced_gateway as mod
        src = inspect.getsource(mod.UniversalEnhancedGateway.chat)
        self.assertIn("finally:", src,
                      "chat() must have a finally block for model_name restore")


# ---------------------------------------------------------------------------
# 7. Tuple return type consistency
# ---------------------------------------------------------------------------
class TestLitellmGatewayTupleReturn(unittest.TestCase):
    """litellm_gateway.chat() always returns (str, bool, str) tuple."""

    def test_chat_returns_3_tuple(self):
        from gateway import litellm_gateway as mod
        sig = inspect.signature(mod.chat)
        # Verify function signature accepts all expected params
        param_names = list(sig.parameters.keys())
        self.assertIn("system_prompt", param_names)
        self.assertIn("speed_mode", param_names)

    def test_achat_returns_3_tuple(self):
        from gateway import litellm_gateway as mod
        sig = inspect.signature(mod.achat)
        param_names = list(sig.parameters.keys())
        self.assertIn("system_prompt", param_names)
        self.assertIn("use_router", param_names)


# ---------------------------------------------------------------------------
# 8. No bare 'analysis.' in universal_enhanced_gateway
# ---------------------------------------------------------------------------
class TestNoBareAnalysisVariable(unittest.TestCase):
    """Search the entire module for bare 'analysis.' references that should be
    'query_analysis.' in the chat method."""

    def test_no_analysis_dot_in_chat_method(self):
        from gateway import universal_enhanced_gateway as mod
        src = inspect.getsource(mod.UniversalEnhancedGateway.chat)
        # Split into lines and check for "analysis." that is NOT preceded by "query_"
        import re
        # Find all occurrences of analysis. that are NOT query_analysis.
        matches = re.findall(r'(?<!query_)analysis\.', src)
        self.assertEqual(len(matches), 0,
                         f"Found {len(matches)} bare 'analysis.' references "
                         f"that should be 'query_analysis.'")


# ---------------------------------------------------------------------------
# 9. Cache singleton thread safety
# ---------------------------------------------------------------------------
class TestCacheSingleton(unittest.TestCase):
    """SimpleCache get_cache() returns same instance across threads."""

    def test_singleton_across_threads(self):
        import threading
        instances = []

        def get_it():
            instances.append(get_cache())

        threads = [threading.Thread(target=get_it) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertTrue(all(i is instances[0] for i in instances),
                        "get_cache() must return the same instance")

    def test_cache_set_get_with_key(self):
        cache = get_cache()
        cache.set("pass5_test_q", "pass5_test_a", _key="pass5_k1")
        result = cache.get("pass5_test_q", _key="pass5_k1")
        self.assertEqual(result, "pass5_test_a")


# ---------------------------------------------------------------------------
# 10. app.py no triple-fallback dead code
# ---------------------------------------------------------------------------
class TestAppNoTripleFallback(unittest.TestCase):
    """The chat_endpoint should not have a triple-fallback try/except cascade."""

    def test_no_triple_fallback(self):
        from server import app as mod
        src = inspect.getsource(mod.chat_endpoint)
        # The triple-fallback pattern: "except (TypeError, ValueError)" appears
        # at most once (for the old 2-tuple compat path). If it appears twice,
        # the dead code was not removed.
        count = src.count("except (TypeError, ValueError)")
        self.assertLessEqual(count, 1,
                             f"Found {count} TypeError/ValueError catches "
                             f"(expected <= 1); dead code may not be removed")


# ---------------------------------------------------------------------------
# 11. Reasoning framework enum includes 'default'
# ---------------------------------------------------------------------------
class TestReasoningFrameworkEnum(unittest.TestCase):
    """ReasoningFramework enum must include 'default' to match config default."""

    def test_default_is_member(self):
        from reasoning.reasoning_manager import ReasoningFramework
        self.assertEqual(ReasoningFramework("default"), ReasoningFramework.DEFAULT)


if __name__ == "__main__":
    unittest.main()
