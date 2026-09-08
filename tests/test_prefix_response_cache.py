"""
Regression tests for the prefix-keyed RESPONSE cache in
gateway/universal_enhanced_gateway.py.

Verifies that the cache:
  1. Returns the cached response on an identical request (cache hit).
  2. Does NOT reuse a stale response when the model changes.
  3. Does NOT reuse a stale response when generation settings change.
  4. Reports a RESPONSE-cache hit, not a backend KV-cache hit.
  5. Preserves the backwards-compatible `prefix_cache` alias.

The cache is a prefix-keyed full-response cache.  It does NOT perform backend
KV-tensor reuse, so no KV-cache metric may be produced from a response-cache hit.
"""

from __future__ import annotations

import hashlib
import threading


def _make_gateway(enable: bool = True, model: str = "phi3:mini"):
    from gateway.universal_enhanced_gateway import UniversalEnhancedGateway
    gw = UniversalEnhancedGateway(
        model_name=model,
        enable_all_optimizations=enable,
        performance_mode="balanced",
    )
    gw.system_prompt = "You are a helpful assistant."
    return gw


def _write(gw, messages, model="phi3:mini", params=None):
    """Manually mirror the write path (same identity construction)."""
    prefix = ""
    if messages and messages[0].get("role") == "system":
        prefix = messages[0]["content"][:200]
    elif messages:
        prefix = messages[0]["content"][:100]
    prefix_key = hashlib.md5(prefix.encode()).hexdigest()
    full_query = messages[-1]["content"]

    ident = {"model": model, "system_prompt": gw.system_prompt}
    for k in ("temperature", "max_tokens", "top_p", "top_k", "max_length"):
        p = params if params is not None else gw.model_params
        if k in p:
            ident[k] = p[k]

    with threading.Lock():
        gw.prefix_response_cache.setdefault(prefix_key, {})[full_query] = {
            "response": "cached response",
            "gen_identity": ident,
        }


class TestResponseCacheHit:
    def test_identical_request_returns_cached_response(self):
        gw = _make_gateway()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ]
        _write(gw, messages)
        result = gw._check_prefix_cache(messages, gw._current_gen_identity())
        assert result == "cached response"


class TestResponseCacheMissOnModelChange:
    def test_changed_model_does_not_reuse(self):
        gw = _make_gateway(model="phi3:mini")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ]
        _write(gw, messages, model="phi3:mini")
        # Same request but a different model now in use.
        gw.model_name = "llama3.2"
        result = gw._check_prefix_cache(messages, gw._current_gen_identity())
        assert result is None


class TestResponseCacheMissOnSettingsChange:
    def test_changed_temperature_does_not_reuse(self):
        gw = _make_gateway(model="phi3:mini")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ]
        _write(gw, messages, params={"temperature": 0.7, "max_tokens": 256})
        # Same request but temperature changed.
        gw.model_params = {"temperature": 1.2, "max_tokens": 256}
        result = gw._check_prefix_cache(messages, gw._current_gen_identity())
        assert result is None


class TestNoFalseKVMetrics:
    def test_hit_reports_response_cache_not_kv_cache(self):
        # Read the counter dynamically from the module each time (an
        # `from ... import` binding would freeze the value at import time).
        import gateway.universal_enhanced_gateway as mod

        gw = _make_gateway()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
        ]
        _write(gw, messages)
        before = mod._prefix_response_cache_hits
        result = gw._check_prefix_cache(messages, gw._current_gen_identity())
        assert result == "cached response"
        assert mod._prefix_response_cache_hits == before + 1

        stats = gw.get_optimization_stats()
        # A response-cache hit must NOT be advertised as a backend KV-cache hit.
        assert stats["prefix_response_cache_hits"] == before + 1
        assert stats.get("kv_cache_reuse_implemented") is False
        # A miss (unreachable full query or wrong identity) must NOT count a hit.
        miss_msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "completely different query"},
        ]
        assert gw._check_prefix_cache(miss_msgs, gw._current_gen_identity()) is None
        assert mod._prefix_response_cache_hits == before + 1


class TestBackCompatAlias:
    def test_prefix_cache_alias_matches(self):
        gw = _make_gateway()
        # Both names must reference the same underlying object when enabled.
        assert gw.prefix_response_cache is not None
        assert gw.prefix_cache is gw.prefix_response_cache
