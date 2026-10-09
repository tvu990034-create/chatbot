"""
Regression tests for API request-flow correctness.

Verifies that every request parameter actually reaches the subsystem that
is supposed to use it.  All external boundaries (LLM, RAG, cache) are
mocked so the tests prove propagation, not integration.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_completion_response(content="hello"):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.model = "ollama/test-model"
    return resp


def _run(coro):
    """Run an async coroutine in a fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def _achat_mocks():
    """Return (patches_dict, started_mocks_dict, setup_fn, teardown_fn).

    Patches all lazy-imported symbols at their source modules.
    Now patches litellm.acompletion (async) instead of litellm.completion.
    """
    patches = {}
    patches["acompletion"] = patch("litellm.acompletion", new_callable=AsyncMock,
                                    return_value=_mock_completion_response("ok"))
    patches["make_cache_identity"] = patch("gateway.opt_core.make_cache_identity", return_value="key")
    patches["is_safe_quick_path"] = patch("gateway.opt_core.is_safe_quick_path", return_value=False)
    patches["quick_arithmetic"] = patch("gateway.opt_core.quick_arithmetic", return_value=None)
    patches["estimate_tokens"] = patch("gateway.opt_core.estimate_tokens", return_value=10)
    patches["build_context_messages"] = patch("gateway.opt_core.build_context_messages")
    patches["select_model"] = patch("gateway.opt_core.select_model", return_value="test-model")
    patches["get_router_state"] = patch("gateway.opt_core.get_router_state")
    patches["ollama_model_id"] = patch("gateway.opt_core.ollama_model_id", side_effect=lambda m: m)
    patches["check_model_available"] = patch("gateway.opt_core.check_model_available", return_value=True)
    patches["get_cache"] = patch("gateway.simple_cache.get_cache")

    m = {}  # started mock instances

    def setup():
        ctx = SimpleNamespace(
            messages=[{"role": "user", "content": "hi"}],
            compressed=False, reasoning="", input_tokens=10,
        )
        m["build_context_messages"] = patches["build_context_messages"].start()
        m["build_context_messages"].return_value = ctx

        router_state = SimpleNamespace(record_start=MagicMock(), record_end=MagicMock())
        m["get_router_state"] = patches["get_router_state"].start()
        m["get_router_state"].return_value = router_state

        m["select_model"] = patches["select_model"].start()
        m["ollama_model_id"] = patches["ollama_model_id"].start()
        m["check_model_available"] = patches["check_model_available"].start()
        m["make_cache_identity"] = patches["make_cache_identity"].start()
        m["is_safe_quick_path"] = patches["is_safe_quick_path"].start()
        m["quick_arithmetic"] = patches["quick_arithmetic"].start()
        m["estimate_tokens"] = patches["estimate_tokens"].start()

        cache_inst = MagicMock()
        cache_inst.get.return_value = None
        m["get_cache"] = patches["get_cache"].start()
        m["get_cache"].return_value = cache_inst

        m["acompletion"] = patches["acompletion"].start()

    def teardown():
        for p in patches.values():
            try:
                p.stop()
            except RuntimeError:
                pass

    return patches, m, setup, teardown


# ===================================================================
# 1. litellm_gateway.chat() — system_prompt injection
# ===================================================================

class TestLitellmGatewayChatSystemPrompt:
    """chat() must inject system_prompt as a system message if not present."""

    @patch("gateway.litellm_gateway.completion")
    @patch("gateway.litellm_gateway._cache")
    def test_injects_system_prompt_when_missing(self, mock_cache, mock_completion):
        mock_cache.get.return_value = None
        mock_completion.return_value = _mock_completion_response("ok")

        from gateway.litellm_gateway import chat
        msgs = [{"role": "user", "content": "hi"}]
        chat(msgs, system_prompt="Be concise.")

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["messages"][0] == {"role": "system", "content": "Be concise."}
        assert call_kwargs["messages"][1] == {"role": "user", "content": "hi"}

    @patch("gateway.litellm_gateway.completion")
    @patch("gateway.litellm_gateway._cache")
    def test_does_not_duplicate_existing_system(self, mock_cache, mock_completion):
        mock_cache.get.return_value = None
        mock_completion.return_value = _mock_completion_response("ok")

        from gateway.litellm_gateway import chat
        msgs = [
            {"role": "system", "content": "Existing prompt."},
            {"role": "user", "content": "hi"},
        ]
        chat(msgs, system_prompt="New prompt.")

        call_kwargs = mock_completion.call_args[1]
        sys_msgs = [m for m in call_kwargs["messages"] if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert sys_msgs[0]["content"] == "Existing prompt."


# ===================================================================
# 2. litellm_gateway.chat() — temperature/max_tokens reach LLM
# ===================================================================

class TestLitellmGatewayChatParams:
    """chat() must pass temperature and max_tokens to litellm.completion."""

    @patch("gateway.litellm_gateway.completion")
    @patch("gateway.litellm_gateway._cache")
    def test_temperature_forwarded(self, mock_cache, mock_completion):
        mock_cache.get.return_value = None
        mock_completion.return_value = _mock_completion_response("ok")

        from gateway.litellm_gateway import chat
        chat([{"role": "user", "content": "test"}], temperature=0.2, max_tokens=100)

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["max_tokens"] == 100

    @patch("gateway.litellm_gateway.completion")
    @patch("gateway.litellm_gateway._cache")
    def test_model_forwarded_with_prefix(self, mock_cache, mock_completion):
        mock_cache.get.return_value = None
        mock_completion.return_value = _mock_completion_response("ok")

        from gateway.litellm_gateway import chat
        chat([{"role": "user", "content": "test"}], model="llama3.2")

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "ollama/llama3.2"


# ===================================================================
# 3. litellm_gateway.chat() — use_cache gating
# ===================================================================

class TestLitellmGatewayChatCache:
    """chat() must respect use_cache for both reads and writes."""

    @patch("gateway.litellm_gateway.completion")
    @patch("gateway.litellm_gateway._cache")
    def test_cache_bypass_on_use_cache_false(self, mock_cache, mock_completion):
        mock_completion.return_value = _mock_completion_response("fresh")

        from gateway.litellm_gateway import chat
        reply, hit, _ = chat(
            [{"role": "user", "content": "test"}],
            use_cache=False,
        )

        assert reply == "fresh"
        assert hit is False
        mock_cache.get.assert_not_called()
        mock_cache.set.assert_not_called()


# ===================================================================
# 3b. litellm_gateway.chat() — speed-mode cache key reflects real params
# ===================================================================

class TestSpeedModeCacheKeyIsolation:
    """BUG: chat() cached the speed-mode (capped 128/0.3) reply under the
    caller's full-budget key, so a later non-speed request got a truncated
    speed answer.  The cache read AND write contexts must use the capped
    parameters that actually produced the response."""

    @patch("gateway.litellm_gateway.completion")
    @patch("gateway.litellm_gateway._cache")
    def test_speed_mode_caches_under_capped_params(self, mock_cache, mock_completion):
        mock_cache.get.return_value = None
        mock_completion.return_value = _mock_completion_response("fast")

        from gateway.litellm_gateway import chat
        chat(
            [{"role": "user", "content": "test"}],
            temperature=0.7,
            max_tokens=512,
            speed_mode=True,
        )

        read_ctx = mock_cache.get.call_args[1]["context"]
        write_ctx = mock_cache.set.call_args[1]["context"]
        # Both the lookup and the store must key on the ACTUAL generation params.
        assert read_ctx["temperature"] == 0.3
        assert read_ctx["max_tokens"] == 128
        assert write_ctx["temperature"] == 0.3
        assert write_ctx["max_tokens"] == 128
        # The LLM actually used the capped budget, not the caller's 512/0.7.
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 128

    @patch("gateway.litellm_gateway.completion")
    @patch("gateway.litellm_gateway._cache")
    def test_non_speed_mode_keeps_caller_params_in_key(self, mock_cache, mock_completion):
        mock_cache.get.return_value = None
        mock_completion.return_value = _mock_completion_response("full")

        from gateway.litellm_gateway import chat
        chat(
            [{"role": "user", "content": "test"}],
            temperature=0.7,
            max_tokens=512,
        )

        read_ctx = mock_cache.get.call_args[1]["context"]
        assert read_ctx["temperature"] == 0.7
        assert read_ctx["max_tokens"] == 512
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 512

    @patch("gateway.litellm_gateway.completion")
    @patch("gateway.litellm_gateway._cache")
    def test_speed_and_baseline_hit_distinct_cache_slots(self, mock_cache, mock_completion):
        mock_cache.get.return_value = None
        mock_completion.return_value = _mock_completion_response("x")

        from gateway.litellm_gateway import chat
        chat([{"role": "user", "content": "same q"}], temperature=0.7, max_tokens=512, speed_mode=True)
        speed_key = mock_cache.set.call_args[1]["context"]

        mock_cache.get.reset_mock()
        chat([{"role": "user", "content": "same q"}], temperature=0.7, max_tokens=512)
        baseline_key = mock_cache.get.call_args[1]["context"]

        assert speed_key["max_tokens"] != baseline_key["max_tokens"] or \
            speed_key["temperature"] != baseline_key["temperature"]


# ===================================================================
# 4–6. langgraph_agent.achat() — model/temperature/max_tokens → LLM
# ===================================================================

class TestAchatModelPropagation:
    """achat() must propagate model to litellm.acompletion."""

    def test_explicit_model_reaches_llm(self):
        _patches, m, setup, teardown = _achat_mocks()
        try:
            setup()
            from agents.langgraph_agent import achat
            _run(achat("hi", model="my-model", use_rag=False, use_cache=False))
            call_kwargs = m["acompletion"].call_args[1]
            assert call_kwargs["model"] == "my-model"
        finally:
            teardown()


class TestAchatTemperaturePropagation:
    """achat() must propagate temperature to litellm.acompletion."""

    def test_explicit_temperature_reaches_llm(self):
        _patches, m, setup, teardown = _achat_mocks()
        try:
            setup()
            from agents.langgraph_agent import achat
            _run(achat("hi", temperature=0.2, use_rag=False, use_cache=False))
            call_kwargs = m["acompletion"].call_args[1]
            assert call_kwargs["temperature"] == 0.2
        finally:
            teardown()


class TestAchatMaxTokensPropagation:
    """achat() must propagate max_tokens to litellm.acompletion."""

    def test_explicit_max_tokens_reaches_llm(self):
        _patches, m, setup, teardown = _achat_mocks()
        try:
            setup()
            from agents.langgraph_agent import achat
            _run(achat("hi", max_tokens=200, use_rag=False, use_cache=False))
            call_kwargs = m["acompletion"].call_args[1]
            assert call_kwargs["max_tokens"] == 200
        finally:
            teardown()


# ===================================================================
# 7. achat() — system_prompt reaches context builder
# ===================================================================

class TestAchatSystemPromptPropagation:
    """achat() must pass system_prompt via _req_ctx to build_context_messages."""

    def test_system_prompt_in_context_builder(self):
        _patches, m, setup, teardown = _achat_mocks()
        try:
            setup()
            from agents.langgraph_agent import achat
            _run(achat(
                "hi", system_prompt="Be a pirate.",
                use_rag=False, use_cache=False,
            ))
            call_kwargs = m["build_context_messages"].call_args[1]
            assert call_kwargs["system_prompt"] == "Be a pirate."
        finally:
            teardown()


# ===================================================================
# 8–9. achat() — use_rag gating
# ===================================================================

class TestAchatUseRag:
    """achat() must honour use_rag for RAG prefetch."""

    def test_use_rag_false_skips_prefetch(self):
        _patches, m, setup, teardown = _achat_mocks()
        rag_mock = patch("agents.langgraph_agent._rag_prefetch", new_callable=AsyncMock)
        try:
            setup()
            rag_inst = rag_mock.start()
            from agents.langgraph_agent import achat
            _run(achat("hi", use_rag=False, use_cache=False))
            rag_inst.assert_not_called()
        finally:
            rag_mock.stop()
            teardown()

    def test_use_rag_true_triggers_prefetch(self):
        _patches, m, setup, teardown = _achat_mocks()
        rag_mock = patch(
            "agents.langgraph_agent._rag_prefetch",
            new_callable=AsyncMock, return_value="RAG context",
        )
        try:
            setup()
            rag_inst = rag_mock.start()
            from agents.langgraph_agent import achat
            _run(achat("hi", use_rag=True, use_cache=False))
            rag_inst.assert_called_once()
        finally:
            rag_mock.stop()
            teardown()


# ===================================================================
# 10. achat() — use_cache gating
# ===================================================================

class TestAchatUseCache:
    """achat() must skip cache read AND write when use_cache=False."""

    def test_cache_bypass(self):
        _patches, m, setup, teardown = _achat_mocks()
        try:
            setup()
            from agents.langgraph_agent import achat
            _run(achat("hi", use_rag=False, use_cache=False))
            cache_inst = m["get_cache"].return_value
            cache_inst.get.assert_not_called()
            cache_inst.set.assert_not_called()
            m["acompletion"].assert_called_once()
        finally:
            teardown()


# ===================================================================
# 11. achat() — use_router gating
# ===================================================================

class TestAchatUseRouter:
    """achat() must skip router when use_router=False."""

    def test_router_disabled_skips_select(self):
        _patches, m, setup, teardown = _achat_mocks()
        try:
            setup()
            m["select_model"].reset_mock()
            from agents.langgraph_agent import achat
            _run(achat("hi", use_router=False, use_rag=False, use_cache=False))
            m["select_model"].assert_not_called()
        finally:
            teardown()


# ===================================================================
# 12. /api/v1/chat — temperature/max_tokens override performance_mode
# ===================================================================

class TestChatV1Endpoint:
    """/api/v1/chat must use caller-provided temp/max_tokens over performance_mode."""

    @patch("agents.langgraph_agent.achat", new_callable=AsyncMock)
    def test_caller_temp_overrides_performance_mode(self, mock_achat):
        mock_achat.return_value = SimpleNamespace(
            reply="ok", cache_hit=False, model_used="test-model",
        )

        from fastapi.testclient import TestClient
        from server.app import app
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/v1/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "temperature": 0.1,
            "max_tokens": 50,
            "performance_mode": "speed",
        })

        assert resp.status_code == 200
        call_kwargs = mock_achat.call_args[1]
        assert call_kwargs["temperature"] == 0.1
        assert call_kwargs["max_tokens"] == 50

        body = resp.json()
        assert body["response"] == "ok"
        assert body["cache_hit"] is False
        assert isinstance(body["duration"], float)
        assert body["optimizations_applied"] >= 1

    @patch("agents.langgraph_agent.achat", new_callable=AsyncMock)
    def test_performance_mode_defaults_when_no_caller_values(self, mock_achat):
        mock_achat.return_value = SimpleNamespace(
            reply="ok", cache_hit=False, model_used="test-model",
        )

        from fastapi.testclient import TestClient
        from server.app import app
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/v1/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
            "model": "test-model",
            "performance_mode": "speed",
        })

        assert resp.status_code == 200
        call_kwargs = mock_achat.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 256


# ===================================================================
# 13. /chat — explicit params reach achat
# ===================================================================

class TestChatEndpointParams:
    """/chat must forward all request params to achat()."""

    @patch("agents.langgraph_agent.achat", new_callable=AsyncMock)
    def test_explicit_params_forwarded(self, mock_achat):
        mock_achat.return_value = SimpleNamespace(
            reply="ok", cache_hit=False, model_used="custom-model",
        )

        from fastapi.testclient import TestClient
        from server.app import app
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/chat", json={
            "message": "hi",
            "model": "custom-model",
            "temperature": 0.3,
            "max_tokens": 128,
            "system_prompt": "Be brief.",
            "use_rag": False,
            "use_cache": True,
            "use_router": True,
            "use_agent": True,
        })

        assert resp.status_code == 200
        call_kwargs = mock_achat.call_args[1]
        assert call_kwargs["model"] == "custom-model"
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 128
        assert call_kwargs["system_prompt"] == "Be brief."
        assert call_kwargs["use_rag"] is False
        assert call_kwargs["use_cache"] is True

        # Unified contract: /chat and /api/v1/chat share the same shape.
        body = resp.json()
        assert body["response"] == "ok"
        assert body["model"] == "custom-model"
        assert isinstance(body["duration"], float)
        assert isinstance(body["cache_hit"], bool)
        assert isinstance(body["optimizations_applied"], int)
        assert call_kwargs["use_router"] is True


# ===================================================================
# 13b. /chat direct path — gateway-first ordering
# ===================================================================

class TestChatDirectGatewayFirst:
    """/chat with use_agent=false must try the thinking-aware gateway
    BEFORE the plain path when generation is not overridden (diag: plain
    first burned 143 s on 12*8 and served 'Empty Response' on qwen3).
    Explicit overrides still go plain-first; empty gateway answers
    still fall back to plain."""

    def _client(self):
        from fastapi.testclient import TestClient
        from server.app import app
        return TestClient(app, raise_server_exceptions=False)

    def _gw(self, reply):
        gw = MagicMock()
        gw.chat.return_value = reply
        gw.model_name = "gw-model"
        return gw

    def test_instant_answer_skips_plain_path(self):
        with patch("gateway.universal_enhanced_gateway.get_universal_gateway",
                   return_value=self._gw("The answer is 96")), \
             patch("gateway.litellm_gateway.achat",
                   new_callable=AsyncMock) as mock_plain:
            resp = self._client().post("/chat", json={
                "message": "What is 12*8?",
                "use_agent": False, "use_rag": False,
            })
        assert resp.status_code == 200
        assert resp.json()["response"] == "The answer is 96"
        mock_plain.assert_not_called()

    def test_empty_gateway_falls_back_to_plain(self):
        with patch("gateway.universal_enhanced_gateway.get_universal_gateway",
                   return_value=self._gw("   ")), \
             patch("gateway.litellm_gateway.achat",
                   new_callable=AsyncMock) as mock_plain:
            mock_plain.return_value = ("plain reply", False, "m")
            resp = self._client().post("/chat", json={
                "message": "What is 12*8?",
                "use_agent": False, "use_rag": False,
            })
        assert resp.status_code == 200
        assert resp.json()["response"] == "plain reply"
        mock_plain.assert_called_once()

    def test_explicit_override_goes_plain_first(self):
        with patch("gateway.universal_enhanced_gateway.get_universal_gateway") as mock_gw, \
             patch("gateway.litellm_gateway.achat",
                   new_callable=AsyncMock) as mock_plain:
            mock_plain.return_value = ("plain reply", False, "m")
            resp = self._client().post("/chat", json={
                "message": "What is 12*8?",
                "use_agent": False, "use_rag": False,
                "temperature": 0.9,
            })
        assert resp.status_code == 200
        mock_gw.assert_not_called()
        assert mock_plain.call_args[1]["temperature"] == 0.9


class TestChatRagDegenerateFallback:
    """A degenerate RAG answer must fall through to the direct path, never
    serve the degenerate text (diag: 12*8 served the literal model text
    'Empty Response' after 143 s with sources pointing at an irrelevant
    note). Stale sources must not survive the fallback either."""

    def _client(self):
        from fastapi.testclient import TestClient
        from server.app import app
        return TestClient(app, raise_server_exceptions=False)

    def _rag(self, answer, sources=None):
        rag = MagicMock()
        rag.aquery = AsyncMock(return_value={
            "answer": answer, "sources": sources or []})
        return rag

    def _gw(self, reply):
        gw = MagicMock()
        gw.chat.return_value = reply
        gw.model_name = "gw-model"
        return gw

    def test_empty_response_falls_back_to_direct(self):
        with patch("rag.llama_index_rag.get_rag",
                   return_value=self._rag("Empty Response", ["note.txt"])), \
             patch("gateway.universal_enhanced_gateway.get_universal_gateway",
                   return_value=self._gw("The answer is 96")), \
             patch("gateway.litellm_gateway.achat",
                   new_callable=AsyncMock) as mock_plain:
            mock_plain.return_value = ("plain reply", False, "m")
            resp = self._client().post("/chat", json={
                "message": "What is 12*8?",
                "use_agent": False, "use_rag": True,
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "The answer is 96"
        assert body["sources"] == []
        mock_plain.assert_not_called()

    def test_good_rag_answer_served_as_is(self):
        with patch("rag.llama_index_rag.get_rag",
                   return_value=self._rag(
                       "The harbor lights mark the entrance.",
                       ["note.txt"])), \
             patch("gateway.universal_enhanced_gateway.get_universal_gateway") as mock_gw, \
             patch("gateway.litellm_gateway.achat",
                   new_callable=AsyncMock) as mock_plain:
            resp = self._client().post("/chat", json={
                "message": "What do the harbor lights mark?",
                "use_agent": False, "use_rag": True,
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["response"] == "The harbor lights mark the entrance."
        assert body["sources"] == ["note.txt"]
        mock_gw.assert_not_called()
        mock_plain.assert_not_called()


# ===================================================================
# 14. Request bounds — oversized payloads rejected before inference
# ===================================================================

class TestRequestBounds:
    """Unbounded history/system_prompt/messages/images let one request burn
    unbounded parse + token-estimate CPU upstream of the context trim."""

    def _client(self):
        from fastapi.testclient import TestClient
        from server.app import app
        return TestClient(app, raise_server_exceptions=False)

    def test_chat_rejects_giant_history(self):
        client = self._client()
        hist = [{"role": "user", "content": "hi"}] * 201
        resp = client.post("/chat", json={"message": "hi", "history": hist})
        assert resp.status_code == 422

    def test_chat_rejects_giant_system_prompt(self):
        client = self._client()
        resp = client.post("/chat", json={"message": "hi",
                                          "system_prompt": "x" * 32_001})
        assert resp.status_code == 422

    def test_v1_rejects_bad_mode_and_giant_images(self):
        client = self._client()
        resp = client.post("/api/v1/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
            "performance_mode": "turbo",
        })
        assert resp.status_code == 422
        resp = client.post("/api/v1/chat", json={
            "messages": [{"role": "user", "content": "hi"}],
            "images": ["x"] * 9,
        })
        assert resp.status_code == 422

    def test_rag_query_rejects_giant_question(self):
        client = self._client()
        resp = client.post("/rag/query", json={
            "question": "x" * 32_001,
            "provider": "haystack",
        })
        assert resp.status_code == 422


class TestRagEndpoints:
    """Ingest must accept files-only (body was required -> 422 on every
    real multipart upload); query must dispatch to the active provider."""

    def test_ingest_files_only(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, patch
        from config import settings, RAGProvider
        monkeypatch.setattr(settings, "rag_provider", RAGProvider.LLAMA_INDEX)
        monkeypatch.setattr(settings, "rag_docs_dir", tmp_path)
        from fastapi.testclient import TestClient
        from server.app import app
        client = TestClient(app, raise_server_exceptions=False)
        fake_rag = MagicMock()
        fake_rag.add_documents.return_value = 1
        with patch("rag.llama_index_rag.get_rag", return_value=fake_rag):
            resp = client.post("/rag/ingest", files=[
                ("files", ("d.txt", b"hello world", "text/plain"))])
        assert resp.status_code == 200
        assert resp.json()["saved_files"]
        fake_rag.add_documents.assert_called_once()

    def test_query_dispatches_to_active_provider(self, monkeypatch):
        from unittest.mock import AsyncMock, MagicMock, patch
        from config import settings, RAGProvider
        monkeypatch.setattr(settings, "rag_provider", RAGProvider.LLAMA_INDEX)
        from fastapi.testclient import TestClient
        from server.app import app
        client = TestClient(app, raise_server_exceptions=False)
        fake_rag = MagicMock()
        fake_rag.aquery = AsyncMock(return_value={"answer": "A",
                                                  "sources": []})
        with patch("rag.llama_index_rag.get_rag", return_value=fake_rag):
            resp = client.post("/rag/query", json={"question": "q?"})
        assert resp.status_code == 200
        assert resp.json()["llama_index"]["answer"] == "A"


class TestOpenAICompat:
    """Training/eval harnesses speak OpenAI's protocol: /v1/chat/completions
    must return the exact shape with the agent answer inside."""

    def _client(self):
        from fastapi.testclient import TestClient
        from server.app import app
        return TestClient(app, raise_server_exceptions=False)

    def test_completions_shape(self):
        from unittest.mock import AsyncMock, patch
        from types import SimpleNamespace
        client = self._client()
        fake = SimpleNamespace(reply="hi there", cache_hit=False,
                               model_used="m", rag_used=False,
                               generation_policy="balanced")
        with patch("agents.langgraph_agent.achat",
                   new_callable=AsyncMock, return_value=fake):
            resp = client.post("/v1/chat/completions", json={
                "model": "qwen3:4b",
                "messages": [{"role": "system", "content": "Be brief."},
                             {"role": "user", "content": "Hi"}],
            })
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "chat.completion"
        assert body["id"].startswith("chatcmpl-")
        assert isinstance(body["created"], int)
        assert body["choices"][0]["message"] == {
            "role": "assistant", "content": "hi there"}
        assert body["choices"][0]["finish_reason"] == "stop"
        usage = body["usage"]
        assert usage["total_tokens"] == (usage["prompt_tokens"]
                                         + usage["completion_tokens"])
        assert usage["completion_tokens"] > 0

    def test_completions_rejects_stream_and_empty(self):
        client = self._client()
        resp = client.post("/v1/chat/completions", json={
            "model": "m",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        })
        assert resp.status_code == 400
        resp = client.post("/v1/chat/completions", json={
            "model": "m", "messages": [],
        })
        assert resp.status_code == 422

    def test_models_falls_back_without_backend(self, monkeypatch):
        import urllib.request
        from fastapi.testclient import TestClient
        from server.app import app
        from config import settings
        client = TestClient(app, raise_server_exceptions=False)

        def boom(*a, **k):
            raise ConnectionRefusedError("down")

        monkeypatch.setattr(urllib.request, "urlopen", boom)
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == settings.default_model


class TestEmptyReplyRecovery:
    """Thinking models return empty content on the plain litellm path;
    the endpoint must recover via the thinking-aware gateway, not serve ''."""

    def test_empty_direct_reply_recovers(self):
        from unittest.mock import AsyncMock, MagicMock, patch
        from fastapi.testclient import TestClient
        from server import app as app_module

        gw = MagicMock()
        gw.chat.return_value = "recovered answer"
        gw.model_name = "qwen3:4b"
        with patch("gateway.litellm_gateway.achat",
                   new_callable=AsyncMock) as mock_achat, \
             patch("gateway.universal_enhanced_gateway.get_universal_gateway",
                   return_value=gw):
            mock_achat.return_value = ("", False, "m")
            client = TestClient(app_module.app, raise_server_exceptions=False)
            resp = client.post("/chat", json={
                "message": "Zxq unique recovery probe quux?",
                "use_agent": False, "use_rag": False, "use_cache": False,
            })
        assert resp.status_code == 200
        assert resp.json()["response"] == "recovered answer"
        gw.chat.assert_called_once()
