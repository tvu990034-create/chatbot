"""
test_perf_regression.py
~~~~~~~~~~~~~~~~~~~~~~~
Performance regression tests for the dev-tool performance goals:
  1. RAG prefetch must be retrieval-ONLY (no LLM generation per RAG call).
  2. check_model_available must be TTL-cached (no HTTP round-trip per request).
  3. Async endpoints must not block the event loop on sync I/O.
"""

import builtins
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# 1. RAG prefetch is retrieval-only (no redundant LLM generation)
# ---------------------------------------------------------------------------


class TestRagPrefetchRetrievalOnly:
    def test_prefetch_uses_retrieve_not_query(self):
        """_rag_prefetch must call retrieve() (embed + fetch), never query()
        (embed + retrieve + LLM generation)."""
        from agents.langgraph_agent import _rag_prefetch
        from config import RAGProvider

        with patch("agents.langgraph_agent.settings") as mock_settings:
            mock_settings.rag_provider = RAGProvider.LLAMA_INDEX

            with patch("rag.llama_index_rag.get_rag") as mock_li:
                mock_li_inst = MagicMock()
                mock_li_inst.retrieve.return_value = {
                    "chunks": ["signal chunk"],
                    "sources": ["signal_doc.py"],
                }
                mock_li.return_value = mock_li_inst

                result = _run(_rag_prefetch("what does signal do?"))

                mock_li_inst.retrieve.assert_called_once()
                mock_li_inst.query.assert_not_called()
                assert "signal chunk" in result

    def test_prefetch_never_uses_haystack_query(self):
        """Haystack RAG prefetch uses retrieve(), never the LLM generator."""
        from agents.langgraph_agent import _rag_prefetch
        from config import RAGProvider

        with patch("agents.langgraph_agent.settings") as mock_settings:
            mock_settings.rag_provider = RAGProvider.HAYSTACK

            with patch("rag.haystack_pipeline.get_haystack_rag") as mock_hs:
                mock_hs_inst = MagicMock()
                mock_hs_inst.retrieve.return_value = {
                    "chunks": ["hay chunk"], "sources": ["doc2"]
                }
                mock_hs.return_value = mock_hs_inst

                result = _run(_rag_prefetch("x"))

                mock_hs_inst.retrieve.assert_called_once()
                mock_hs_inst.query.assert_not_called()
                assert "hay chunk" in result

    def test_llama_index_retrieve_has_no_llm_synthesis(self):
        """LlamaIndexRAG.retrieve() must not call the query engine (no LLM)."""
        from rag.llama_index_rag import LlamaIndexRAG

        rag = LlamaIndexRAG.__new__(LlamaIndexRAG)
        rag.top_k = 3
        rag._index = MagicMock()
        rag._query_engine = MagicMock()  # non-None -> no build_index()
        rag.build_index = MagicMock()
        retriever = MagicMock()
        rag._index.as_retriever.return_value = retriever
        node = MagicMock()
        node.get_content.return_value = "raw chunk text"
        node.metadata = {"file_name": "x.py"}
        retriever.retrieve.return_value = [node]
        rag._eq8_gate = lambda nodes: (0.9, 100)

        result = rag.retrieve("Q")

        assert result["chunks"] == ["raw chunk text"]
        assert result["answer"] == ""
        # If it had synthesised, it would have built a query engine:
        rag.build_index.assert_not_called()

    def test_haystack_retrieve_skips_llm_node(self):
        """HaystackRAG.retrieve() embeds + retrieves only; never runs llm."""
        from rag.haystack_pipeline import HaystackRAG

        rag = HaystackRAG.__new__(HaystackRAG)
        rag._is_built = True
        rag.meta_get = None
        embedder = MagicMock()
        embedder.run.return_value = {"embedding": [0.1, 0.2]}
        retriever = MagicMock()
        doc = MagicMock()
        doc.content = "raw hay chunk"
        doc.meta = {"file_path": "y.md"}
        retriever.run.return_value = {"documents": [doc]}
        qp = MagicMock()
        qp.get_component.side_effect = lambda n: {
            "embedder": embedder, "retriever": retriever,
        }.get(n)
        rag._query_pipeline = qp
        rag.top_k = 3

        result = rag.retrieve("Q")

        assert result["chunks"] == ["raw hay chunk"]
        assert result["answer"] == ""
        embedder.run.assert_called_once()
        retriever.run.assert_called_once()


# ---------------------------------------------------------------------------
# 2. check_model_available is TTL-cached
# ---------------------------------------------------------------------------


class TestModelAvailabilityCache:
    def test_second_call_does_not_reach_network(self):
        from gateway import opt_core
        opt_core.clear_model_availability_cache()
        opt_core._MODEL_AVAIL_TTL = 60.0
        calls = {"n": 0}

        real_urlopen = builtins.__import__("urllib.request").request.urlopen

        def fake_urlopen(*a, **k):
            calls["n"] += 1
            import io
            import json
            payload = json.dumps({
                "models": [{"name": "phi3:mini"}],
            }).encode()
            return io.BytesIO(payload)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            assert opt_core.check_model_available("phi3:mini") is True
            assert calls["n"] == 1
            # Second call within TTL: cache hit, no HTTP
            assert opt_core.check_model_available("phi3:mini") is True
            assert calls["n"] == 1
            # A different model -> new HTTP call (different cache key)
            assert opt_core.check_model_available(
                "phi3:mini", api_base="http://localhost:11435") is True
            assert calls["n"] == 2
        opt_core.clear_model_availability_cache()

    def test_cache_expiry_invalidates(self):
        from gateway import opt_core
        opt_core.clear_model_availability_cache()
        opt_core._MODEL_AVAIL_TTL = 0.0  # force expiry

        import io
        import json
        real_urlopen = builtins.__import__("urllib.request").request.urlopen

        def fake_urlopen(*a, **k):
            payload = json.dumps({"models": [{"name": "phi3:mini"}]}).encode()
            return io.BytesIO(payload)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen) as mocked:
            assert opt_core.check_model_available("phi3:mini") is True
            assert opt_core.check_model_available("phi3:mini") is True
            assert mocked.call_count == 2  # TTL=0 never caches
        opt_core.clear_model_availability_cache()
        opt_core._MODEL_AVAIL_TTL = 60.0

    def test_failure_cached_as_available(self):
        from gateway import opt_core
        opt_core.clear_model_availability_cache()
        opt_core._MODEL_AVAIL_TTL = 60.0

        calls = {"n": 0}

        def boom(*a, **k):
            calls["n"] += 1
            raise OSError("backend down")

        with patch("urllib.request.urlopen", side_effect=boom):
            assert opt_core.check_model_available("phi3:mini") is True
            assert opt_core.check_model_available("phi3:mini") is True
            assert calls["n"] == 1  # failure cached, no repeat timeouts
        opt_core.clear_model_availability_cache()

    def test_missing_tag_not_matched_by_prefix(self):
        """A specific tag (phi3:3.8b) must NOT match an installed model whose
        name merely shares the family prefix (phi3:mini).  Previously
        ``n.startswith(bare.split(':')[0])`` made every 'phi3*' model 'available'
        and RouteLLM routed to uninstalled phi3:3.8b -> APIConnectionError."""
        from gateway import opt_core
        opt_core.clear_model_availability_cache()
        opt_core._MODEL_AVAIL_TTL = 60.0

        import io
        import json

        def fake_urlopen(*a, **k):
            payload = json.dumps({
                "models": [{"name": "phi3:mini"}, {"name": "gemma2:2b"}],
            }).encode()
            return io.BytesIO(payload)

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            assert opt_core.check_model_available("phi3:mini") is True
            assert opt_core.check_model_available("phi3:3.8b") is False, \
                "uninstalled tag must not be reported available via prefix match"
            assert opt_core.check_model_available("phi3") is True, \
                "tag-less name should match the default-tagged installed model"
            assert opt_core.check_model_available("gemma2:2b") is True
            assert opt_core.check_model_available("qwen2.5:3b") is False
        opt_core.clear_model_availability_cache()


# ---------------------------------------------------------------------------
# 3. Async endpoints do not block on sync I/O
# ---------------------------------------------------------------------------


class TestAsyncEndpointsNonBlocking:
    def test_backend_status_runs_in_thread(self):
        import server.app as app_module
        import asyncio

        with patch("server.app.asyncio.to_thread",
                   new=AsyncMock(return_value={"ok": True})) as mock_thread:
            coro = app_module.backend_status()
            result = asyncio.run(coro)
            mock_thread.assert_awaited_once()
            assert result == {"ok": True}

    def test_rag_query_runs_haystack_in_thread(self):
        import server.app as app_module
        import asyncio
        from config import RAGProvider, settings

        req = MagicMock()
        req.provider = "haystack"
        req.question = "Q"

        with patch("server.app.settings") as mock_settings:
            mock_settings.rag_provider = RAGProvider.HAYSTACK
            with patch("rag.haystack_pipeline.get_haystack_rag") as mock_hs:
                mock_hs_inst = MagicMock()
                mock_hs_inst.query.return_value = {"answer": "A", "sources": []}
                mock_hs.return_value = mock_hs_inst

                with patch("server.app.asyncio.to_thread",
                           new=AsyncMock(side_effect=lambda fn: fn())) as mt:
                    out = asyncio.run(app_module.rag_query(req))
                    mt.assert_awaited_once()
                    assert out["haystack"] == {"answer": "A", "sources": []}


def _run(coro):
    import asyncio
    return asyncio.run(coro)