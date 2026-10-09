"""
Test RAG functionality.
"""

import pytest
from pathlib import Path


def test_llama_index_imports():
    """Test that LlamaIndex RAG can be imported."""
    try:
        from rag.llama_index_rag import LlamaIndexRAG
        assert LlamaIndexRAG is not None
    except ImportError as e:
        pytest.skip(f"LlamaIndex not available: {e}")


def test_haystack_imports():
    """Test that Haystack RAG can be imported."""
    try:
        from rag.haystack_pipeline import HaystackRAG
        assert HaystackRAG is not None
    except ImportError as e:
        pytest.skip(f"Haystack not available: {e}")


@pytest.mark.skipif(
    not Path("data/docs").exists(),
    reason="Test documents directory not found"
)
def test_llama_index_initialization():
    """Test LlamaIndex RAG initialization."""
    try:
        from rag.llama_index_rag import LlamaIndexRAG
        
        rag = LlamaIndexRAG()
        assert rag.docs_dir is not None
        assert rag.top_k > 0
    except ImportError as e:
        pytest.skip(f"LlamaIndex not available: {e}")


def test_rag_config_settings():
    """Test that RAG settings are properly configured."""
    from config import settings
    
    assert settings.rag_provider is not None
    assert settings.rag_chunk_size > 0
    assert settings.rag_top_k > 0
    assert settings.embedding_model is not None


def test_data_directories():
    """Test that data directories exist or can be created."""
    from config import settings
    from pathlib import Path
    
    # Test that directories can be accessed
    docs_dir = settings.rag_docs_dir
    index_dir = settings.rag_index_dir
    chroma_dir = settings.chroma_persist_dir
    
    # Parent directories should exist
    assert docs_dir.parent.exists() or docs_dir.parent.parent.exists()
    assert index_dir.parent.exists() or index_dir.parent.parent.exists()
    assert chroma_dir.parent.exists() or chroma_dir.parent.parent.exists()


def test_public_paths_share_retrieve_nodes_seam():
    """retrieve()/query()/aquery() must all funnel node fetching through
    _retrieve_nodes (was triplicated inline): one seam for retrieve-level
    wiring, no behavior change."""
    from unittest.mock import patch
    from rag.llama_index_rag import LlamaIndexRAG

    rag = LlamaIndexRAG()
    rag._query_engine = object()  # skip index build; seam is mocked below
    with patch.object(LlamaIndexRAG, "_retrieve_nodes",
                      return_value=[]) as seam:
        out = rag.retrieve("q?")
        assert seam.call_count == 1
        assert out["chunks"] == []
        assert out["retrieval_confidence"] == 0.0
        out = rag.query("q?")
        assert seam.call_count == 2
        # Empty retrieval -> Eq8 hard gate refuses without any LLM call.
        assert "could not find" in out["answer"]
        import asyncio
        out = asyncio.run(rag.aquery("q?"))
        assert seam.call_count == 3
        assert "could not find" in out["answer"]


def test_llama_empty_index_skips_embedding(tmp_path):
    """An index known to be empty must not pay a query encode (~50-200ms
    CPU) per request: retrieve short-circuits before retrieval."""
    from unittest.mock import patch
    try:
        from rag.llama_index_rag import LlamaIndexRAG
    except ImportError as e:
        pytest.skip(f"LlamaIndex stack not installed: {e}")
        return

    docs = tmp_path / "docs"
    docs.mkdir()
    persist = tmp_path / "chroma"
    try:
        rag = LlamaIndexRAG(docs_dir=docs, persist_dir=persist)
        rag.build_index()
    except ImportError as e:
        pytest.skip(f"LlamaIndex stack not installed: {e}")
        return
    assert rag._empty_index is True
    with patch.object(LlamaIndexRAG, "_retrieve_nodes",
                      side_effect=AssertionError("must not retrieve")):
        out = rag.retrieve("anything at all?")
    assert out == {"answer": "", "chunks": [], "sources": [],
                   "retrieval_confidence": 0.0}


def test_llama_add_documents_clears_empty_flag(tmp_path):
    try:
        from rag.llama_index_rag import LlamaIndexRAG
    except ImportError as e:
        pytest.skip(f"LlamaIndex stack not installed: {e}")
        return

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.txt").write_text(
        "The harbor lights mark the entrance. " * 20, encoding="utf-8")
    persist = tmp_path / "chroma"
    try:
        rag = LlamaIndexRAG(docs_dir=docs, persist_dir=persist)
        rag.build_index()
    except ImportError as e:
        pytest.skip(f"LlamaIndex stack not installed: {e}")
        return
    assert rag._empty_index is False
    assert rag.retrieve("harbor lights")["chunks"]


def test_haystack_empty_store_skips_embedding():
    """Same short-circuit for the Haystack provider."""
    try:
        from rag.haystack_pipeline import HaystackRAG
        rag = HaystackRAG()
        out = rag.retrieve("anything at all?")
    except ImportError as e:
        pytest.skip(f"Haystack not installed: {e}")
        return
    assert out == {"answer": "", "chunks": [], "sources": []}


def test_haystack_llm_points_at_ollama():
    """The query generator must target the local ollama OpenAI endpoint,
    not a hypothetical :4000 proxy (every generation failed there)."""
    import pathlib
    src = pathlib.Path("rag/haystack_pipeline.py").read_text(encoding="utf-8")
    assert "http://127.0.0.1:11434" in src
    assert "localhost:4000" not in src


class TestRagLazyBuild:
    """build_index must check for retrievable content BEFORE configuring
    embeddings: _configure_llama_settings() downloads a ~133 MB model,
    and every agent call on a docs-less box paid it before the user got
    any response (user-perceived latency)."""

    def _rag(self, tmp_path, docs=True):
        from rag.llama_index_rag import LlamaIndexRAG
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        if docs:
            (docs_dir / "note.txt").write_text(
                "The harbor lights mark the entrance.", encoding="utf-8")
        return LlamaIndexRAG(
            docs_dir=docs_dir, persist_dir=tmp_path / "chroma")

    def test_empty_docs_skips_embedding_config(self, tmp_path):
        from unittest.mock import patch
        rag = self._rag(tmp_path, docs=False)
        with patch.object(
                rag, "_configure_llama_settings") as mock_cfg:
            rag.build_index()
            mock_cfg.assert_not_called()
        assert rag._empty_index is True
        assert rag._query_engine is None

    def test_empty_retrieve_is_instant_and_empty(self, tmp_path):
        from unittest.mock import patch
        rag = self._rag(tmp_path, docs=False)
        with patch.object(rag, "_configure_llama_settings") as mock_cfg:
            out = rag.retrieve("What do the harbor lights mark?")
            mock_cfg.assert_not_called()
        assert out == {"answer": "", "chunks": [], "sources": [],
                       "retrieval_confidence": 0.0}

    def test_docs_present_reports_content_without_deps(self, tmp_path):
        rag = self._rag(tmp_path, docs=True)
        assert rag._has_retrievable_content() is True

    def test_empty_dir_reports_no_content(self, tmp_path):
        rag = self._rag(tmp_path, docs=False)
        assert rag._has_retrievable_content() is False
