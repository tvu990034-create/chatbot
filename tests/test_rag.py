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
    from rag.llama_index_rag import LlamaIndexRAG

    docs = tmp_path / "docs"
    docs.mkdir()
    persist = tmp_path / "chroma"
    rag = LlamaIndexRAG(docs_dir=docs, persist_dir=persist)
    assert rag._empty_index is False
    rag.build_index()
    assert rag._empty_index is True
    with patch.object(LlamaIndexRAG, "_retrieve_nodes",
                      side_effect=AssertionError("must not retrieve")):
        out = rag.retrieve("anything at all?")
    assert out == {"answer": "", "chunks": [], "sources": [],
                   "retrieval_confidence": 0.0}


def test_llama_add_documents_clears_empty_flag(tmp_path):
    from rag.llama_index_rag import LlamaIndexRAG

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "note.txt").write_text(
        "The harbor lights mark the entrance. " * 20, encoding="utf-8")
    persist = tmp_path / "chroma"
    rag = LlamaIndexRAG(docs_dir=docs, persist_dir=persist)
    rag.build_index()
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
