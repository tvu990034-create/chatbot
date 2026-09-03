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
