"""
Test configuration management.
"""

import os
import pytest
from pathlib import Path


def test_config_imports():
    """Test that config module can be imported."""
    try:
        from config import settings
        assert settings is not None
    except ImportError as e:
        pytest.fail(f"Failed to import config: {e}")


def test_settings_defaults():
    """Test that settings have sensible defaults."""
    from config import settings
    
    assert settings.app_name == "Local Chatbot"
    assert settings.api_port == 8000
    assert settings.ui_port == 7860
    # The actual model may vary based on .env configuration, just check it's set
    assert settings.default_model is not None and len(settings.default_model) > 0
    assert settings.rag_provider.value in ["llama_index", "haystack", "both", "none"]


def test_performance_equations_toggle():
    """Test that performance equations can be disabled."""
    from config import settings
    
    # Should be disabled by default for safety
    assert settings.enable_performance_equations is False


def test_paths_exist():
    """Test that configured paths are valid."""
    from config import settings
    
    # Base directory should exist
    assert Path(__file__).parent.parent.exists()
    
    # Data directories should be configurable
    assert settings.rag_docs_dir is not None
    assert settings.rag_index_dir is not None
    assert settings.chroma_persist_dir is not None
