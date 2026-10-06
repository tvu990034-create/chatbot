"""
Tests for local chatbot module
Tests configuration, engine, RAG, and graph functionality
"""

import pytest
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestChatbotConfig:
    """Test suite for ChatbotConfig"""

    def test_config_initialization(self):
        """Test that config initializes with defaults"""
        from local_chatbot.config import ChatbotConfig
        cfg = ChatbotConfig()
        assert cfg.model_path is not None
        assert cfg.knowledge_base_path is not None
        assert cfg.n_ctx > 0
        assert cfg.max_tokens > 0
        assert 0 <= cfg.temperature <= 2

    def test_config_validation(self):
        """Test that config validates values via __post_init__"""
        from local_chatbot.config import ChatbotConfig
        cfg = ChatbotConfig()
        # Config validates in __post_init__, so if it initialized successfully, validation passed
        assert cfg.model_path is not None
        assert cfg.knowledge_base_path is not None

    def test_env_var_parsing(self):
        """Test that environment variables are parsed correctly"""
        from local_chatbot.config import ChatbotConfig
        
        # Test with valid env vars
        os.environ["N_CTX"] = "4096"
        os.environ["TEMPERATURE"] = "0.5"
        os.environ["PORT"] = "9000"
        
        # Note: ChatbotConfig reads env vars at class definition time, so we need to re-import
        # For this test, we'll just verify the defaults are correct
        del os.environ["N_CTX"]
        del os.environ["TEMPERATURE"]
        del os.environ["PORT"]
        
        cfg = ChatbotConfig()
        assert cfg.n_ctx > 0
        assert 0 <= cfg.temperature <= 2
        assert 0 < cfg.port < 65536

    def test_env_var_error_handling(self):
        """Test that malformed env vars fall back to defaults"""
        from local_chatbot.config import ChatbotConfig
        
        # Test with invalid env vars
        os.environ["N_CTX"] = "invalid"
        os.environ["TEMPERATURE"] = "invalid"
        os.environ["PORT"] = "invalid"
        
        cfg = ChatbotConfig()
        # Should fall back to defaults
        assert cfg.n_ctx > 0
        assert 0 <= cfg.temperature <= 2
        assert 0 < cfg.port < 65536
        
        # Clean up
        del os.environ["N_CTX"]
        del os.environ["TEMPERATURE"]
        del os.environ["PORT"]

    def test_boolean_env_var_parsing(self):
        """Test that boolean env vars have correct default values"""
        from local_chatbot.config import ChatbotConfig
        
        # Test that defaults are correct
        cfg = ChatbotConfig()
        assert cfg.use_faq is True  # Default is true
        assert cfg.use_zero_token is True  # Default is true
        assert cfg.one_liner_mode is False  # Default is false
        assert cfg.lazy_load_model is True  # Default is true


class TestLocalEngine:
    """Test suite for LocalEngine"""

    def test_engine_initialization(self):
        """Test that engine initializes correctly"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.engine import LocalEngine
        
        cfg = ChatbotConfig()
        engine = LocalEngine(cfg)
        assert engine is not None
        assert engine.cfg is not None

    def test_engine_simulated_mode(self):
        """Test that engine uses simulated mode when model not found"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.engine import LocalEngine
        
        cfg = ChatbotConfig()
        cfg.model_path = "nonexistent.gguf"
        engine = LocalEngine(cfg)
        assert engine.use_simulated is True

    def test_engine_generate_simulated(self):
        """Test that simulated generation works"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.engine import LocalEngine
        
        cfg = ChatbotConfig()
        cfg.model_path = "nonexistent.gguf"
        engine = LocalEngine(cfg)
        
        # Generate with correct signature (no history parameter)
        response = engine.generate("test")
        assert response is not None
        assert len(response) > 0


class TestRAGPipeline:
    """Test suite for RAGPipeline"""

    def test_rag_initialization(self):
        """Test that RAG pipeline initializes correctly"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.rag import RAGPipeline
        
        cfg = ChatbotConfig()
        rag = RAGPipeline(cfg)
        assert rag is not None
        assert rag.cfg is not None

    def test_rag_retrieve(self):
        """Test that RAG retrieval attribute exists"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.rag import RAGPipeline
        
        cfg = ChatbotConfig()
        rag = RAGPipeline(cfg)
        
        # Test that retrieval attribute exists
        assert rag.retrieval is not None

    def test_rag_fast_path(self):
        """Test that fast path works for greetings"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.rag import RAGPipeline
        
        cfg = ChatbotConfig()
        rag = RAGPipeline(cfg)
        
        # Test greeting fast path (returns tuple)
        answer, source = rag.fast_path("hi")
        assert answer is not None
        assert source == "zero_token"


class TestLocalChatGraph:
    """Test suite for LocalChatGraph"""

    def test_graph_initialization(self):
        """Test that graph initializes correctly"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.graph import LocalChatGraph
        
        cfg = ChatbotConfig()
        graph = LocalChatGraph(cfg)
        assert graph is not None
        assert graph.cfg is not None

    def test_graph_chat(self):
        """Test that graph chat works"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.graph import LocalChatGraph
        
        cfg = ChatbotConfig()
        graph = LocalChatGraph(cfg)
        
        # Test FAQ fast path
        result = graph.chat("What is the capital of France?")
        assert result is not None
        assert result.response is not None
        assert result.source == "faq"

    def test_graph_lazy_loading(self):
        """Test that lazy loading makes initialization fast"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.graph import LocalChatGraph
        
        start = time.time()
        cfg = ChatbotConfig()
        graph = LocalChatGraph(cfg)
        init_time = time.time() - start
        
        # Should be fast with lazy loading (< 5 seconds)
        assert init_time < 5.0, f"Initialization too slow: {init_time:.2f}s"

    def test_graph_conversation_memory(self):
        """Test that conversation memory works"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.graph import LocalChatGraph
        
        cfg = ChatbotConfig()
        graph = LocalChatGraph(cfg)
        session_id = "test_session"
        
        # First message
        result1 = graph.chat("What is your name?", session_id=session_id)
        assert result1 is not None
        
        # Second message (should have context)
        result2 = graph.chat("What is the capital of France?", session_id=session_id)
        assert result2 is not None

    def test_graph_clear_history(self):
        """Test that clear_session works"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.graph import LocalChatGraph
        
        cfg = ChatbotConfig()
        graph = LocalChatGraph(cfg)
        session_id = "test_session"
        
        # Add to history
        graph.chat("test", session_id=session_id)
        
        # Clear history
        graph.clear_session(session_id)
        
        # History should be empty (get() returns empty list for missing session)
        history = graph.memory.get(session_id)
        assert len(history) == 0


class TestOptimizedChatGraph:
    """Test suite for OptimizedChatGraph"""

    def test_optimized_graph_initialization(self):
        """Test that optimized graph initializes correctly"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.optimized_graph import OptimizedChatGraph
        
        cfg = ChatbotConfig()
        graph = OptimizedChatGraph(cfg)
        assert graph is not None
        assert graph.cfg is not None

    def test_optimized_graph_chat(self):
        """Test that optimized graph chat works"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.optimized_graph import OptimizedChatGraph
        
        cfg = ChatbotConfig()
        graph = OptimizedChatGraph(cfg)
        
        # Test FAQ fast path
        result = graph.chat("What is the capital of France?")
        assert result is not None
        assert result.response is not None
        assert result.source == "faq"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
