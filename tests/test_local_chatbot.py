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
        """Test that config validates values"""
        from local_chatbot.config import ChatbotConfig
        cfg = ChatbotConfig()
        # Test valid config
        assert cfg.validate() is True

    def test_env_var_parsing(self):
        """Test that environment variables are parsed correctly"""
        from local_chatbot.config import ChatbotConfig
        
        # Test with valid env vars
        os.environ["N_CTX"] = "4096"
        os.environ["TEMPERATURE"] = "0.5"
        os.environ["PORT"] = "9000"
        
        cfg = ChatbotConfig()
        assert cfg.n_ctx == 4096
        assert cfg.temperature == 0.5
        assert cfg.port == 9000
        
        # Clean up
        del os.environ["N_CTX"]
        del os.environ["TEMPERATURE"]
        del os.environ["PORT"]

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
        """Test that boolean env vars are parsed correctly"""
        from local_chatbot.config import ChatbotConfig
        
        # Test with various boolean values
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
        ]
        
        for value, expected in test_cases:
            os.environ["USE_FAQ"] = value
            cfg = ChatbotConfig()
            assert cfg.use_faq == expected, f"Failed for value: {value}"
        
        # Clean up
        del os.environ["USE_FAQ"]


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
        
        response = engine.generate("test", history=[])
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
        """Test that RAG retrieval works"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.rag import RAGPipeline
        
        cfg = ChatbotConfig()
        rag = RAGPipeline(cfg)
        
        # Test retrieval
        docs = rag.retrieve("What is BM25?")
        assert docs is not None
        assert isinstance(docs, list)

    def test_rag_fast_path(self):
        """Test that fast path works for greetings"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.rag import RAGPipeline
        
        cfg = ChatbotConfig()
        rag = RAGPipeline(cfg)
        
        # Test greeting fast path
        result = rag.fast_path("hi")
        assert result is not None
        assert result["source"] == "zero_token"


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
        """Test that clear_history works"""
        from local_chatbot.config import ChatbotConfig
        from local_chatbot.graph import LocalChatGraph
        
        cfg = ChatbotConfig()
        graph = LocalChatGraph(cfg)
        session_id = "test_session"
        
        # Add to history
        graph.chat("test", session_id=session_id)
        
        # Clear history
        graph.clear_history(session_id)
        
        # History should be empty
        assert len(graph.memory.get(session_id, [])) == 0


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
