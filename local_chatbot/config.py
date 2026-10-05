"""Configuration for the local chatbot."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ChatbotConfig:
    model_path: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_PATH", "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
        )
    )
    knowledge_base_path: str = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_BASE_PATH", "data/knowledge_base")
    )
    n_ctx: int = int(os.getenv("N_CTX", "2048"))
    n_threads: int = int(os.getenv("N_THREADS", "0")) or (os.cpu_count() or 4)
    n_gpu_layers: int = int(os.getenv("N_GPU_LAYERS", "-1"))
    n_batch: int = int(os.getenv("N_BATCH", "512"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.1"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))
    chunk_limit: int = int(os.getenv("CHUNK_LIMIT", "3"))
    use_faq: bool = os.getenv("USE_FAQ", "true").lower() in ("true", "1", "yes")
    use_zero_token: bool = os.getenv("USE_ZERO_TOKEN", "true").lower() in ("true", "1", "yes")
    one_liner_mode: bool = os.getenv("ONE_LINER_MODE", "false").lower() in ("true", "1", "yes")
    lazy_load_model: bool = os.getenv("LAZY_LOAD_MODEL", "true").lower() in ("true", "1", "yes")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    system_prompt: str = (
        "You are a helpful local AI assistant. Answer clearly and accurately "
        "using the provided context when available."
    )

    def __post_init__(self):
        """Validate configuration after initialization (UX Bug #36 fix)."""
        if not self.model_path:
            raise ValueError("MODEL_PATH cannot be empty")
        if not self.knowledge_base_path:
            raise ValueError("KNOWLEDGE_BASE_PATH cannot be empty")
        if self.n_ctx <= 0:
            raise ValueError("N_CTX must be positive")
        if self.max_tokens <= 0:
            raise ValueError("MAX_TOKENS must be positive")
        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("TEMPERATURE must be between 0 and 2")
        if self.chunk_limit <= 0:
            raise ValueError("CHUNK_LIMIT must be positive")
        if self.port <= 0 or self.port > 65535:
            raise ValueError("PORT must be between 1 and 65535")
