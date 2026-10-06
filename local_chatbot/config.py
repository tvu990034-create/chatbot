"""Configuration for the local chatbot."""

import logging
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Magic Number Bug #46: Use named constants for default values
DEFAULT_MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
DEFAULT_KNOWLEDGE_BASE_PATH = "data/knowledge_base"
DEFAULT_N_CTX = 2048
DEFAULT_N_GPU_LAYERS = -1  # -1 means use all available GPU layers
DEFAULT_N_BATCH = 512
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 512
DEFAULT_CHUNK_LIMIT = 3
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def _env_int(name: str, default: int) -> int:
    """Parse integer environment variable with error handling (Environment Bug #50 fix)."""
    raw = os.getenv(name, str(default))
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Invalid integer value for {name}={raw}, using default {default}")
        return default


def _env_float(name: str, default: float) -> float:
    """Parse float environment variable with error handling (Environment Bug #50 fix)."""
    raw = os.getenv(name, str(default))
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Invalid float value for {name}={raw}, using default {default}")
        return default


@dataclass
class ChatbotConfig:
    model_path: str = field(
        default_factory=lambda: os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    )
    knowledge_base_path: str = field(
        default_factory=lambda: os.getenv("KNOWLEDGE_BASE_PATH", DEFAULT_KNOWLEDGE_BASE_PATH)
    )
    n_ctx: int = _env_int("N_CTX", DEFAULT_N_CTX)
    n_threads: int = _env_int("N_THREADS", 0) or (os.cpu_count() or 4)
    n_gpu_layers: int = _env_int("N_GPU_LAYERS", DEFAULT_N_GPU_LAYERS)
    n_batch: int = _env_int("N_BATCH", DEFAULT_N_BATCH)
    temperature: float = _env_float("TEMPERATURE", DEFAULT_TEMPERATURE)
    max_tokens: int = _env_int("MAX_TOKENS", DEFAULT_MAX_TOKENS)
    chunk_limit: int = _env_int("CHUNK_LIMIT", DEFAULT_CHUNK_LIMIT)
    use_faq: bool = os.getenv("USE_FAQ", "true").strip().lower() in ("true", "1", "yes")
    use_zero_token: bool = os.getenv("USE_ZERO_TOKEN", "true").strip().lower() in ("true", "1", "yes")
    one_liner_mode: bool = os.getenv("ONE_LINER_MODE", "false").strip().lower() in ("true", "1", "yes")
    lazy_load_model: bool = os.getenv("LAZY_LOAD_MODEL", "true").strip().lower() in ("true", "1", "yes")
    host: str = os.getenv("HOST", DEFAULT_HOST)
    port: int = _env_int("PORT", DEFAULT_PORT)
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
