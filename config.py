"""
config.py – Central configuration for local-chatbot.
Now includes all tuning knobs for all 9 performance optimizations.

All settings are read from environment variables (or a .env file).
Import the singleton `settings` everywhere else in the project:

    from config import settings
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BackendType(str, Enum):
    VLLM   = "vllm"
    SGLANG = "sglang"
    OLLAMA = "ollama"
    NONE   = "none"


class RAGProvider(str, Enum):
    LLAMA_INDEX = "llama_index"
    HAYSTACK    = "haystack"
    BOTH        = "both"
    NONE        = "none"


class EmbeddingProvider(str, Enum):
    HUGGINGFACE = "huggingface"
    OPENAI      = "openai"
    LITELLM     = "litellm"


class FinetuningMethod(str, Enum):
    LORA    = "lora"
    QLORA   = "qlora"
    FULL    = "full"
    ADA_LORA = "ada_lora"
    DORA    = "dora"


class QuantizationType(str, Enum):
    NF4 = "nf4"
    FP4 = "fp4"
    INT8 = "int8"
    INT4 = "int4"


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent


class Settings(BaseSettings):
    """All runtime configuration, sourced from .env or environment.
    BUG 14 FIX: Added field validators for malformed environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    @field_validator('api_port', 'ui_port', 'local_backend_port')
    @classmethod
    def validate_port(cls, v):
        """Validate port numbers are in valid range."""
        if not isinstance(v, int) or v < 1 or v > 65535:
            raise ValueError(f"Invalid port number: {v}. Must be between 1 and 65535.")
        return v
    
    @field_validator('litellm_temperature')
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature is in valid range."""
        if not isinstance(v, (int, float)) or v < 0 or v > 2:
            raise ValueError(f"Invalid temperature: {v}. Must be between 0 and 2.")
        return float(v)
    
    @field_validator('litellm_max_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        """Validate max_tokens is positive."""
        if not isinstance(v, int) or v < 1:
            raise ValueError(f"Invalid max_tokens: {v}. Must be a positive integer.")
        return v

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    app_name:    str = Field("Local Chatbot")
    app_version: str = Field("0.1.0")
    debug:       bool = Field(False)
    log_level:   Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")

    # ------------------------------------------------------------------
    # FastAPI server
    # ------------------------------------------------------------------
    api_host:    str  = Field("0.0.0.0")
    api_port:    int  = Field(8000)
    api_workers: int  = Field(1)
    api_reload:  bool = Field(False)

    # ------------------------------------------------------------------
    # Gradio UI
    # ------------------------------------------------------------------
    ui_host:  str  = Field("0.0.0.0")
    ui_port:  int  = Field(7860)
    ui_share: bool = Field(False)
    ui_theme: str  = Field("soft")
    
    # UI rendering optimization parameters
    ui_debounce_threshold: float = Field(
        0.050,
        description="Debounce threshold in seconds between UI updates (default: 50ms)"
    )
    ui_batch_size: int = Field(
        4,
        description="Minimum tokens before UI update (default: 4)"
    )
    ui_max_typing_cycles: int = Field(
        10,
        description="Maximum typing indicator animation cycles (default: 10)"
    )
    ui_agent_timeout: int = Field(
        30,
        description="Agent execution timeout in seconds (default: 30)"
    )

    # ------------------------------------------------------------------
    # LiteLLM gateway
    # ------------------------------------------------------------------
    # CENTRALIZED MODEL CONFIGURATION - Single source of truth
    default_model: str = Field(
        "ollama/tinyllama:latest",  # Smallest model for reliability
        description="LiteLLM model string used by default - CENTRALIZED MODEL CONFIG",
    )
    litellm_api_base:    str | None = Field(None)
    litellm_timeout:     int        = Field(15)  # Increased timeout for quality responses
    litellm_max_tokens:  int        = Field(1024)  # Increased for better quality answers
    litellm_temperature: float      = Field(0.3)  # Balanced determinism - CENTRALIZED TEMP
    litellm_stream:      bool       = Field(True)
    fallback_models: list[str] = Field(
        default_factory=lambda: ["ollama/phi3:mini", "ollama/gemma2:2b"],
        description="Fallback models when default model is unavailable",
    )

    # ------------------------------------------------------------------
    # Advanced Reasoning Configuration
    # ------------------------------------------------------------------
    enable_advanced_reasoning: bool = Field(
        True,
        description="Enable advanced reasoning techniques by default"
    )
    use_cutting_edge_techniques: bool = Field(
        True,
        description="Use cutting-edge techniques (geodesic flow, quantum reasoning, etc.)"
    )
    reasoning_max_steps: int = Field(
        10,
        description="Maximum number of reasoning steps"
    )
    reasoning_confidence_threshold: float = Field(
        0.95,
        description="Confidence threshold for early stopping"
    )
    enable_semantic_drift_detection: bool = Field(
        True,
        description="Enable semantic drift detection"
    )
    enable_causal_pruning: bool = Field(
        True,
        description="Enable causal intervention pruning"
    )
    enable_constitutional_alignment: bool = Field(
        True,
        description="Enable constitutional alignment filtering"
    )

    # ------------------------------------------------------------------
    # API keys
    # ------------------------------------------------------------------
    openai_api_key:      str | None = Field(None)
    anthropic_api_key:   str | None = Field(None)
    openrouter_api_key:  str | None = Field(None)
    cohere_api_key:      str | None = Field(None)

    # ------------------------------------------------------------------
    # Local backend (vLLM / SGLang / Ollama)
    # ------------------------------------------------------------------
    local_backend:                    BackendType = Field(BackendType.OLLAMA)
    local_model_name:                 str         = Field("tinyllama:latest")  # MATCHES DEFAULT_MODEL
    local_backend_host:               str         = Field("127.0.0.1")
    local_backend_port:               int         = Field(8081)
    local_backend_gpu_memory_utilization: float   = Field(0.90)
    local_backend_max_model_len:      int         = Field(8192)
    local_backend_dtype:              str         = Field("auto")
    local_backend_quantization:       str | None  = Field(None)  # Disabled for reliability
    
    # vLLM-specific optimizations (from research papers) - ready for migration
    vllm_enable_prefix_caching: bool = Field(
        True,
        description="Enable prefix caching (#160, #201) - reuses system prompts"
    )
    vllm_enable_chunked_prefill: bool = Field(
        True,
        description="Enable chunked prefill for long prompts"
    )
    vllm_enable_paged_attention: bool = Field(
        True,
        description="Enable PagedAttention (#64) - automatic KV cache management"
    )
    vllm_kv_cache_dtype: str = Field(
        "auto",
        description="KV cache data type: auto, fp8, fp16, bf16"
    )
    vllm_max_num_batched_tokens: int = Field(
        8192,
        description="Maximum number of batched tokens"
    )
    vllm_enforce_eager: bool = Field(
        False,
        description="Enforce eager execution for debugging"
    )

    @property
    def local_backend_base_url(self) -> str:
        return f"http://{self.local_backend_host}:{self.local_backend_port}/v1"

    # ------------------------------------------------------------------
    # Embedding model
    # ------------------------------------------------------------------
    embedding_provider:  EmbeddingProvider = Field(EmbeddingProvider.HUGGINGFACE)
    embedding_model:     str               = Field("BAAI/bge-small-en-v1.5")
    embedding_batch_size: int              = Field(32)
    embedding_device:    str               = Field("cpu")

    # ------------------------------------------------------------------
    # RAG
    # ------------------------------------------------------------------
    rag_provider:         RAGProvider = Field(RAGProvider.LLAMA_INDEX)
    rag_docs_dir:         Path        = Field(BASE_DIR / "data" / "docs")
    rag_index_dir:        Path        = Field(BASE_DIR / "data" / "indexes")
    rag_collection_name:  str         = Field("local_chatbot_docs")
    rag_chunk_size:       int         = Field(512)
    rag_chunk_overlap:    int         = Field(64)
    rag_top_k:            int         = Field(5)
    rag_similarity_threshold: float   = Field(0.3)
    chroma_host:          str         = Field("localhost")
    chroma_port:          int         = Field(8001)
    chroma_persist_dir:   Path        = Field(BASE_DIR / "data" / "chroma")

    # ------------------------------------------------------------------
    # Agent / LangGraph
    # ------------------------------------------------------------------
    agent_max_iterations:  int  = Field(10)
    agent_recursion_limit: int  = Field(25)
    agent_memory_window:   int  = Field(
        20,
        description=(
            "Eq1 – last N messages kept in working memory. "
            "Larger → more context; smaller → lower TTFB via KV-cache recycling. "
            "BUG 25 NOTE: Currently message-count based, not token-based. "
            "Long messages may exceed actual token budget."
        ),
    )
    # BUG 29 FIX: Tool loop budgets
    agent_max_tool_calls: int = Field(
        10,
        description="Maximum number of tool calls per agent turn (BUG 29 FIX)"
    )
    agent_max_tool_time: int = Field(
        60,
        description="Maximum time in seconds for tool execution per turn (BUG 29 FIX)"
    )
    agent_max_wall_time: int = Field(
        120,
        description="Maximum wall-clock time for entire agent execution (BUG 29 FIX)"
    )
    # BUG 58 FIX: Tool output limits
    agent_max_tool_output_chars: int = Field(
        5000,
        description="Maximum characters for tool output to prevent context explosion (BUG 58 FIX)"
    )
    agent_max_tool_output_tokens: int = Field(
        1000,
        description="Maximum tokens for tool output to prevent context explosion (BUG 58 FIX)"
    )
    agent_system_prompt: str = Field(
        "You are a helpful, accurate, and concise AI assistant. "
        "Use the provided tools to retrieve relevant information before answering. "
        "Always cite your sources when using retrieved context.",
    )

    # ------------------------------------------------------------------
    # Aider code tool
    # ------------------------------------------------------------------
    aider_repo_path:   Path = Field(Path.cwd())
    aider_auto_commit: bool = Field(False)
    aider_read_only:   bool = Field(True)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    cors_origins:        list[str] = Field(default_factory=lambda: ["*"])
    max_history_messages: int      = Field(50)
    request_timeout:     int       = Field(60)
    # BUG 76 FIX: Centralized timeout configuration
    # Connect timeout - for establishing connections
    connect_timeout: int = Field(
        5,
        description="Connection timeout in seconds for HTTP requests"
    )
    # Generation timeout - for LLM generation
    generation_timeout: int = Field(
        15,
        description="Timeout in seconds for LLM generation calls"
    )
    # Tool timeout - for tool execution
    tool_timeout: int = Field(
        10,
        description="Timeout in seconds for tool execution"
    )
    # Request deadline - overall request deadline
    request_deadline: int = Field(
        120,
        description="Overall request deadline in seconds"
    )
    slack_webhook_url:   str | None = Field(
        None, description="Slack webhook for Eq9 anomaly alerts"
    )
    enable_performance_equations: bool = Field(
        True,
        description="Enable advanced performance optimizations - ENABLED for maximum performance"
    )
    # BUG 33 FIX: Best-of-N resource budgeting
    bon_max_n: int = Field(
        3,
        description="Maximum N for Best-of-N sampling (BUG 33 FIX)"
    )
    bon_max_wall_time: int = Field(
        30,
        description="Maximum wall-clock time for Best-of-N in seconds (BUG 33 FIX)"
    )
    
    # Legitimate Research Paper Optimizations
    enable_legitimate_optimizations: bool = Field(
        True,
        description="Enable legitimate research paper optimizations"
    )
    
    # Prompt Compression (LLMLingua #31, Selective Context #32)
    prompt_compression_enabled: bool = Field(
        True,
        description="Enable prompt compression based on research papers"
    )
    prompt_compression_method: str = Field(
        "simple_token_pruning",
        description="Compression method: simple_token_pruning, llmlingua, selective_context"
    )
    prompt_compression_ratio: float = Field(
        0.5,
        description="Target compression ratio (0.0-1.0)"
    )
    
    # Difficulty-based Routing (CALM #45, LayerSkip #50)
    difficulty_routing_enabled: bool = Field(
        True,
        description="Enable difficulty-based request routing"
    )
    difficulty_threshold: float = Field(
        0.5,
        description="Difficulty threshold for routing (0.0-1.0)"
    )
    
    # Response Caching
    response_cache_enabled: bool = Field(
        True,
        description="Enable response caching for repeated queries"
    )
    response_cache_ttl: int = Field(
        3600,
        description="Time-to-live for cached responses (seconds)"
    )
    semantic_cache_enabled: bool = Field(
        True,
        description="Enable semantic (second-stage cosine) cache fallback beside exact-match"
    )
    semantic_cache_threshold: float = Field(
        0.70,
        description="Similarity threshold for semantic cache fallback (0.0-1.0)"
    )

    # ------------------------------------------------------------------
    # LoRA/QLoRA Fine-Tuning Configuration
    # ------------------------------------------------------------------
    
    # Fine-tuning method selection
    finetuning_method: str = Field(
        "lora",
        description="Fine-tuning method: lora, qlora, full, ada_lora, dora"
    )
    
    # LoRA parameters (original LoRA paper: https://arxiv.org/abs/2106.09685)
    lora_r: int = Field(
        16,
        description="LoRA rank (r) - dimension of low-rank matrices"
    )
    lora_alpha: int = Field(
        32,
        description="LoRA alpha (α) - scaling factor for LoRA weights"
    )
    lora_dropout: float = Field(
        0.05,
        description="Dropout probability for LoRA layers"
    )
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Target modules for LoRA adaptation"
    )
    lora_bias: str = Field(
        "none",
        description="Bias training strategy for LoRA: none, all, lora_only"
    )
    lora_task_type: str = Field(
        "CAUSAL_LM",
        description="Task type for PEFT: CAUSAL_LM, SEQ_2_SEQ_LM, etc."
    )
    
    # QLoRA parameters (QLoRA paper: https://arxiv.org/abs/2305.14314)
    qlora_bits: int = Field(
        4,
        description="Quantization bits for QLoRA: 4, 8"
    )
    qlora_quantization_type: str = Field(
        "nf4",
        description="Quantization type: nf4 (NormalFloat4), fp4"
    )
    qlora_double_quant: bool = Field(
        True,
        description="Double quantization for additional memory savings"
    )
    qlora_compute_dtype: str = Field(
        "bfloat16",
        description="Compute dtype: bfloat16, float16, float32"
    )
    qlora_use_gradient_checkpointing: bool = Field(
        True,
        description="Use gradient checkpointing for memory efficiency"
    )
    
    # AdaLoRA parameters (AdaLoRA paper: https://arxiv.org/abs/2303.10512)
    adalora_init_r: int = Field(
        16,
        description="Initial rank for AdaLoRA"
    )
    adalora_target_r: int = Field(
        8,
        description="Target rank for AdaLoRA"
    )
    adalora_tota_step: int = Field(
        1000,
        description="Total training steps for AdaLoRA rank scheduling"
    )
    adalora_delta_t: int = Field(
        10,
        description="Time interval for AdaLoRA rank updates"
    )
    
    # DoRA parameters (DoRA paper: https://arxiv.org/abs/2402.09353)
    dora_weight_decompose: bool = Field(
        False,
        description="Enable DoRA weight decomposition"
    )
    
    # Training hyperparameters
    finetuning_learning_rate: float = Field(
        2e-4,
        description="Learning rate for fine-tuning"
    )
    finetuning_batch_size: int = Field(
        4,
        description="Training batch size"
    )
    finetuning_gradient_accumulation_steps: int = Field(
        1,
        description="Gradient accumulation steps"
    )
    finetuning_num_epochs: int = Field(
        3,
        description="Number of training epochs"
    )
    finetuning_warmup_steps: int = Field(
        100,
        description="Warmup steps for learning rate scheduler"
    )
    finetuning_lr_scheduler_type: str = Field(
        "cosine",
        description="LR scheduler: cosine, linear, polynomial"
    )
    finetuning_weight_decay: float = Field(
        0.01,
        description="Weight decay for optimizer"
    )
    finetuning_max_grad_norm: float = Field(
        1.0,
        description="Maximum gradient norm for clipping"
    )
    
    # Fine-tuning data configuration
    finetuning_dataset_name: str = Field(
        "gsm8k",
        description="Dataset name for fine-tuning (HuggingFace dataset)"
    )
    finetuning_dataset_split: str = Field(
        "train",
        description="Dataset split to use for training"
    )
    finetuning_max_samples: int = Field(
        -1,
        description="Maximum samples to use (-1 for all)"
    )
    finetuning_max_seq_length: int = Field(
        512,
        description="Maximum sequence length for training"
    )
    finetuning_template: str = Field(
        "alpaca",
        description="Instruction template: alpaca, vicuna, chatml, etc."
    )
    
    # Fine-tuning paths
    finetuning_output_dir: Path = Field(
        BASE_DIR / "data" / "adapters",
        description="Output directory for fine-tuned adapters"
    )
    finetuning_cache_dir: Path = Field(
        BASE_DIR / "data" / "cache",
        description="Cache directory for datasets and models"
    )
    finetuning_logging_dir: Path = Field(
        BASE_DIR / "data" / "logs",
        description="Logging directory for training metrics"
    )
    
    # Multi-LoRA serving (LoRA-Hub: https://arxiv.org/abs/2307.13269)
    multi_lora_enabled: bool = Field(
        False,
        description="Enable multi-LoRA serving with dynamic adapter loading"
    )
    multi_lora_max_adapters: int = Field(
        8,
        description="Maximum number of LoRA adapters to load simultaneously"
    )
    multi_lora_adapter_dir: Path = Field(
        BASE_DIR / "data" / "adapters",
        description="Directory containing LoRA adapters"
    )
    
    # Fine-tuning evaluation
    finetuning_eval_split: str = Field(
        "test",
        description="Dataset split for evaluation"
    )
    finetuning_eval_steps: int = Field(
        500,
        description="Evaluation frequency in steps"
    )
    finetuning_save_steps: int = Field(
        500,
        description="Model checkpoint frequency in steps"
    )
    finetuning_save_total_limit: int = Field(
        3,
        description="Maximum number of checkpoints to keep"
    )

    # ------------------------------------------------------------------
    # Model Merging Configuration
    # ------------------------------------------------------------------
    
    # Merging method selection
    merging_method: str = Field(
        "ties",
        description="Default merging method: ties, dare, task_arithmetic, slerp, git_rebasin, fisher, regmean, model_stock"
    )
    
    # Merging output configuration
    merging_output_dir: Path = Field(
        BASE_DIR / "data" / "merged_models",
        description="Output directory for merged models"
    )
    
    # TIES-Merging parameters (https://arxiv.org/abs/2306.01708)
    ties_trim_threshold: float = Field(
        0.1,
        description="TIES trim threshold for removing small parameters"
    )
    ties_k_ratio: float = Field(
        0.1,
        description="TIES k ratio for top-k parameter selection"
    )
    
    # DARE parameters (https://arxiv.org/abs/2311.03099)
    dare_drop_rate: float = Field(
        0.2,
        description="DARE drop rate for random parameter dropping"
    )
    dare_rescale: bool = Field(
        True,
        description="DARE rescaling to preserve expectation"
    )
    
    # Task Arithmetic parameters (https://arxiv.org/abs/2212.04089)
    task_arithmetic_default_operation: str = Field(
        "add",
        description="Default operation for task arithmetic: add, subtract"
    )
    
    # SLERP parameters (https://arxiv.org/abs/2401.02905)
    slerp_interpolation: float = Field(
        0.5,
        description="SLERP interpolation parameter (0 to 1)"
    )
    
    # Git Re-Basin parameters (https://arxiv.org/abs/2209.04836)
    git_rebasin_max_iterations: int = Field(
        100,
        description="Git Re-Basin maximum iterations for permutation alignment"
    )
    
    # Fisher Merging parameters (https://arxiv.org/abs/2111.09832)
    fisher_damping: float = Field(
        0.01,
        description="Fisher merging damping parameter for numerical stability"
    )
    fisher_num_samples: int = Field(
        100,
        description="Number of samples for Fisher information estimation"
    )
    
    # RegMean parameters (https://arxiv.org/abs/2212.09849)
    regmean_regularization: float = Field(
        0.01,
        description="RegMean regularization parameter"
    )
    
    # Model Stock parameters (https://arxiv.org/abs/2403.19522)
    model_stock_max_iterations: int = Field(
        100,
        description="Model Stock maximum iterations for geometric median"
    )
    model_stock_tolerance: float = Field(
        1e-6,
        description="Model Stock convergence tolerance"
    )
    
    # General merging settings
    merging_device: str = Field(
        "auto",
        description="Device for model merging: auto, cuda, cpu"
    )
    merging_dtype: str = Field(
        "float16",
        description="Data type for model merging: float16, float32, bfloat16"
    )
    merging_save_tokenizer: bool = Field(
        True,
        description="Whether to save tokenizer with merged model"
    )

    # ------------------------------------------------------------------
    # A-Tier Advanced Optimizations
    # ------------------------------------------------------------------
    
    # Continuous Thought Vectors (#51)
    enable_continuous_thoughts: bool = Field(
        True,
        description="Enable continuous thought vectors for reasoning compression"
    )
    n_thoughts: int = Field(
        4,
        description="Number of latent thought vectors"
    )
    
    # Reasoning Depth Budget (#54)
    enable_reasoning_depth_budget: bool = Field(
        True,
        description="Enable reasoning depth budget penalty"
    )
    reasoning_lambda: float = Field(
        0.1,
        description="Lambda parameter for depth budget penalty"
    )
    
    # Variational Thought Compression (#55)
    enable_variational_compression: bool = Field(
        True,
        description="Enable variational thought compression"
    )
    variational_beta: float = Field(
        1.0,
        description="Beta parameter for VAE-style compression"
    )
    
    # Thought-Level Early Exiting (#57)
    enable_thought_early_exit: bool = Field(
        True,
        description="Enable thought-level early exiting"
    )
    exit_threshold: float = Field(
        0.9,
        description="Confidence threshold for early exit"
    )
    
    # Attention Span Prediction (#101)
    enable_attention_span_prediction: bool = Field(
        True,
        description="Enable attention span prediction"
    )
    
    # Query-Conditional Attention Sparsity (#102)
    enable_query_sparse_attention: bool = Field(
        True,
        description="Enable query-conditional attention sparsity"
    )
    
    # Linear Attention with Exponential Kernel (#106)
    enable_linear_attention: bool = Field(
        True,
        description="Enable linear attention with exponential kernel"
    )
    linear_attention_m: int = Field(
        128,
        description="Feature dimension for linear attention"
    )
    
    # Adaptive Vocabulary Shrinkage (#5)
    enable_adaptive_vocab: bool = Field(
        True,
        description="Enable adaptive vocabulary shrinkage"
    )
    vocab_shrink_k: int = Field(
        2000,
        description="K parameter for vocabulary shrinkage"
    )
    
    # Token-Grouped Decoding (#36)
    enable_grouped_decoding: bool = Field(
        True,
        description="Enable token-grouped decoding"
    )
    n_groups: int = Field(
        100,
        description="Number of vocabulary groups"
    )
    
    # Dynamic Token-Wise Top-K (#282)
    enable_dynamic_topk: bool = Field(
        True,
        description="Enable dynamic token-wise top-k"
    )
    max_dynamic_k: int = Field(
        100,
        description="Maximum dynamic k value"
    )
    
    # Token-Wise Precision Scheduling (#32)
    enable_token_precision_scheduling: bool = Field(
        True,
        description="Enable token-wise precision scheduling"
    )
    
    # Dynamic Weight Quantisation (#168)
    enable_dynamic_weight_quant: bool = Field(
        True,
        description="Enable dynamic weight quantisation"
    )
    
    # Dynamic Precision Router (#259)
    enable_precision_router: bool = Field(
        True,
        description="Enable dynamic precision router"
    )
    
    # SSM Reasoning (#72)
    enable_ssm_reasoning: bool = Field(
        True,
        description="Enable state-space model reasoning"
    )
    ssm_d_state: int = Field(
        256,
        description="State dimension for SSM"
    )
    
    # GLU Reasoning (#88)
    enable_glu_reasoning: bool = Field(
        True,
        description="Enable GLU-based latent reasoning"
    )
    
    # Feedback Draft (#206)
    enable_feedback_draft: bool = Field(
        True,
        description="Enable draft model with execution feedback"
    )

    # ------------------------------------------------------------------
    # S-Tier Advanced Optimizations
    # ------------------------------------------------------------------
    
    # Speculative Decoding
    enable_speculative_decoding: bool = Field(
        True,
        description="Enable full-sequence speculative decoding"
    )
    speculative_k: int = Field(
        5,
        description="Draft length for speculative decoding"
    )
    enable_hierarchical_speculative: bool = Field(
        True,
        description="Enable hierarchical speculative decoding"
    )
    enable_adaptive_draft_length: bool = Field(
        True,
        description="Enable adaptive draft length control"
    )
    pid_kp: float = Field(
        0.5,
        description="PID controller proportional gain"
    )
    pid_ki: float = Field(
        0.1,
        description="PID controller integral gain"
    )
    pid_kd: float = Field(
        0.05,
        description="PID controller derivative gain"
    )
    target_acceptance: float = Field(
        0.8,
        description="Target acceptance rate for speculative decoding"
    )
    
    # KV-Cache Optimizations
    enable_kv_free: bool = Field(
        True,
        description="Enable KV-cache-free generation"
    )
    kv_state_size: int = Field(
        128,
        description="State size for KV compression"
    )
    enable_hierarchical_cache: bool = Field(
        True,
        description="Enable hierarchical KV-cache"
    )
    cache_fine_window: int = Field(
        512,
        description="Fine window size for hierarchical cache"
    )
    cache_block_size: int = Field(
        16,
        description="Block size for coarse cache"
    )
    
    # Prompt Optimizations
    enable_prompt_minimisation: bool = Field(
        True,
        description="Enable prompt minimisation"
    )
    minimisation_threshold: float = Field(
        0.95,
        description="Similarity threshold for prompt minimisation"
    )
    enable_prefix_cache: bool = Field(
        True,
        description="Enable prefix KV-cache precomputation"
    )
    cache_dir: str = Field(
        "./cache",
        description="Directory for cache storage"
    )
    
    # Routing and Early Exit
    enable_early_exit: bool = Field(
        True,
        description="Enable complexity-aware early exit"
    )
    enable_cascade_routing: bool = Field(
        True,
        description="Enable confidence-based cascade routing"
    )
    cascade_threshold: float = Field(
        0.9,
        description="Confidence threshold for cascade routing"
    )
    enable_rl_routing: bool = Field(
        True,
        description="Enable routing with reinforcement learning"
    )
    
    # Syntax Optimizations
    enable_syntax_masking: bool = Field(
        True,
        description="Enable syntax-aware token masking"
    )
    enable_syntax_beam_search: bool = Field(
        True,
        description="Enable syntax-guided beam search"
    )
    beam_width: int = Field(
        5,
        description="Beam width for syntax-guided search"
    )
    
    # Attention Optimizations
    enable_distilled_attention: bool = Field(
        True,
        description="Enable attention map distillation"
    )
    attention_summary_size: int = Field(
        128,
        description="Summary size for distilled attention"
    )
    enable_cutoff_bias: bool = Field(
        True,
        description="Enable cutoff relative position bias"
    )
    max_cutoff_dist: int = Field(
        1024,
        description="Maximum distance for cutoff bias"
    )

    # ------------------------------------------------------------------
    # Advanced Token Optimizations (Deep Dives)
    # ------------------------------------------------------------------
    
    # Recursive Token Contraction (#30)
    enable_token_contraction: bool = Field(
        True,
        description="Enable recursive token contraction"
    )
    contraction_threshold: int = Field(
        2,
        description="Minimum phrase length for contraction"
    )
    
    # Context-Free Token Expansion (#31)
    enable_token_expansion: bool = Field(
        True,
        description="Enable context-free token expansion"
    )
    
    # Template-Aware Token Abbreviation (#22)
    enable_template_abbreviation: bool = Field(
        True,
        description="Enable template-aware token abbreviation"
    )
    template_threshold: float = Field(
        0.9,
        description="Threshold for template matching"
    )
    
    # Hieroglyphic Tokenisation (#6)
    enable_hieroglyph: bool = Field(
        True,
        description="Enable hieroglyphic tokenisation"
    )
    hieroglyph_threshold: int = Field(
        3,
        description="Frequency threshold for hieroglyph learning"
    )
    max_ngram: int = Field(
        4,
        description="Maximum n-gram length for hieroglyph"
    )

    # ------------------------------------------------------------------
    # B-Tier Optimizations
    # ------------------------------------------------------------------
    
    # Code-Aware Token Blending (#17)
    enable_code_blending: bool = Field(
        True,
        description="Enable code-aware token blending"
    )

    # ------------------------------------------------------------------
    # Phase 2 Smart Routing & Gating
    # ------------------------------------------------------------------
    
    # Difficulty & Complexity Estimation
    enable_difficulty_estimation: bool = Field(
        True,
        description="Enable difficulty estimation for queries"
    )
    
    # Routing
    enable_query_routing: bool = Field(
        True,
        description="Enable query routing based on characteristics"
    )
    
    # Confidence Estimation
    enable_confidence_estimation: bool = Field(
        True,
        description="Enable confidence estimation for responses"
    )
    
    # Self-Consistency
    enable_self_consistency: bool = Field(
        True,
        description="Enable self-consistency voting"
    )

    # ------------------------------------------------------------------
    # Optimized Configuration (Conflict-Resolved)
    # ------------------------------------------------------------------
    
    # Core Foundational (Phase 1)
    enable_reasoning_enhancements: bool = Field(
        True,
        description="Enable reasoning enhancements (Phase 1)"
    )
    
    enable_attention_optimizations: bool = Field(
        True,
        description="Enable attention optimizations (Phase 1)"
    )
    
    enable_vocabulary_optimizations: bool = Field(
        True,
        description="Enable vocabulary optimizations (Phase 1)"
    )
    
    enable_precision_optimizations: bool = Field(
        True,
        description="Enable precision optimizations (Phase 1)"
    )
    
    # Smart Routing (Optimized)
    enable_optimized_routing: bool = Field(
        True,
        description="Enable optimized routing with conflict resolution"
    )
    
    complexity_threshold: float = Field(
        0.5,
        description="Complexity threshold for routing decisions"
    )
    
    abstention_threshold: float = Field(
        0.3,
        description="Confidence threshold for abstention"
    )
    
    # Budget Management
    enable_budget_routing: bool = Field(
        True,
        description="Enable budget-aware routing"
    )
    
    max_budget: float = Field(
        10.0,
        description="Maximum budget for query processing"
    )
    
    # Adaptive Features
    enable_adaptive_temperature: bool = Field(
        True,
        description="Enable adaptive temperature when user not specified"
    )
    
    base_temp: float = Field(
        0.1,
        description="Base temperature for adaptive adjustment"
    )
    
    # Self-Consistency
    enable_self_consistency: bool = Field(
        True,
        description="Enable self-consistency voting"
    )
    
    # Similarity Routing
    enable_similarity_routing: bool = Field(
        True,
        description="Enable query similarity routing"
    )
    
    similarity_threshold: float = Field(
        0.7,
        description="Similarity threshold for query matching"
    )

    # ------------------------------------------------------------------
    # Factuality and Hallucination Reduction Configuration
    # ------------------------------------------------------------------
    
    # Factuality method selection
    factuality_method: str = Field(
        "contrastive",
        description="Default factuality method: contrastive, dola, cad, iti, selfcheck, factual_nucleus"
    )
    
    # Contrastive Decoding parameters (https://arxiv.org/abs/2210.15097)
    contrastive_amateur_temperature: float = Field(
        0.7,
        description="Temperature for amateur model in contrastive decoding"
    )
    contrastive_expert_temperature: float = Field(
        1.0,
        description="Temperature for expert model in contrastive decoding"
    )
    contrastive_coefficient: float = Field(
        0.5,
        description="Contrastive coefficient λ for logit subtraction"
    )
    
    # DoLa parameters (https://arxiv.org/abs/2309.03883)
    dola_early_layer: int = Field(
        5,
        description="Early layer index for DoLa decoding"
    )
    dola_late_layer: int = Field(
        -1,
        description="Late layer index for DoLa decoding (-1 for final layer)"
    )
    dola_contrasting_coefficient: float = Field(
        0.5,
        description="Contrasting coefficient α for DoLa"
    )
    
    # CAD parameters (https://arxiv.org/abs/2305.14703)
    cad_context_coefficient: float = Field(
        0.5,
        description="Context coefficient β for CAD decoding"
    )
    
    # ITI parameters (https://arxiv.org/abs/2306.03341)
    iti_intervention_strength: float = Field(
        0.5,
        description="Intervention strength for ITI"
    )
    iti_truth_direction_path: Optional[str] = Field(
        None,
        description="Path to pre-computed truth direction for ITI"
    )
    
    # SelfCheckGPT parameters (https://arxiv.org/abs/2303.08896)
    selfcheck_num_samples: int = Field(
        5,
        description="Number of samples for SelfCheckGPT consistency checking"
    )
    selfcheck_temperature: float = Field(
        0.7,
        description="Temperature for SelfCheckGPT sampling"
    )
    selfcheck_consistency_threshold: float = Field(
        0.5,
        description="Consistency threshold for hallucination detection"
    )
    
    # Factual-Nucleus parameters (https://arxiv.org/abs/2106.07447)
    factual_nucleus_base_top_p: float = Field(
        0.9,
        description="Base top-p for factual-nucleus sampling"
    )
    factual_nucleus_strength: float = Field(
        0.5,
        description="Factuality strength for entropy-based top-p adjustment"
    )
    factual_nucleus_entropy_threshold: float = Field(
        2.0,
        description="Entropy threshold for factual-nucleus sampling"
    )
    
    # General factuality settings
    factuality_enabled: bool = Field(
        False,
        description="Enable factuality improvements during generation"
    )
    factuality_confidence_threshold: float = Field(
        0.5,
        description="Confidence threshold for low-confidence detection"
    )
    factuality_uncertainty_threshold: float = Field(
        0.5,
        description="Uncertainty threshold for high-uncertainty detection"
    )
    factuality_log_hallucinations: bool = Field(
        True,
        description="Log detected hallucinations for analysis"
    )

    # ------------------------------------------------------------------
    # Knowledge Graph and Symbolic Reasoning Configuration
    # ------------------------------------------------------------------
    
    # Knowledge Graph settings
    kg_enabled: bool = Field(
        True,
        description="Enable knowledge graph features"
    )
    kg_storage_dir: Path = Field(
        BASE_DIR / "data" / "knowledge_graphs",
        description="Directory for storing knowledge graphs"
    )
    kg_format: str = Field(
        "networkx",
        description="Knowledge graph storage format: networkx, gexf, graphml, json"
    )
    
    # GraphRAG settings (https://github.com/microsoft/graphrag)
    graph_rag_enabled: bool = Field(
        False,
        description="Enable GraphRAG for graph-based retrieval"
    )
    graph_rag_max_hops: int = Field(
        2,
        description="Maximum hops for GraphRAG retrieval"
    )
    graph_rag_top_k: int = Field(
        10,
        description="Number of results to return from GraphRAG"
    )
    graph_rag_community_resolution: float = Field(
        1.0,
        description="Resolution parameter for community detection"
    )
    
    # KGQA settings (Knowledge Graph Question Answering)
    kgqa_enabled: bool = Field(
        False,
        description="Enable KGQA for knowledge graph question answering"
    )
    kgqa_max_entities: int = Field(
        5,
        description="Maximum entities to extract from questions"
    )
    
    # Symbolic Reasoning settings
    symbolic_reasoning_enabled: bool = Field(
        True,
        description="Enable symbolic reasoning with SymPy"
    )
    symbolic_max_complexity: int = Field(
        100,
        description="Maximum complexity for symbolic operations"
    )
    
    # Neuro-Symbolic settings
    neuro_symbolic_enabled: bool = Field(
        False,
        description="Enable neuro-symbolic reasoning"
    )
    neuro_symbolic_use_llm: bool = Field(
        True,
        description="Use LLM for natural language to symbolic conversion"
    )
    
    # Entity Extraction settings
    entity_extraction_enabled: bool = Field(
        True,
        description="Enable entity extraction for KG construction"
    )
    entity_types: list = Field(
        ["PERSON", "ORG", "LOC", "MISC"],
        description="Entity types to extract"
    )
    
    # Knowledge Graph construction
    kg_auto_build: bool = Field(
        False,
        description="Automatically build KG from documents"
    )
    kg_build_threshold: float = Field(
        0.5,
        description="Confidence threshold for adding relations to KG"
    )

    # ------------------------------------------------------------------
    # Prompt Compression Configuration
    # ------------------------------------------------------------------
    
    # Prompt compression settings
    prompt_compression_enabled: bool = Field(
        False,
        description="Enable prompt compression for long contexts"
    )
    prompt_compression_method: str = Field(
        "llmlingua",
        description="Compression method: llmlingua, selective_context, token_pruning, summarization, recursive"
    )
    
    # LLMLingua settings (https://arxiv.org/abs/2310.05736)
    llmlingua_target_ratio: float = Field(
        0.5,
        description="Target compression ratio for LLMLingua"
    )
    llmlingua_perplexity_threshold: float = Field(
        100.0,
        description="Perplexity threshold for LLMLingua"
    )
    
    # Selective Context settings (https://arxiv.org/abs/2304.12102)
    selective_context_target_ratio: float = Field(
        0.5,
        description="Target compression ratio for Selective Context"
    )
    selective_context_self_info_threshold: float = Field(
        2.0,
        description="Self-information threshold for Selective Context"
    )
    
    # Token Pruning settings (https://arxiv.org/abs/2309.11535)
    token_pruning_target_ratio: float = Field(
        0.5,
        description="Target compression ratio for token pruning"
    )
    token_pruning_attention_threshold: float = Field(
        0.1,
        description="Attention threshold for token pruning"
    )
    
    # Summarization settings (https://arxiv.org/abs/2310.04408)
    summarization_target_ratio: float = Field(
        0.5,
        description="Target compression ratio for summarization"
    )
    summarization_summary_ratio: float = Field(
        0.3,
        description="Summary ratio for summarization-based compression"
    )
    
    # Recursive Compression settings (https://arxiv.org/abs/2205.11346)
    recursive_target_ratio: float = Field(
        0.5,
        description="Target compression ratio for recursive compression"
    )
    recursive_chunk_size: int = Field(
        1000,
        description="Chunk size for recursive compression"
    )
    recursive_max_levels: int = Field(
        3,
        description="Maximum recursion levels for recursive compression"
    )
    
    # General compression settings
    compression_preserve_structure: bool = Field(
        True,
        description="Preserve sentence structure during compression"
    )
    compression_auto_apply: bool = Field(
        False,
        description="Automatically apply compression to long prompts"
    )
    compression_auto_threshold: int = Field(
        4096,
        description="Token count threshold for auto-compression"
    )

    # ------------------------------------------------------------------
    # Advanced Performance Optimizations
    # ------------------------------------------------------------------
    
    # FlashAttention
    flash_attention_enabled: bool = Field(
        False,
        description="Enable FlashAttention for 2-4x faster attention computation",
    )
    
    # Linear Attention (Performer)
    performer_attention_enabled: bool = Field(
        False,
        description="Enable Performer linear attention for O(L) complexity",
    )
    performer_nb_features: int = Field(
        64,
        description="Number of random features for Performer attention",
    )
    
    # Linear Attention (Linformer)
    linformer_attention_enabled: bool = Field(
        False,
        description="Enable Linformer low-rank attention",
    )
    linformer_low_rank_dim: int = Field(
        256,
        description="Low-rank dimension for Linformer attention",
    )
    
    # Speculative Decoding
    speculative_decoding_enabled: bool = Field(
        False,
        description="Enable speculative decoding for 2-3x faster generation",
    )
    speculative_spec_len: int = Field(
        5,
        description="Number of tokens to draft in speculative decoding",
    )
    speculative_verify_every: int = Field(
        1,
        description="Verification frequency for speculative decoding",
    )
    speculative_method: str = Field(
        "standard",
        description="Speculative decoding method: standard, medusa, eagle, self_speculative, dynamic",
    )
    speculative_num_branches: int = Field(
        4,
        description="Number of branches for tree-based speculative decoding",
    )
    speculative_num_heads: int = Field(
        4,
        description="Number of heads for Medusa speculative decoding",
    )
    speculative_target_acceptance: float = Field(
        0.8,
        description="Target acceptance rate for dynamic speculative decoding",
    )
    
    # KV Cache Compression
    kv_cache_compression_enabled: bool = Field(
        True,
        description="Enable advanced KV cache compression",
    )
    kv_eviction_policy: str = Field(
        "h2o",
        description="KV cache eviction policy: h2o, streaming_llm, scissorhands, etc.",
    )
    kv_compression_ratio: float = Field(
        0.5,
        description="Target KV cache compression ratio",
    )
    kv_compression_method: str = Field(
        "h2o",
        description="KV cache compression method",
    )
    kv_quantization_method: str = Field(
        "kivi",
        description="KV cache quantization method: kivi, gear, zipcache, etc.",
    )
    kv_quantization_bits: int = Field(
        4,
        description="KV cache quantization bits",
    )
    kv_attention_sink_threshold: float = Field(
        0.1,
        description="Attention sink threshold for KV cache eviction",
    )
    
    # KV Cache Optimization
    kv_cache_size: int = Field(
        1000,
        description="Maximum number of KV cache entries",
    )
    kv_page_size: int = Field(
        16,
        description="Page size for paged KV cache",
    )
    kv_max_blocks: int = Field(
        100,
        description="Maximum number of cache blocks",
    )
    
    # Continuous Batching
    continuous_batch_size: int = Field(
        32,
        description="Maximum batch size for continuous batching",
    )
    max_sequence_length: int = Field(
        2048,
        description="Maximum sequence length for batching",
    )
    
    # Advanced Batching Configuration
    enable_continuous_batching: bool = Field(
        False,
        description="Enable Orca-style continuous batching with iteration-level scheduling",
    )
    enable_dynamic_batching: bool = Field(
        False,
        description="Enable dynamic batching with queue management and timeout",
    )
    enable_phase_splitting: bool = Field(
        False,
        description="Enable prefill/decode phase splitting (Splitwise, SARATHI-style)",
    )
    enable_cache_aware_batching: bool = Field(
        False,
        description="Enable cache-aware batch scheduling with prefix reuse",
    )
    enable_slo_aware_batching: bool = Field(
        False,
        description="Enable SLO-aware scheduling with latency constraints",
    )
    enable_throughput_optimization: bool = Field(
        False,
        description="Enable throughput/latency modeling and optimization",
    )
    
    # Batching Parameters
    max_prefill_batch_size: int = Field(
        8,
        description="Maximum batch size for prefill phase",
    )
    max_decode_batch_size: int = Field(
        32,
        description="Maximum batch size for decode phase",
    )
    batch_timeout_ms: float = Field(
        10.0,
        description="Timeout for dynamic batch formation (milliseconds)",
    )
    min_batch_size: int = Field(
        1,
        description="Minimum batch size for dynamic batching",
    )
    scheduling_policy: str = Field(
        "fifo",
        description="Scheduling policy: fifo, sjf, edf, mlfq, slo_aware, cache_aware",
    )
    prefill_decode_ratio: float = Field(
        0.3,
        description="Ratio of compute resources for prefill vs decode (0.0-1.0)",
    )
    cache_hit_bonus: float = Field(
        2.0,
        description="Priority multiplier for cache hits in cache-aware scheduling",
    )
    slo_latency_p50: float = Field(
        100.0,
        description="50th percentile latency target (milliseconds)",
    )
    slo_latency_p99: float = Field(
        500.0,
        description="99th percentile latency target (milliseconds)",
    )
    target_throughput: float = Field(
        100.0,
        description="Target throughput (tokens per second)",
    )
    
    # Grammar Constraints Configuration
    enable_regex_constraints: bool = Field(
        False,
        description="Enable regex/FSM-based constrained generation",
    )
    regex_pattern: str = Field(
        "",
        description="Regular expression pattern for constrained generation",
    )
    enable_json_constraints: bool = Field(
        False,
        description="Enable JSON schema-based constrained generation",
    )
    enable_grammar_constraints: bool = Field(
        False,
        description="Enable GBNF grammar-based constrained generation",
    )
    gbnf_grammar: str = Field(
        "",
        description="GBNF grammar string for constrained generation",
    )
    enable_stop_token_control: bool = Field(
        False,
        description="Enable stop token control for generation termination",
    )
    stop_tokens: list[str] = Field(
        default_factory=lambda: ["</s>", "<|end_of_text|>"],
        description="List of stop token strings",
    )
    stop_strings: list[str] = Field(
        default_factory=list,
        description="List of stop strings to detect in generated text",
    )
    enable_thinking_control: bool = Field(
        False,
        description="Enable thinking tag control for reasoning models (DeepSeek-style)",
    )
    suppress_thinking: bool = Field(
        False,
        description="Suppress thinking tags in output",
    )
    thinking_start_tag: str = Field(
        "<think>",
        description="Start tag for thinking sections",
    )
    thinking_end_tag: str = Field(
        "</think>",
        description="End tag for thinking sections",
    )
    enable_token_masking: bool = Field(
        False,
        description="Enable token masking for constrained generation",
    )
    enable_logit_biasing: bool = Field(
        False,
        description="Enable logit biasing for constrained generation",
    )
    mask_strategy: str = Field(
        "hard",
        description="Token masking strategy: hard or soft",
    )
    enable_lexical_constraints: bool = Field(
        False,
        description="Enable lexical constraints (forced/forbidden phrases)",
    )
    forced_phrases: list[str] = Field(
        default_factory=list,
        description="Phrases that must appear in output",
    )
    forbidden_phrases: list[str] = Field(
        default_factory=list,
        description="Phrases that must not appear in output",
    )
    
    # Threading and NUMA Optimizations
    enable_speedup_modeling: bool = Field(
        False,
        description="Enable parallel speedup modeling (Amdahl, Gustafson)",
    )
    parallel_fraction: float = Field(
        0.95,
        description="Parallel fraction for speedup modeling (0-1)",
    )
    enable_numa_aware: bool = Field(
        False,
        description="Enable NUMA-aware memory allocation",
    )
    numa_policy: str = Field(
        "interleave",
        description="NUMA memory policy: local, interleave, preferred",
    )
    numa_balancing: bool = Field(
        False,
        description="Enable automatic NUMA balancing",
    )
    thread_count: int = Field(
        0,
        description="Thread count (0 = auto-detect)",
    )
    enable_hyperthreading: bool = Field(
        True,
        description="Enable hyper-threading for threading",
    )
    max_threads_per_core: int = Field(
        2,
        description="Maximum threads per core",
    )
    enable_cpu_affinity: bool = Field(
        False,
        description="Enable CPU affinity and thread pinning",
    )
    affinity_policy: str = Field(
        "compact",
        description="CPU affinity policy: compact, scatter, balanced",
    )
    enable_bandwidth_optimization: bool = Field(
        False,
        description="Enable memory bandwidth optimization",
    )
    target_bandwidth_utilization: float = Field(
        0.8,
        description="Target memory bandwidth utilization (0-1)",
    )
    enable_numa_scheduling: bool = Field(
        False,
        description="Enable NUMA-aware task scheduling",
    )
    scheduling_policy: str = Field(
        "work_stealing",
        description="Task scheduling policy: work_stealing, static, dynamic",
    )
    
    # Prompt Engineering and Context Management
    enable_prompt_compression: bool = Field(
        False,
        description="Enable LLMLingua-style prompt compression",
    )
    compression_method: str = Field(
        "perplexity_based",
        description="Compression method: perplexity_based, entropy_based, attention_based",
    )
    compression_ratio: float = Field(
        0.5,
        description="Target compression ratio (0-1)",
    )
    min_prompt_length: int = Field(
        100,
        description="Minimum prompt length after compression",
    )
    enable_token_importance: bool = Field(
        False,
        description="Enable token importance scoring for truncation",
    )
    importance_method: str = Field(
        "h2o",
        description="Importance scoring method: h2o, streaming_llm, snapkv",
    )
    importance_threshold: float = Field(
        0.1,
        description="Token importance threshold (0-1)",
    )
    top_k_tokens: int = Field(
        100,
        description="Number of top-k tokens to keep",
    )
    enable_history_truncation: bool = Field(
        False,
        description="Enable chat history truncation",
    )
    truncation_strategy: str = Field(
        "hybrid",
        description="Truncation strategy: recency, importance, hybrid, rag, summary",
    )
    max_history_length: int = Field(
        4096,
        description="Maximum history length in tokens",
    )
    history_window_size: int = Field(
        10,
        description="History window size for sliding window",
    )
    enable_dynamic_context: bool = Field(
        False,
        description="Enable dynamic context window management",
    )
    context_window_size: int = Field(
        8192,
        description="Maximum context window size",
    )
    token_budget: int = Field(
        4096,
        description="Token budget for history",
    )
    rag_token_budget: int = Field(
        2000,
        description="Token budget for RAG-retrieved context",
    )
    sliding_window_overlap: int = Field(
        512,
        description="Sliding window overlap in tokens",
    )
    enable_sampling_optimization: bool = Field(
        False,
        description="Enable sampling parameters optimization",
    )
    # REMOVED: General temperature field - use litellm_temperature for consistency
    # temperature: float = Field(0.7, description="Sampling temperature (0-2)")
    top_k: int = Field(
        50,
        description="Top-k sampling parameter",
    )
    top_p: float = Field(
        0.9,
        description="Top-p (nucleus) sampling parameter (0-1)",
    )
    repetition_penalty: float = Field(
        1.0,
        description="Repetition penalty factor (1.0 = no penalty)",
    )
    frequency_penalty: float = Field(
        0.0,
        description="Frequency penalty (0-2)",
    )
    presence_penalty: float = Field(
        0.0,
        description="Presence penalty (0-2)",
    )
    enable_auto_optimization: bool = Field(
        False,
        description="Enable automatic prompt optimization",
    )
    optimization_method: str = Field(
        "gradient_based",
        description="Optimization method: gradient_based, prefix_tuning",
    )
    optimization_iterations: int = Field(
        10,
        description="Number of optimization iterations",
    )
    enable_entropy_truncation: bool = Field(
        False,
        description="Enable entropy-based truncation",
    )
    entropy_threshold: float = Field(
        2.0,
        description="Entropy threshold for truncation",
    )
    mutual_information_threshold: float = Field(
        0.5,
        description="Mutual information threshold for truncation",
    )
    
    # Kernel Fusion
    kernel_fusion_enabled: bool = Field(
        False,
        description="Enable torch.compile for kernel fusion",
    )
    torch_compile_mode: str = Field(
        "reduce-overhead",
        description="torch.compile mode (reduce-overhead, max-autotune, etc)",
    )
    
    # Quantization
    enable_awq: bool = Field(
        False,
        description="Enable AWQ quantization (3x+ speedup)",
    )
    awq_bits: int = Field(
        4,
        description="AWQ quantization bits",
    )
    awq_group_size: int = Field(
        128,
        description="AWQ group size for quantization",
    )
    awq_clip_ratio: float = Field(
        0.01,
        description="AWQ outlier clipping ratio",
    )
    
    enable_gptq: bool = Field(
        False,
        description="Enable GPTQ quantization (3.25-4.5x speedup)",
    )
    gptq_bits: int = Field(
        4,
        description="GPTQ quantization bits",
    )
    gptq_group_size: int = Field(
        128,
        description="GPTQ group size for quantization",
    )
    gptq_damp_percent: float = Field(
        0.01,
        description="GPTQ damping factor for Hessian computation",
    )
    
    enable_llm_int8: bool = Field(
        False,
        description="Enable LLM.int8() quantization (2x memory reduction)",
    )
    llm_int8_threshold: float = Field(
        6.0,
        description="LLM.int8() outlier threshold",
    )
    
    enable_spqr: bool = Field(
        False,
        description="Enable SpQR quantization",
    )
    spqr_bits: int = Field(
        4,
        description="SpQR quantization bits",
    )
    spqr_sparsity: float = Field(
        0.1,
        description="SpQR sparsity ratio",
    )
    
    quantization_method: str = Field(
        "awq",
        description="Active quantization method (awq, gptq, llm_int8, smoothquant, spqr, zeroquant, zeroquant_fp, qlora, omniquant, bitnet, llm_fp8)",
    )
    
    # Model Conversion
    model_conversion_format: str = Field(
        "gguf",
        description="Default model conversion format (gguf, tensorrt, mlc, onnx)",
    )
    model_conversion_target: str = Field(
        "auto",
        description="Target hardware for model conversion (auto, nvidia-gpu, amd-gpu, cpu, apple-silicon)",
    )

    # ------------------------------------------------------------------
    # Advanced ML Optimizations (SAFE IMPLEMENTATIONS)
    # These apply optimization CONCEPTS to orchestration, not LLM architecture
    # All disabled by default - enable only if you understand the implications
    # ------------------------------------------------------------------
    
    # HRR-inspired history compression
    hrr_compression_enabled: bool = Field(
        False,
        description="Enable HRR-inspired conversation history compression",
    )
    hrr_history_size: int = Field(
        10,
        description="Number of recent messages to keep in HRR buffer",
    )
    hrr_compression_dim: int = Field(
        128,
        description="Dimension of compressed context representation",
    )
    
    # MoSE-inspired request segmentation
    mose_segmentation_enabled: bool = Field(
        False,
        description="Enable MoSE-inspired request segmentation for parallel processing",
    )
    mose_segment_length: int = Field(
        2000,
        description="Maximum characters per request segment",
    )
    mose_num_segments: int = Field(
        4,
        description="Number of segments to split requests into",
    )
    
    # Koopman-inspired context mixing
    koopman_mixing_enabled: bool = Field(
        False,
        description="Enable Koopman-inspired global context mixing for RAG results",
    )
    koopman_mix_dim: int = Field(
        64,
        description="Dimension for Koopman mixing projection",
    )
    
    # Neural ODE-inspired adaptive iterations
    ode_adaptive_iterations_enabled: bool = Field(
        False,
        description="Enable Neural ODE-inspired adaptive iteration for agent loops",
    )
    ode_max_iterations: int = Field(
        10,
        description="Maximum iterations for ODE-inspired agent loop",
    )
    ode_tolerance: float = Field(
        0.1,
        description="Convergence tolerance for adaptive iteration (0.0-1.0)",
    )

    # ==================================================================
    # REMOVED: Old Eq1-Eq9 and Eq19 (non-functional fake optimizations)
    # These have been replaced with legitimate research paper optimizations
    # See: ENABLE_LEGITIMATE_OPTIMIZATIONS, PROMPT_COMPRESSION_ENABLED, etc.
    # ==================================================================

    # ==================================================================
    # ADVANCED RESEARCH QUANTIZATION (new techniques from recent papers)
    # These are cutting-edge methods with proven research backing
    # ==================================================================

    # SmoothQuant - Post-training quantization
    enable_smoothquant: bool = Field(
        False,
        description="SmoothQuant: Post-training quantization for better accuracy",
    )
    smoothquant_alpha: float = Field(
        0.5,
        description="SmoothQuant: Channel-wise scaling factor (0.0-1.0)",
    )

    # ZeroQuant - Layer-by-layer quantization
    enable_zeroquant: bool = Field(
        False,
        description="ZeroQuant: Layer-by-layer quantization with INT8/FP16 support",
    )
    zeroquant_bits: int = Field(
        8,
        description="ZeroQuant: Quantization bits (4, 8, or 16)",
    )
    zeroquant_method: str = Field(
        "int8",
        description="ZeroQuant: Quantization method (int8, fp16, mixed)",
    )

    # ZeroQuant-FP - FP8 and INT4 quantization
    enable_zeroquant_fp: bool = Field(
        False,
        description="ZeroQuant-FP: FP8 and INT4 quantization with advanced scaling",
    )
    zeroquant_fp_bits: int = Field(
        4,
        description="ZeroQuant-FP: Quantization bits (4 or 8)",
    )
    zeroquant_fp_format: str = Field(
        "int4",
        description="ZeroQuant-FP: Quantization format (int4, fp8)",
    )

    # QLoRA - NF4 data type and double quantization
    enable_qlora: bool = Field(
        False,
        description="QLoRA: NF4 data type and double quantization for efficient finetuning",
    )
    qlora_bits: int = Field(
        4,
        description="QLoRA: Quantization bits (typically 4)",
    )
    qlora_double_quant: bool = Field(
        True,
        description="QLoRA: Enable double quantization of scaling factors",
    )
    qlora_nf4: bool = Field(
        True,
        description="QLoRA: Use NF4 data type instead of standard INT4",
    )

    # OmniQuant - Learnable weight clipping and equivalent transformations
    enable_omniquant: bool = Field(
        False,
        description="OmniQuant: Learnable weight clipping and equivalent transformations",
    )
    omniquant_bits: int = Field(
        4,
        description="OmniQuant: Quantization bits",
    )
    omniquant_learnable_clip: bool = Field(
        True,
        description="OmniQuant: Enable learnable clipping thresholds",
    )
    omniquant_let_transform: bool = Field(
        True,
        description="OmniQuant: Enable equivalent transformation (LET)",
    )

    # BitNet b1.58 - Ternary quantization
    enable_bitnet: bool = Field(
        False,
        description="BitNet b1.58: Ternary quantization for extreme compression",
    )
    bitnet_ternary: bool = Field(
        True,
        description="BitNet: Use ternary quantization {-1, 0, +1}",
    )
    bitnet_learnable_scale: bool = Field(
        False,
        description="BitNet: Learn scaling factor instead of using mean",
    )

    # LLM-FP8 - FP8 quantization with scaling factors
    enable_llm_fp8: bool = Field(
        False,
        description="LLM-FP8: FP8 quantization with per-token/per-channel scaling",
    )
    llm_fp8_per_token: bool = Field(
        True,
        description="LLM-FP8: Enable per-token scaling for activations",
    )
    llm_fp8_per_channel: bool = Field(
        True,
        description="LLM-FP8: Enable per-channel scaling for weights",
    )

    # ==================================================================
    # ADVANCED PROMPT CACHING & KV CACHE MANAGEMENT (from research papers)
    # These optimize memory usage and accelerate inference through intelligent caching
    # ==================================================================

    # Prefix Caching (PagedAttention-style)
    enable_prefix_caching: bool = Field(
        False,
        description="Prefix Caching: Cache common prompt prefixes to avoid recomputation",
    )
    prefix_cache_size: int = Field(
        1000,
        description="Prefix Caching: Maximum number of prefixes to cache",
    )
    prefix_hash_algorithm: str = Field(
        "sha256",
        description="Prefix Caching: Hash algorithm for cache keys (sha256, md5)",
    )

    # KV Cache Compression & Eviction
    enable_kv_cache_compression: bool = Field(
        False,
        description="KV Cache Compression: Enable intelligent KV cache eviction",
    )
    kv_cache_policy: str = Field(
        "h2o",
        description="KV Cache: Eviction policy (h2o, streaming, snapkv, lru, lfu)",
    )
    kv_cache_size: int = Field(
        4096,
        description="KV Cache: Maximum number of tokens to keep in cache",
    )

    # KV Cache Quantization
    enable_kv_quantization: bool = Field(
        False,
        description="KV Cache Quantization: Compress KV cache with quantization",
    )
    kv_quantization_bits: int = Field(
        8,
        description="KV Cache: Quantization bits (2, 4, or 8)",
    )
    kv_quantization_method: str = Field(
        "kivi",
        description="KV Cache: Quantization method (kivi, gear)",
    )

    # Attention Sinks & Sliding Window (StreamingLLM)
    enable_attention_sinks: bool = Field(
        False,
        description="Attention Sinks: Enable StreamingLLM-style attention sinks",
    )
    attention_sink_tokens: int = Field(
        4,
        description="Attention Sinks: Number of initial tokens to keep as sinks",
    )
    sliding_window_size: int = Field(
        1024,
        description="Attention Sinks: Sliding window size for recent tokens",
    )

    # Prompt Compression (LLMLingua-inspired)
    enable_prompt_compression: bool = Field(
        False,
        description="Prompt Compression: Compress prompts while preserving important information",
    )
    compression_ratio: float = Field(
        0.5,
        description="Prompt Compression: Target compression ratio (0.0-1.0)",
    )
    compression_method: str = Field(
        "importance",
        description="Prompt Compression: Compression method (importance, perplexity, random)",
    )

    # Session-based KV Cache Reuse
    enable_session_reuse: bool = Field(
        False,
        description="Session Reuse: Enable cross-session KV cache reuse",
    )
    session_similarity_threshold: float = Field(
        0.8,
        description="Session Reuse: Similarity threshold for cache reuse (0.0-1.0)",
    )
    max_sessions: int = Field(
        100,
        description="Session Reuse: Maximum number of sessions to cache",
    )

    # ==================================================================
    # HIGH-IMPACT MODEL-LEVEL OPTIMIZATIONS (from research papers)
    # These provide massive speedups but require model architecture changes
    # ==================================================================

    # HRR Attention - 10x speedup, 1000x memory reduction
    hrr_attention_enabled: bool = Field(
        False,
        description="HRR Attention: 10x speedup, 1000x memory reduction via circular convolution",
    )
    hrr_attention_heads: int = Field(
        8,
        description="Number of attention heads for HRR",
    )

    # Tensor Ring Linear - 60x parameter compression, 5x speedup
    tensor_ring_enabled: bool = Field(
        False,
        description="Tensor Ring decomposition: 60x parameter compression, 5x speedup",
    )
    tensor_ring_rank: int = Field(
        4,
        description="Tensor ring rank for decomposition",
    )

    # Performer Linear Attention - 9x speedup
    performer_attention_enabled: bool = Field(
        False,
        description="Performer linear attention: 9x speedup via random features",
    )
    performer_nb_features: int = Field(
        64,
        description="Number of random features for Performer",
    )

    # Neural ODE-Driven Depth - 2.5x speedup
    neural_ode_enabled: bool = Field(
        False,
        description="Neural ODE adaptive depth: 2.5x speedup via adaptive step size",
    )
    neural_ode_max_steps: int = Field(
        6,
        description="Maximum ODE steps",
    )
    neural_ode_tolerance: float = Field(
        1.0,
        description="Convergence tolerance for ODE",
    )

    # FlashAttention - 2-4x faster attention
    flash_attention_enabled: bool = Field(
        False,
        description="FlashAttention: 2-4x faster attention via online softmax tiling",
    )

    # Speculative Decoding - 2-3x speedup
    speculative_decoding_enabled: bool = Field(
        False,
        description="Speculative decoding: 2-3x faster generation via draft model",
    )
    speculative_draft_model: str = Field(
        "ollama/phi3:mini",
        description="Draft model for speculative decoding",
    )
    speculative_spec_len: int = Field(
        5,
        description="Number of tokens to speculate",
    )

    # ==================================================================
    # REMOVED: Old Eq20-Eq32 (overlapping with vLLM built-in features)
    # These are handled by vLLM automatically when using vLLM backend
    # ==================================================================

    # ==================================================================
    # REMOVED: Old Eq26-Eq30 (production optimizations)
    # These are handled by the new legitimate optimization system
    # See: ENABLE_LEGITIMATE_OPTIMIZATIONS, RESPONSE_CACHE_ENABLED, etc.
    # ==================================================================

    # ==================================================================
    # chatbot-phase1 Excellent Optimizations (OPTIONAL - all disabled by default)
    # These are proven techniques from the old project that worked well.
    # The chatbot works perfectly without them. Enable based on your needs.
    # ==================================================================

    # Eq31: SimHash Cache (chatbot-phase1 pattern)
    simhash_cache_enabled: bool = Field(
        False,
        description="Eq31: Enable SimHash-based fuzzy cache from chatbot-phase1.",
    )
    simhash_max_entries: int = Field(
        5000,
        description="Eq31: Maximum entries in SimHash cache.",
    )
    simhash_initial_threshold: int = Field(
        4,
        description="Eq31: Initial Hamming distance threshold for similarity.",
    )
    simhash_target_hit_rate: float = Field(
        0.9,
        description="Eq31: Target cache hit rate for auto-tuning threshold.",
    )

    # Eq32: Speculative Retrieval (chatbot-phase1 pattern)
    speculative_retrieval_enabled: bool = Field(
        False,
        description="Eq32: Enable speculative retrieval from chatbot-phase1.",
    )
    speculative_max_branches: int = Field(
        3,
        description="Eq32: Maximum number of retrieval branches to execute.",
    )
    speculative_timeout_ms: int = Field(
        200,
        description="Eq32: Timeout for each retrieval branch in milliseconds.",
    )

    # ==================================================================
    # Adaptive Timeout Settings
    # ==================================================================
    adaptive_t_min: float = Field(
        0.1,
        description="Minimum adaptive timeout in seconds.",
    )
    adaptive_t_max: float = Field(
        1.1,
        description="Maximum adaptive timeout in seconds.",
    )

    # ==================================================================
    # Model Architecture Settings (for performance calculations)
    # ==================================================================
    model_n_kv_heads: int = Field(
        8,
        description="Number of key-value heads for KV-cache calculations.",
    )
    model_d_head: int = Field(
        128,
        description="Dimension of each head for KV-cache calculations.",
    )
    model_n_layers: int = Field(
        32,
        description="Number of transformer layers for performance modeling.",
    )
    gpu_memory_mb: int = Field(
        16384,
        description="GPU memory size in MB for memory management calculations.",
    )

    # ==================================================================
    # Advanced Reasoning Techniques
    # ==================================================================
    
    # Tree-of-Thoughts (ToT) Configuration
    tot_enabled: bool = Field(
        True,
        description="Enable Tree-of-Thoughts reasoning for complex problem solving",
    )
    tot_max_depth: int = Field(
        5,
        description="Maximum depth for ToT reasoning tree",
    )
    tot_max_branching: int = Field(
        3,
        description="Maximum branching factor for ToT",
    )
    tot_max_thoughts: int = Field(
        10,
        description="Maximum total thoughts to generate in ToT",
    )
    tot_search_strategy: str = Field(
        "best_first",
        description="ToT search strategy: bfs, dfs, best_first, beam_search",
    )
    tot_beam_width: int = Field(
        2,
        description="Beam width for ToT beam search",
    )
    tot_early_termination_threshold: float = Field(
        0.9,
        description="Threshold for early termination in ToT (0-1)",
    )
    
    # Reflexion Configuration
    reflexion_enabled: bool = Field(
        True,
        description="Enable Reflexion self-reflection framework",
    )
    reflexion_max_attempts: int = Field(
        3,
        description="Maximum attempts in Reflexion",
    )
    reflexion_max_episodes: int = Field(
        20,
        description="Maximum episodes to store in Reflexion memory",
    )
    reflexion_temperature: float = Field(
        0.7,
        description="Temperature for Reflexion reflection generation",
    )
    reflexion_memory_file: str = Field(
        "data/reflexion_episodes.json",
        description="Path to Reflexion episodic memory file",
    )
    
    # ReAct Configuration
    react_enabled: bool = Field(
        True,
        description="Enable ReAct reasoning + acting framework",
    )
    react_max_steps: int = Field(
        10,
        description="Maximum steps per ReAct iteration",
    )
    react_max_iterations: int = Field(
        5,
        description="Maximum iterations in ReAct",
    )
    react_temperature: float = Field(
        0.7,
        description="Temperature for ReAct action generation",
    )
    react_allow_self_correction: bool = Field(
        True,
        description="Enable self-correction in ReAct",
    )
    react_verbose_thoughts: bool = Field(
        True,
        description="Enable verbose thought output in ReAct",
    )
    
    # Graph-of-Thoughts (GoT) Configuration
    got_enabled: bool = Field(
        True,
        description="Enable Graph-of-Thoughts reasoning",
    )
    got_max_nodes: int = Field(
        15,
        description="Maximum nodes in GoT graph",
    )
    got_max_depth: int = Field(
        5,
        description="Maximum depth for GoT reasoning",
    )
    got_max_branching: int = Field(
        3,
        description="Maximum branching factor for GoT",
    )
    got_aggregation_method: str = Field(
        "merge",
        description="GoT aggregation method: concatenate, merge, vote, hierarchical",
    )
    got_enable_aggregation: bool = Field(
        True,
        description="Enable thought aggregation in GoT",
    )
    got_enable_transformation: bool = Field(
        True,
        description="Enable thought transformation in GoT",
    )
    got_evaluation_threshold: float = Field(
        0.5,
        description="Threshold for thought transformation in GoT",
    )
    got_top_k_keep: int = Field(
        3,
        description="Number of top thoughts to keep in GoT",
    )
    
    # Self-Refine Configuration
    self_refine_enabled: bool = Field(
        True,
        description="Enable Self-Refine framework",
    )
    self_refine_max_iterations: int = Field(
        3,
        description="Maximum refinement iterations in Self-Refine",
    )
    self_refine_min_improvement: float = Field(
        0.1,
        description="Minimum improvement threshold for Self-Refine",
    )
    self_refine_temperature: float = Field(
        0.7,
        description="Temperature for Self-Refine generation",
    )
    self_refine_evaluation_temperature: float = Field(
        0.3,
        description="Temperature for Self-Refine evaluation",
    )
    self_refine_stop_threshold: float = Field(
        0.95,
        description="Stop threshold for Self-Refine",
    )
    
    # Self-Consistency Configuration
    self_consistency_enabled: bool = Field(
        True,
        description="Enable Self-Consistency framework",
    )
    self_consistency_num_samples: int = Field(
        5,
        description="Number of samples for Self-Consistency",
    )
    self_consistency_temperature: float = Field(
        0.7,
        description="Temperature for Self-Consistency sampling",
    )
    self_consistency_aggregation_strategy: str = Field(
        "majority_vote",
        description="Self-Consistency aggregation strategy: majority_vote, weighted_vote, best_score, consensus, average",
    )
    self_consistency_answer_normalization: str = Field(
        "whitespace",
        description="Self-Consistency answer normalization: exact, case_insensitive, whitespace, numeric, fuzzy",
    )
    self_consistency_min_agreement_threshold: float = Field(
        0.6,
        description="Minimum agreement threshold for Self-Consistency",
    )
    self_consistency_enable_confidence_scoring: bool = Field(
        True,
        description="Enable confidence scoring in Self-Consistency",
    )
    self_consistency_max_reasoning_length: int = Field(
        1000,
        description="Maximum reasoning length for Self-Consistency",
    )
    
    # General Reasoning Configuration
    reasoning_framework: str = Field(
        "adaptive",
        description="Primary reasoning framework: default, tot, got, reflexion, self_refine, self_consistency, react, adaptive, geodesic_flow, adaptive_depth, abductive_leap, meta_controller, recursive_critique, quantum_reasoning, active_inference",
    )
    reasoning_timeout: int = Field(
        60,
        description="Default timeout for reasoning operations in seconds",
    )
    tot_timeout: int = Field(
        120,
        description="Timeout for Tree-of-Thoughts reasoning in seconds",
    )
    got_timeout: int = Field(
        120,
        description="Timeout for Graph-of-Thoughts reasoning in seconds",
    )
    reflexion_timeout: int = Field(
        180,
        description="Timeout for Reflexion reasoning in seconds",
    )
    self_refine_timeout: int = Field(
        90,
        description="Timeout for Self-Refine reasoning in seconds",
    )
    self_consistency_timeout: int = Field(
        150,
        description="Timeout for Self-Consistency reasoning in seconds",
    )
    react_timeout: int = Field(
        90,
        description="Timeout for ReAct reasoning in seconds",
    )
    enable_reasoning_cache: bool = Field(
        True,
        description="Enable caching of reasoning results",
    )
    reasoning_cache_ttl: int = Field(
        600,
        description="TTL for reasoning cache in seconds (default 10 minutes)",
    )
    reasoning_cache_ttl_short: int = Field(
        300,
        description="Short TTL for time-sensitive reasoning cache in seconds (default 5 minutes)",
    )
    
    # ==================================================================
    # Advanced Optimization-Based Reasoning Techniques
    # ==================================================================
    
    # Reasoning Geodesic Flow Configuration
    geodesic_flow_enabled: bool = Field(
        True,
        description="Enable Reasoning Geodesic Flow for optimal reasoning paths",
    )
    geodesic_hidden_dim: int = Field(
        512,
        description="Hidden dimension for geodesic flow metric learning",
    )
    geodesic_regularization_weight: float = Field(
        0.1,
        description="Regularization weight for geodesic loss",
    )
    geodesic_max_iterations: int = Field(
        100,
        description="Maximum iterations for metric learning",
    )
    
    # Adaptive Depth Controller Configuration
    adaptive_depth_enabled: bool = Field(
        True,
        description="Enable Adaptive Thought Depth controller",
    )
    adaptive_depth_cost_lambda: float = Field(
        0.01,
        description="Cost parameter for depth control",
    )
    adaptive_depth_min_steps: int = Field(
        1,
        description="Minimum reasoning steps",
    )
    adaptive_depth_max_steps: int = Field(
        10,
        description="Maximum reasoning steps",
    )
    adaptive_depth_confidence_threshold: float = Field(
        0.95,
        description="Confidence threshold for early stopping",
    )
    
    # Abductive Leap Configuration
    abductive_leap_enabled: bool = Field(
        True,
        description="Enable Abductive Leap Operator",
    )
    abductive_lambda_simplicity: float = Field(
        0.1,
        description="Simplicity weight for abductive reasoning",
    )
    abductive_num_hypotheses: int = Field(
        5,
        description="Number of candidate hypotheses to generate",
    )
    
    # Counterfactual Robustness Configuration
    counterfactual_robustness_enabled: bool = Field(
        True,
        description="Enable Counterfactual Robustness evaluation",
    )
    counterfactual_num_perturbations: int = Field(
        5,
        description="Number of perturbed versions to test",
    )
    counterfactual_perturbation_strength: float = Field(
        0.1,
        description="Strength of input perturbations",
    )
    counterfactual_similarity_threshold: float = Field(
        0.8,
        description="Minimum similarity for robustness",
    )
    
    # Meta Controller Configuration
    meta_controller_enabled: bool = Field(
        True,
        description="Enable Meta-Reasoning Controller",
    )
    meta_controller_stop_threshold: float = Field(
        0.05,
        description="Threshold for stopping reasoning",
    )
    meta_controller_hidden_dim: int = Field(
        64,
        description="Hidden dimension for meta controller network",
    )
    
    # Recursive Critique Configuration
    recursive_critique_enabled: bool = Field(
        True,
        description="Enable Recursive Self-Critique Improvement",
    )
    recursive_critique_iterations: int = Field(
        3,
        description="Number of critique iterations",
    )
    recursive_critique_learning_rate: float = Field(
        0.01,
        description="Learning rate for gradient-based improvement",
    )
    
    # Quantum Reasoning Configuration
    quantum_reasoning_enabled: bool = Field(
        True,
        description="Enable Quantum Superposition reasoning",
    )
    quantum_state_dim: int = Field(
        8,
        description="Dimension of quantum state space",
    )
    quantum_enable_interference: bool = Field(
        True,
        description="Enable quantum interference effects",
    )
    quantum_decoherence_rate: float = Field(
        0.001,
        description="Rate of quantum decoherence",
    )
    
    # Active Inference Configuration
    active_inference_enabled: bool = Field(
        True,
        description="Enable Active Inference Free Energy reasoning",
    )
    active_inference_complexity_weight: float = Field(
        0.5,
        description="Weight for complexity penalty in free energy",
    )
    active_inference_planning_horizon: int = Field(
        3,
        description="Planning horizon for action selection",
    )
    active_inference_prior_strength: float = Field(
        1.0,
        description="Strength of prior beliefs",
    )

    # ==================================================================
    # Speculative Decoding Settings
    # ==================================================================
    speculative_enabled: bool = Field(
        False,
        description="Enable speculative decoding for faster inference.",
    )
    speculative_gamma: int = Field(
        4,
        description="Speculative decoding gamma parameter (draft tokens per step).",
    )
    speculative_alpha: float = Field(
        0.75,
        description="Speculative decoding acceptance rate threshold.",
    )
    speculative_cost_ratio: float = Field(
        0.1,
        description="Cost ratio of draft model to target model.",
    )

    # ============================================================================
    # BENCHMARK INTEGRATION CONFIGURATION
    # Research-based enhancements for target benchmarks while maintaining speed
    # ============================================================================
    
    # Benchmark Integration Layer
    enable_benchmark_integration: bool = Field(
        True,
        description="Enable benchmark integration layer for selective enhancement"
    )
    benchmark_integration_path: str = Field(
        "benchmark_integrations/benchmark_integration_layer.py",
        description="Path to benchmark integration layer module"
    )
    
    # Math Reasoning Enhancement
    enable_math_reasoning: bool = Field(
        True,
        description="Enable math reasoning enhancement with CoT templates"
    )
    math_reasoning_config: str = Field(
        '{"enable_cot_templates": true, "max_reasoning_steps": 3}',
        description="JSON configuration for math reasoning"
    )
    
    # Knowledge Graph Integration
    enable_knowledge_graph: bool = Field(
        True,
        description="Enable knowledge graph integration for domain queries"
    )
    knowledge_graph_config: str = Field(
        '{"max_entities_per_query": 3, "cache_graph_queries": true}',
        description="JSON configuration for knowledge graph"
    )
    
    # Uncertainty Calibration
    enable_uncertainty_calibration: bool = Field(
        True,
        description="Enable uncertainty calibration for better self-assessment"
    )
    uncertainty_config: str = Field(
        '{"confidence_threshold": 0.4, "uncertainty_threshold": 0.5}',
        description="JSON configuration for uncertainty calibration"
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("rag_docs_dir", "rag_index_dir", "chroma_persist_dir", mode="before")
    @classmethod
    def _ensure_dir(cls, v: str | Path) -> Path:
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

# Rebuild the model to resolve forward references (Pydantic v2 requirement)
Settings.model_rebuild()

settings = Settings()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_litellm_model() -> str:
    """Return the correct LiteLLM model string based on active backend."""
    if settings.local_backend in (BackendType.VLLM, BackendType.SGLANG):
        return f"openai/{settings.local_model_name}"
    return settings.default_model


def export_api_keys() -> None:
    """Push API keys into the environment so LiteLLM can find them."""
    key_map = {
        "OPENAI_API_KEY":     settings.openai_api_key,
        "ANTHROPIC_API_KEY":  settings.anthropic_api_key,
        "OPENROUTER_API_KEY": settings.openrouter_api_key,
        "COHERE_API_KEY":     settings.cohere_api_key,
    }
    for env_var, value in key_map.items():
        if value:
            os.environ[env_var] = value
