"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file before any config is read
load_dotenv()

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"Invalid integer value for {name}={raw}, using default {default}")
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(slots=True)
class AdvancedConfig:
    # --- Existing features ---
    use_history_trimming: bool = True
    history_tau: float = 0.15
    history_top_k: int = 10
    history_max_tokens: int = 512
    use_speculative: bool = False
    use_frgl: bool = False

    use_pagerank_boost: bool = False       # kept for backward compat, but newer use_pagerank_prune is preferred
    pagerank_boost_gamma: float = 0.1

    use_kv_pruning: bool = False
    kv_pruning_theta: float = 0.85
    kv_pruning_layer: int = 2

    use_tome: bool = False
    tome_threshold: float = 0.95
    tome_merge_every: int = 2

    # --- New features from Equation bundle ---
    # System prompt state caching (Eq 19) – dramatically reduces TTFB after first call
    use_system_prompt_cache: bool = True

    # Dynamic max tokens (Eq 12) – short questions get tiny budgets
    use_dynamic_max_tokens: bool = True
    dynamic_tokens_base: int = 20
    dynamic_tokens_multiplier: float = 2.0
    dynamic_tokens_max_cap: int = 256
    dynamic_tokens_long_answer_cap: int = 150

    # Boredom stopper (Eq 13) – stops rambling early
    use_boredom_stopper: bool = True
    boredom_threshold: float = 0.15
    boredom_patience: int = 3
    boredom_min_tokens: int = 5

    # Dynamic index switching (Eq 8) – skip dense/reranker for short queries
    use_dynamic_index_switch: bool = True
    short_query_max_words: int = 2

    # Essential keyword truncation (Eq 2)
    use_essential_keywords: bool = True
    essential_keywords_alpha: float = 0.3

    # PageRank multiplicative pruning (Eq 3) – supersedes simple boost
    use_pagerank_prune: bool = True
    pagerank_prune_beta: float = 0.5
    pagerank_prune_keep_k: int = 80
    pagerank_prune_score_floor: float = 0.01

    # Single‑chunk retrieval (Eq 21)
    chunk_limit: int = 1                   # 5 is default for full context; 1 = single chunk

    # Score‑based early termination (Eq 7) – don't call LLM if retrieval is weak
    use_score_gating: bool = True
    score_threshold: float = 0.5

    # Minimal prompt (Eq 20)
    minimal_prompt: bool = True

    # One‑liner mode (Eq 17)
    one_liner_mode: bool = False

    # Zero‑token responder (Eq 14)
    zero_token_enabled: bool = True

    # FAQ database (Eq 15)
    faq_enabled: bool = True
    faq_db_path: str = "faq_database.json"

    # --- Equation 1: Adaptive Early Exit ---
    use_early_exit: bool = False
    early_exit_patience: int = 3
    early_exit_min_layer: int = 5
    early_exit_thresholds_path: str = "early_exit_thresholds.json"

    # --- Equation 2: ESN Draft Model ---
    use_esn_draft: bool = False
    esn_config_path: str = "esn_config.json"
    esn_max_draft_len: int = 20

    # --- Equation 3: Prompt Compiler ---
    use_prompt_compiler: bool = False
    compiler_threshold: float = 0.5
    compiler_k_eff: float = 1.2
    compiler_eta: float = 0.8

    # --- Equation 5: Lightning ESN Performance ---
    # Analytical only - no config flags needed

    # --- Equation 7: Multi-core CPU Speedup ---
    # Analytical only - no config flags needed

    # --- Equation 8: KG Routing ---
    use_kg_routing: bool = False
    kg_threshold: float = 0.7
    kg_alpha: float = 0.5
    kg_triples_path: str = "kg_triples.json"

    # --- Equation 9: Neural Hash Cache ---
    use_neural_hash_cache: bool = False
    hash_model_path: str = "hash_model.pt"
    hash_table_path: str = "hash_table.json"
    hash_threshold: float = 0.85

    # --- Equation 10: Template Forest ---
    # Analytical only - no config flags needed

    # --- Equation 11: Speculative Worlds ---
    use_speculative_worlds: bool = False
    spec_worlds_max_branches: int = 3
    spec_worlds_timeout_ms: int = 500

    # --- Equation 12: Draft Speedup ---
    # Analytical only - no config flags needed

    # --- Equation 13: Speculative Cache ---
    use_speculative_cache: bool = False
    spec_cache_max_entries: int = 1000
    spec_cache_ttl_ms: int = 3600000

    # --- Equation 14: mmap Loader ---
    # Analytical only - no config flags needed

    # --- Equation 15: Structured Generation ---
    # Analytical only - no config flags needed

    # --- Equation 16: One-Shot Answer ---
    use_one_shot_answer: bool = False
    one_shot_threshold: float = 0.8
    one_shot_mapper_path: str = "one_shot_mapper.pt"
    one_shot_index_path: str = "one_shot_index.faiss"
    one_shot_answers_path: str = "one_shot_answers.json"

    # --- Equation 17: Infinite Context ---
    use_infinite_context: bool = False
    max_spine_tokens: int = 2048
    max_window_tokens: int = 4096
    summarize_every_n_turns: int = 5
    summarizer_model: str = "gpt-3.5-turbo"

    # --- Equation 18: IVF-PQ Retrieval ---
    use_ivf_pq_retrieval: bool = False
    ivf_nlist: int = 4096
    ivf_nprobe: int = 64
    pq_m: int = 64
    pq_bits: int = 8

    @classmethod
    def from_env(cls) -> "AdvancedConfig":
        return cls(
            use_history_trimming=_env_bool("USE_HISTORY_TRIMMING", True),
            history_tau=_env_float("HISTORY_TAU", 0.15),
            history_top_k=_env_int("HISTORY_TOP_K", 10),
            history_max_tokens=_env_int("HISTORY_MAX_TOKENS", 512),
            use_pagerank_boost=_env_bool("USE_PAGERANK_BOOST", False),
            pagerank_boost_gamma=_env_float("PAGERANK_BOOST_GAMMA", 0.1),
            use_kv_pruning=_env_bool("USE_KV_PRUNING", False),
            kv_pruning_theta=_env_float("KV_PRUNING_THETA", 0.85),
            kv_pruning_layer=_env_int("KV_PRUNING_LAYER", 2),
            use_tome=_env_bool("USE_TOME", False),
            tome_threshold=_env_float("TOME_THRESHOLD", 0.95),
            tome_merge_every=_env_int("TOME_MERGE_EVERY", 2),
            # new flags
            use_dynamic_max_tokens=_env_bool("USE_DYNAMIC_MAX_TOKENS", True),
            dynamic_tokens_base=_env_int("DYNAMIC_TOKENS_BASE", 20),
            dynamic_tokens_multiplier=_env_float("DYNAMIC_TOKENS_MULTIPLIER", 2.0),
            dynamic_tokens_max_cap=_env_int("DYNAMIC_TOKENS_MAX_CAP", 256),
            dynamic_tokens_long_answer_cap=_env_int("DYNAMIC_TOKENS_LONG_ANSWER_CAP", 150),
            use_boredom_stopper=_env_bool("USE_BOREDOM_STOPPER", True),
            boredom_threshold=_env_float("BOREDOM_THRESHOLD", 0.15),
            boredom_patience=_env_int("BOREDOM_PATIENCE", 3),
            boredom_min_tokens=_env_int("BOREDOM_MIN_TOKENS", 5),
            use_dynamic_index_switch=_env_bool("USE_DYNAMIC_INDEX_SWITCH", True),
            short_query_max_words=_env_int("SHORT_QUERY_MAX_WORDS", 2),
            use_essential_keywords=_env_bool("USE_ESSENTIAL_KEYWORDS", True),
            essential_keywords_alpha=_env_float("ESSENTIAL_KEYWORDS_ALPHA", 0.3),
            use_pagerank_prune=_env_bool("USE_PAGERANK_PRUNE", True),
            pagerank_prune_beta=_env_float("PAGERANK_PRUNE_BETA", 0.5),
            pagerank_prune_keep_k=_env_int("PAGERANK_PRUNE_KEEP_K", 80),
            pagerank_prune_score_floor=_env_float("PAGERANK_PRUNE_SCORE_FLOOR", 0.01),
            chunk_limit=_env_int("CHUNK_LIMIT", 1),
            use_score_gating=_env_bool("USE_SCORE_GATING", True),
            score_threshold=_env_float("SCORE_THRESHOLD", 0.5),
            minimal_prompt=_env_bool("MINIMAL_PROMPT", True),
            one_liner_mode=_env_bool("ONE_LINER_MODE", False),
            zero_token_enabled=_env_bool("ZERO_TOKEN_ENABLED", True),
            faq_enabled=_env_bool("FAQ_ENABLED", True),
            faq_db_path=os.getenv("FAQ_DB_PATH", "faq_database.json"),
            use_speculative=_env_bool("USE_SPECULATIVE", False),
            use_frgl=_env_bool("USE_FRGL", False),
            # Equation 1: Adaptive Early Exit
            use_early_exit=_env_bool("USE_EARLY_EXIT", False),
            early_exit_patience=_env_int("EARLY_EXIT_PATIENCE", 3),
            early_exit_min_layer=_env_int("EARLY_EXIT_MIN_LAYER", 5),
            early_exit_thresholds_path=os.getenv("EARLY_EXIT_THRESHOLDS_PATH", "early_exit_thresholds.json"),
            # Equation 2: ESN Draft Model
            use_esn_draft=_env_bool("USE_ESN_DRAFT", False),
            esn_config_path=os.getenv("ESN_CONFIG_PATH", "esn_config.json"),
            esn_max_draft_len=_env_int("ESN_MAX_DRAFT_LEN", 20),
            # Equation 3: Prompt Compiler
            use_prompt_compiler=_env_bool("USE_PROMPT_COMPILER", False),
            compiler_threshold=_env_float("COMPILER_THRESHOLD", 0.5),
            compiler_k_eff=_env_float("COMPILER_K_EFF", 1.2),
            compiler_eta=_env_float("COMPILER_ETA", 0.8),
            # Equation 8: KG Routing
            use_kg_routing=_env_bool("USE_KG_ROUTING", False),
            kg_threshold=_env_float("KG_THRESHOLD", 0.7),
            kg_alpha=_env_float("KG_ALPHA", 0.5),
            kg_triples_path=os.getenv("KG_TRIPLES_PATH", "kg_triples.json"),
            # Equation 9: Neural Hash Cache
            use_neural_hash_cache=_env_bool("USE_NEURAL_HASH_CACHE", False),
            hash_model_path=os.getenv("HASH_MODEL_PATH", "hash_model.pt"),
            hash_table_path=os.getenv("HASH_TABLE_PATH", "hash_table.json"),
            hash_threshold=_env_float("HASH_THRESHOLD", 0.85),
            # Equation 11: Speculative Worlds
            use_speculative_worlds=_env_bool("USE_SPECULATIVE_WORLDS", False),
            spec_worlds_max_branches=_env_int("SPEC_WORLDS_MAX_BRANCHES", 3),
            spec_worlds_timeout_ms=_env_int("SPEC_WORLDS_TIMEOUT_MS", 500),
            # Equation 13: Speculative Cache
            use_speculative_cache=_env_bool("USE_SPECULATIVE_CACHE", False),
            spec_cache_max_entries=_env_int("SPEC_CACHE_MAX_ENTRIES", 1000),
            spec_cache_ttl_ms=_env_int("SPEC_CACHE_TTL_MS", 3600000),
            # Equation 16: One-Shot Answer
            use_one_shot_answer=_env_bool("USE_ONE_SHOT_ANSWER", False),
            one_shot_threshold=_env_float("ONE_SHOT_THRESHOLD", 0.8),
            one_shot_mapper_path=os.getenv("ONE_SHOT_MAPPER_PATH", "one_shot_mapper.pt"),
            one_shot_index_path=os.getenv("ONE_SHOT_INDEX_PATH", "one_shot_index.faiss"),
            one_shot_answers_path=os.getenv("ONE_SHOT_ANSWERS_PATH", "one_shot_answers.json"),
            # Equation 17: Infinite Context
            use_infinite_context=_env_bool("USE_INFINITE_CONTEXT", False),
            max_spine_tokens=_env_int("MAX_SPINE_TOKENS", 2048),
            max_window_tokens=_env_int("MAX_WINDOW_TOKENS", 4096),
            summarize_every_n_turns=_env_int("SUMMARIZE_EVERY_N_TURNS", 5),
            summarizer_model=os.getenv("SUMMARIZER_MODEL", "gpt-3.5-turbo"),
            # Equation 18: IVF-PQ Retrieval
            use_ivf_pq_retrieval=_env_bool("USE_IVF_PQ_RETRIEVAL", False),
            ivf_nlist=_env_int("IVF_NLIST", 4096),
            ivf_nprobe=_env_int("IVF_NPROBE", 64),
            pq_m=_env_int("PQ_M", 64),
            pq_bits=_env_int("PQ_BITS", 8),
        )
    
use_model_router: bool = True
concurrency_cap: int = 10
token_batch_size: int = 4
use_confidence_token_cap: bool = True
confidence_theta: float = 5.0
use_flat_prompt: bool = True
use_dynamic_system_prompt: bool = True
slo_threshold_ms: float = 1000.0
anomaly_k_factor: float = 3.0