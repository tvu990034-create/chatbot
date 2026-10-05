"""Pipeline configuration (passed explicitly; server reads env and builds this)."""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class SpeedConfig:
    # Pre-filter
    use_bloom: bool = True
    use_faq: bool = True
    use_zero_token: bool = True
    entity_known_prior: float = 0.5
    bloom_bits_per_element: int = 8
    idk_response: str = "I'm sorry, I don't have information about that."

    # Cache (EQ-PREWARM-CACHE, EQ-SIMHASH)
    use_prewarm_cache: bool = True
    cache_max_entries: int = 50_000
    simhash_threshold: int = 4
    cache_target_hit_rate: float = 0.95
    bm25_cache_score_threshold: float = 0.3

    # Retrieval
    use_dynamic_index_switch: bool = True
    short_query_max_words: int = 2
    bm25_min_results: int = 3
    bm25_score_threshold: float = 0.0
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    bm25_top_k: int = 100
    dense_top_k: int = 50
    use_query_truncation: bool = True
    truncation_alpha: float = 0.3
    use_score_gating: bool = True
    score_threshold: float = 0.5
    use_pagerank_prune: bool = True
    pagerank_beta: float = 0.5
    pagerank_keep_k: int = 80
    pagerank_score_floor: float = 0.01
    chunk_limit: int = 1

    # Prompt
    minimal_prompt: bool = True
    one_liner_mode: bool = False
    dynamic_tokens_base: int = 20
    dynamic_tokens_multiplier: float = 2.0
    dynamic_tokens_max_cap: int = 256
    dynamic_tokens_long_answer_cap: int = 150

    # Generation
    boredom_threshold: float = 0.15
    boredom_patience: int = 3
    boredom_min_tokens: int = 5
    simulated_cloud_ms: float = 25.0
    simulated_token_ms: float = 2.0

    # Data injected at startup
    faq: Dict[str, str] = field(default_factory=dict)
    zero_token_phrases: Dict[str, str] = field(default_factory=dict)
    bloom_entities: frozenset = field(default_factory=frozenset)
    documents: tuple = field(default_factory=tuple)
    pagerank: Dict[str, float] = field(default_factory=dict)
    df_dict: Dict[str, int] = field(default_factory=dict)
