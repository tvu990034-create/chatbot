"""
Prime Optimizations - Special Logic and Equations from Provided Files
Implements 6 key equations for speed and relevance improvements
"""

import time
import hashlib
import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Dict, List, Optional, Tuple
import json


# ============================================================
# 1st Equation: Session Cache TTFT Speedup
# ============================================================
def session_cache_ttft_speedup(
    history_tokens: int,            # H – total prefix length (no-cache)
    new_query_tokens: int,          # u – tokens in new user message
    suffix_tokens: int = 5,         # δ – "\nassistant:" overhead
    prefill_time_per_token: float = 2.0,  # c (ms) – from benchmarks
    load_overhead_ms: float = 15.0,       # t_load (ms) – load state + hash
    hit_rate: float = 1.0,                # p_hit – 1.0 for steady sessions
) -> dict:
    """
    Predict TTFT speedup from session KV‑cache reuse.
    
    Returns a dict with:
        - ttft_no_cache_ms: float
        - ttft_cached_ms: float
        - expected_ttft_ms: float (weighted by hit_rate)
        - speedup_factor: float (≥1)
        - reduction_pct: float (0‑100, how much of the original TTFT is eliminated)
    """
    # Time to prefill the whole history from scratch
    ttft_no = history_tokens * prefill_time_per_token

    # Time when cache is valid
    ttft_cache = load_overhead_ms + (new_query_tokens + suffix_tokens) * prefill_time_per_token

    # Expected TTFT given cache hit probability
    expected_ttft = hit_rate * ttft_cache + (1 - hit_rate) * ttft_no

    # Speedup factor
    speedup = ttft_no / expected_ttft if expected_ttft > 0 else float('inf')

    # Reduction in TTFT (percentage of original that is removed)
    reduction_pct = 100.0 * (1 - expected_ttft / ttft_no) if ttft_no > 0 else 0.0

    return {
        "ttft_no_cache_ms": round(ttft_no, 2),
        "ttft_cached_ms": round(ttft_cache, 2),
        "expected_ttft_ms": round(expected_ttft, 2),
        "speedup_factor": round(speedup, 2),
        "reduction_pct": round(reduction_pct, 1),
    }


def overall_conv_speedup(turns: list[dict]) -> dict:
    """Calculate overall conversation speedup from multiple turns."""
    total_no = 0.0
    total_ex = 0.0
    for t in turns:
        # t: dict with history_tokens, new_query_tokens, hit_rate (1.0 after first)
        res = session_cache_ttft_speedup(**t)
        total_no += res["ttft_no_cache_ms"]
        total_ex += res["expected_ttft_ms"]
    return {
        "total_no_cache_ms": total_no,
        "total_cached_ms": total_ex,
        "overall_speedup": total_no / total_ex if total_ex > 0 else 0,
        "avg_reduction_pct": 100 * (1 - total_ex / total_no) if total_no > 0 else 0,
    }


# ============================================================
# 2nd Equation: CachedLLM for System Prompt Caching
# ============================================================
class CachedLLM:
    """
    Engine that caches the KV state after a fixed system prompt,
    so later requests skip re‑computing it.
    """

    def __init__(self, model, system_prompt: str, n_ctx: int = 2048):
        self.model = model
        self.system_prompt = system_prompt
        self.n_ctx = n_ctx
        # The constant prefix that will be saved
        self.prefix = f"system: {self.system_prompt}\nuser: "
        # Cached state buffer (None until first request)
        self.cached_state: Optional[bytes] = None
        self.prefix_hash = hashlib.sha256(self.prefix.encode()).hexdigest()[:8]

    def _build_base_state(self):
        """
        Process only the prefix (max_tokens=0) and save its state.
        This is a placeholder - actual implementation depends on model type.
        """
        # For llama-cpp-python: self.model.create_completion(self.prefix, max_tokens=0)
        # For other models: similar prefill-only operation
        self.cached_state = b"cached_state_placeholder"  # Placeholder
        print(f"Base state saved for prefix hash: {self.prefix_hash}")

    def generate(self, user_message: str, max_tokens: int = 128) -> str:
        """Generate response using cached system prompt state."""
        # First request: build and cache the base state
        if self.cached_state is None:
            self._build_base_state()

        # Load the saved state – this is T_restore, near‑zero cost
        # For llama-cpp-python: self.model.load_state(self.cached_state)
        
        # Build the suffix that changes every time
        suffix = f"{user_message}\nassistant: "
        
        # Generate response (model-specific implementation)
        # This is a placeholder - actual implementation depends on model type
        return f"Cached response to: {user_message}"

    def invalidate_cache(self):
        """Invalidate cache when system prompt changes."""
        self.cached_state = None


# ============================================================
# 3rd Equation: One-Liner Mode
# ============================================================
def simulate_generation(output_tokens, time_per_token_ms=10, overhead_ms=50):
    """
    Simulate LLM generation latency.
    - output_tokens: number of tokens generated
    - time_per_token_ms: milliseconds per token (inference + decoding)
    - overhead_ms: fixed cost (prompt processing, network, etc.)
    Returns total latency in ms.
    """
    return overhead_ms + output_tokens * time_per_token_ms


# Typical numbers from a small model (e.g., TinyLlama on CPU)
NORMAL_TOKENS = 100
ONE_LINER_TOKENS = 18


def get_one_liner_speedup():
    """Calculate speedup factor for one-liner mode."""
    latency_normal = simulate_generation(NORMAL_TOKENS)
    latency_one_liner = simulate_generation(ONE_LINER_TOKENS)
    speedup = latency_normal / latency_one_liner
    return {
        "normal_mode_ms": latency_normal,
        "one_liner_mode_ms": latency_one_liner,
        "speedup_factor": round(speedup, 1),
    }


# ============================================================
# 4th Equation: Model Routing/Cascade
# ============================================================
def predict_length(query: str) -> int:
    """Crude proxy: answer length roughly 3× query length for factual questions."""
    words = len(query.split())
    if any(w in query.lower() for w in ['explain', 'describe', 'how to', 'compare']):
        return words * 25   # complex → long
    return words * 8        # simple → short


def route_model(query, tau=80):
    """Route query to appropriate model based on predicted answer length."""
    L = predict_length(query)
    return 'small' if L <= tau else 'large'


def generate_with_routing(small_model_fn, large_model_fn, prompt, max_small_tokens=50):
    """
    Cascade routing: try small model with token cap, fallback to large if truncated.
    Returns the answer string, and a flag indicating which model was used.
    """
    # 1. Try small model
    try:
        output_small = small_model_fn(prompt, max_tokens=max_small_tokens)
        answer = output_small.strip()
        # Heuristic for truncation: output ends mid‑sentence, or reached exactly max_tokens
        token_count = len(answer.split())
        if token_count < max_small_tokens and not answer.endswith(('...', '..')):
            # Looks complete → simple query
            return answer, 'small'
    except Exception:
        pass
    
    # 2. Fallback to large model (no token cap)
    output_large = large_model_fn(prompt, max_tokens=256)
    return output_large.strip(), 'large'


def estimate_speedup(frac_simple, time_small, time_large):
    """
    Estimate overall speedup from model routing.
    - frac_simple: fraction of queries routed to small model
    - time_small: average total time for small model (ms)
    - time_large: average total time for large model (ms)
    """
    avg_routed = frac_simple * time_small + (1 - frac_simple) * time_large
    avg_baseline = time_large   # if you always use large model
    return avg_baseline / avg_routed if avg_routed > 0 else 0


# ============================================================
# 5th Equation: Cache-Aware Conciseness
# ============================================================
def cost_delta(
    L_add: int,
    delta_L_out: int,
    c_prompt_per_token: float = 0.0,   # 0 if cached
    c_output_per_token: float = 1.0,    # cost or time per output token
) -> float:
    """
    Returns negative if optimization saves cost/latency.
    Usage: how much we save by appending a conciseness instruction.
    """
    input_cost = c_prompt_per_token * L_add
    output_saving = c_output_per_token * delta_L_out
    return input_cost - output_saving   # negative = saving


SYSTEM_PROMPT = "Answer briefly."
SYSTEM_HASH = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:8]


def build_cache_aware_prompt(query, context=""):
    """Build prompt with cache-aware system prompt."""
    # Cached prefix ends after system prompt
    return f"{SYSTEM_PROMPT}\n{context}\nUser: {query}\nAssistant:"


# ============================================================
# 6th Equation: Optimized FAQ Database
# ============================================================
def normalize(text: str) -> str:
    """Canonical form for exact matching."""
    # 1. Lowercase
    t = text.lower().strip()
    # 2. Remove punctuation (keep letters, digits, spaces)
    t = re.sub(r'[^\w\s]', '', t)
    # 3. Collapse multiple whitespaces into single space
    t = re.sub(r'\s+', ' ', t).strip()
    return t


@dataclass
class FAQRecord:
    question_raw: str
    normalized: str
    answer: str
    hash: str                # SHA‑256 of normalized+answer
    timestamp_added: float = field(default_factory=lambda: time.time())

    @staticmethod
    def compute_hash(norm_question: str, answer: str) -> str:
        return hashlib.sha256(f"{norm_question}|{answer}".encode()).hexdigest()


class OptimizedFAQDatabase:
    """Optimized FAQ database with thread safety and hashing."""
    def __init__(self):
        self._data: Dict[str, FAQRecord] = {}         # Fast O(1) lookup
        self._questions: List[str] = []               # For debugging
        self._lock = RLock()                          # Thread safety

    def add(self, question: str, answer: str) -> None:
        norm_q = normalize(question)
        rec = FAQRecord(
            question_raw=question,
            normalized=norm_q,
            answer=answer,
            hash=FAQRecord.compute_hash(norm_q, answer)
        )
        with self._lock:
            self._data[norm_q] = rec
            self._questions.append(norm_q)

    def get(self, question: str) -> Optional[str]:
        """Return answer if found, else None."""
        norm_q = normalize(question)
        with self._lock:
            rec = self._data.get(norm_q)
            return rec.answer if rec else None

    def contains(self, question: str) -> bool:
        return normalize(question) in self._data

    def remove(self, question: str) -> bool:
        norm_q = normalize(question)
        with self._lock:
            if norm_q in self._data:
                del self._data[norm_q]
                self._questions.remove(norm_q)
                return True
        return False

    def to_json(self, filepath: str) -> None:
        with self._lock:
            records = [{
                "question_raw": r.question_raw,
                "normalized": r.normalized,
                "answer": r.answer,
                "hash": r.hash,
                "timestamp_added": r.timestamp_added
            } for r in self._data.values()]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(records, f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'OptimizedFAQDatabase':
        db = cls()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                records = json.load(f)
            for rec in records:
                db.add(rec.get("question_raw", ""), rec.get("answer", ""))
        except Exception:
            pass
        return db


# ============================================================
# Prime Optimization Manager
# ============================================================
class PrimeOptimizationManager:
    """
    Manages all prime optimizations for maximum speed and relevance.
    """
    def __init__(self):
        self.session_cache_enabled = True
        self.system_prompt_cache_enabled = True
        self.one_liner_mode_enabled = False
        self.model_routing_enabled = False
        self.cache_aware_conciseness_enabled = True
        self.optimized_faq_enabled = True
        
        self.faq_db = OptimizedFAQDatabase()
        self.cached_llm = None  # Will be initialized with model
        
        # Track session history for cache speedup
        self.session_history: Dict[str, List[int]] = {}  # session_id -> token counts
        
    def get_session_speedup_prediction(self, session_id: str, history_tokens: int, new_query_tokens: int):
        """Predict speedup for a session using KV-cache."""
        return session_cache_ttft_speedup(
            history_tokens=history_tokens,
            new_query_tokens=new_query_tokens,
            hit_rate=1.0 if session_id in self.session_history else 0.0
        )
    
    def enable_one_liner_mode(self, enabled: bool = True):
        """Enable or disable one-liner mode for faster responses."""
        self.one_liner_mode_enabled = enabled
        return get_one_liner_speedup()
    
    def route_query_to_model(self, query: str):
        """Route query to appropriate model based on complexity."""
        if not self.model_routing_enabled:
            return 'large'  # Default to large model
        return route_model(query)
    
    def build_optimized_prompt(self, query: str, context: str = "") -> str:
        """Build prompt with cache-aware optimizations."""
        if self.cache_aware_conciseness_enabled:
            return build_cache_aware_prompt(query, context)
        return f"{context}\nUser: {query}\nAssistant:"
    
    def get_faq_answer(self, question: str) -> Optional[str]:
        """Get answer from optimized FAQ database."""
        if self.optimized_faq_enabled:
            return self.faq_db.get(question)
        return None
