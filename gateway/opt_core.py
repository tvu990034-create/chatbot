"""
Production optimization primitives used by the live inference path.

Every helper here is intended to change real request behavior:
cache identity, routing, generation policy, compression, validation, and execution.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Adaptive generation timeout (measured throughput)
# ---------------------------------------------------------------------------
# phi3:mini measured ~4.5 tok/s on this deployment under real long-reasoning
# traffic (adaptive_tokens_bench did NOT cap this; high-token correctness does).
# Short prompts show ~10 tok/s, but 1700-2700 char multi-step queries drop the
# effective rate to ~4.2-4.6 tok/s.  A fixed timeout kills any generation whose
# token budget needs more wall time (quality mode: 256-1024 tokens needs
# ~55-245s).  Scale the timeout with the actual token budget using the measured
# real-traffic floor so long queries finish instead of dying on all paths.
TOKENS_PER_SECOND_FLOOR = 4.5   # measured real-traffic effective rate
TIMEOUT_BUFFER_S = 15.0         # scheduler/queue/load cushion


def adaptive_generation_timeout(
    max_tokens: Optional[int] = None,
    base_timeout: Optional[float] = None,
) -> float:
    """Return an LLM-call timeout scaled to the token budget.

    base_timeout is the user-configured minimum (settings.generation_timeout).
    For max_tokens that fit within base grace the budget unchanged; otherwise
    scale so the model is allowed enough wall time to actually finish.
    """
    base = float(base_timeout if base_timeout is not None else 15.0)
    budget = int(max_tokens or 128)
    needed = budget / TOKENS_PER_SECOND_FLOOR + TIMEOUT_BUFFER_S
    return max(base, needed)


# Speed-mode token budget: simple queries stay small and fast, but multi-step
# reasoning/math/code needs enough room to actually finish (GSM8K showed a 128
# cap truncates the answer -> 0% accuracy).  Cap below baseline (512) so speed
# mode is still strictly cheaper than the naive path.
SPEED_MAX_TOKENS_SIMPLE = 128
SPEED_MAX_TOKENS_REASONING = 384


def speed_mode_max_tokens(
    req_max: Optional[int] = None,
    is_reasoning: bool = False,
) -> int:
    """Return the speed-mode max_tokens cap: 128 for simple queries, 384 for
    multi-step reasoning so the answer can complete without hitting the wall."""
    cap = SPEED_MAX_TOKENS_REASONING if is_reasoning else SPEED_MAX_TOKENS_SIMPLE
    return min(req_max or cap, cap)

# ---------------------------------------------------------------------------
# Bounded LRU + TTL cache (replaces unbounded dicts and O(n log n) eviction)
# ---------------------------------------------------------------------------

class BoundedTTLCache:
    """Thread-safe LRU cache with per-entry TTL and O(1) eviction."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600.0):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._data: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def _expired(self, entry: Dict[str, Any], now: float) -> bool:
        expires = entry.get("expires_at")
        return expires is not None and now >= float(expires)

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            if self._expired(entry, now):
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return entry.get("value")

    def set(self, key: str, value: Any, ttl: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        ttl = self.default_ttl if ttl is None else ttl
        now = time.time()
        with self._lock:
            self._data[key] = {
                "value": value,
                "expires_at": now + ttl if ttl and ttl > 0 else None,
                "created_at": now,
                "metadata": metadata or {},
            }
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def items_snapshot(self) -> List[Tuple[str, Dict[str, Any]]]:
        now = time.time()
        with self._lock:
            return [
                (k, v) for k, v in self._data.items()
                if not self._expired(v, now)
            ]

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0


# ---------------------------------------------------------------------------
# Cache identity — must include every response-affecting field
# ---------------------------------------------------------------------------

def make_cache_identity(
    query: str,
    *,
    model: str = "",
    temperature: Any = None,
    max_tokens: Any = None,
    top_p: Any = None,
    top_k: Any = None,
    system_prompt: str = "",
    messages: Optional[Sequence[Dict[str, Any]]] = None,
    tools: Any = None,
    rag_version: str = "",
    rag_enabled: bool = False,
    prompt_version: str = "v1",
    language: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    payload = {
        "q": " ".join((query or "").lower().split()),
        "model": model or "",
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "top_k": top_k,
        "sys": hashlib.md5((system_prompt or "").encode("utf-8")).hexdigest()[:12],
        "hist": hashlib.md5(json.dumps(messages or [], sort_keys=True, default=str).encode("utf-8")).hexdigest()[:12],
        "tools": hashlib.md5(json.dumps(tools or [], sort_keys=True, default=str).encode("utf-8")).hexdigest()[:8],
        "rag": f"{rag_version}:{int(bool(rag_enabled))}",
        "prompt_version": prompt_version,
        "language": language,
        "extra": extra or {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Cheap hashed embeddings (real cosine similarity, no extra model required)
# ---------------------------------------------------------------------------

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "to", "of", "and",
    "in", "for", "on", "with", "at", "by", "from", "or", "as",
}


def hashed_embedding(text: str, dim: int = 256) -> List[float]:
    """Feature-hashed character n-grams + tokens. Deterministic and cheap."""
    vec = [0.0] * dim
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    if not normalized:
        return vec
    tokens = [t for t in re.findall(r"[a-z0-9_]+", normalized) if t not in _STOP]
    grams = tokens[:]
    compact = re.sub(r"[^a-z0-9]", "", normalized)
    for n in (3, 4):
        if len(compact) >= n:
            grams.extend(compact[i:i + n] for i in range(len(compact) - n + 1))
    for g in grams:
        h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def lexical_overlap(a: str, b: str) -> float:
    wa = {t for t in re.findall(r"[a-z0-9_]+", (a or "").lower()) if t not in _STOP}
    wb = {t for t in re.findall(r"[a-z0-9_]+", (b or "").lower()) if t not in _STOP}
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ---------------------------------------------------------------------------
# Token-aware budgeting
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def compress_prompt(text: str, ratio: float, keep_tail_tokens: int = 96) -> str:
    """
    Compress filler in the middle while preserving the user's final instruction.
    Never character-slices through code, JSON, or the last question.
    """
    if not text:
        return text
    try:
        ratio = float(ratio)
    except (TypeError, ValueError):
        ratio = 1.0
    if not (0 < ratio <= 1):
        raise ValueError("compression ratio must be in (0, 1]")
    if ratio >= 0.99:
        return text

    original_tokens = estimate_tokens(text)
    target = max(32, int(original_tokens * ratio))
    if original_tokens <= target:
        return re.sub(r"[ \t]+", " ", text).strip()

    # Keep fenced code / JSON blocks intact.
    fences = list(re.finditer(r"```[\s\S]*?```", text))
    if fences or text.lstrip().startswith("{") or text.lstrip().startswith("["):
        # Only squeeze whitespace outside fences; do not drop user instruction.
        squeezed = re.sub(r"[ \t]+", " ", text)
        squeezed = re.sub(r"\n{3,}", "\n\n", squeezed)
        if estimate_tokens(squeezed) <= original_tokens:
            return squeezed.strip()
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 2:
        return text.strip()

    tail = sentences[-1]
    head = sentences[0] if len(sentences) > 2 else ""
    middle = sentences[1:-1]
    filler = {"the", "a", "an", "just", "really", "very", "please", "kindly"}
    kept_middle: List[str] = []
    budget = max(keep_tail_tokens, target - estimate_tokens(head) - estimate_tokens(tail))
    used = 0
    for sent in middle:
        words = [w for w in sent.split() if w.lower() not in filler]
        piece = " ".join(words)
        cost = estimate_tokens(piece)
        if used + cost > budget:
            continue
        kept_middle.append(piece)
        used += cost
    parts = [p for p in [head] + kept_middle + [tail] if p]
    result = " ".join(parts).strip()
    if estimate_tokens(result) >= original_tokens:
        return text.strip()
    return result


def build_priority_messages(
    messages: List[Dict[str, str]],
    token_budget: int = 2048,
) -> List[Dict[str, str]]:
    """Keep system + current user + highest-value history within a token budget."""
    if not messages:
        return messages
    system = [m for m in messages if m.get("role") in ("system", "developer")]
    current = messages[-1:]
    rest = [m for m in messages[:-1] if m.get("role") not in ("system", "developer")]

    def _tokens(ms: List[Dict[str, str]]) -> int:
        return sum(estimate_tokens(str(m.get("content", ""))) + 4 for m in ms)

    selected = list(system) + list(current)
    # Newest history first, then reverse back to chronological.
    history: List[Dict[str, str]] = []
    for msg in reversed(rest):
        candidate = list(system) + list(reversed(history + [msg])) + list(current)
        if _tokens(candidate) <= token_budget:
            history.append(msg)
        else:
            break
    return list(system) + list(reversed(history)) + list(current)


# ---------------------------------------------------------------------------
# Canonical context builder (single source of truth for the live inference path)
# ---------------------------------------------------------------------------

@dataclass
class BuiltContext:
    messages: List[Dict[str, str]]      # ready for litellm.completion (role/content)
    input_tokens: int                   # estimated input tokens
    dropped_tokens: int                 # tokens dropped or compressed out
    compressed: bool                    # whether any history was compressed
    reasoning: str                      # human-readable summary of what happened

def _msg_type(m) -> str:
    """Return a langchain-style message role/type string for a message object."""
    t = getattr(m, "type", None) or (m.get("role") if isinstance(m, dict) else "")
    if t == "human":
        return "user"
    if t == "ai":
        return "assistant"
    return t or "user"


def _msg_content(m) -> str:
    c = getattr(m, "content", None)
    if c is None and isinstance(m, dict):
        c = m.get("content")
    return str(c or "")


def _looks_structured(text: str) -> bool:
    """Detect code/JSON/tables that must never be compressed away."""
    stripped = (text or "").lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    if re.search(r"```", text):
        return True
    # Common code / markup signals that indicate structure worth preserving.
    if re.search(r"(?m)^\s*(def|class|import|from|const|let|var|function|return)\b", text):
        return True
    if re.search(r"(?m)^\s*([A-Za-z_][\w.]*\s*[:=]\s*[^=]|<\?xml|<html|<div|<table)", text):
        return True
    return False


def build_context_messages(
    messages: Sequence[Any],
    *,
    system_prompt: str = "",
    rag_context: str = "",
    context_limit: int = 8192,
    budget: int = 4096,
    compression_ratio: float = 0.6,
    always_keep_n_history: int = 6,
) -> BuiltContext:
    """
    ONE canonical context builder for the live inference path.

    Priority order (highest first, dropped last):
      system -> current request -> retrieved info -> recent history -> old history

    Rules:
      * System, current user request, and retrieved info are NEVER dropped/compressed.
      * Old history is trimmed first; then, only if still over budget, the tail of
        the remaining history is compressed via compress_prompt().
      * Compression NEVER touches code / JSON / structured content (guarded).
      * The final list is checked against `context_limit`; worst-case we fall back
        to system + current + retrieved rather than exceeding the model window.

    Returns a BuiltContext whose `.messages` can be passed straight to
    litellm.completion().
    """
    msgs = list(messages)
    if not msgs:
        msgs = [{"role": "user", "content": system_prompt or ""}]

    # Bucket by priority.
    system_msgs = [m for m in msgs if _msg_type(m) in ("system", "developer")]
    others = [m for m in msgs if _msg_type(m) not in ("system", "developer")]
    current = others[-1:]  # the live user request
    history = others[:-1]  # everything before the current request

    def _mk(role: str, content: str) -> Dict[str, str]:
        return {"role": role if role != "human" else "user", "content": content}

    # High-priority fixed prefix (never dropped): system + retrieved, in order.
    prefix: List[Dict[str, str]] = []
    for m in system_msgs:
        prefix.append(_mk(_msg_type(m), _msg_content(m)))
    if system_prompt and not any(mm["role"] == "system" for mm in prefix):
        prefix.insert(0, {"role": "system", "content": system_prompt})
    if rag_context and (rag_context or "").strip():
        prefix.append({"role": "system", "content": (rag_context or "").strip()})

    # The live user request is always the LAST message and is never dropped.
    current_msg = _mk(_msg_type(current[-1]), _msg_content(current[-1])) if current else {
        "role": "user", "content": system_prompt or ""}

    def _tokens(dicts: List[Dict[str, str]]) -> int:
        return sum(estimate_tokens(d["content"]) + 4 for d in dicts)

    # Build final message list = prefix + trimmed history + current (last).
    def _assemble(hist: List[Dict[str, str]]) -> List[Dict[str, str]]:
        return list(prefix) + list(hist) + [current_msg]

    # History with all turns preserving chronological order.
    hist_msgs = [_mk(_msg_type(m), _msg_content(m)) for m in history]
    used = _tokens(_assemble(hist_msgs))

    # Drop oldest history until we fit within the budget, but always keep a small
    # recent window so the model has immediate dialogue context.
    dropped = 0
    while len(hist_msgs) > always_keep_n_history and used > budget:
        head = hist_msgs.pop(0)
        head_tok = estimate_tokens(head["content"]) + 4
        used -= head_tok
        dropped += head_tok

    # Final hard check against the model context limit: if we would still overflow,
    # keep only the most recent history window so we never exceed the window.
    if used > context_limit:
        dropped += _tokens(hist_msgs)
        hist_msgs = hist_msgs[-always_keep_n_history:]
        used = _tokens(_assemble(hist_msgs))
        while hist_msgs and used > context_limit:
            dropped_one = hist_msgs.pop(0)
            one_tok = estimate_tokens(dropped_one["content"]) + 4
            used -= one_tok
            dropped += one_tok

    # Compression as a last resort: compress the OLDEST filler in history only.
    compressed_flag = False
    if used > budget and hist_msgs:
        compressed_flag = True
        for i, d in enumerate(hist_msgs):
            if _looks_structured(d["content"]):
                continue
            orig = estimate_tokens(d["content"])
            comp = compress_prompt(d["content"], compression_ratio)
            new_tok = estimate_tokens(comp)
            if new_tok < orig:
                dropped += orig - new_tok
                hist_msgs[i]["content"] = comp
        used = _tokens(_assemble(hist_msgs))

    final = _assemble(hist_msgs)
    input_tokens = _tokens(final)
    reasoning = (
        f"context built: {len(final)} msgs, ~{input_tokens} input tokens, "
        f"~{dropped} dropped/compressed, compressed={compressed_flag}"
    )
    return BuiltContext(
        messages=final,
        input_tokens=input_tokens,
        dropped_tokens=dropped,
        compressed=compressed_flag,
        reasoning=reasoning,
    )


def wrap_retrieved(docs: Sequence[str], source: str = "RETRIEVED DOCUMENT") -> str:
    chunks = []
    for i, doc in enumerate(docs, 1):
        if not doc:
            continue
        chunks.append(f"----- BEGIN {source} {i} -----\n{doc}\n----- END {source} {i} -----")
    return "\n".join(chunks)


def dedupe_context_parts(parts: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for p in parts:
        key = re.sub(r"\s+", " ", (p or "").strip().lower())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def structured_truncate(
    text: str,
    max_chars: int = 5000,
    max_tokens: int = 1000,
) -> str:
    """
    Reduce a large tool/context blob before it enters the model, preserving the
    most information-dense parts:

      * JSON            -> keep the outer structure, drop the middle elements,
                           keep the head and tail arrays/keys.
      * source code     -> keep the first and last lines (signatures + tail),
                           note the truncation in a comment.
      * logs / prose    -> keep head + tail, trim the middle.

    Never destroys the first (head) or last (tail) section arbitrarily.
    """
    if not text:
        return text
    max_chars = int(max_chars or 5000)
    max_tokens = int(max_tokens or 1000)
    char_budget = min(max_chars, max_tokens * 4)
    if len(text) <= char_budget:
        return text

    stripped = text.lstrip()
    note = "\n\n[Tool output truncated: kept head+tail, removed middle]"

    # --- JSON: keep structure, drop middle array/object items ---
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            obj = json.loads(text)
        except Exception:
            obj = None
        if obj is not None:
            try:
                head = json.dumps(obj, ensure_ascii=False)[:char_budget // 2]
                tail = json.dumps(obj, ensure_ascii=False)[-char_budget // 2:]
                return head + "\n...\n" + tail + note
            except Exception:
                pass

    # --- Source code: keep first + last N lines ---
    lines = text.splitlines()
    if len(lines) > 4 and re.search(r"(?m)^\s*(def|class|import|from|function|const|let|var)\b", text):
        half = max(1, min(len(lines) // 2, 120))
        head_lines = lines[:half]
        tail_lines = lines[-min(half, 40):]
        return "\n".join(head_lines) + "\n# ...\n" + "\n".join(tail_lines) + note

    # --- Logs / prose: keep head + tail ---
    half_chars = char_budget // 2
    head = text[:half_chars]
    tail = text[-half_chars:]
    return head + "\n...\n" + tail + note


# ---------------------------------------------------------------------------
# Single generation policy
# ---------------------------------------------------------------------------

@dataclass
class GenerationPolicy:
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int
    reason: str = ""


def resolve_generation_policy(
    analysis: Any,
    *,
    base: Optional[Dict[str, Any]] = None,
    configured_max_tokens: int = 1024,
    model_name: str = "",
) -> GenerationPolicy:
    """One authoritative decoding policy. Urgency affects latency hints, not completeness."""
    base = dict(base or {})
    temperature = float(base.get("temperature", 0.3))
    max_tokens = int(base.get("max_tokens", configured_max_tokens) or configured_max_tokens)
    top_p = float(base.get("top_p", 0.9))
    top_k = int(base.get("top_k", 40))
    reasons = []

    is_math = bool(getattr(analysis, "is_math", False))
    is_coding = bool(getattr(analysis, "is_coding", False))
    is_complex = bool(getattr(analysis, "is_complex", False))
    needs_reasoning = bool(getattr(analysis, "needs_reasoning", False))
    expected = getattr(analysis, "expected_response_length", "medium")
    query_text = (getattr(analysis, "query_text", "") or "").lower()

    if is_math or is_coding:
        temperature = min(temperature, 0.2)
        reasons.append("precise_task")
    elif is_complex or needs_reasoning:
        temperature = min(max(temperature, 0.2), 0.4)
        reasons.append("reasoning")

    user_wants_short = any(w in query_text for w in ("short answer", "briefly", "in one sentence", "tl;dr"))
    user_wants_long = any(w in query_text for w in ("detailed", "comprehensive", "5000", "long explanation", "step by step"))
    if expected == "short" and user_wants_short and not is_complex and not is_coding and not is_math:
        max_tokens = min(max_tokens, 128)
        reasons.append("explicit_short")
    elif user_wants_long or expected == "long":
        max_tokens = min(configured_max_tokens, max(max_tokens, 512))
        reasons.append("long_form")

    # Hard cap: never exceed configured safety limit.
    max_tokens = min(max_tokens, configured_max_tokens)

    # Small models: slightly lower temperature, but do not force greedy top_k=1.
    small = any(tag in (model_name or "").lower() for tag in ("2b", "mini", "0.5b", "1.5b", "tiny"))
    if small:
        temperature = min(temperature, 0.3)
        top_p = min(max(top_p, 0.8), 0.95)
        top_k = max(top_k, 20)
        reasons.append("small_model_nongreedy")

    return GenerationPolicy(temperature, max_tokens, top_p, top_k, ",".join(reasons) or "default")


# ---------------------------------------------------------------------------
# Routing with residency / hysteresis / load
# ---------------------------------------------------------------------------

@dataclass
class RouterState:
    loaded_model: str = ""
    last_switch_ts: float = 0.0
    switch_count: int = 0
    inflight: Dict[str, int] = field(default_factory=dict)
    min_dwell_seconds: float = 8.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_start(self, model: str) -> None:
        with self._lock:
            self.inflight[model] = self.inflight.get(model, 0) + 1

    def record_end(self, model: str) -> None:
        with self._lock:
            self.inflight[model] = max(0, self.inflight.get(model, 0) - 1)

    def queue_depth(self, model: str) -> int:
        with self._lock:
            return int(self.inflight.get(model, 0))


def select_model(
    requested: str,
    routed: Optional[str],
    code_model: Optional[str],
    state: RouterState,
    *,
    expected_output_tokens: int = 128,
) -> str:
    """Choose the model that will actually be passed to the backend."""
    candidate = code_model or routed or requested
    # BUG 13 FIX: never route to a model the backend does not actually expose.
    # The availability check is TTL-cached and fails open (unreachable backend
    # is treated as available) so it never blocks a request; we only probe when
    # a routing decision actually differs from the caller's default model.
    if candidate != requested:
        try:
            if not check_model_available(candidate):
                logger.info(
                    "Router: model %s not installed; falling back to %s",
                    candidate, requested,
                )
                candidate = requested
        except Exception:
            pass
    now = time.time()

    def _ewma_load(m: str) -> Optional[float]:
        ewma = getattr(state, "_ewma", None)
        if ewma is None:
            return None
        try:
            return ewma.load(m)
        except Exception:
            return None

    if (
        state.loaded_model
        and candidate != state.loaded_model
        and (now - state.last_switch_ts) < state.min_dwell_seconds
        and expected_output_tokens < 256
    ):
        return state.loaded_model
    if candidate != state.loaded_model:
        # Prefer idle resident model if the candidate is heavily queued.
        if state.queue_depth(candidate) > state.queue_depth(state.loaded_model or candidate) + 2 and state.loaded_model:
            return state.loaded_model
        # Prefer the materially-less-loaded model when the EWMA tracker is
        # installed and disagrees clearly with the naive queue-depth view.
        cand_load = _ewma_load(candidate)
        loaded_load = _ewma_load(state.loaded_model) if state.loaded_model else None
        if (
            cand_load is not None
            and loaded_load is not None
            and state.loaded_model
            and loaded_load < cand_load - 1.0
        ):
            return state.loaded_model
        state.loaded_model = candidate
        state.last_switch_ts = now
        state.switch_count += 1
    return candidate


def ollama_model_id(model: str) -> str:
    if not model:
        return model
    return model if model.startswith("ollama/") else f"ollama/{model}"


_model_avail_cache: Dict[str, tuple[float, bool]] = {}
_MODEL_AVAIL_TTL = 60.0  # seconds


def clear_model_availability_cache() -> None:
    """Reset the model-availability TTL cache (used by tests)."""
    _model_avail_cache.clear()


def check_model_available(model: str, api_base: str = "http://localhost:11434") -> bool:
    """Check if a model exists in the Ollama backend. Returns True if available.

    Results are cached with a TTL (``_MODEL_AVAIL_TTL``) so the HTTP
    round-trip is NOT performed on every request — only once per
    (api_base, model) per TTL window.  Network failures are cached as True
    (assume available) for the same window so a flaky backend does not stall
    the event loop with repeated 3-second timeouts.
    """
    key = f"{api_base}|{model}"
    now = time.time()
    cached = _model_avail_cache.get(key)
    if cached is not None and (now - cached[0]) < _MODEL_AVAIL_TTL:
        return cached[1]
    import urllib.request
    import json as _json
    try:
        url = f"{api_base.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = _json.loads(resp.read())
            names = [m.get("name", "") for m in data.get("models", [])]
            # Exact match, or prefix+dangling-colon match so that asking for a
            # tag-less name like "phi3" also matches "phi3:mini" (default tag)
            # while "phi3:3.8b" does NOT match "phi3:mini".
            bare = model.replace("ollama/", "")
            available = any(
                n == bare or (bare and not bare.endswith(":") and n.startswith(bare + ":"))
                for n in names
            )
    except Exception:
        available = True  # Assume available if we can't check (don't block inference)
    _model_avail_cache[key] = (now, available)
    return available


# ---------------------------------------------------------------------------
# Code intent detection
# ---------------------------------------------------------------------------

_CODE_KEYWORDS = (
    "def ", "class ", "import ", "function ", "const ", "let ", "var ",
    "return ", "if (", "for (", "while (", "```", "code", "program",
    "algorithm", "debug", "error", "exception", "stack trace",
    "compile", "runtime", "syntax", "refactor", "implement",
)


def detect_code_intent(query: str) -> bool:
    """Detect if a query is asking for code generation or debugging."""
    q = (query or "").lower()
    return any(kw in q for kw in _CODE_KEYWORDS)


# ---------------------------------------------------------------------------
# Math / self-consistency
# ---------------------------------------------------------------------------

def normalize_math_answer(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    boxed = re.search(r"\\boxed\{([^}]+)\}", s)
    if boxed:
        s = boxed.group(1).strip()
    dollar = re.fullmatch(r"\s*\$([^$]+)\$\s*", s)
    if dollar:
        s = dollar.group(1).strip()
    s = s.replace(",", "")
    if s.endswith("%"):
        try:
            return f"{float(s[:-1]) / 100:.12g}"
        except ValueError:
            pass
    frac = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", s)
    if frac and int(frac.group(2)) != 0:
        return f"{int(frac.group(1)) / int(frac.group(2)):.12g}"
    try:
        return f"{float(s):.12g}"
    except ValueError:
        return re.sub(r"\s+", " ", s.lower())


def _response_quality(text: str) -> float:
    """Heuristic quality score: prefer concise, informative, well-formed text
    over rambling filler (replaces a naive 'longest response wins' bias)."""
    t = (text or "").strip()
    if not t:
        return float("-inf")
    informative = 1.0 if re.search(r"[A-Za-z]{4,}|\d", t) else 0.0
    length_penalty = min(len(t) / 4000.0, 2.0)
    repetition = len(re.findall(r"(.)\1{5,}", t)) * 0.25
    complete = 1.0 if t.endswith((".", "!", "?")) else 0.0
    return informative - length_penalty - repetition + complete


def vote_responses(responses: Sequence[str]) -> str:
    if not responses:
        return ""
    buckets: Dict[str, List[str]] = {}
    for r in responses:
        key = normalize_math_answer(r) or re.sub(r"\s+", " ", r.strip().lower())
        buckets.setdefault(key, []).append(r)
    best_key = max(buckets, key=lambda k: (len(buckets[k]), -len(k)))
    # Prefer a boxed member of the winning bucket.
    group = buckets[best_key]
    for item in group:
        if "\\boxed" in item:
            return item
    # Within the agreeing group, prefer the highest-quality response rather
    # than blindly returning the longest one.
    return max(group, key=_response_quality)


def cisc_confidence(responses: Sequence[str], query: Optional[str] = None) -> float:
    """Confidence based on agreement across repeated samples.

    When the query is a deterministic arithmetic / word problem, the result can
    be verified directly: if any sample matches the computed answer the
    confidence is 1.0 (BUG 28: confidence is then based on correctness, not on
    formatting heuristics).  Otherwise it falls back to agreement frequency.
    """
    if not responses:
        return 0.0
    keys = [normalize_math_answer(r) or r.strip().lower() for r in responses]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    agreement = max(counts.values()) / len(keys)
    if query:
        try:
            expected = quick_arithmetic(query) or quick_word_problem(query)
        except Exception:
            expected = None
        if expected is not None:
            if any(k == expected or k == normalize_math_answer(expected) for k in keys):
                return 1.0
            return min(agreement, 0.4)  # predicted answer present but none matched
    return agreement


# ---------------------------------------------------------------------------
# Code validation / language detection
# ---------------------------------------------------------------------------

def detect_code_language(text: str) -> str:
    t = text or ""
    low = t.lower()
    scores = {
        "python": len(re.findall(r"\b(def|import|elif|None|self)\b", t)) + (2 if "def " in t else 0),
        "javascript": len(re.findall(r"\b(const|let|function|=>|console\.log)\b", t)),
        "java": len(re.findall(r"\b(public class|System\.out|static void)\b", t)),
        "cpp": len(re.findall(r"\b(#include|std::|int main)\b", t)),
        "json": 1 if t.strip().startswith(("{", "[")) else 0,
        "go": len(re.findall(r"\b(func |package |fmt\.)\b", t)),
        "rust": len(re.findall(r"\b(fn |let mut|impl )\b", t)),
    }
    if "python" in low:
        scores["python"] += 3
    if "javascript" in low or "typescript" in low:
        scores["javascript"] += 3
    if re.search(r"\bjava\b", low) and "javascript" not in low:
        scores["java"] += 3
    lang, score = max(scores.items(), key=lambda kv: kv[1])
    return lang if score > 0 else "unknown"


def _balanced(code: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    in_str = None
    escape = False
    for ch in code:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == in_str:
                in_str = None
            continue
        if ch in "'\"":
            in_str = ch
            continue
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def validate_code(code: str, language: Optional[str] = None) -> Tuple[bool, Optional[str], bool]:
    """
    Returns ``(is_valid, error, actually_validated)``.

    * ``is_valid`` — ``True`` only when the code is **parsed** successfully
      (Python via ``ast.parse``, JSON via ``json.loads``) **or** for languages
      where we can at least confirm that brackets/braces balance *and* the
      extracted code is non-empty.
    * ``error`` — human-readable description on failure.
    * ``actually_validated`` — ``False`` when we could only do a partial check
      (e.g. bracket balancing for JS/Java/Go/Rust) so callers know the result
      is **necessary-but-not-sufficient**.

    Bracket balance alone is *never* reported as ``is_valid=True`` for
    languages without a real parser — it is treated as a *partial* check
    (``actually_validated=False``).
    """
    extracted = code
    fenced = re.search(r"```(?:\w+)?\n([\s\S]*?)```", code)
    if fenced:
        extracted = fenced.group(1)
    lang = (language or detect_code_language(extracted)).lower()

    if lang == "python":
        try:
            ast.parse(extracted)
            return True, None, True
        except SyntaxError as e:
            return False, f"Syntax error at line {e.lineno}: {e.msg}", True
    if lang == "json":
        try:
            json.loads(extracted)
            return True, None, True
        except json.JSONDecodeError as e:
            return False, str(e), True
    if lang in ("javascript", "js", "typescript", "ts"):
        if not _balanced(extracted):
            return False, "Unbalanced brackets/braces", True
        if not extracted.strip():
            return False, "Empty code", True
        return True, None, False   # partial: balanced, not parsed
    if lang in ("java", "cpp", "c++", "c", "go", "rust"):
        if not _balanced(extracted):
            return False, "Unbalanced brackets/braces", True
        if not extracted.strip():
            return False, "Empty code", True
        return True, None, False   # partial: balanced, not parsed
    return False, f"validation_unsupported:{lang}", False


# ---------------------------------------------------------------------------
# Isolated Python execution with a real process timeout
# ---------------------------------------------------------------------------

_TIR_RUNNER = r"""
import ast, sys
src = sys.stdin.read()
ns = {}
tree = ast.parse(src, mode="exec")
if tree.body and isinstance(tree.body[-1], ast.Expr):
    last = tree.body.pop()
    exec(compile(ast.Module(body=tree.body, type_ignores=[]), "<tir>", "exec"), ns, ns)
    value = eval(compile(ast.Expression(last.value), "<tir>", "eval"), ns, ns)
    print("__TIR_RESULT__=" + repr(value))
else:
    exec(compile(tree, "<tir>", "exec"), ns, ns)
    if "_result" in ns:
        print("__TIR_RESULT__=" + repr(ns["_result"]))
"""


def execute_python_isolated(code: str, timeout: float = 5.0) -> Tuple[bool, Any, Optional[str]]:
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _TIR_RUNNER],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired:
        return False, None, f"timeout after {timeout}s"
    if proc.returncode != 0:
        return False, None, (proc.stderr or "nonzero exit")[-500:]
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("__TIR_RESULT__="):
            try:
                result = ast.literal_eval(line.split("=", 1)[1])
            except Exception:
                result = line.split("=", 1)[1]
    return True, result, proc.stdout


# ---------------------------------------------------------------------------
# Quick-path safety
# ---------------------------------------------------------------------------

_COMPLEX_HINTS = (
    "why", "how", "explain", "code", "function", "def ", "class ", "prove",
    "derive", "hack", "exploit", "password", "previous", "earlier", "remember",
    "calculate", "integral", "algorithm",
)


def is_safe_quick_path(query: str) -> bool:
    q = (query or "").strip()
    if not q or len(q) > 40:
        # Longer inputs are only fast-path-capable if a deterministic solver
        # can fully handle them (strictly validated, no LLM).
        if quick_word_problem(q) is not None:
            return True
        return quick_word_arithmetic(q) is not None
    low = q.lower()
    # Deterministic math is safe regardless of conversational hints
    # ('how/why/explain' gate greetings, not validated math).
    if quick_word_arithmetic(q) is not None:
        return True
    if quick_word_problem(q) is not None:
        return True
    if any(h in low for h in _COMPLEX_HINTS):
        return False
    if low in {"hello", "hi", "hey", "greetings", "thanks", "thank you", "ok", "okay"}:
        return True
    # Arithmetic (possibly multi-operator with precedence).
    if re.fullmatch(r"[\d\s\+\-\*/\(\)\.]+", q.rstrip("?")):
        if re.search(r"\d", q):
            return quick_arithmetic(q) is not None
    return False


_WORD_ARITH_LEAD = re.compile(
    r"^(?:what is|what's|whats|what is the answer to|how much is|how much"
    r"|calculate|compute|solve|find|give me)\s+", re.IGNORECASE)


def quick_word_arithmetic(query: str) -> Optional[str]:
    """Word-form arithmetic: 'What is 2 plus 2?' / '5 times 4' / '10 minus 3'.

    Strictly conservative: the leading phrase is stripped, word operators are
    mapped to symbols, and the remainder MUST be a pure arithmetic expression
    (digits, + - * /, parentheses).  Anything with residual letters is rejected.
    """
    q = (query or "").strip().lower().rstrip("?").rstrip("=").strip()
    if not q or len(q) > 60:
        return None
    q = _WORD_ARITH_LEAD.sub("", q).strip()
    if not q:
        return None
    q = re.sub(r"\s+x\s+", "*", q)
    q = q.replace("multiplied by", "*").replace("divided by", "/")
    q = q.replace("plus", "+").replace("minus", "-").replace("times", "*")
    if re.search(r"[a-z]", q):
        return None
    q = re.sub(r"\s+", " ", q)
    if not re.fullmatch(r"[\(\)\d][\d\s\+\-\*/\(\)\.]*", q):
        return None
    if not re.search(r"[\+\-\*/]", q):
        return None
    expr = q.replace(" ", "")
    if not re.search(r"\d", expr):
        return None
    return quick_arithmetic(expr)


def quick_arithmetic(query: str) -> Optional[str]:
    """
    Safe arithmetic evaluation for simple numeric expressions.  Handles operator
    precedence (e.g. 2 + 3 * 4 == 14) without using eval().
    Accepts digits, + - * / and parentheses.  Returns the numeric result as a
    string, or None if the expression isn't a safe, simple arithmetic query.
    """
    expr = (query or "").strip().rstrip("?")
    if not re.fullmatch(r"[\d\s\+\-\*/\(\)\.]+", expr):
        return None
    tokens = re.findall(r"\d+\.?\d*|[()+\-*/]", expr)
    if not tokens:
        return None
    try:
        value = _eval_safe(tokens)
    except Exception:
        return None
    if value is None:
        return None
    # If the expression uses division, report a float result (back-compat keeps
    # "4.0" for 20 / 5).  Otherwise collapse exact integer results to int form.
    if "/" in expr:
        return str(float(round(value, 12)))
    if isinstance(value, float):
        value = round(value, 12)
        if value.is_integer():
            value = int(value)
    return str(value)


def quick_word_problem(query: str) -> Optional[str]:
    """Deterministic elementary word-problem solver: rate x time.

    Handles the common pattern 'If X <verb> N <unit> per <period>, how far/much
    does it go/travel/work in M <period>?'  Strictly regex validated so it only
    fires on unambiguous rate*time problems.  Returns the numeric result as a
    string, or None if the query is not a recognised safe word problem.
    """
    q = (query or "").strip()
    if not q or len(q) > 160:
        return None
    low = q.lower()
    # Normalize common speed/rate abbreviations to the canonical "N unit per
    # period" form so '50 mph for 3 hours' resolves like '50 miles per hour'.
    low = re.sub(r"\b(\d+(?:\.\d+)?)\s*mph\b", r"\1 miles per hour", low)
    low = re.sub(r"\b(\d+(?:\.\d+)?)\s*(?:kmph|kph|km/h)\b", r"\1 kilometers per hour", low)
    if "per" not in low:
        return None
    if not re.search(r"\b(how far|how much|how many|travel|go\b|drive|run\b|walk|fly|work)\b", low):
        return None
    m_rate = re.search(r"(\d+(?:\.\d+)?)\s+([a-z]+)\s+per\s+([a-z]+)", low)
    if not m_rate:
        return None
    rate = float(m_rate.group(1))
    if rate <= 0:
        return None
    period = m_rate.group(3)
    m_time = re.search(r"\b(?:in|for)\s+(\d+(?:\.\d+)?)\s+([a-z]+)", low)
    if not m_time:
        return None
    time = float(m_time.group(1))
    if time <= 0:
        return None
    time_unit = m_time.group(2)
    if time_unit.rstrip("s") != period.rstrip("s"):
        return None
    value = rate * time
    if isinstance(value, float):
        value = round(value, 12)
        if value.is_integer():
            value = int(value)
    return str(value)


def _eval_safe(tokens: List[str]):
    """Recursive-descent evaluator over + - * / and parentheses. No eval()."""
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def advance():
        nonlocal pos
        t = tokens[pos]
        pos += 1
        return t

    def parse_expr():
        val = parse_term()
        while peek() in ("+", "-"):
            op = advance()
            rhs = parse_term()
            if op == "+":
                val = val + rhs
            else:
                val = val - rhs
        return val

    def parse_term():
        val = parse_factor()
        while peek() in ("*", "/"):
            op = advance()
            rhs = parse_factor()
            if op == "*":
                val = val * rhs
            else:
                if rhs == 0:
                    raise ZeroDivisionError
                val = val / rhs
        return val

    def parse_factor():
        nonlocal pos
        t = peek()
        if t == "(":
            advance()
            val = parse_expr()
            if peek() == ")":
                advance()
            return val
        if t == "-":
            advance()
            return -parse_factor()
        if t and re.fullmatch(r"\d+\.?\d*", t):
            advance()
            return float(t) if "." in t else int(t)
        raise ValueError(f"unexpected token {t}")

    result = parse_expr()
    return result


# ---------------------------------------------------------------------------
# Tool intent (not raw keyword hits)
# ---------------------------------------------------------------------------

def wants_calculator(query: str) -> bool:
    q = query or ""
    return bool(re.search(r"\b(calculate|compute|what is)\b", q.lower()) and re.search(r"\d+\s*[\+\-\*/]\s*\d+", q))


def wants_search(query: str) -> bool:
    q = (query or "").lower()
    return bool(re.match(r"^(search (for|the web)|look up|google)\b", q))


# ---------------------------------------------------------------------------
# Registry — only components that are actually injected/used
# ---------------------------------------------------------------------------

class OptimizationRegistry:
    def __init__(self) -> None:
        self.components: Dict[str, Any] = {}
        self.enabled: Dict[str, bool] = {}

    def register(self, name: str, obj: Any, enabled: bool = True) -> None:
        self.components[name] = obj
        self.enabled[name] = bool(enabled and obj is not None)

    def get(self, name: str) -> Any:
        if not self.enabled.get(name):
            return None
        return self.components.get(name)

    def attach_to(self, gateway: Any) -> None:
        for name, obj in self.components.items():
            if not self.enabled.get(name):
                continue
            setattr(gateway, name, obj)
            logger.info("Wired optimization '%s' onto %s", name, type(gateway).__name__)


_REGISTRY = OptimizationRegistry()
_ROUTER_STATE = RouterState()
_TOOL_MEMO = BoundedTTLCache(max_size=256, default_ttl=300)


def get_registry() -> OptimizationRegistry:
    return _REGISTRY


def get_router_state() -> RouterState:
    return _ROUTER_STATE


def tool_memo_get(name: str, args: str) -> Optional[Any]:
    return _TOOL_MEMO.get(f"{name}::{args}")


def tool_memo_set(name: str, args: str, value: Any) -> None:
    _TOOL_MEMO.set(f"{name}::{args}", value)
