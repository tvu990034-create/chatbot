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
    now = time.time()
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
        state.loaded_model = candidate
        state.last_switch_ts = now
        state.switch_count += 1
    return candidate


def ollama_model_id(model: str) -> str:
    if not model:
        return model
    return model if model.startswith("ollama/") else f"ollama/{model}"


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
    return group[0]


def cisc_confidence(responses: Sequence[str]) -> float:
    if not responses:
        return 0.0
    keys = [normalize_math_answer(r) or r.strip().lower() for r in responses]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    return max(counts.values()) / len(keys)


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
    Returns (is_valid, error, actually_validated).
    Unsupported languages never report is_valid=True.
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
        return True, None, True
    if lang in ("java", "cpp", "c++", "c", "go", "rust"):
        if not _balanced(extracted):
            return False, "Unbalanced brackets/braces", True
        return True, None, True
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
    if not q or len(q) > 24:
        return False
    low = q.lower()
    if any(h in low for h in _COMPLEX_HINTS):
        return False
    if low in {"hello", "hi", "hey", "greetings", "thanks", "thank you", "ok", "okay"}:
        return True
    if re.fullmatch(r"\d+\s*[\+\-\*/]\s*\d+", low):
        return True
    return False


def quick_arithmetic(query: str) -> Optional[str]:
    m = re.fullmatch(r"\s*(\d+)\s*([\+\-\*/])\s*(\d+)\s*", query or "")
    if not m:
        return None
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    if op == "+":
        return str(a + b)
    if op == "-":
        return str(a - b)
    if op == "*":
        return str(a * b)
    if op == "/" and b != 0:
        return str(a / b)
    return None


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
