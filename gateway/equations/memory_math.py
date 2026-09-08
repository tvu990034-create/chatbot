"""
gateway/equations/memory_math.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Memory / context-window / attention-heuristic / token-budget equations ported
from the user's documentation.  Corresponds to "Untitled document (9)" and
"Tài liệu (29)" (memory & attention equations), "1.txt" (context-window
compaction) and "Tài liệu (8)" (token-pruning / early-exit heuristics).

What is feasible in pure Python:
  * recency-weighted context priority and per-message pruning scores
  * consolidation / rehearsal (long-term importance) heuristics
  * token-stopping / summary triggers
  * self-contained context compaction and token-budget compression
  * structural token pruning guarded against code / JSON
  * attention *importance* heuristics (structural placeholders only)
  * response-length policies, semantic turn retention, sliding windows

What is NOT feasible without a trained model / GPU kernels (marked with
NOT_IMPLEMENTABLE_* constants below): softmax attention, GQA/MQA head splits,
FlashAttention, sliding-window attention kernels, RoPE, layer-wise attention
reuse, and a MemGPT-style *trained* memory compactor.

Pure stdlib.  The only cross-module imports are ``estimate_tokens`` and
``cosine_similarity`` from ``gateway.opt_core``.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from gateway.opt_core import cosine_similarity, estimate_tokens

# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp to [lo, hi]; non-finite values resolve to a bound (no NaN)."""
    if not math.isfinite(x):
        return hi if x > 0 else lo
    return max(lo, min(hi, x))


# ---------------------------------------------------------------------------
# 1. Recency-weighted context priority
# (source: "Untitled document (9)" — keep/what-to-drop priority math)
# ---------------------------------------------------------------------------


def recency_weight(position: int, total: int, gamma: float = 0.9) -> float:
    """Eq M1 — recency weight for an item at 0-indexed ``position`` out of
    ``total``.  Newest position (``total - 1``) gets weight 1.0; every step
    backward in time decays by ``gamma``: ``gamma^(total-1-position)``.
    ("src", "Untitled document (9)")"""
    total = max(1, int(total or 1))
    position = max(0, int(position or 0))
    gamma = _clamp(float(gamma or 0.0), 0.0, 1.0)
    exponent = max(0, total - 1 - min(position, total - 1))
    return math.pow(gamma, exponent)


def priority_score(recency: float, relevance: float, saliency: float,
                   w_relevance: float = 0.6, w_saliency: float = 0.4) -> float:
    """Eq M2 — weighted blend for what to keep.  A message's retention
    priority is its recency times a relevance/saliency blend, all clamped to
    [0, 1].  Higher = keep first.
    ("src", "Untitled document (9)")"""
    wr = _clamp(float(w_relevance or 0.0), 0.0, 1.0)
    ws = _clamp(float(w_saliency or 0.0), 0.0, 1.0)
    if wr + ws <= 0.0:
        wr, ws = 0.6, 0.4
    blend = (wr * _clamp(relevance) + ws * _clamp(saliency)) / (wr + ws)
    return _clamp(_clamp(recency) * blend)


# ---------------------------------------------------------------------------
# 2. Per-message memory score / pruning window
# (source: "Untitled document (9)", "Tài liệu (29)" — memory management)
# ---------------------------------------------------------------------------

_SALIENT_MARKERS = (
    "important", "urgent", "remember", "critical", "must", "key",
    "task", "user", "please", "required", "deadline", "do not",
)


def message_priority(msg_dict: Dict, now: float, age: float,
                     mention_count: int, half_life: float = 3600.0) -> float:
    """Eq M3 — per-message memory score combining recency (exponential decay
    over ``age``), keyword saliency (imperative / user-directed words in the
    content), ``mention_count`` amplification and turn depth.  ``now`` is the
    reference clock timestamp (kept for signature symmetry with the docs).
    Higher = more worth keeping.  ("src", "Tài liệu (29)")"""
    msg = msg_dict if isinstance(msg_dict, dict) else {}
    content = str(msg.get("content", ""))
    half_life = half_life if half_life and half_life > 0 else 3600.0
    age = max(0.0, float(age or 0.0))
    recency = math.pow(0.5, age / half_life)

    low = content.lower()
    hits = sum(1 for marker in _SALIENT_MARKERS if marker in low)
    saliency = _clamp(math.log1p(hits) / math.log1p(4.0)) if hits else 0.0

    mention = _clamp(math.log1p(max(0, int(mention_count or 0))) / math.log1p(9.0))

    depth = max(0, int(msg.get("turn_depth", 0) or 0))
    depth_factor = _clamp(1.0 / (1.0 + depth))

    score = 0.40 * recency + 0.30 * saliency + 0.20 * mention + 0.10 * depth_factor
    return _clamp(score)


class MemoryWindow:
    """Eq M4 — stateful sliding memory of message dicts scored by
    :func:`message_priority`.  ``trim_to_budget`` keeps the newest message
    unconditionally and fills the remaining token budget with the highest-value
    survivors (utility eviction, not just FIFO).  Deterministic; exposes
    ``reset()`` and ``stats()``.  ("src", "Tài liệu (29)")"""

    def __init__(self, half_life: float = 3600.0):
        self.half_life = half_life if half_life and half_life > 0 else 3600.0
        self._entries: Dict[int, Dict] = {}
        self._meta: Dict[int, Dict] = {}
        self._order: Dict[int, int] = {}
        self._counter = 0
        self._dropped_total = 0

    def add(self, msg: Dict, meta: Optional[Dict] = None) -> int:
        """Add a message dict (role/content).  ``meta`` may carry ``age``
        (seconds since creation), ``mention_count``, ``turn_depth`` and
        ``tokens`` (overrides the token estimate)."""
        self._counter += 1
        eid = self._counter
        self._entries[eid] = msg if isinstance(msg, dict) else {}
        self._meta[eid] = dict(meta or {})
        self._order[eid] = self._counter
        return eid

    def _tokens(self, msg: Dict, meta: Dict) -> int:
        explicit = meta.get("tokens")
        if explicit is not None:
            return max(0, int(explicit))
        return estimate_tokens(str(msg.get("content", ""))) + 4

    def _score(self, eid: int) -> float:
        msg, meta = self._entries[eid], self._meta[eid]
        return message_priority(
            msg, now=0.0,
            age=float(meta.get("age", 0.0) or 0.0),
            mention_count=int(meta.get("mention_count", 0) or 0),
            half_life=self.half_life,
        )

    def sorted_by_value(self, n: int) -> List[Dict]:
        """Return the top-``n`` message dicts by priority, descending."""
        n = max(0, int(n or 0))
        ranked = sorted(self._entries, key=self._score, reverse=True)
        return [self._entries[eid] for eid in ranked[:n]]

    def trim_to_budget(self, token_budget: int) -> List[Dict]:
        """Drop the lowest-value messages until the window fits ``token_budget``,
        always keeping the newest message.  Returns the list of dropped message
        dicts (mutating this window's state only)."""
        budget = max(0, int(token_budget or 0))
        if not self._entries:
            return []
        ordered_ids = sorted(self._entries, key=lambda e: self._order[e])
        newest_id = ordered_ids[-1]
        used = self._tokens(self._entries[newest_id], self._meta[newest_id])
        kept = [newest_id]
        ranked = sorted((e for e in self._entries if e != newest_id),
                        key=self._score, reverse=True)
        for eid in ranked:
            cost = self._tokens(self._entries[eid], self._meta[eid])
            if used + cost <= budget:
                kept.append(eid)
                used += cost
        kept_set = set(kept)
        dropped_ids = [eid for eid in list(self._entries) if eid not in kept_set]
        dropped_msgs = [self._entries[eid] for eid in dropped_ids]
        for eid in dropped_ids:
            self._dropped_total += 1
            del self._entries[eid]
            del self._meta[eid]
            del self._order[eid]
        return dropped_msgs

    def reset(self) -> None:
        self._entries.clear()
        self._meta.clear()
        self._order.clear()
        self._counter = 0
        self._dropped_total = 0

    def stats(self) -> Dict:
        total_tokens = sum(
            self._tokens(self._entries[e], self._meta[e]) for e in self._entries)
        newest_content = ""
        if self._entries:
            newest_id = max(self._entries, key=lambda e: self._order[e])
            newest_content = str(self._entries[newest_id].get("content", ""))
        return {
            "count": len(self._entries),
            "tokens_used": total_tokens,
            "dropped_total": self._dropped_total,
            "newest_content": newest_content,
        }


# ---------------------------------------------------------------------------
# 3. Consolidation (memory consolidation / rehearsal)
# (source: "Tài liệu (29)" — long-term memory formation)
# ---------------------------------------------------------------------------


def consolidation_score(detail: float, age: float, revisit_count: int,
                        half_life: float = 86400.0) -> float:
    """Eq M5 — consolidation score: older, frequently-revisited memories are
    more important for long-term retention.  Age saturates toward 1 over one
    day; revisits enter log-scaled.  Higher = strengthens into long-term.
    ("src", "Tài liệu (29)")"""
    detail = _clamp(detail)
    half_life = half_life if half_life and half_life > 0 else 86400.0
    age = max(0.0, float(age or 0.0))
    age_factor = 1.0 - math.pow(0.5, age / half_life)
    revisit = _clamp(math.log1p(max(0, int(revisit_count or 0))) / math.log1p(10.0))
    return _clamp(detail * (0.55 + 0.30 * age_factor + 0.15 * revisit))


def exponential_rehearsal(detail: float, revisit_count: int) -> float:
    """Eq M6 — exponential rehearsal: each revisit multiplicatively strengthens
    a memory's retention.  Deterministic, bounded [0, detail * 2].
    ("src", "Tài liệu (29)")"""
    detail = _clamp(detail)
    r = max(0, int(revisit_count or 0))
    boost = min(2.0, 1.0 + 0.5 * math.log1p(r))
    return _clamp(detail * boost)


# ---------------------------------------------------------------------------
# 4. Token-stopping / summary triggers
# (source: "1.txt" — context-window compaction)
# ---------------------------------------------------------------------------


def should_summarize(total_tokens: float, budget: float,
                     threshold_frac: float = 0.7) -> bool:
    """Eq M7 — trigger summarization once the current context token count
    reaches ``threshold_frac`` of the budget.  Empty/degenerate budgets resolve
    to "summarize if anything is pending".  ("src", "1.txt")"""
    total_tokens = max(0.0, float(total_tokens or 0.0))
    frac = _clamp(float(threshold_frac or 0.7), 0.0, 1.0)
    try:
        budget = float(budget)
    except (TypeError, ValueError):
        return total_tokens > 0.0
    if budget <= 0.0:
        return total_tokens > 0.0
    return total_tokens >= frac * budget


def truncate_decay_stop(sequence: Sequence[Tuple[float, int]], budget: int,
                        decay: float, min_margin: float = 0.05) -> Tuple[List, List]:
    """Eq M8 — drop the oldest ``(value, tokens)`` items from ``sequence``
    while over ``budget``, but STOP early once the next drop's marginal value
    (value * ``decay``^k) falls below ``min_margin`` times the first item's
    value — the remaining overflow is cheaper than the information loss.
    Returns ``(remaining, dropped)``; empty sequence -> ``([], [])``.
    ("src", "1.txt")"""
    budget = max(0, int(budget or 0))
    decay = _clamp(float(decay or 0.0), 0.0, 1.0)
    min_margin = _clamp(float(min_margin or 0.0), 0.0, 1.0)
    remaining: List[Tuple[float, int]] = [
        (float(v), max(0, int(t))) for v, t in (sequence or [])]
    if not remaining:
        return [], []

    def _tokens(items: Sequence[Tuple[float, int]]) -> int:
        return sum(t for _, t in items)

    if _tokens(remaining) <= budget:
        return remaining, []
    first_value = abs(remaining[0][0])
    dropped: List[Tuple[float, int]] = []
    k = 0
    while remaining and _tokens(remaining) > budget:
        value, _t = remaining[0]
        marginal = abs(value) * math.pow(decay, k)
        if first_value > 0.0 and marginal < min_margin * first_value:
            break
        dropped.append(remaining.pop(0))
        k += 1
    return remaining, dropped


# ---------------------------------------------------------------------------
# 5. Context compaction  (self-contained mirror of opt_core.build_context_messages)
# (source: "1.txt" — context-window compaction)
# ---------------------------------------------------------------------------


def _head_tail_slice(text: str, budget_tokens: int) -> str:
    """Best-effort head+tail character slice with an ellipsis marker."""
    if budget_tokens <= 0:
        return ""
    chars = max(8, int(budget_tokens) * 4)
    if len(text) <= chars:
        return text
    half = chars // 2
    head = text[:half].rstrip()
    tail = text[-half:].lstrip()
    return (head + " ... " + tail).strip()


def is_structured_text(text: str) -> bool:
    """Eq M9 — guard: detect code / JSON / markup that must never be pruned or
    compressed.  Self-contained mirror of ``opt_core._looks_structured``.
    ("src", "1.txt")"""
    stripped = (text or "").lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    if re.search(r"```", text or ""):
        return True
    if re.search(r"(?m)^\s*(def|class|import|from|const|let|var|function|return)\b",
                 text or ""):
        return True
    if re.search(
            r"(?m)^\s*([A-Za-z_][\w.]*\s*[:=]\s*[^=]|<\?xml|<html|<div|<table)",
            text or ""):
        return True
    return False


def compress_to_token_budget(text: str, budget_tokens: int) -> str:
    """Eq M10 — reduce ``text`` to fit ``budget_tokens``.  Structured content
    (code / JSON) is only whitespace-squeezed, never shredded.  Prose keeps its
    tail sentence, fills from the head, and falls back to a head+tail slice.
    Empty/zero budgets return a safe default.  ("src", "1.txt")"""
    if not text:
        return text
    try:
        budget = int(budget_tokens)
    except (TypeError, ValueError):
        budget = int(budget_tokens) if budget_tokens else 0
    if budget <= 0:
        return ""
    if estimate_tokens(text) <= budget:
        return text
    if is_structured_text(text):
        squeezed = re.sub(r"[ \t]+", " ", text)
        squeezed = re.sub(r"\n{3,}", "\n\n", squeezed)
        return squeezed.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) <= 2:
        return _head_tail_slice(text, budget)
    head, tail = sentences[0], sentences[-1]
    base = estimate_tokens(head) + estimate_tokens(tail)
    if base > budget:
        return _head_tail_slice(text, budget)
    mids: List[str] = []
    used = base
    for sent in sentences[1:-1]:
        cost = estimate_tokens(sent)
        if used + cost <= budget:
            mids.append(sent)
            used += cost
    result = " ".join([head] + mids + [tail]).strip()
    if estimate_tokens(result) < estimate_tokens(text):
        return result
    return _head_tail_slice(text, budget)


class ContextCompactor:
    """Eq M11 — context-window compaction as a pure function over a list of
    ``{role, content}`` dicts.  Mirrors ``opt_core.build_context_messages``
    (system + current request never dropped; oldest history trimmed first;
    oldest non-structured filler compressed as a last resort) but is fully
    self-contained.  ("src", "1.txt")"""

    def __init__(self, compression_ratio: float = 0.5,
                 always_keep_n_history: int = 4):
        self.compression_ratio = _clamp(float(compression_ratio or 0.5), 0.05, 1.0)
        self.always_keep_n_history = max(1, int(always_keep_n_history or 4))

    def compact(self, messages: Sequence[Dict], budget: int,
                keep_ratio: float = 0.6) -> Tuple[List[Dict], int, str]:
        """Compress ``messages`` toward ``budget`` tokens.  Returns
        ``(compact_msgs, dropped_tokens, summary_hint)``.  ``keep_ratio``
        controls how aggressively per-message filler is compressed (0, 1].
        Never mutates the caller's message dicts."""
        budget = max(0, int(budget or 0))
        keep_ratio = _clamp(float(keep_ratio or 0.0), 0.05, 1.0)
        msgs = [m for m in (messages or [])
                if isinstance(m, dict) and "content" in m]
        if not msgs:
            return ([], 0, "empty: nothing to compact")

        system = [m for m in msgs if str(m.get("role", "")).strip().lower()
                  in ("system", "developer")]
        current = msgs[-1]
        history = [
            dict(m) for m in msgs[:-1]
            if str(m.get("role", "")).strip().lower() not in ("system", "developer")
        ]

        def _tokens(dicts: Sequence[Dict]) -> int:
            return sum(estimate_tokens(str(d.get("content", ""))) + 4 for d in dicts)

        def _assemble(hist: List[Dict]) -> List[Dict]:
            return list(system) + list(hist) + [current]

        dropped = 0
        used = _tokens(_assemble(history))
        # 1) Trim oldest history until we fit, always keeping a recent window.
        while len(history) > self.always_keep_n_history and used > budget:
            head = history.pop(0)
            dropped += estimate_tokens(str(head.get("content", ""))) + 4
            used = _tokens(_assemble(history))

        # 2) Compress oldest non-structured filler as a last resort.
        if used > budget and history:
            for i, msg in enumerate(history):
                if is_structured_text(str(msg.get("content", ""))):
                    continue
                content = str(msg.get("content", ""))
                orig = estimate_tokens(content) + 4
                target = max(4, int(orig * keep_ratio))
                comp = compress_to_token_budget(content, target)
                new_tok = estimate_tokens(comp) + 4
                if new_tok < orig:
                    dropped += orig - new_tok
                    history[i] = {**msg, "content": comp}
            used = _tokens(_assemble(history))

        # 3) Final hard check: drop oldest remaining history rather than overflow.
        while history and _tokens(_assemble(history)) > budget:
            head = history.pop(0)
            dropped += estimate_tokens(str(head.get("content", ""))) + 4

        compact_msgs = _assemble(history)
        final_tokens = _tokens(compact_msgs)
        hint = (
            f"kept {len(compact_msgs)} msgs (~{final_tokens} tok); "
            f"dropped/compressed ~{dropped} tok; trimmed oldest history, "
            f"compressed filler"
        )
        return compact_msgs, dropped, hint


# ---------------------------------------------------------------------------
# 6. Attention / importance heuristics  (structural placeholders only)
# (source: "Untitled document (9)", token heuristics: "Tài liệu (8)")
# ---------------------------------------------------------------------------


def attention_importance(token_count: int, position: int, score: float) -> float:
    """Eq M12 — small attention heuristic: importance prefers salient scores,
    longer (information-dense) spans and earlier window positions.  This is a
    structural proxy — no real attention weights exist in pure Python.
    ("src", "Untitled document (9)")"""
    tok_norm = _clamp(math.log1p(max(0, int(token_count or 0))) / math.log1p(512.0))
    position = max(0, int(position or 0))
    pos_norm = _clamp(1.0 / (1.0 + position / 64.0))
    return _clamp(0.4 * _clamp(score) + 0.4 * tok_norm + 0.2 * pos_norm)


def token_importance(logit_magnitude: float, position: int, sep: float = 0.0) -> float:
    """Eq M13 — placeholder attention-importance for a single token, mapping an
    (unknown) logit magnitude through a logistic centered at the separator
    threshold ``sep`` and mildly discounting deep positions.  Provenance:
    "Tài liệu (8)" token-level early-exit heuristics — no real attention
    weights are available in pure Python.  ("src", "Tài liệu (8)")"""
    magnitude = abs(float(logit_magnitude or 0.0))
    sep = float(sep or 0.0)
    try:
        logistic = 1.0 / (1.0 + math.exp(-(magnitude - sep)))
    except OverflowError:
        logistic = 1.0 if magnitude > sep else 0.0
    position = max(0, int(position or 0))
    recency_nudge = math.pow(0.97, position)
    return _clamp(logistic * (0.5 + 0.5 * recency_nudge))


# ---------------------------------------------------------------------------
# 7. Token pruning heuristic (purely structural, no learned model)
# (source: "Tài liệu (8)" — token pruning)
# ---------------------------------------------------------------------------

_DEFAULT_FILLER = frozenset({
    "the", "a", "an", "and", "or", "but", "of", "for", "on", "with",
    "at", "by", "from", "as", "to", "in", "is", "are", "was", "were",
    "be", "it", "this", "that", "very", "really", "just", "quite",
})


def prune_filler_tokens(text: str, filler_words: Optional[Sequence[str]] = None) -> str:
    """Eq M14 — structurally remove stopword-like filler from prose so the
    context carries more information per token.  Never touches code / JSON /
    structured text (guarded by :func:`is_structured_text`).  Filler words are
    matched case-insensitively as whole tokens.  ("src", "Tài liệu (8)")"""
    if is_structured_text(text):
        return text
    fillers = {str(w).strip().lower() for w in (filler_words or _DEFAULT_FILLER)
               if str(w).strip()}
    if not fillers:
        return text
    kept = [tok for tok in (text or "").split() if tok.lower() not in fillers]
    return " ".join(kept).strip()


# ---------------------------------------------------------------------------
# 8. Early-stop / length heuristics
# (source: "Tài liệu (8)" — early-exit heuristics)
# ---------------------------------------------------------------------------


def desired_output_length(query_length: int, is_coding: bool = False,
                          is_math: bool = False) -> int:
    """Eq M15 — target output length in tokens from the query length, boosted
    for coding / math requests (they need more room for code and derivation).
    Bounded to a sane range.  ("src", "Tài liệu (8)")"""
    query = max(0, int(query_length or 0))
    base = max(64.0, query * 1.2 + 96.0)
    boost = 1.0
    if is_coding:
        boost *= 1.4
    if is_math:
        boost *= 1.25
    return int(min(4096.0, max(64.0, base * boost)))


def response_length_policy(expected: float, min_tokens: int,
                           max_tokens: int) -> int:
    """Eq M16 — clamp the expected response length into ``[min_tokens,
    max_tokens]``.  Degenerate orderings (min > max) are swapped into a sane
    interval rather than raised.  ("src", "Tài liệu (8)")"""
    lo = max(0, int(min_tokens or 0))
    hi = max(0, int(max_tokens or 0))
    if lo > hi:
        lo, hi = hi, lo
    expected_int = int(float(expected or 0.0))
    return max(lo, min(hi, expected_int))


# ---------------------------------------------------------------------------
# 9. Semantic retention (cosine-based turn filtering)
# (source: "Tài liệu (29)" — provenance memory docs)
# ---------------------------------------------------------------------------


def should_keep_turn(turn_emb: Sequence[float], theme_emb: Sequence[float],
                     threshold: float) -> bool:
    """Eq M17 — keep a conversation turn only if its embedding is similar
    enough to the task theme; drop off-theme turns first.  Uses the
    ``cosine_similarity`` primitive from ``gateway.opt_core`` (expects
    pre-normalized vectors, as produced by ``hashed_embedding``).
    Empty/mismatched embeddings compare at 0.0 and are dropped unless the
    threshold is at or below 0.  ("src", "Tài liệu (29)")"""
    sim = cosine_similarity(turn_emb, theme_emb)
    return sim >= _clamp(float(threshold or 0.0))


# ---------------------------------------------------------------------------
# 10. Sliding-window identity
# (source: "Untitled document (9)" / "1.txt" — context-window slides)
# ---------------------------------------------------------------------------


class SlidingWindow:
    """Eq M18 — trivially sliding context: add items, read the last ``n``.
    An optional ``maxlen`` caps total storage.  Exposes ``reset()``/``stats()``.
    ("src", "1.txt")"""

    def __init__(self, maxlen: Optional[int] = None):
        self.maxlen = maxlen if maxlen is None else max(1, int(maxlen))
        self._items: List = []
        self._added_total = 0

    def add(self, item) -> None:
        self._items.append(item)
        self._added_total += 1
        if self.maxlen is not None and len(self._items) > self.maxlen:
            self._items = self._items[-self.maxlen:]

    def window(self, n: int) -> List:
        """Return the last ``n`` items (fewer if not enough history)."""
        n = max(0, int(n or 0))
        if n <= 0:
            return []
        return self._items[-n:]

    def reset(self) -> None:
        self._items.clear()
        self._added_total = 0

    def stats(self) -> Dict:
        return {"count": len(self._items), "added_total": self._added_total}


def drop_oldest(items: Sequence, n_keep: int) -> List:
    """Eq M19 — keep only the newest ``n_keep`` items of a sequence.
    Empty input or n_keep <= 0 -> safe default.  ("src", "1.txt")"""
    n_keep = max(0, int(n_keep or 0))
    if n_keep <= 0:
        return []
    if not items:
        return []
    return list(items[-n_keep:])


# ---------------------------------------------------------------------------
# NOT-IMPLEMENTABLE: real KV-cache / attention internals and neural memory.
# These need trained model internals, GPU kernels or forward passes that do not
# exist in pure stdlib Python.  Callers must guard with
# ``getattr(settings, "enable_...", False)``.
# ---------------------------------------------------------------------------

# NOT_IMPLEMENTABLE: causal softmax attention needs per-head Q/K/V tensor ops
# and a trained model's weight matrices — none exist in a pure-Python runtime.
NOT_IMPLEMENTABLE_SOFTMAX_ATTENTION = (
    "causal-softmax-attention:needs-qkv-forward-pass-and-trained-weights"
)

# NOT_IMPLEMENTABLE: GQA/MQA head grouping is defined by trained head
# projection matrices (and KV-sharing topologies) that must be read off weights.
NOT_IMPLEMENTABLE_GQA_MQA_SPLIT = (
    "gqa/mqa-head-split:needs-trained-head-projections-and-kv-sharing-topology"
)

# NOT_IMPLEMENTABLE: FlashAttention is a GPU tiling / memory-layout kernel;
# there is no equivalent data-structure-level operation in stdlib Python.
NOT_IMPLEMENTABLE_FLASH_ATTENTION = (
    "flash-attention:tiling-kernel-requires-gpu-sram-memory-layout"
)

# NOT_IMPLEMENTABLE: sliding-window attention kernels are GPU tile loops over
# causal blocks; not representable as a host-side pure-Python expression.
NOT_IMPLEMENTABLE_SLIDING_WINDOW_KERNEL = (
    "sliding-window-attention-kernel:gpu-tile-loop-not-representable-in-stdlib"
)

# NOT_IMPLEMENTABLE: RoPE applies rotation matrices to trained position/inv-freq
# arrays inside the attention forward pass; requires real model tensors.
NOT_IMPLEMENTABLE_ROPE = (
    "rotary-position-embedding:requires-model-tensor-ops-and-inv-freq-arrays"
)

# NOT_IMPLEMENTABLE: layer-wise attention reuse reflows intermediate attention
# activations between layers — needs the model's forward activations.
NOT_IMPLEMENTABLE_LAYER_ATTENTION_REUSE = (
    "layer-wise-attention-reuse:needs-intermediate-activations-of-trained-layers"
)

# NOT_IMPLEMENTABLE: a MemGPT-style trained memory compactor is a learned neural
# compressor; explicit-summary truncation is available here, learned compression
# is not.
NOT_IMPLEMENTABLE_NEURAL_MEMORY_COMPACTOR = (
    "memgpt-style-learned-compactor:needs-trained-neural-compressor"
)


__all__ = [
    "recency_weight", "priority_score",
    "message_priority", "MemoryWindow",
    "consolidation_score", "exponential_rehearsal",
    "should_summarize", "truncate_decay_stop",
    "is_structured_text", "compress_to_token_budget", "ContextCompactor",
    "attention_importance", "token_importance",
    "prune_filler_tokens",
    "desired_output_length", "response_length_policy",
    "should_keep_turn",
    "SlidingWindow", "drop_oldest",
    "NOT_IMPLEMENTABLE_SOFTMAX_ATTENTION",
    "NOT_IMPLEMENTABLE_GQA_MQA_SPLIT",
    "NOT_IMPLEMENTABLE_FLASH_ATTENTION",
    "NOT_IMPLEMENTABLE_SLIDING_WINDOW_KERNEL",
    "NOT_IMPLEMENTABLE_ROPE",
    "NOT_IMPLEMENTABLE_LAYER_ATTENTION_REUSE",
    "NOT_IMPLEMENTABLE_NEURAL_MEMORY_COMPACTOR",
]