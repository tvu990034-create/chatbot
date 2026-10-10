# Speed methods — every way this stack buys latency, measured

Rule of the house: wall-clock wins must come from **skipped work**,
never from truncated answers. `bench` prints ms/char next to every
ratio so brevity can't masquerade as throughput (measured: no
systematic tok/s edge either way — CPU physics).

## 1. Zero-call answers (measured: 0 ms vs 23–47 s raw)

- Greeting templates (`hello`, `hi there!`, …) on first messages.
- Deterministic solvers: safe arithmetic (no `eval`, precedence
  correct), word problems, follow-up conjunctions (`And 13*13?`),
  unit-safe forms. Multi-turn safe: confirmations (`yes`/`ok`) only
  answer instantly on first messages, never mid-conversation.
- Where: `gateway/opt_core.py` (`quick_arithmetic`,
  `is_safe_quick_path`), `universal_enhanced_gateway._get_quick_response`,
  agent quick path (sync + stream).

## 2. Cache tiers (measured: repeat questions ~0 ms)

Exact dict → semantic second stage (mode-scoped embeddings, persisted
sidecar) → disk. Dirty-flag saves, atexit flush, corrupt quarantine,
TTL + popularity eviction. Repeats never touch the model.

## 3. Fast-model routing (measured: phi3 trivials in seconds)

RouteLLM complexity analysis sends trivia to the small model (higher
tok/s on the same CPU) and reserves qwen3 for reasoning. First
installed tier candidate wins; tiers documented in README.

## 4. Think-off (measured: cleaner + faster on thinking models)

qwen3 burns its whole budget on hidden chain-of-thought and can return
empty content. Think-off primary (384-token measured budget) skips the
hidden trace; a capped raw retry inherits only leftover deadline.
`num_predict` travels in ollama `options` (top-level `max_tokens`
alongside it returns EMPTY — handled in the gateway).

## 5. Token budgets (bounded worst case, never truncation)

Speed caps (explicit, cache-key-scoped), dynamic server defaults
(`budget_math`: short questions stop rambling, explainers keep room,
reasoning uncapped), shared per-item deadlines (retry inherits
remaining = cap − elapsed; worst item == cap exactly, never stacked
windows).

## 6. Lazy everything (measured: agent start without 133 MB download)

RAG index builds only with retrievable content (docs or persisted
collection); embedding model never downloads on docs-less boxes.
Heavy modules stay lazy imports; gateway ctor ~0.4 ms, cache key ~2 µs,
routing ~16 µs (all negligible next to generation).

## 7. Overlap + residency

Eq2 RAG prefetch overlaps retrieval with graph setup; `keep_alive=30m`
on all local calls (plus `OLLAMA_KEEP_ALIVE=30m` server-side) avoids
the ~60 s cold model load; one warmup probe per bench moves load out
of measured rows.

## 8. Server ordering (measured: 143 s Empty → 6.6 s answer)

Gateway-first on the direct path (plain only for overrides/fallback),
degenerate RAG answers fall through instead of serving model
gibberish, follow-up math resolves instantly, children terminate on
every shutdown path (no port squatters).
