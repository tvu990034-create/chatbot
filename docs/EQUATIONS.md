# Equations library — what each set models and where it runs

All sets live in `gateway/equations/`, are pure functions (stdlib +
numpy), and each has a test module (`tests/test_equations_*`). LIVE
marks sets executing on request paths; the rest are available
machinery with tests but no current caller.

## LIVE sets

- **cache_math** — semantic-cache scoring: `SemanticCacheMath.rank`
  (cosine rank over candidates), `AdaptiveThreshold` (mu/sigma gate
  that learns from observed similarities; base 0.92, bounds
  0.60–0.95), `evict_scores` (frequency × recency vs token cost),
  `cos_sim`, `temporal_decay`, `kl_divergence`. Runs on every
  semantic-cache lookup via `equation_wiring`.
- **calibration_math** — confidence scaler + deferral rule feeding the
  speed verify gate: low-confidence drafts get re-checked, high
  confidence passes through.
- **latency_math** — peak-centered num_predict→latency framework:
  latency brackets, 3-point measurement card, measured-b\* selection.
  The think-off budget (384 default) is measured, not hardcoded —
  re-probe with `benchmark_results/probe_num_predict.py --three-point`
  when model, hardware, or ollama changes.
- **memory_math** — conversation window trim (`recency_weight`,
  `compress_to_token_budget`, `ContextCompactor`).
- **retrieval_math** — RRF (reciprocal rank fusion) combining
  LlamaIndex + Haystack rankings in the agent prefetch.
- **routing_math** — `majority_consensus` (CISC votes),
  `confidence_token_budget`, `LoadBalancer` (with dwell + EWMA),
  `difficulty_score`, `should_route_to_strong`, Thompson/UCB
  bandits (Thompson live via temperature adapter).
- **budget_math** — length-based default token budgets
  (`dynamic_max_tokens`: base + multiplier × words, explainer long
  cap). Reasoning queries return **no cap**: thinking models burn ~10x
  visible budget in hidden chain-of-thought, so any small cap
  truncates into garbage. Wired into the server default `max_tokens`;
  caller values always win.
- **perf_math** (`gateway/perf_math.py`) — Eq8 confidence→budget
  mapping for RAG synthesis; Eq9 anomaly detector over endpoint
  latency (O(1) bounded deque).

## Available (tested, no live caller)

- **bandit_math** — Thompson sampler, UCB1, epsilon-greedy selects.
- **planning_math** — multi-sample consensus helpers, `plan_budgets`
  (shared-deadline multiplier 2.0).
- **not_implementable** — the deliberate boundary: ideas audited and
  rejected (e.g. speculative decoding without logprob access), so
  nobody re-litigates them.

## Legacy reference

- `SPEED_EQUATIONS.md` — original design notes; superseded by this
  file and the module docstrings where they disagree.
- The archived `archive/legacy-chatbot` branch has its own
  `speed_engine/` equation set (Bloom/FAQ/SimHash/BM25); audited for
  porting — its adaptive threshold, SimHash, and prompt budgets are
  covered here by strictly better / mode-safe versions, and its
  generation engine is a hardcoded simulator. Nothing was ported;
  nothing was lost.
