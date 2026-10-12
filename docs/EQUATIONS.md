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

---

## Complete function reference

Shared numerically-safe helpers (every module): `_clamp` (bounds),
`_normal_pdf/_cdf`, `_inv_normal_cdf`, `_erf` (own implementations —
no scipy dependency on purpose).

### bandit_math — exploration vs exploitation

| Function | Formula / role |
|---|---|
| `ThompsonSampler` | Beta-posterior sampling per arm, live via temperature adapter |
| `UCB1` / `ucb` / `ucb1_tuned` | Upper-confidence-bound arm selection |
| `epsilon_greedy_select` / `epsilon_anneal` / `explore_fraction` | Epsilon schedules with decay |
| `EXP3` | Adversarial bandit weights |
| `acquisition_EI/UCB/PI`, `SimpleGP` | Bayesian-optimization acquisition over a tiny GP |
| `regret` | Cumulative best-vs-actual gap |

### cache_math — semantic cache scoring (LIVE)

| Function | Formula / role |
|---|---|
| `cos_sim` | Cosine similarity of embedding vectors |
| `temporal_decay` | 0.5^((now − created)/half_life), default 1 h half-life |
| `AdaptiveThreshold` | Mu/sigma gate learning from observed similarities (base 0.92, bounds 0.60–0.95) |
| `SemanticCacheMath.rank` | Cosine rank over candidates, top-k |
| `evict_scores` | Frequency × recency vs token-cost eviction scores |
| `logistic_eviction_probability` | Sigmoid eviction curve |
| `kl_divergence`, `staleness_score` | Distribution drift between cache generations |
| `EntryScore` | Per-entry score record |

### calibration_math — confidence for the verify gate (LIVE)

| Function | Formula / role |
|---|---|
| `softmax` / `temperature_scale` / `TemperatureScaler` | Temperature-scaled probabilities |
| `ece`, `brier_score` | Calibration error metrics |
| `calibrated_confidence` | Temperature-adjusted confidence feeding the gate |
| `PlattScaler`, `isotonic_regression` | Post-hoc calibration maps |
| `ConformalSetPredictor` | Quantile-based prediction sets |
| `entropy`, `max_entropy`, `entropy_abstention_decision` | Uncertainty → abstain decisions |
| `energy_ood_score`, `is_ood` | Out-of-distribution detection |
| `max_confidence`, `top2_margin`, `predicted_probability_ratio`, `EnsembleUncertainty` | Confidence readouts |
| `should_defer_to_verify` | conf < threshold (0.6 math/code, 0.4 other) → verify |
| `temperature_to_meet_entropy` | Solve temperature for a target entropy |

### latency_math — measurement framework (LIVE)

| Function | Formula / role |
|---|---|
| `latency`, `median_latency` | Per-budget medians over observations |
| `peak_budget` / `peak_bracket` | Plateau-tolerant peak (think-off b\*) |
| `parabola_vertex`, `refine_from_bracket` | Quadratic refinement of the peak |
| `ucb_chart`, `suggest_next_budget` | Which budget to probe next |
| `welch_t`, `jzs_bf10`, `savings_test` | Is the speedup statistically real |
| `plan_budgets` | Shared-deadline budget ladder (mult 2.0) |
| `sample_size_needed` | Power analysis for bench runs |

### memory_math — context window management (LIVE trim)

| Function | Formula / role |
|---|---|
| `recency_weight` | gamma^position decay |
| `priority_score`, `message_priority`, `attention_importance`, `token_importance` | Keep/drop scoring |
| `MemoryWindow`, `SlidingWindow`, `drop_oldest` | Window mechanics |
| `consolidation_score`, `exponential_rehearsal` | Revisit-based retention |
| `should_summarize`, `truncate_decay_stop`, `_head_tail_slice` | Transform-or-cut decisions |
| `is_structured_text`, `compress_to_token_budget`, `ContextCompactor` | Budget-fitting compression |
| `prune_filler_tokens` | Filler-word stripping |
| `desired_output_length`, `response_length_policy`, `should_keep_turn` | Length + turn policies |

### planning_math — multi-sample decisions

| Function | Formula / role |
|---|---|
| `uct_score`, `expand_if_promising`, `max_q`, `mean_q`, `depth_limit_knowledge` | MCTS-style search control |
| `simulation_budget` | Sims that fit remaining ms |
| `agreement_fraction`, `should_verify`, `cisc_style_confidence`, `answer_consensus_check` | Vote → verify decisions |
| `best_of_n_expected`, `optimal_n`, `best_candidate`, `verification_score`, `prob_pass` | Sample-count economics |

### retrieval_math — RAG ranking (RRF LIVE)

| Function | Formula / role |
|---|---|
| `rrf_score` / `rrf` | 1/(k + rank) fusion across rankers (k=60) |
| `weighted_reciprocal_rank` | Weighted variant |
| `mmr` / `mmr_score`, `greedy_diverse_select`, `novelty_score`, `curiosity_score` | Diversity-aware selection |
| `min_max_normalize`, `z_score_normalize`, `sigmoid_normalize`, `normalize_scores`, `cosine_to_similarity` | Score alignment |
| `pointwise_mutual_information`, `mi_gate`, `relevance_gate`, `hybrid_relevance` (lexical + semantic, α=0.5) | Relevance gates |
| `qf_similarity`, `chunk_cache_key`, `embedding_similarity_from_embeddings`, `topk_stable` | Query/chunk utilities |

### routing_math — model selection (LIVE)

| Function | Formula / role |
|---|---|
| `LoadBalancer`, `select_idle_model`, `should_switch` | Least-load routing with dwell + hysteresis |
| `BudgetAwareRouter` | Cost-aware routing |
| `_rare_word_density`, `difficulty_score` | Query complexity estimate |
| `should_route_to_strong` | difficulty ≥ 0.6 → strong model |
| `confidence_token_budget` | M_min + (M_max − M_min)(1 − e^(−θc)), gate 0.45 |
| `zipf_popularity`, `cache_hit_ratio`, `zipf_capacity_hit` | Cache sizing curves |
| `deadline_scheduler`, `feasible_under_deadline` | Stage budgeting under a cap |
| `load_adjusted_score` | EWMA load penalty |
| `majority_consensus`, `consensus_clearance` | Vote winners + clearance bar |

### budget_math — default token budgets (LIVE)

| Function | Formula / role |
|---|---|
| `dynamic_max_tokens` | base + multiplier × words within [floor, max_cap]; explainer triggers → long cap; **reasoning markers → None** (no safe small cap for hidden chain-of-thought — configured default applies) |

### perf_math (`gateway/perf_math.py`, LIVE)

| Function | Formula / role |
|---|---|
| `eq8_confidence_to_max_tokens` | Same M(c) curve gating RAG synthesis budgets |
| Eq9 anomaly detector | O(1) bounded-deque mean/variance per endpoint |

### opt_core math helpers (LIVE, in `gateway/opt_core.py`)

Safe `quick_arithmetic` (precedence, no `eval`, lead-in + follow-up
stripping) · word arithmetic/problem solvers · `hashed_embedding` +
`cosine_similarity` (stdlib embeddings) · `make_cache_identity`
(canonical sha256 over model/budget/system/history) · adaptive
generation timeouts with thinking cushion · sandboxed TIR
subprocess runner.
