# Speed & Performance Equations

Complete, cited reference for **every equation used in local-chatbot**, with a
deep-dive on the **speed / latency / throughput** equations that drive the live
inference path. Every entry names the source file and function so you can jump
to the code.

Related (older, Eq1–Eq19 catalog): `ALL_OPTIMIZATION_EQUATIONS.md`.

---

## Part A — Live speed path (the equations that actually run)

### A1. Measured throughput floor (the baseline every timeout is built on)

`gateway/opt_core.py` (module constants, `opt_core.py:37`)

```
TOKENS_PER_SECOND_FLOOR = 4.5      # measured real-traffic effective rate
TIMEOUT_BUFFER_S        = 15.0     # scheduler / queue / load cushion
```

Rationale in the code: phi3:mini measures ~10 tok/s on short prompts, but
1700–2700-character multi-step queries drop the *effective* rate to ~4.2–4.6
tok/s. A fixed timeout therefore kills any generation whose token budget (256–
1024 tokens ≈ 55–245 s) needs more wall time than the timeout allows.

### A2. `adaptive_generation_timeout` — token-budget-scaled timeout

`gateway/opt_core.py:41`

```
T(m, b) = max( b ,  m / TOKENS_PER_SECOND_FLOOR + TIMEOUT_BUFFER_S )
        = max( b ,  m / 4.5 + 15 )
```

- `m` = `max_tokens` budget (default 128).
- `b` = `settings.generation_timeout` (default 15 s).
- Short generations fit inside the base grace period unchanged; long ones get
  the wall time they actually need to finish.

Used by: `agents/langgraph_agent.py`, orchestrator and agent node LLM calls.

### A3. `_adaptive_timeout` — timeout from the *actual* effective budget

`gateway/universal_enhanced_gateway.py:375`

```
effective  = options['num_predict']  (thinking models)  else params['max_tokens']
             else settings.litellm_max_tokens
needed     = effective / 4.5 + 15
needed     += 120   if options present          # thinking-model cold start
needed     += 120   if effective >= 768        # slow prefill / long prompts
result     = max(base, needed)                 # base = settings.generation_timeout
result     = min(result, ceiling)              # only when a ceiling is passed
```

The cold-start term exists because qwen3 takes ~60 s just to load into VRAM
before emitting its first token, and that load time counts against the timeout.

### A4. `_main_generation_timeout` — 220 s cap for speed+thinking

`gateway/universal_enhanced_gateway.py:3165`

```
t = _adaptive_timeout(params, settings)
return min(t, 220.0)   if performance_mode == "speed"
                       and model is a THINKING model (qwen3 family)
return t               otherwise
```

Why: in the 61-question run the speed path on a thinking model burned a
~6-minute identical 592 s timeout on a hard question. Capping at 220 s makes a
stuck generation **fail fast** so the plain retry (proven SMART params for
thinking models) recovers a real answer instead of an apology.

### A5. Balanced-mode reasoning budget — the "tool" think-off recipe

`gateway/universal_enhanced_gateway.py:3326` (`_thinkoff_call`, `3300` area)

For hard/reasoning items in `balanced` mode the tool calls the model exactly the
way the raw baseline does (proven to converge on GSM8K):

```
budget      = 160                            # max_tokens AND options['num_predict']
hard_deadline = max(240, settings.generation_timeout)
attempts    = 1  then retry ONCE if text is None   # worst case ~2 × generation_timeout
call        = litellm_gateway.chat(
                  messages=[user: query], model=self.model_name,
                  max_tokens=160, use_cache=False,
                  options={"think": False, "num_predict": 160})
decoding    = policy temp 0.2, top_p 0.9     # calm generation policy for numeric answers
```

- `think: False` moves qwen3's chain-of-thought off the reply so content is
  always present (streaming `/api/generate` ignores `options.think`; litellm
  `/api/chat` honours it).
- The single retry bounds the tail at ~2× `generation_timeout` and never loops.
- Easy/chat items use the routed fast model with `/no_think` and "Answer briefly
  and directly in one short sentence."

### A6. Speed-mode token caps

`gateway/opt_core.py:61`

```
c = SPEED_MAX_TOKENS_REASONING = 384   if is_reasoning
c = SPEED_MAX_TOKENS_SIMPLE    = 128   otherwise
max_tokens = min(req_max or c, c)
```

GSM8K showed a 128-token cap truncates the answer → 0% accuracy, so reasoning
gets 384; both stay strictly below the 512 baseline so speed mode is still
cheaper.

### A7. Cache-hit rate

`gateway/fixed_enhanced_gateway.py:607` and `optimizer_cli._stats`

```
cache_hit_rate = cache_hits / total_requests
"cache-like fast" rows = requests whose latency < 50 ms
```

A `warm_ms` ≈ 0 ms in `run_bench` means the answer was served from a cache.

### A8. Utility math for model comparison

`gateway/advanced_optimization.py:63,79`

```
U(acc, lat) = ln(acc) − β · ln(lat)          # Cobb–Douglas latency–accuracy trade-off
              → −∞ when acc ≤ 0 or lat ≤ 0
LNS(score, lat) = score / (lat + ε)          # ε = 1e-6
```

- `β` default `0.3` (higher = penalise latency harder).
- `LNS` = latency-normalized score = quality per unit of time.

### A9. Average latency aggregation

`reasoning/gateway_integration.py:245`

```
avg_latency_ms = total_latency_ms / total_calls
latency_ms     = (time.time() − start_time) · 1000
```

### A10. `perf_math` fast-path formulas (Eq1–Eq19)

`gateway/perf_math.py`

| Eq | Name | Formula |
|----|------|---------|
| Eq1 | TTFB reduction fraction | 0.8 on cache hit, 0.3 with pre-warming, else 0 (ESTIMATED) |
| Eq2 | Prefetch speedup | `1 + (t_prefill − t_rag)/t_llm` when `t_rag < t_prefill`, else 1 |
| Eq2 | Visible latency (prefetch) | `t_prefill + t_decode − min(t_rag, t_prefill)` |
| Eq5 | Amdahl pre-warm speedup | `1/(s + p/N) · f^α · (1 + w)` (defaults s=0.1, p=0.9, N=5, f=0.555, α=100, w=0.05) |
| Eq8 | Confidence→token budget | `None` if `c < g`; else `m_min + ((c−g)/(1−g))^θ · (m_max − m_min)` (m_min=20, m_max=256, θ=5, g=0.45) |
| Eq18 | Query complexity | `0.4·min(len/300,1) + 0.4·min(kw/4,1) + 0.2·min(qmarks/3,1)` |
| Eq19 | Best-of-N expected quality | `μ + σ·Φ⁻¹((n − 0.375)/(n + 0.25))` (Beasley–Springer–Moro inverse CDF) |
| Eq19 | Optimal N | heuristic: `1` if σ < 0.1 else `min(1 + 10σ, max_n)` |
| Eq19 | Use Best-of-N? | `complexity > 0.6 and cost_per_sample < 0.02` |

### A11. Benchmark timing & report equations

`optimizer_cli.py:234` (`run_bench`), `optimizer_cli.py:666` (`run_eval`),
`benchmark_results/eval100_run.py`, `benchmark_results/eval100_report.py`

```
latency_ms   = (time.perf_counter() − t0) · 1000        # every code path
ratio        = optimized_ms / max(baseline_ms, 0.001)   # "vs raw", lower = faster
accuracy     = correct / answered
avg_ms       = Σ ms / n
p_k          = sorted(ms)[ ⌊n·k⌋ ]                      # p25, p50 (median), p75, p95
total_s      = Σ ms / 1000
opt_faster_count = #{ i : opt_ms[i] < raw_ms[i] }
agree_frac   = #{ i : opt_correct == raw_correct } / #{ i : both answered }
accuracy delta  = optimized.accuracy − baseline.accuracy
```

Our committed 100-question GSM8K mode sets `generation_timeout = max(·, 420)`
and runs raw with `max_tokens = 160` for parity.

### A12. Hardware sizing equations (model fits this machine?)

`gateway/hardware_benchmark.py:36,116`

```
model_bytes   = params_billions · 10⁹ · bytes_per_param
bytes_per_param: Q4 = 0.55 | Q5 = 0.70 | Q8 = 1.00 | FP16 = 2.00
budget_cpu    = 0.6 · physical RAM            # reserve ~40% for the OS
budget_gpu    = VRAM bytes                    # detected or simulated look-up
fit           = model_bytes <= budget
tok_est       = base / params_b               # base: 260 GPU, 22 CPU
speed_filter  = "fast" needs tok_est ≥ 20 ; "usable" needs ≥ 5
sort          = quality_score desc, then tok_est desc
```

### A13. SLO-anchored latency anomaly detection (Eq9)

`server/anomaly_detector.py:162` and `gateway/perf_math.py` (`Eq9AnomalyDetector`)

```
μ = mean(window)     σ = population std of window     (window ≤ 200, min 20 samples)
is_anomaly = (latest > slo_ms)  AND  (latest > μ + k·σ)
             with slo_ms default 1000 ms, k = 3.0
cooldown   = 300 s between alerts
feedback   = fallback_fraction += 0.05   (capped at 0.30)   # Eq9 → Eq7
anomaly_rate = total_anomalies / total_requests
```

Slow-but-within-SLO requests are suppressed by design (noise suppression).

### A14. Answer-extraction / correctness equations (numeric scoring)

`optimizer_cli.py:467,543,554,567,587,600,615,637,651`

```
gold = after "#### " marker, else last number in the answer text
pred = trailing bare-number line, else last number outside parentheses  (commas stripped)
_equal(pred, gold) = float(pred) == float(gold)   (numeric mode)
pred_value: numeric → number | letter → first A–E | word → yes/no/true/false | exact → token-normalized
is_correct: numeric → number equality; exact → token-insensitive equality or substring;
            letter/word → case-insensitive string equality
```

---

## Part B — All equation families (`gateway/equations/*`)

Pure-stdlib math modules ported from the optimization catalog. Each entry is the
main formula; defaults are given where the code parameterises them.

### B1. Cache math — `cache_math.py`

| Eq | Name | Formula (defaults) |
|----|------|--------------------|
| — | `cos_sim` | `a·b / (‖a‖·‖b‖)`; 0 on empty/mismatch, never NaN |
| T1 | `temporal_decay` | `2^(−dt/half_life)` with `half_life = 3600 s` |
| T2 | `AdaptiveThreshold.hit` | `sim ≥ (base + w·(μ−base) + z·σ) − margin`; base=0.92, w=1.0, z=2.0, clamp [0.70, 0.99]; window ≤ 500 scores |
| T3 | `EntryScore.score` | `0.45·recency + 0.30·log1p(freq)/log1p(100) + 0.15·log1p(tokens)/log1p(4096) + 0.10·demand`, each clamped |
| T4 | `logistic_eviction_probability` | `p = σ(k·(cap − v))`, k=12, cap=0.5 |
| — | `kl_divergence` | `Σ p·ln(p/q)`, q floor 1e-12, ≥ 0 |
| T5 | `staleness_score` | `KL(old‖new) · (0.5 + 0.5·clamp(age/ttl))`, ttl=3600 |
| T6–T8 | `SemanticCacheMath` | rank by `cos_sim` above adaptive threshold, top-k; blend length-weighted: `target = Σ w·len / Σ w`, `w = sim^α` (α=0.7); exact answer wins unconditionally |
| T9 | `insert_score` | admit if `EntryScore(benefit) > 0` |
| T10 | `evict_scores` | evict lowest-value entries up to `(1−keep_ratio)·n`, keep_ratio=0.7 |

### B2. Routing math — `routing_math.py`

| Eq | Name | Formula (defaults) |
|----|------|--------------------|
| R1 | `LoadBalancer` EWMA | `ewma = α·cur + (1−α)·ewma`, α=0.5; recommend = min EWMA |
| R2 | `select_idle_model` | switch candidate unless dwell window (`now−last_switch ≥ min_dwell`) violated and queues not `cq > lq + 2` |
| R3 | `BudgetAwareRouter.cost_estimate` | `cpt · (input + output)`; route: strong if affordable, then fast, else fallback; `best_model_for_budget` = max `value/fixed_cost` under budget |
| R4 | `difficulty_score` | `0.35·clamp(tokens/300) + 0.30·rare_word_density + 0.35·clamp(hard_keyword_hits/3)` |
| R5 | `should_route_to_strong` | `difficulty ≥ 0.6` |
| R6 | `confidence_token_budget` | `None` if `c < gate=0.45`; `m_min + ((c−0.45)/0.55)^θ · (m_max−m_min)`, θ=5, [20, 256] |
| R7 | `zipf_popularity` | `R(rank) = 1 / rank^s` (s=1) |
| R8 | `cache_hit_ratio` | `Σ₁ᵏ R(r) / Σ₁ⁿ R(r)` over top-`k` of `n` |
| R9 | `zipf_capacity_hit` | smallest `k` with cumulative mass ≥ target |
| R10 | `deadline_scheduler` | `budget_i = deadline · w_i / Σw` (proportional split) |
| R11 | `feasible_under_deadline` | `Σ est_i ≤ deadline` |
| R12 | `should_switch` | `new − old ≥ hysteresis` (0.1) — anti-thrash |
| R13/R14 | `majority_consensus` / `consensus_clearance` | top count / n; `frac ≥ 0.5` |
| R15 | `load_adjusted_score` | `base − penalty·max(0, EWMA_load)` (penalty 1.0) |

### B3. Bandit math — `bandit_math.py`

| Eq | Name | Formula (defaults) |
|----|------|--------------------|
| 217 | `ThompsonSampler` | Beta-Bernoulli: sample `θ_i ~ Beta(a_i, b_i)`, pick max; conjugate update `a+=y, b+=(1−y)`; prior (1,1) |
| 227 | `UCB1` | `score = mean + sqrt(c·ln(t)/count)`, c=2.0; unseen = +∞ |
| 228 | `UCB1-Tuned` | `mean + sqrt( ln(t)/count · min(0.25, V_i) )`, `V_i = variance + sqrt(2·ln(t)/count)` |
| 229/230 | ε-greedy / anneal | random with prob ε else best mean; `ε = ε_end + (ε_start−ε_end)·min(1, step/decay_steps)` |
| 231 | `EXP3` | softmax over `η·w`; `w_i *= exp(η·r̄_i/p_i)`; `η = sqrt(ln n/(n·t))` |
| 233 | acquisition EI | `(μ−best−c)·Φ(z) + σ·φ(z)`, `z = (μ−best−c)/σ`, ≥ 0 |
| 234 | acquisition UCB | `μ + κ·σ` (κ=2) |
| 235 | acquisition PI | `Φ((μ − best)/σ)` |
| 236/237 | `SimpleGP` | squared-exponential kernel `exp(−½d²/ℓ²)`; `μ = k·K⁻¹y`, `σ² = k(x,x) − k·K⁻¹k`, jitter 1e-9, noise 1e-6 |
| 238 | `explore_fraction` | `exp(−step/decay)`, decay=100 |
| 239 | `regret` | `max(0, best_cum − actual_cum)` |

### B4. Retrieval math — `retrieval_math.py`

| Eq | Name | Formula (defaults) |
|----|------|--------------------|
| R1/R1b | `mmr` / `mmr_score` | select argmax `λ·sim_doc[i] − (1−λ)·max_j sim_docs[i][j]`, λ=0.7 |
| R2 | normalize | minmax `(s−lo)/(hi−lo)`; z-score → clamp[−3,3]; sigmoid `1/(1+e^(−scale·(s−mid)))` |
| R2d | `cosine_to_similarity` | `(cos + 1)/2` maps [−1,1]→[0,1] |
| R3 | `rrf` | fused score `Σ 1/(k + rank)`, k=60; weighted variant multiplies per-list weight (R4) |
| R5a/b | PMI / MI gate | `log2(p_xy/(p_x·p_y))`; gate `(1 − H_signal)/H_noise ≥ 0.5` |
| R6 | novelty / curiosity | novelty `1 − max cos_sim`; curiosity `(α·nov + β·unc)/(α+β)` |
| R7 | `greedy_diverse_select` | like MMR with `score` and `sim_matrix`, λ=0.7 |
| R8 | `chunk_cache_key` | SHA-256 of whitespace-normalized, lowercased text |
| R9 | `relevance_gate` | `score ≥ threshold` (`>` strict), NaN ⇒ False |
| R10 | `qf_similarity` | `0.5·((cos_hashed+1)/2) + 0.5·Jaccard`; feature hash = MD5 char 3/4-grams, dim 128 |
| R11 | `topk_stable` | top-k by score, ties broken by index |
| R12 | `select_within_token_budget` | greedy by `score/tokens` descending, never exceed budget |

### B5. Planning math — `planning_math.py`

| Eq | Name | Formula (defaults) |
|----|------|--------------------|
| EqP1 | `uct_score` | `win_frac + c·sqrt(ln(parent_visits)/(node_visits+1))`, c=1.41; unseen = +∞ |
| EqP2 | `simulation_budget` | `min(max_sims, ⌊time_remaining / per_sim⌋)` |
| EqP3–P5 | agreement/self-consistency | majority fraction of math-normalized votes; verify-before-commit if `agreement < 0.6` |
| EqP7 | `best_of_n_expected` | `μ + σ·Φ⁻¹((n−0.375)/(n+0.25))` |
| EqP8 | `optimal_n` | argmax over `n∈[1,max_n]` of `best_of_n_expected − n·cost` |
| EqP10 | `verification_score` | weighted fraction of passed checks, each weight default 1.0 |
| EqP11 | `prob_pass` | `0.5 + (score−0.5)·(1 + 0.25·log1p(n_checks))` |
| EqP12/P13 | `max_q` / `mean_q` | discounted sum `Σ γ^i·v_i` (γ=0.99) / arithmetic mean |
| EqP15 | `depth_limit_knowledge` | `max(1, round(max_depth·0.5·(1 + difficulty)))` |

### B6. Memory / context math — `memory_math.py`

| Eq | Name | Formula (defaults) |
|----|------|--------------------|
| M1 | `recency_weight` | `γ^(total−1−position)`, γ=0.9 |
| M2 | `priority_score` | `recency · (w_rel·relevance + w_sal·saliency)/(w_rel+w_sal)`, 0.6/0.4 |
| M3 | `message_priority` | `0.40·2^(−age/H) + 0.30·saliency + 0.20·mention + 0.10·depth_factor` (H=3600) |
| M5 | `consolidation_score` | `detail·(0.55 + 0.30·(1−2^(−age/H)) + 0.15·revisit)`, H=86400 |
| M6 | `exponential_rehearsal` | `detail·min(2, 1 + 0.5·log1p(revisits))` |
| M7 | `should_summarize` | `total_tokens ≥ 0.7·budget` |
| M8 | `truncate_decay_stop` | drop oldest while over budget, stop when `|v|·decay^k < min_margin·|v₁|` |
| M10 | `compress_to_token_budget` | head+tail slice (≈4 chars/token), never shred code/JSON (`is_structured_text`), prose keeps first+last sentence then fills mids |
| M12/M13 | attention/importance proxies | `0.4·score + 0.4·log1p(tokens)/log1p(512) + 0.2·1/(1+pos/64)`; `σ(logit−sep)·(0.5+0.5·0.97^pos)` |
| M15 | `desired_output_length` | `clamp( (max(64, len·1.2+96) · boost), 64, 4096)`, boost 1.4 coding / 1.25 math |
| M16 | `response_length_policy` | clamp into `[min, max]` |
| M17 | `should_keep_turn` | `cos_sim(turn, theme) ≥ threshold` |
| M18/M19 | `SlidingWindow` / `drop_oldest` | keep last `n` items |

### B7. Calibration math — `calibration_math.py`

| Eq | Name | Formula (defaults) |
|----|------|--------------------|
| C1 | `softmax` | `e^(z_i/T) / Σ e^(z_j/T)`, max-subtracted for stability |
| C2/C3 | temperature scaling | `softmax(logits/T)`; `TemperatureScaler` fits scalar `T` by golden-section search on binned ECE or NLL over `T ∈ [0.01, 20]` |
| C4 | `ece` | `Σ_b (n_b/N)·|acc_b − conf_b|`, 10 bins |
| C5 | `brier_score` | `mean over samples of Σ_k (p_k − y_k)²`, ∈ [0, 2] |
| C6 | `calibrated_confidence` | `conf^(1/T)` |
| C7 | `PlattScaler` | `p(y=1|z) = σ(a·z + b)`, fitted by gradient descent on binary cross-entropy (lr 0.1, 2000 epochs); falls back to base rate |
| — | isotonic | pool-adjacent-violators monotone regression |
| — | conformal | quantile-based prediction sets with `ConformalSetPredictor` |
| — | entropy / abstention | `H = −Σ p·ln p`; abstain if normalized `H/H_max ≥ threshold`; `max_entropy = ln(n_classes)` |
| — | energy OOD | `E = T·log Σ e^(z_i/T)`; OOD if `E` deviates from in-distribution mean by `k·std` |
| — | top-2 margin / prob ratio / max-conf | margins on the probability vector to gate deferral |

### B8. `opt_core` primitives (shared by everything above)

| Name | Formula (defaults) |
|------|--------------------|
| `hashed_embedding` | MD5 feature-hash of tokens + char 3/4-grams into dim-256 vector, L2-normalized |
| `cosine_similarity` | dot product (inputs must be pre-normalized) |
| `estimate_tokens` | `max(1, ⌈len(text)/4⌉)` = `(len+3)//4` |
| `lexical_overlap` | Jaccard of lowercase token sets (stopwords removed) |
| `compress_prompt` | keep fenced code/JSON intact; compress filler toward `max(32, ⌈n·ratio⌉)` |
| `resolve_generation_policy` | temp: math/coding ≤ 0.2, reasoning ∈ [0.2, 0.4], small models ≤ 0.3, top_p ≤ 0.95, top_k ≥ 20; short-request cap 128, long-request ≥ 512, never above `configured_max_tokens` |
| `normalize_math_answer` | canonicalize numbers / `\boxed{}` so `\boxed{7}` == `7.0` |
| `vote_responses` / `cisc_confidence` | majority answer / majority fraction |

### B9. Not-implementable catalog

`gateway/equations/not_implementable.py` lists everything that needs a training
loop, GPU kernels, or model weights (cross-entropy/KL/distillation losses,
FlashAttention-style kernels, learned calibration, RL/PPO loops, neural
compressors…) as `NOT_IMPLEMENTABLE_*` constants. They exist so a GPU-porting
auditor sees exactly what is missing — none are callable in this runtime.