# Repository Guide — every file in this project, what it does

A **measured optimizer layer** for local Ollama models (no cloud):
instant routing, deterministic solvers, response caches, model tiers,
and budgeted reasoning — with a chatbot CLI/API/UI on top as the
interface. ~290 tracked files. Status tags used below: **LIVE**
(executes on request paths), **LAZY** (loaded on demand), **TESTED**
(covered by `tests/`), **DORMANT** (no importers; kept, not executed),
**SCRATCH** (gitignored local tooling/output).

Branches: `main` is this project (the optimizer). `main-local` tracks
the same tip. `archive/legacy-chatbot` preserves the unrelated older
snapshot that used to occupy `main` (a TinyLlama FAQ/canned-reply
chatbot; see its own docs there, not here).

## How a request flows (read this first)

1. **CLI** (`main.py chat` balanced direct path, or `optimizer_cli.py`
   turbo commands) **or server** (`server/app.py` → agent `achat` by
   default, direct `litellm achat`, or pure-RAG).
2. **Gateway** (`gateway/universal_enhanced_gateway.py`): quick templates
   → prefix cache → exact cache → semantic second stage → analysis →
   routing → generation (think-off for reasoning) → verify (speed) →
   write caches → fallbacks on failure.
3. **Agent** (`agents/langgraph_agent.py`): cache → quick path → LangGraph
   (RAG prefetch overlapped, router, generation policy) → store.
4. **Caches** (`gateway/simple_cache.py` singleton, per-process dicts):
   exact → semantic (embedding index) → disk persistence.

## Entry points (start here as a user)

| File | Role |
|---|---|
| `main.py` | **LIVE** CLI: `chat` (answer-only by default, `--verbose` for logs), `run` REPL, `api`, `both` (children terminated on every exit path), `ui`, `ingest`, `status`, `benchmark`, `finetune`, `adapters`, `merge`, `advanced-benchmark` |
| `optimizer_cli.py` | **LIVE** turbo CLI: `chat`, `run` REPL, `check` (30 module smoke checks), `stats`, `bench` (cold/warm/raw + alternating order + reply lengths + ms/char readout), `eval` (HF benchmarks + accuracy) |
| `server/app.py` | **LIVE** FastAPI: `/chat`, `/api/v1/chat`, `/v1/chat/completions` (OpenAI shape), `/chat/stream` (SSE), `/rag/ingest`, `/rag/query`, health/models/docs routes |
| `deploy.py` | One-shot deploy/health-check script (**SCRATCH**-adjacent, manual use) |

## Configuration

| File | Role |
|---|---|
| `config.py` | **LIVE** all settings (pydantic-settings, env-overridable, everything defaulted). NOTE: `.env` is committed in this repo (unusual) — it holds non-secret defaults; keep real secrets out |
| `.env.example` | Template matching real model pair (`qwen3:4b`/`phi3:mini`) |
| `config_manager.py` | Layered config get/merge helper |
| `config.json` / `config.py.backup` | Legacy snapshots, not loaded |
| `pyproject.toml` / `requirements.txt` | Build + deps (`pip install -r requirements.txt` is the primary path) |
| `pytest.ini` | `testpaths=tests`, strict markers |

## Gateway core (the hot path)

| File | Role |
|---|---|
| `gateway/universal_enhanced_gateway.py` (~4.2k lines) | **LIVE** the whole pipeline: greeting/solver fast paths, 3 cache tiers, RouteLLM analysis + tier routing, staircase generation (think-off primary, raw fallback), speed verify gate, degradation ladder, per-item shared deadlines, telemetry. Owns model params per mode |
| `gateway/opt_core.py` | **LIVE** shared primitives: safe arithmetic solver (`quick_arithmetic`, no `eval`), word problems, context builder with budget/compression, model router w/ dwell + EWMA hooks, adaptive timeouts (incl. thinking cushion), cache identity (`make_cache_identity`: canonical sha256 over model/budget/system/history), hashed embeddings, TIR sandboxed subprocess runner |
| `gateway/litellm_gateway.py` | **LIVE** thin litellm adapter: `chat`/`achat`/`achat_stream` with cache, speed caps, think-off options handling, adaptive timeouts, 127.0.0.1 default |
| `gateway/simple_cache.py` | **LIVE** persistent response cache: TTL+LRU store, dirty-flag saves, atexit flush, corrupt-file quarantine, process singleton |
| `gateway/semantic_cache.py` | **TESTED**, standalone alternative cache backend (no live callers; live semantic path goes through the wiring layer onto SimpleCache) |
| `gateway/equation_wiring.py` | **LIVE** opt-in patch layer: semantic second stage (mode-scoped embedding index, persisted), EWMA router tracker, Koopman RAG mixer; `install_all()` runs in the gateway factory |
| `gateway/adaptive_temperature.py` | **LIVE** online-gradient-descent temperature; learns from verify verdicts |
| `gateway/thompson_temperature.py` | **TESTED** Thompson-sampling temperature adapter over `bandit_math` (no live callers; OGD policy is live) |
| `gateway/calibration_metrics.py` | **LIVE** lightweight confidence heuristics feeding the verify gate |
| `gateway/query_classifier.py`, `mathematical_enhancer.py`, `adversarial_detector.py` | **LAZY** controller plugins (regex/keyword analysis, no model calls) |
| `gateway/optimization_controller.py` | **LIVE** per-request analysis; query rewriting OFF by default |
| `gateway/perf_math.py` | **LIVE** eq8 confidence-gated budgets (RAG), eq9 anomaly detector (server), eq18/19 smoke-covered |
| `gateway/hardware_benchmark.py` | Model table behind `main.py benchmark` |
| `gateway/knowledge_base.py`, `long_term_memory.py` | **DORMANT** (only referenced by legacy gateways) |
| `gateway/enhanced_gateway.py`, `fixed_enhanced_gateway.py` | **DORMANT** legacy gateways (only uncollected root scripts touch them) |
| `gateway/safety_module.py`, `extended_monitoring.py`, `advanced_monitoring.py`, `training_diagnostics.py` | **DORMANT** monitoring/safety research modules |
| `gateway/*_optimization.py`, `academic_module.py`, `human_like_intelligence.py`, `arc_optimization.py`, `truthfulqa_optimization.py`, `advanced_vision_techniques.py`, `meta_reasoning.py`, `inference_optimization.py`, `performance_optimizer.py` | **DORMANT** paper-pattern modules (torch imports, no live callers) |

## Equations library (`gateway/equations/` — all TESTED; live ones marked)

`bandit_math` (Thompson/UCB/epsilon-greedy; Thompson live via adapter) ·
`cache_math` (**LIVE** semantic rank/threshold/eviction) ·
`calibration_math` (**LIVE** scaler/confidence/deferral in verify gate) ·
`latency_math` (**LIVE** peak/bracket/3-point card/measured-b\* selection) ·
`memory_math` (**LIVE** window trim) · `planning_math` (consensus available;
CISC uses routing consensus) · `retrieval_math` (**LIVE** RRF fusion) ·
`routing_math` (**LIVE** consensus/clearance/load balancer) ·
`budget_math` (**LIVE** length-based default token budgets; reasoning
queries return no cap so hidden chain-of-thought is never truncated) ·
`not_implementable` (documents the deliberate boundary).

## Agents, RAG, server

| File | Role |
|---|---|
| `agents/langgraph_agent.py` | **LIVE** `achat`/`chat`/streaming: cache (speed-scoped), quick path (greeting tolerance, follow-up-conjunction math), parallel RAG prefetch with RRF fusion, router, generation policy, thinking-aware timeouts |
| `agents/__init__.py` | Package marker |
| `rag/llama_index_rag.py` | **LIVE** retrieve/query/aquery over one `_retrieve_nodes` seam, Eq8-gated budgets, empty-index short-circuit; `build_index` checks for retrievable content BEFORE configuring embeddings (no 133 MB download on docs-less boxes), ingest force-rebuilds |
| `rag/haystack_pipeline.py` | **LIVE** alt provider (embed+retrieve; generator targets local Ollama) |
| `rag/self_rag.py`, `rag/advanced_embeddings.py` | **DORMANT** self-reflective RAG / hybrid retriever (no live callers) |
| `server/app.py` | **LIVE** (see entry points); direct path is gateway-first (plain only for overrides/fallback), degenerate RAG answers fall through instead of serving model gibberish, dynamic default token budgets, request bounds, upload limits, SSE |
| `server/anomaly_detector.py` | **DORMANT** per-endpoint anomaly detectors, healthy code, no callers |
| `tools/aider_tool.py` | **LAZY** code tools for the agent (empty list + debug note when aider missing) |
| `backends/model_server.py` | Backend health checker used by `status` |

## Research collections (all DORMANT, paper implementations, no live callers)

`merging/` (DARE, Fisher, Git Re-Basin, Model Stock, RegMean, SLERP, Task
Arithmetic + base/manager), `finetuning/` (LoRA/QLoRA trainers, adapters,
dataset prep, math recipes), `factuality/` (CAD, conformal, contrastive
decoding, DoLa, factual nucleus, hallucination detector, ITI, logit lens,
PMI processor, SelfCheckGPT, semantic uncertainty), `prompt_compression/`
(LLMLingua 1+2, selective context, recursive/file/summarization
compressors, token pruning), `reasoning/` (CoT, reflexion, self-refine,
self-consistency, ReAct, efficient reasoning, cache, manager, zero-shot;
`gateway_integration` lazy-loads optional deps).

## Tests (`tests/`, 26 files, ~705 tests, all green in CI)

`test_optimizations` (largest: gateway behaviors + regression tests) ·
`test_optimizer_cli` (bench/eval/CLI) · `test_request_flow` (server
contracts + bounds) · `test_langgraph_audit` (agent paths) ·
`test_tool_execution` (tools + TIR sandbox) · `test_rag` (providers +
seams + lazy build) · `test_gateway` · `test_cache_concurrency` (threads/singleton) ·
`test_prefix_response_cache` · `test_pass2/3/4/5` (solver/router/context) ·
`test_perf_regression` · `test_config` · `test_repo_hygiene` (BOM,
import case, no-prints lint) · `test_equations_*` (one per math family,
incl. `budget`) ·
`test_equation_wiring_smoke` (hermetic: tmp cache dirs).

## Data, runtime, and scratch (not in git)

`cache/*.json` (response/semantic caches + embedding sidecar, rewritten
constantly) · `data/` (RAG docs) · `.hf_cache/` (downloaded datasets) ·
`benchmark_results/` (probe/bench harnesses `*.py` + run outputs) ·
`test_eval/` (3 legacy eval-result JSONs) · `monitoring_logs/`, `ml_data/`,
`analytics_data/` (legacy outputs) · root `test_*.py` + helpers
(`track_score.py`, `clear_cache.py`, …) are uncollected manual scripts,
not the suite.

## Docs and project files

`README.md` (copy-paste usage for every command) · `BUSINESS.md`
(company integration guide) · `SPEED_EQUATIONS.md` (legacy design notes) ·
`LICENSE` (MIT) · `Dockerfile` + `docker-compose.yml` (container deploy) ·
`.github/workflows/ci.yml` (test suite on push to main + PRs; node24
actions, ubuntu-24.04, offline env) · `ALL_15_STEPS_COMPLETE.md`,
`*_REPOSITORIES_INTEGRATION_COMPLETE.md`, assorted `*_REPORT.md` (historical
status notes, informational only) · `react-ui/` (77-file web UI) ·
`static/index.html` · `ui/` + `ui_components/` (Gradio + dashboard pieces).
