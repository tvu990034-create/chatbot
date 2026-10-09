# ⚡ Turbo Optimizer — faster + smarter local AI

An **optimizer layer** that sits in front of local Ollama models and makes
every request faster, cheaper, or smarter: instant answers for trivial
questions (no model call), deterministic solvers for arithmetic, model
routing by complexity, response caching, bounded think-off reasoning, and
time-boxed generations that never hang. No API keys, no cloud, no GPU.

It ships with a **chatbot interface** (terminal chat, REPL, REST API,
Gradio UI, RAG ingest) so you can use the optimizer as a daily driver —
but the chatbot is the demo surface. The product is the stack underneath:
every claim below is measured **optimized vs raw baseline** (see
`benchmark_results/`).

---

## Copy–paste command reference

Setup (once):

```bash
git clone https://github.com/tvu990034-create/chatbot.git
cd chatbot
ollama pull qwen3:4b
ollama pull phi3:mini
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
Copy-Item .env.example .env   # Windows PowerShell
# cp .env.example .env        # macOS / Linux
```

Smoke test (proves the install works):

```bash
python optimizer_cli.py check --quiet
python main.py chat "Hello there!"
python main.py chat "What is 12 * 8?"
```

Chat:

```bash
python main.py chat "What is the capital of France?"
python main.py chat --model phi3:mini "What is 7*6?"
python main.py chat --use-agent "Tell me about LangGraph"
python main.py chat --use-agent --no-rag "What is 7*6?"
python optimizer_cli.py run                    # interactive loop (/quit to exit)
python optimizer_cli.py run --mode speed
python optimizer_cli.py chat "Hi" --mode balanced --baseline
```

Server + API:

```bash
python main.py api --port 8000 --no-reload
# docs: http://127.0.0.1:8000/docs   health: http://127.0.0.1:8000/health
python main.py both                              # API + Gradio UI together
```

RAG (documents):

```bash
mkdir data\docs
echo "The harbor lights mark the entrance." > data\docs\note.txt
python main.py ingest --path data\docs
python main.py ingest --path data\docs --rebuild
```

Measure:

```bash
python optimizer_cli.py bench --n 5
python optimizer_cli.py bench --n 5 --mode speed
python optimizer_cli.py eval --n 5
python optimizer_cli.py stats
python main.py status
python main.py benchmark --top 3
```

Maintain:

```bash
del cache\*.json              # Windows: cold restart (forget cached answers)
# rm cache/*.json             # macOS / Linux
python -m pytest tests -q     # full test suite (needs: pip install -e ".[dev]")
```

Advanced (`finetune`, `adapters`, `merge`, `advanced-benchmark`) each
document themselves — run any of them with `--help`.

---

## Quick start

### 1. Install Ollama + a model pair

Download Ollama from https://ollama.com, then pull the recommended pair:

```bash
ollama pull qwen3:4b    # strong small reasoning model (for hard questions)
ollama pull phi3:mini   # fast model for trivia (auto-detected)
```

Only pulling one model is fine too — simple chat will just use that model.

### 2. Install Python (real one, on PATH)

You need Python 3.10–3.13. On Windows install from
[python.org](https://www.python.org/downloads/) and tick
**"Add python.exe to PATH"** — the Microsoft Store shortcut alone fails
with `Python was not found`. Verify first:

```bash
python --version     # must print 3.10+ (not open the Store)
```

### 3. Install dependencies

```bash
cd chatbot
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

(`pip install -e .` also works. If you previously saw
`BackendUnavailable: Cannot import 'setuptools...'`, pull the latest code
— the build backend is fixed.)

### 4. Configure

```bash
Copy-Item .env.example .env   # Windows PowerShell
# cp .env.example .env        # macOS / Linux
```

In `.env` ensure:

```
DEFAULT_MODEL=ollama/qwen3:4b
```

That is all — balanced mode is the default and needs nothing else.
(There is no `PERFORMANCE_MODE` setting: speed/balanced/quality is chosen
per call via `--mode`, not the environment.)

---

## CLI usage

```
python main.py <command>

Commands:
  chat       Ask a single question (terminal)
  status     Show config and backend health
  benchmark  Benchmark hardware and get model recommendations
```

### Chat

```bash
python main.py chat "How many days are in a week?"
python main.py chat "Prove that sqrt(2) is irrational"
python main.py chat --model qwen3:4b "Why is the sky blue?"   # use a specific model
python main.py chat --use-agent "Tell me about LangGraph"     # LangGraph agent (tools/RAG)
python main.py chat --use-agent --no-rag "..."                 # agent without retrieval
```

(`--no-rag` applies to the agent path; the direct path never retrieves.
`--stream` is accepted for compatibility but output prints at once.)

Each `chat` call runs the balanced gateway: greetings and simple arithmetic
answer instantly, other trivia in seconds on a fast model, and hard reasoning
runs to a real final answer instead of being cut off mid-thought.

### Status

```bash
python main.py status
```

### Benchmark

```bash
python main.py benchmark                        # CPU recommendations
python main.py benchmark --gpu "RTX 4090"        # simulate a GPU
python main.py benchmark --gpu "RTX 4090" --top 3 --speed fast
python main.py benchmark --json                  # JSON output
```

Run `python main.py --help` for the full list of options.

### Turbo bench / eval (optimized vs raw)

```bash
python optimizer_cli.py bench --n 5 --mode speed      # 5 curated questions, timed
python optimizer_cli.py bench --n 5 --mode balanced   # default smart path
python optimizer_cli.py bench --n 2 --no-baseline --questions "6*7|8+9"
python optimizer_cli.py eval --n 5                     # grade 5 GSM8K items, report to benchmark_results/
```

Modes: `speed` (tight token caps, short drafts get a verification pass),
`balanced` (default; bounded think-off reasoning), `quality` (generous
budgets, multi-model selection on hard queries).

### Interactive chat (REPL)

```bash
python optimizer_cli.py run                # type messages, live answers + timings
python optimizer_cli.py run --mode speed   # same loop, speed budgets
```

In-chat commands: `/quit` (or `/exit`), `/reset` (clear history),
`/stats` (cache hits, timings), `/help`.

### Health check + telemetry

```bash
python optimizer_cli.py check              # 30 module checks, offline, ~seconds
python optimizer_cli.py stats              # live cache hits, hit rate, timings
python optimizer_cli.py stats --json       # machine-readable
```

---

## Server (REST API) step by step

Start it (default `http://127.0.0.1:8000`):

```bash
python main.py api                          # REST server
python main.py api --port 9000 --no-reload  # custom port, no auto-reload
python main.py both                          # API + Gradio UI, two processes
```

Health and docs (no model needed):

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/openapi.json   # full schema
# Interactive docs: http://127.0.0.1:8000/docs
```

Chat (Windows PowerShell shown; any HTTP client works):

```powershell
# Basic chat (agent path with tools/RAG by default)
Invoke-WebRequest http://127.0.0.1:8000/chat -Method Post `
  -ContentType "application/json" `
  -Body '{"message": "What is 12*8?", "use_agent": true}'

# Multi-turn: pass history along
Invoke-WebRequest http://127.0.0.1:8000/chat -Method Post `
  -ContentType "application/json" `
  -Body '{"message": "And 13*13?", "history": [{"role": "user", "content": "What is 12*8?"}, {"role": "assistant", "content": "96"}]}'

# Speed mode for one request
Invoke-WebRequest http://127.0.0.1:8000/chat -Method Post `
  -ContentType "application/json" `
  -Body '{"message": "What is 12*8?", "speed_mode": true}'

# OpenAI-style endpoint
Invoke-WebRequest http://127.0.0.1:8000/api/v1/chat -Method Post `
  -ContentType "application/json" `
  -Body '{"messages": [{"role": "user", "content": "Hi"}], "performance_mode": "balanced"}'

# Streaming (server-sent events, one JSON object per line)
curl -N -X POST http://127.0.0.1:8000/chat/stream `
  -H "Content-Type: application/json" `
  -d '{"message": "Hello there!"}'
```

Limits are enforced before inference: message ≤ 32k chars, history ≤ 200
turns, oversized payloads get HTTP 422, failures return HTTP 500 with the
error (never a hang).

### For training teams (OpenAI-compatible API)

Point any OpenAI-compatible harness at the server with no other changes —
eval loops, data-gen pipelines, and agent trainers just work, and every
request gets the optimizer stack (cache, routing, think-off generation):

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")

print([m.id for m in client.models.list()])   # live backend models
r = client.chat.completions.create(
    model="qwen3:4b",
    messages=[{"role": "user", "content": "What is 14*14?"}],
    temperature=0.1, max_tokens=64)
print(r.choices[0].message.content, r.usage.total_tokens)
```

`POST /v1/chat/completions` (non-streaming) and `GET /v1/models` follow
the OpenAI shape (`chatcmpl-*` ids, `choices[].message`, `usage` token
counts from the same estimator that drives context budgeting). Why route
training traffic through it: repeat prompts (templates, few-shot prefixes,
retries) are served from cache in milliseconds instead of re-generated;
greetings/arithmetic never touch the model at all; per-request wall clocks
are bounded so a wedged generation can't stall a training batch. Point
`lm-eval-harness` style runners at `/v1` the same way.

### RAG step by step

```bash
mkdir data\docs                                   # Windows (default docs dir)
echo "The harbor lights mark the entrance." > data\docs\harbor.txt
python main.py ingest --path data\docs            # build the vector index
python main.py ingest --path data\docs --rebuild  # force rebuild
```

Then ask with retrieval (agent path, or `use_rag` on the endpoints).
`/rag/ingest` accepts files-only multipart uploads (max 20 files, 10MB
each, 50MB total); `/rag/query` takes `{"question": "..."}`.

---

## Troubleshooting

- **Ollama not reachable** (`Connection refused` / hangs): start it first
  (`ollama serve`), then `ollama pull qwen3:4b` (reasoning) and/or
  `ollama pull phi3:mini` (fast trivial answers). The CLI validates the
  model is installed and falls back with a message when it is not.
- **First answer is slow**: the model loads into memory on first use
  (~60s for qwen3:4b on CPU). Later answers reuse the loaded model
  (`keep_alive`), and repeats are served from cache instantly.
- **A call times out**: CPU generation is ~1-4 tokens/s; large budgets
  need minutes. Timeouts scale with the token budget automatically.
- **Stale or wrong cached answer**: caches live under `cache/` (plus a
  `semantic_embs.json` sidecar). Stop the app and delete `cache/*.json`
  to start cold — corrupt files are quarantined to `.bak` automatically.
- **A dependency is missing**: reinstall with `pip install -r requirements.txt`
  (add `[dev]` via `pip install -e ".[dev]"` if you run the test suite).
- **`Python was not found`** (Windows): you have only the Store shortcut —
  install real Python from python.org with PATH enabled, close and reopen
  the terminal, then `python --version`.

---

## How the optimizer answers questions

Every question makes ONE adaptive call to the local model (no fallback
chains), classified and routed on the fly. (Exposed as `main.py chat`;
the same stack serves the REPL, the REST API, and the Gradio UI.)

| Query type | Example | Model | Latency |
|---|---|---|---|
| Instant | "Hello there!" / "What is 12 * 8?" | none (template / safe math) | <1s |
| Trivial / factual | "How many days are in a week?" | fast "simple" model | ~5-10s |
| Reasoning | "Prove sqrt(2) is irrational" | selected reasoning model | ~1-4 min on CPU |
| Repeat | any question you already asked | answer cache | ~0s |

Measured on qwen3:4b CPU (5-question diag): balanced factual 9.6s vs raw
35s; explainer 33s vs raw 34-74s; speed reasoning runs hotter (bigger
budgets + verification pass), so `balanced` is the default fast path and
`speed` wins on short/trivial traffic. Latency SLOs: instant <1s,
factual <60s, explainer <120s, code <300s (balanced, warm model).

### User experience vs baseline (measured)

Same questions, optimized server vs raw unoptimized path (qwen3:4b,
CPU-only box; full logs in `benchmark_results/readme_sweep_report.txt`):

| Question | Optimized | Raw baseline |
|---|---|---|
| `12*8` (direct) | 6.6 s, "The answer is 96" | 47 s essay |
| `And 13*13?` (agent + history) | 0.1 s, "169" | 33 s essay |
| `Hello there!` (stream, time-to-first-byte) | 5.9 s | 27 s greeting |
| `What is the capital of France?` | 12 s, "Paris." | 38–159 s essay |

CLI answers greetings and simple arithmetic in ~1 s with no model call
at all; the REPL answers two questions plus `/stats` in ~2 s total;
the API serves `/health` ~2 s after boot. Raw has no routing, so every
question pays a full generation. Ratios swing with backend load -- remedial
detail: the server used to serve these same questions in 79–200+ s
(empty/timeout); gateway-first ordering, follow-up math normalization,
and degenerate-RAG fallback fixed that (see report).

Keep the model resident: ollama unloads models after 5 idle minutes by
default (next request pays ~60s cold load). All local calls request
`keep_alive=30m`; for full effect also set `OLLAMA_KEEP_ALIVE=30m` in the
ollama *server* environment.

The think-off token budget is measured, not hardcoded: run
`python benchmark_results/probe_num_predict.py --three-point` on a quiet
machine (20-40 min CPU) and the gateway adopts the resulting b* from
`benchmark_results/num_predict_scan.json` automatically. Re-probe when
the model, hardware, or ollama version changes. (Benchmarks don't run in
CI: shared runners are too noisy for latency assertions.)

Model routing tiers (the first **installed** candidate of the right tier is
used):

| Tier | Models |
|---|---|
| `simple` | `phi3:mini`, `gemma2:2b`, `qwen2.5:0.5b`, `tinyllama` |
| `medium` | `phi3:3.8b`, `qwen2.5:3b`, `gemma2:9b` |
| `complex` (selected) | `qwen2.5:7b`, `qwen2.5:14b`, `llama3.2`, `deepseek-llm:7b` |

Hard reasoning on a thinking model uses a generous token budget and wall-clock
cap so the model can finish its (hidden) chain of thought and emit the real
final answer — measured: qwen3:4b needs ~2000+ hidden tokens on GSM8K, so a
short box returned truncated reasoning and scored 0/3, while letting it finish
scores 3/3 (raw baseline: 2/3). An answer cut mid-sentence still gets one short
continuation pass. When a call fails it degrades gracefully — optimized → plain
retry → raw generator → a neutral `"I couldn't finish, please retry"`. Answers
are always the model's real final `response`, not its hidden reasoning text.
Repeat questions are served from cache in ~0ms.

---

## Tests

```bash
pytest
```

---

## Licence

MIT