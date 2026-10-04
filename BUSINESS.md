# Local Chatbot for Business — faster local inference that cuts AI operating cost

A fully local, MIT-licensed chatbot and inference-optimization stack. Companies
point their existing OpenAI-compatible harnesses at it and immediately spend
less time and compute per answer — which is more evaluation throughput, more
synthetic training data per day, and no per-token vendor bills.

## The business case in one paragraph

Most production LLM traffic is repetitive: prompt templates, few-shot
examples, retries, eval repeats, support macros. This stack serves those in
milliseconds from cache instead of re-generating them, answers greetings and
math instantly without touching a model, routes trivial questions to a small
fast model and hard reasoning to a strong one, and bounds every request so
one slow generation can never stall a batch. Measured on CPU hardware:
template/solver answers in ~0ms vs ~30s raw, factual answers at ~0.3x raw
latency, repeat questions free — with zero crashes across hundreds of live
runs and a 667-test regression suite guarding every release.

## How companies use it

**1. Drop-in OpenAI endpoint (recommended).** No harness changes — only the
base URL:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
r = client.chat.completions.create(model="qwen3:4b", messages=[...])
```

Works with eval harnesses, data-generation loops, and agent trainers.
`GET /v1/models` lists the live backend models.

**2. REST server.** `python main.py api` — chat, streaming, RAG ingest/query,
health and OpenAPI docs included. Horizontal scaling is standard FastAPI
behind any load balancer.

**3. Embedded library.** Import the gateway, optimizer CLI, or equation
modules directly into existing Python pipelines (see README).

## Proven use cases

- **Eval harnesses**: run benchmarks locally at a fraction of vendor cost,
  with deterministic caching across repeated items.
- **Synthetic training data**: generate graded Q&A datasets offline
  (`optimizer_cli.py eval` writes accuracy reports per batch).
- **Support copilots**: instant answers for greetings, FAQs, and arithmetic;
  RAG grounding over company docs (`main.py ingest`).
- **Air-gapped deployments**: no API keys, no cloud, no data leaving the
  building — everything runs on local Ollama models.

## Requirements

- Python 3.10–3.13, Ollama, ~5GB for two starter models (`qwen3:4b`,
  `phi3:mini`), ~2–3GB Python packages. CPU-only works; a GPU simply makes
  the same calls faster. RAM note: keep ≥4GB free alongside the loaded
  models for comfortable multi-turn sessions.

## Honest scope

- Throughput is single-model bound: ideal for eval queues, offline
  data-gen, and team copilots — not thousand-QPS serving (put GPU
  replicas behind a balancer for that).
- Depth-vs-speed is a real tradeoff: short correct answers are the
  default; raw verbose mode exists where thoroughness matters.
- Evals are English-centric; reasoning budgets are calibrated for the
  bundled models (re-probe after changing either).

## Commercial

MIT licensed — use it commercially, modify it, ship it. For integration
support, custom model routing, or domain tuning, open a discussion on the
repository and describe the workload.
