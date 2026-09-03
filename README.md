# 🤖 Local Chatbot

A fully local, production-grade AI chatbot that integrates the best open-source
libraries into one cohesive application with advanced performance optimizations.

| Component | Library | Role |
|---|---|---|
| LLM serving (GPU) | **vLLM** / **SGLang** | High-throughput local inference |
| LLM gateway | **LiteLLM** | Unified API for 100+ providers |
| Agent orchestration | **LangGraph** | Stateful, tool-using agent |
| RAG (option A) | **LlamaIndex** | Document ingestion + retrieval |
| RAG (option B) | **Haystack** | Pipeline-based RAG |
| Vector store | **ChromaDB** | Persistent embedding store |
| Code awareness | **Aider** (optional) | Repo map, file read, code search |
| REST API | **FastAPI** | OpenAI-compatible HTTP backend |
| Chat UI | **Gradio** | Interactive web interface |

## ✨ What's New

- **Simplified Configuration** - Streamlined `.env.example` with advanced options separated
- **Modular Architecture** - Refactored gateway into focused, maintainable modules
- **Optional Performance Features** - Advanced optimizations can be toggled on/off
- **Enhanced Code Tools** - Lightweight code awareness without full Aider dependency
- **Test Suite** - Basic test structure for core functionality
- **Better Documentation** - Comprehensive development guide
- **Advanced ML Optimizations** - Safe implementations of cutting-edge concepts (HRR, MoSE, Koopman, Neural ODE)

---

## Project structure

```
local-chatbot/
├── main.py                  # CLI entry point (Typer)
├── config.py                # All settings via Pydantic + .env
├── .env.example             # Template – copy to .env
├── requirements.txt
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
│
├── gateway/
│   └── litellm_gateway.py   # LiteLLM wrapper (sync + async, streaming, fallback)
│
├── backends/
│   └── model_server.py      # vLLM / SGLang / Ollama subprocess launcher
│
├── rag/
│   ├── llama_index_rag.py   # LlamaIndex vector index + query engine
│   └── haystack_pipeline.py # Haystack indexing + query pipelines
│
├── agents/
│   └── langgraph_agent.py   # LangGraph graph (RAG node + agent node + tools)
│
├── tools/
│   └── aider_tool.py        # RepoMapTool, FileReadTool, CodeSearchTool, AiderEditTool
│
├── server/
│   └── app.py               # FastAPI app with REST endpoints
│
├── ui/
│   └── gradio_ui.py         # Gradio Blocks UI
│
└── data/
    ├── docs/                # Drop documents here for RAG ingestion
    ├── indexes/             # LlamaIndex persistent index
    └── chroma/              # ChromaDB persistent store
```

---

## Quick start (CPU, no GPU needed)

### 1. Install dependencies

```bash
cd local-chatbot
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Optional:** For enhanced code awareness capabilities:
```bash
pip install -r requirements-code-tools.txt
```

```bash
cd local-chatbot
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# The defaults use Ollama with llama3 – no API keys required
```

For advanced performance tuning, see `.env.advanced.example` and copy relevant sections to your `.env` file.

### 3. Install and start Ollama

Download Ollama from https://ollama.com, then:

```bash
ollama pull llama3
```

### 4. Launch

```bash
# Gradio UI only (http://localhost:7860)
python main.py ui

# FastAPI only (http://localhost:8000)
python main.py api

# Both at the same time
python main.py both
```

---

## Using a GPU backend (vLLM or SGLang)

### vLLM

```bash
pip install vllm          # NVIDIA GPU + CUDA required
```

In `.env`:

```
LOCAL_BACKEND=vllm
LOCAL_MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
LOCAL_BACKEND_PORT=8080
DEFAULT_MODEL=openai/meta-llama/Meta-Llama-3-8B-Instruct
LITELLM_API_BASE=http://localhost:8080/v1
```

Start the backend first, then launch the chatbot:

```bash
python main.py status   # check config
python main.py both
```

### SGLang

```bash
pip install sglang        # NVIDIA / AMD GPU required
```

In `.env`:

```
LOCAL_BACKEND=sglang
LOCAL_MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
LOCAL_BACKEND_PORT=8080
DEFAULT_MODEL=openai/meta-llama/Meta-Llama-3-8B-Instruct
LITELLM_API_BASE=http://localhost:8080/v1
```

---

## Using cloud models (OpenAI / Anthropic / OpenRouter)

Add the relevant key to `.env` and set the model:

```bash
OPENAI_API_KEY=sk-...
DEFAULT_MODEL=openai/gpt-4o
```

```bash
ANTHROPIC_API_KEY=sk-ant-...
DEFAULT_MODEL=anthropic/claude-3-5-sonnet-20241022
```

LiteLLM handles all routing — no other code changes needed.

---

## RAG (document retrieval)

### Add documents

Drop any `.txt`, `.pdf`, `.md`, `.py` files into `data/docs/`, then:

```bash
python main.py ingest
```

Or upload via the **Documents** tab in the UI, or POST to the API:

```bash
curl -X POST http://localhost:8000/rag/ingest \
  -F "files=@my-document.pdf"
```

### Switch RAG provider

In `.env`:

```
RAG_PROVIDER=llama_index   # or: haystack | both | none
```

---

## CLI reference

```
python main.py --help

Commands:
  ui         Launch Gradio chat UI
  api        Launch FastAPI REST server
  both       Launch API + UI concurrently
  ingest     Ingest documents into the RAG index
  chat       Send a single message (terminal)
  status     Show config and backend health
  benchmark  Run hardware benchmark and model recommendations
```

```bash
# Examples
python main.py ui --share                         # public Gradio link
python main.py api --reload                       # dev mode with auto-reload
python main.py chat "Explain RAG in one sentence"
python main.py ingest --path ./my-docs --rebuild
python main.py status
python main.py benchmark                        # show CPU recommendations
python main.py benchmark --gpu "RTX 4090"        # simulate GPU
python main.py benchmark --gpu "RTX 4090" --top 3 --speed fast
python main.py benchmark --json                  # JSON output
```

---

## REST API reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | App info |
| GET | `/health` | Liveness probe |
| GET | `/models` | List available models |
| POST | `/chat` | Single-turn chat (JSON) |
| POST | `/chat/stream` | Streaming chat (SSE) |
| POST | `/rag/ingest` | Upload + ingest documents |
| POST | `/rag/query` | Direct RAG query |
| GET | `/backend/status` | Local backend health |

Interactive docs: http://localhost:8000/docs

### Example chat request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is PagedAttention?",
    "use_rag": true,
    "use_agent": true
  }'
```

### Streaming (SSE)

```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about LangGraph"}' \
  --no-buffer
```

---

## Docker deployment

```bash
# CPU stack (Ollama + ChromaDB + FastAPI + Gradio)
docker compose up -d

# Pull a model into Ollama
docker compose exec ollama ollama pull llama3

# View logs
docker compose logs -f chatbot-api

# Stop everything
docker compose down
```

For GPU (vLLM or SGLang), uncomment the relevant service block in
`docker-compose.yml` and ensure `nvidia-container-toolkit` is installed.

---

## Configuration reference

All settings live in `.env` (or environment variables). Key options:

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_MODEL` | `ollama/llama3` | LiteLLM model string |
| `LOCAL_BACKEND` | `ollama` | `vllm` / `sglang` / `ollama` / `none` |
| `LOCAL_MODEL_NAME` | `meta-llama/Meta-Llama-3-8B-Instruct` | HuggingFace model ID |
| `LITELLM_API_BASE` | _(auto)_ | Override provider base URL |
| `RAG_PROVIDER` | `llama_index` | `llama_index` / `haystack` / `both` / `none` |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence-transformers model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` / `cuda` / `mps` |
| `RAG_CHUNK_SIZE` | `512` | Tokens per chunk |
| `RAG_TOP_K` | `5` | Chunks retrieved per query |
| `AGENT_MAX_ITERATIONS` | `10` | LangGraph max tool calls |
| `AIDER_REPO_PATH` | _(cwd)_ | Repo for code tools |
| `AIDER_READ_ONLY` | `true` | Disable file writes |
| `OPENAI_API_KEY` | _(blank)_ | OpenAI key (optional) |
| `ANTHROPIC_API_KEY` | _(blank)_ | Anthropic key (optional) |

See `.env.example` for the full list. For advanced performance tuning options, see `.env.advanced.example`.

---

## Development

For development setup, testing, and contribution guidelines, see [DEVELOPMENT.md](DEVELOPMENT.md).

### Advanced ML Optimizations

The chatbot includes safe implementations of advanced ML optimization concepts:

- **HRR-Inspired History Compression** - Fixed-size conversation context
- **MoSE-Inspired Request Segmentation** - Parallel processing of long inputs  
- **Koopman-Inspired Context Mixing** - Unified RAG result aggregation
- **Neural ODE-Inspired Adaptive Iterations** - Convergence-based agent loops

⚠️ **Important:** These optimizations apply concepts to the orchestration layer only and do NOT modify LLM architectures. All are disabled by default. See [ADVANCED_OPTIMIZATIONS.md](ADVANCED_OPTIMIZATIONS.md) for details.

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py

# Run with coverage
pytest --cov=. --cov-report=html
```

---

## Architecture overview

```
User
 │
 ▼
Gradio UI (port 7860)
 │    or
 ▼
FastAPI (port 8000)
 │
 ▼
LangGraph Agent
 ├── RAG node  ──► LlamaIndex ──► ChromaDB
 │               └► Haystack  ──► InMemory store
 │
 ├── Agent node ──► LiteLLM gateway
 │                   ├── Ollama  (local CPU)
 │                   ├── vLLM   (local GPU)
 │                   ├── SGLang (local GPU)
 │                   ├── OpenAI (cloud)
 │                   └── Anthropic / OpenRouter / …
 │
 └── Tool node
      ├── RepoMapTool   (Aider repomap)
      ├── FileReadTool  (safe file reader)
      ├── CodeSearchTool (grep / regex)
      └── AiderEditTool  (write mode only)
```

---

## Licence

MIT
