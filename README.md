# Local AI Chatbot - Terminal CLI with Optimization Layer

A high-performance, local AI chatbot with an **advanced optimization layer** for faster and smarter responses using multi-layer caching, advanced retrieval, and intelligent fast paths.

## 🚀 Key Features

### Optimization Layer (speed_engine)
- **Multi-Layer Caching**: Exact match → SimHash → BM25 cascade
- **Advanced Retrieval**: BM25 + FAISS dense + PageRank + cross-encoder reranking
- **Smart Fast Paths**: FAQ database (~4ms) and zero-token responder (~2ms)
- **Score Gating**: Filter low-confidence retrieval results
- **Dynamic Token Allocation**: Adjust tokens based on query complexity
- **Query Truncation**: Essential keyword extraction
- **PageRank Pruning**: Graph-based result filtering

### Core Features
- **Terminal CLI Interface**: Chat directly from your command line
- **Lazy Loading**: Instant startup (~7s), model loads only when needed
- **Local Inference**: Runs on your machine with llama-cpp-python
- **Web Server Option**: Optional FastAPI web interface
- **LangGraph Workflow**: Modern conversation graph with retrieve-then-generate
- **Production Ready**: Docker deployment, health checks, CI/CD

## Project Structure

```
chatbot-phase1/
├── local_chatbot/         # Main CLI application package
│   ├── __init__.py
│   ├── server.py          # FastAPI web server (optional)
│   ├── config.py          # Configuration management
│   ├── engine.py          # Local LLM engine with lazy loading
│   ├── graph.py           # LangGraph conversation workflow
│   ├── rag.py             # RAG pipeline integration
│   └── static/            # Web UI assets
├── speed_engine/          # Advanced retrieval and optimization
│   ├── retrieval.py       # BM25 + FAISS retrieval
│   ├── prefilter.py       # FAQ and zero-token fast paths
│   └── prompt.py          # Minimal prompt builders
├── app/                   # Legacy cloud app (for reference)
├── cli_chat.py            # Terminal CLI entry point
├── run_local_chatbot.py   # Web server entry point
├── requirements.txt       # Python dependencies
├── .env                   # Environment configuration
├── render.yaml            # Render deployment config
└── .github/
    └── workflows/
        ├── ci.yml         # CI pipeline
        └── deploy.yml     # Deployment pipeline
```

## Quick Start

### Local Development

1. **Install dependencies:**
```bash
python -m venv .venv312
.venv312\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment (optional):**
```bash
# The .env file already exists with default settings
# Edit .env to customize model path, knowledge base, etc.
```

3. **Run the CLI:**
```bash
python cli_chat.py
```

4. **Or run the web server:**
```bash
python run_local_chatbot.py
```

5. **Health check (web server):**
```bash
curl http://localhost:8000/health
```

### Docker Deployment (Web Server)

1. **Build the image:**
```bash
docker build -t local-chatbot .
```

2. **Run the container:**
```bash
docker run -p 8000:8000 local-chatbot
```

3. **Access the web UI:**
Open http://localhost:8000 in your browser

## 📊 Performance Metrics

**Terminal CLI Performance:**
- **Startup Time**: < 1s (lazy loading)
- **FAQ Responses**: ~30ms (no model load)
- **LLM Responses**: ~2-3s (includes model load + generation)
- **Knowledge Base**: 44,707 documents loaded in ~3.4s

**Web Server Performance:**
- **Startup Time**: < 1s (lazy loading)
- **Health Check**: Instant
- **Chat API**: Same performance as CLI

**Architecture Optimizations:**
- Multi-layer caching (exact match, simhash, normalized phrases)
- Lazy initialization with eager loading for small datasets
- Dynamic index switching (BM25 vs dense based on query complexity)
- Essential keyword truncation (Eq 2)
- PageRank-based result pruning (Eq 3)
- Query embedding caching
- Pre-compiled regex patterns
- Combined fast-path lookups (cache, zero-token, FAQ)

## 🏗️ Architecture

### Core Components

1. **Local Engine** (`local_chatbot/engine.py`)
   - llama-cpp-python integration
   - Lazy loading for instant startup
   - Simulated mode fallback

2. **RAG Pipeline** (`local_chatbot/rag.py`)
   - Integration with speed_engine retrieval
   - FAQ database for fast responses
   - Zero-token responder for trivial inputs

3. **Conversation Graph** (`local_chatbot/graph.py`)
   - LangGraph-based workflow
   - Fast path → retrieve → generate
   - In-memory conversation history

4. **Speed Engine** (`speed_engine/`)
   - BM25 + FAISS dense retrieval
   - Cross-encoder reranking
   - PageRank-based scoring
   - Dynamic index switching

### API Endpoints (Web Server)

#### Health Check
```http
GET /health
```
Returns server status and model loading state.

#### Chat Endpoint (Streaming)
```http
POST /chat
Content-Type: application/json

{
  "message": "What is KV cache?",
  "session_id": "user123"
}
```
Returns streaming response with Server-Sent Events (SSE).

#### Chat Endpoint (Sync)
```http
POST /chat/sync
Content-Type: application/json

{
  "message": "What is KV cache?",
  "session_id": "user123"
}
```
Returns complete response in one call.

## ⚙️ Configuration

The application uses environment variables for configuration. See `.env` for all available options.

### Key Configuration Options

- `MODEL_PATH`: Path to GGUF model file (default: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)
- `KNOWLEDGE_BASE_PATH`: Path to knowledge base directory (default: data/knowledge_base)
- `N_CTX`: Context window size (default: 2048)
- `N_THREADS`: Number of CPU threads (default: 0 = auto)
- `N_GPU_LAYERS`: Number of GPU layers (default: -1 = all)
- `MAX_TOKENS`: Maximum tokens to generate (default: 512)
- `CHUNK_LIMIT`: Number of retrieved chunks (default: 3)
- `USE_FAQ`: Enable FAQ fast path (default: true)
- `USE_ZERO_TOKEN`: Enable zero-token responder (default: true)
- `LAZY_LOAD_MODEL`: Load model on first LLM request (default: true)
- `ONE_LINER_MODE`: Use one-liner prompt format (default: false)
- `PORT`: Web server port (default: 8000)

### Performance Modes

**Lazy Loading (default, LAZY_LOAD_MODEL=true):**
- Instant startup (~7s for RAG only)
- FAQ responses: ~4ms (457x faster)
- Zero-token: ~2ms (964x faster)
- First LLM: ~77s (includes model load)
- Best for FAQ-heavy workloads

**Eager Loading (LAZY_LOAD_MODEL=false):**
- Slower startup (~17-20s for RAG + model)
- FAQ responses: ~4ms (457x faster)
- Zero-token: ~2ms (964x faster)
- LLM responses: ~18s (model already loaded)
- Best for LLM-heavy workloads

## CI/CD

The project includes GitHub Actions workflows (`.github/workflows/`):

### CI Workflow (`ci.yml`)
- Checks out the code
- Sets up Python 3.12
- Installs dependencies
- Tests CLI imports
- Tests CLI initialization (lazy loading)
- Tests fast path (FAQ responses)
- Tests web server startup
- Tests model path handling

### Deploy Workflow (`deploy.yml`)
- Builds and tests the application
- Builds and pushes Docker image
- Deploys to Kubernetes (optional)
- Deploys to Render (optional)
- Runs security scans







## 🔌 Integration Examples

### Python Client (Web API)
```python
import requests
import json

def chat(message, session_id="default"):
    response = requests.post(
        "http://localhost:8000/chat/sync",
        json={"message": message, "session_id": session_id}
    )
    return response.json()

result = chat("What is KV cache?")
print(result["response"])
print(f"Source: {result['source']}")
print(f"Latency: {result['latency_ms']}ms")
```

### Programmatic CLI Usage
```python
from local_chatbot.config import ChatbotConfig
from local_chatbot.graph import LocalChatGraph

cfg = ChatbotConfig()
chatbot = LocalChatGraph(cfg)

result = chatbot.chat("What is the capital of France?")
print(result.response)
```

## 📈 Monitoring & Observability

### Health Endpoint
- `/health` - Basic health check, returns model loading state

### Performance Monitoring
Track these key metrics:
- Startup time (lazy loading)
- FAQ response latency
- LLM response latency
- Source of response (faq, zero_token, llm)

## 🤝 Support

For issues or questions, please open an issue on GitHub.

## 📝 License

[Add your license information here - e.g., MIT, Apache 2.0]

## 🙏 Acknowledgments

Built with:
- FastAPI for the web framework
- llama-cpp-python for local inference
- LangGraph for conversation workflow
- FAISS for efficient similarity search
- Sentence Transformers for embeddings
