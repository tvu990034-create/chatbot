# Local AI Chatbot - Terminal CLI

A fast, local AI chatbot that runs from your terminal. No API keys, no cloud, no GPU required.

**IMPORTANT**: This README provides commands for both Windows PowerShell and macOS/Linux. Use only the commands for your platform!

## 🚀 Quick Start (Copy & Paste)

### Step 1: Clone the repository
```bash
git clone https://github.com/tvu990034-create/chatbot.git
cd chatbot-phase1
```

### Step 2: Download the model (636MB)

**Option A: Automated download (recommended)**

**Windows PowerShell:**
```powershell
# Create models directory (may already exist - ignore error)
New-Item -ItemType Directory -Path models -ErrorAction SilentlyContinue

# Download model,bigger and smarter model can give better ressult tinyllama is jusn an example
Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" -OutFile "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
```

**macOS/Linux:**
```bash
# Create models directory
mkdir -p models

# Download model
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -P models/
```

**Option B: Manual download**
- Visit: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
- Download: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
- Save to: `models/` folder

**Option C: Use a different GGUF model**
- Any llama.cpp compatible GGUF model will work
- Place it in the `models/` folder
- Update `MODEL_PATH` in `.env` if filename differs

### Step 3: Install Python dependencies

**Windows Prerequisites (Important)**:
- Python 3.14 on Windows requires [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to compile llama-cpp-python
- During installation, select "Desktop development with C++"
- **Alternative**: Use Python 3.11 or 3.12 (has pre-built wheels, no C++ Build Tools needed)
- **Alternative**: Download pre-compiled wheel from [llama-cpp-python releases](https://github.com/abetlen/llama-cpp-python/releases)

**Windows PowerShell:**
```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Note**: If you need the legacy cloud app dependencies, use `requirements-cloud.txt` instead (requires Microsoft C++ Build Tools on Windows).

### Step 4: Configure environment

**Windows PowerShell:**
```powershell
Copy-Item .env.example .env
```

**macOS/Linux:**
```bash
cp .env.example .env
```

### Step 5: Run the chatbot
```bash
python cli_chat.py
```

### Step 6: Chat!
```
You: hi
Bot: Hello! How can I help you today?
  [Source: zero_token, Latency: 2ms]

You: What is the capital of France?
Bot: The capital of France is Paris.
  [Source: faq, Latency: 4ms]

You: help
Available commands:
  quit/exit/q - Exit the chatbot
  help/h/? - Show this help message
  clear - Clear conversation history

You: clear
Conversation history cleared.

You: quit
```

**Available CLI Commands:**
- `quit/exit/q` - Exit the chatbot
- `help/h/?` - Show available commands
- `clear` - Clear conversation history

## 📋 All Available Commands

### Terminal CLI
```bash
# Basic chat (lazy loading - instant startup)
python cli_chat.py

# Optimized chat (with full optimization layer)
python cli_chat_optimized.py
```

### Web Server

**Windows PowerShell:**
```powershell
python run_local_chatbot.py
# Then open: http://localhost:8000 in your browser
```

**macOS/Linux:**
```bash
python run_local_chatbot.py
# Then open: http://localhost:8000 in your browser
```

### Benchmarking
```bash
# Test basic performance
python benchmark_cli.py

# Test after model is loaded
python benchmark_cli_after_load.py

# Compare basic vs optimized
python benchmark_optimized.py

# Test cache performance
python benchmark_cache.py

# Test real user speed
python test_user_speed.py
```

## 🎯 Performance

| Feature | Speed |
|---------|-------|
| Startup | ~7 seconds |
| FAQ responses | ~4ms (457x faster) |
| Zero-token | ~2ms (964x faster) |
| First LLM | ~77s (includes model load) |
| Subsequent LLM | ~18s |
| Cache hit | 2.37x faster |

## ⚙️ Configuration

Edit `.env` to customize:

```bash
# Model settings
MODEL_PATH=models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
KNOWLEDGE_BASE_PATH=data/knowledge_base

# Performance mode
LAZY_LOAD_MODEL=true    # true = instant startup, false = load model at startup

# Retrieval
CHUNK_LIMIT=3
USE_FAQ=true
USE_ZERO_TOKEN=true

# Server
PORT=8000
```

## 📖 Advanced Usage

### Switch to Eager Loading (for LLM-heavy usage)
```bash
# Edit .env
LAZY_LOAD_MODEL=false
```

### Use the Optimization Layer
```bash
python cli_chat_optimized.py
```

Features:
- Multi-layer caching (exact → SimHash → BM25)
- Advanced retrieval (BM25 + FAISS + PageRank)
- Smart fast paths (FAQ, zero-token)
- Score gating
- Dynamic token allocation

## 🔧 Troubleshooting

### "Python was not found"
Install Python 3.10+ from https://python.org (not Microsoft Store)

### "Module not found: llama_cpp"
```bash
pip install llama-cpp-python
```

### "Model not found"
- Ensure GGUF file is in `models/` folder
- Check `MODEL_PATH` in `.env`

### Slow first response
Normal - model loads on first LLM question (~77s). Subsequent responses are faster.

## 📚 Documentation

- [MODES_EXPLAINED.md](MODES_EXPLAINED.md) - Performance modes explained
- [USER_SPEED_TEST_RESULTS.md](USER_SPEED_TEST_RESULTS.md) - User speed test results
- [SETUP.md](SETUP.md) - Detailed setup guide
- [QUICK_START.md](QUICK_START.md) - Quick start guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide (Docker, CI/CD)
- [BUG_SUMMARY.md](BUG_SUMMARY.md) - Comprehensive bug audit summary

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details
