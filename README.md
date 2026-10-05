# Local AI Chatbot - Terminal CLI

A fast, local AI chatbot that runs from your terminal. No API keys, no cloud, no GPU required.

## 🚀 Quick Start (Copy & Paste)

### Step 1: Clone the repository
```bash
git clone https://github.com/tvu990034-create/chatbot.git
cd chatbot-phase1
```

### Step 2: Download the model (636MB)

**Option A: Download directly**
```bash
# Create models directory
mkdir models

# Download from this link:
# https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# Save it as: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

**Option B: Use a different GGUF model**
- Any llama.cpp compatible GGUF model will work
- Place it in the `models/` folder
- Update `MODEL_PATH` in `.env` if filename differs

### Step 3: Install Python dependencies
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Step 4: Configure environment
```bash
Copy-Item .env.example .env   # Windows PowerShell
# cp .env.example .env        # macOS/Linux
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
```bash
# Start web server
python run_local_chatbot.py

# Then open: http://localhost:8000
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

## 🤝 Contributing

Feel free to open issues or submit pull requests!

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details
