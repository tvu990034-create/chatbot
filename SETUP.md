# Setup Guide for Users

## Quick Start (5 minutes)

### 1. Clone the repository
```bash
git clone https://github.com/tvu990034-create/chatbot.git
cd chatbot
```

### 2. Create virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the model (or use your own)
```bash
# Option A: Download tinyllama (recommended for CPU)
mkdir models
# Download from: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
# Place the .gguf file in models/ directory

# Or use the provided model if included
```

### 5. Configure
```bash
# The .env file is already configured with defaults
# Edit .env if you want to change settings
```

### 6. Run the CLI
```bash
python cli_chat.py
```

## Alternative: Use the Optimized Version

For better performance with multi-layer caching:

```bash
python cli_chat_optimized.py
```

## Troubleshooting

### "Python was not found"
- Install Python 3.10+ from https://python.org
- Make sure to check "Add Python to PATH" during installation

### "Module not found: llama_cpp"
- Install llama-cpp-python:
```bash
pip install llama-cpp-python
```

### Model not found
- Check that the GGUF file is in the `models/` directory
- Update `MODEL_PATH` in `.env` to point to your model file

### Slow performance
- Use `LAZY_LOAD_MODEL=true` (default) for instant startup
- Use a smaller model if CPU-only
- Consider GPU acceleration if available

## Performance Modes

### Lazy Loading (default, LAZY_LOAD_MODEL=true)
- Instant startup (~7s)
- FAQ responses: ~4ms
- First LLM: ~77s (includes model load)
- Best for FAQ-heavy workloads

### Eager Loading (LAZY_LOAD_MODEL=false)
- Slower startup (~130s)
- FAQ responses: ~4ms
- LLM responses: ~18s (model already loaded)
- Best for LLM-heavy workloads

## Web Server (Optional)

```bash
python run_local_chatbot.py
```

Then open http://localhost:8000 in your browser.
