# Quick Start Guide

## For Users Who Just Want to Use It

### Option 1: Download and Run (Simplest)

1. **Download the model** (636MB):
   - Go to: https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
   - Download: `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
   - Place it in a `models/` folder

2. **Install Python** (if not installed):
   - Download from: https://python.org
   - Version: 3.10 or higher
   - IMPORTANT: Check "Add Python to PATH" during installation

3. **Clone and setup**:
```bash
git clone https://github.com/tvu990034-create/chatbot.git
cd chatbot
python -m venv .venv
.venv\Scripts\activate    # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

4. **Run the chatbot**:
```bash
python cli_chat.py
```

### Option 2: Use Without Cloning (Download ZIP)

1. Download the ZIP from GitHub
2. Extract to a folder
3. Follow steps 1-4 from Option 1 (skip the git clone part)

## What You Need

- **Python 3.10+**
- **GGUF model file** (636MB for tinyllama, or any llama.cpp compatible model)
- **8GB RAM minimum** (16GB recommended)

## First Run

When you first run `python cli_chat.py`:
1. It will load the knowledge base (~7 seconds)
2. The model will load on first LLM question (~77 seconds)
3. Subsequent questions will be fast

## Commands

### Chat in terminal:
```bash
python cli_chat.py
```

### Chat with optimizations:
```bash
python cli_chat_optimized.py
```

### Web server:
```bash
python run_local_chatbot.py
# Then open http://localhost:8000
```

## Common Issues

**"Python was not found"**: Install Python from python.org (not Microsoft Store)

**"Module not found"**: Make sure you activated the virtual environment with `.venv\Scripts\activate`

**"Model not found"**: Download the GGUF file and place it in `models/` folder

**Slow first response**: Normal - model loads on first LLM question (~77s). Subsequent responses are faster.

## Need Help?

Open an issue on GitHub: https://github.com/tvu990034-create/chatbot/issues
