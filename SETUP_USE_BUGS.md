# Setup and Use Bugs

## Summary

Found **8 setup and use bugs** that could prevent users from successfully setting up and using the chatbot.

**Total**: 8 bugs
**Priority**: High (blocks user onboarding)
**Status**: Documented for fixing

---

## Setup and Use Bugs

### Setup Bug #56: No Automated Model Download

**Location**: README.md lines 13-23

**Issue**: Model download instructions are manual - user must:
1. Manually create `models/` directory
2. Manually visit HuggingFace URL
3. Manually download file
4. Manually save with correct filename

**Impact**: High friction for users, prone to errors

**Current State**:
```bash
# Create models directory
mkdir models

# Download from this link:
# https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
# Save it as: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

**Fix**: Add wget/curl command or Python script to download model automatically:
```bash
# Option A: wget
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -P models/

# Option B: curl
curl -L https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf -o models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Option C: Python script
python download_model.py
```

---

### Setup Bug #57: No Directory Creation in Setup Instructions

**Location**: README.md

**Issue**: Setup instructions don't explicitly mention creating required directories:
- `models/` - for GGUF model file
- `data/knowledge_base/` - for knowledge base (though it exists in repo)

**Impact**: Users may fail to create directories, leading to "not found" errors

**Current State**: Only mentions creating `models/` in model download section

**Fix**: Add explicit directory creation step:
```bash
# Create required directories
mkdir models
mkdir -p data/knowledge_base
```

---

### Setup Bug #58: No .env File Validation Before Running

**Location**: CLI entry points (cli_chat.py, run_local_chatbot.py)

**Issue**: If user forgets to copy `.env.example` to `.env`, the chatbot will still run with defaults, but user won't know they're missing configuration options

**Impact**: User may not know about available configuration options

**Current State**: load_dotenv() silently fails if .env doesn't exist, uses defaults

**Fix**: Add validation and helpful message:
```python
from pathlib import Path
if not Path(".env").exists():
    print("Warning: .env file not found. Using default configuration.")
    print("To customize settings, copy .env.example to .env")
```

---

### Setup Bug #59: No Check for llama-cpp-python Installation

**Location**: CLI entry points

**Issue**: If llama-cpp-python is not installed, the chatbot will silently fall back to simulated mode without clear explanation

**Impact**: User may not realize the model isn't actually being used

**Current State**: Engine catches ImportError and sets use_simulated=True with logging only

**Fix**: Add user-facing warning:
```python
try:
    from llama_cpp import Llama
except ImportError:
    print("Warning: llama-cpp-python not installed.")
    print("Install with: pip install llama-cpp-python")
    print("Falling back to simulated mode for testing.")
    self.use_simulated = True
```

---

### Setup Bug #60: No Model Existence Check at Startup

**Location**: cli_chat.py, local_chatbot/server.py

**Issue**: Chatbot initializes successfully even if model file doesn't exist, only fails when first question is asked

**Impact**: User doesn't know model is missing until they try to use it

**Current State**: Model validation happens in engine __init__ but only logs warning

**Fix**: Add explicit check at startup:
```python
if not os.path.exists(cfg.model_path):
    print(f"Error: Model file not found at {cfg.model_path}")
    print("Please download the model. See README.md for instructions.")
    sys.exit(1)
```

---

### Setup Bug #61: No Setup Script to Automate Setup

**Location**: Root directory

**Issue**: No setup script (setup.py, setup.sh, setup.bat) to automate:
- Virtual environment creation
- Dependency installation
- Model download
- .env file creation
- Directory creation

**Impact**: Users must manually run multiple commands, increasing error risk

**Current State**: Users must run 5+ manual commands

**Fix**: Create setup script:
```bash
#!/bin/bash
# setup.sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
mkdir -p models data/knowledge_base
wget https://huggingface.co/.../tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -P models/
cp .env.example .env
echo "Setup complete! Run: python cli_chat.py"
```

---

### Setup Bug #62: Python Version Not Explicitly Specified

**Location**: README.md line 170

**Issue**: Troubleshooting says "Install Python 3.10+" but main setup doesn't specify version

**Impact**: Users may install wrong Python version (e.g., 3.9 or 3.14 which may have compatibility issues)

**Current State**: Just says "Python 3.10+"

**Fix**: Add explicit version requirement in main setup:
```bash
### Step 0: Install Python 3.10-3.12
# Download from https://python.org/downloads/
# Recommended: Python 3.12
```

---

### Setup Bug #63: No .gitignore for Sensitive Files

**Location**: Root directory

**Issue**: .gitignore exists but may not cover all sensitive files:
- .env (should never be committed)
- models/*.gguf (large files, shouldn't be in repo)
- test_*.db (test databases)
- cache/ (cache files)
- *.log (log files)

**Impact**: Risk of accidentally committing sensitive data or large files

**Current State**: Need to verify .gitignore contents

**Fix**: Ensure .gitignore includes:
```
.env
models/*.gguf
*.db
cache/
*.log
__pycache__/
*.pyc
.venv/
```

---

## Priority Recommendations

### High Priority (Blocks Onboarding)
1. **Setup Bug #56**: Add automated model download (wget/curl or script)
2. **Setup Bug #57**: Add directory creation to setup instructions
3. **Setup Bug #60**: Add model existence check at startup
4. **Setup Bug #61**: Create setup script to automate setup

### Medium Priority (Improves UX)
5. **Setup Bug #58**: Add .env file validation
6. **Setup Bug #59**: Add llama-cpp-python installation check
7. **Setup Bug #62**: Explicitly specify Python version in main setup

### Low Priority (Best Practices)
8. **Setup Bug #63**: Verify and update .gitignore

---

## Implementation Status

- [ ] Setup Bug #56: Add automated model download
- [ ] Setup Bug #57: Add directory creation to setup instructions
- [ ] Setup Bug #58: Add .env file validation
- [ ] Setup Bug #59: Add llama-cpp-python installation check
- [ ] Setup Bug #60: Add model existence check at startup
- [ ] Setup Bug #61: Create setup script
- [ ] Setup Bug #62: Explicitly specify Python version
- [ ] Setup Bug #63: Verify and update .gitignore

---

## Expected Impact

After fixing setup and use bugs:
- Faster onboarding (automated setup)
- Fewer user errors (directory creation, file downloads)
- Better error messages (missing dependencies, missing model)
- Clearer setup process (explicit steps)
- Professional first impression (setup script)
