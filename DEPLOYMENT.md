# Deployment Guide

This guide covers deployment options for the local AI chatbot.

## Table of Contents

- [Local Deployment](#local-deployment)
- [Docker Deployment](#docker-deployment)
- [GitHub Actions CI/CD](#github-actions-cicd)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

---

## Local Deployment

### Prerequisites

- Python 3.10 or higher
- GGUF model file (TinyLlama or compatible)
- 4GB+ RAM minimum (8GB+ recommended)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/tvu990034-create/chatbot.git
   cd chatbot-phase1
   ```

2. **Download the model**
   ```bash
   mkdir models
   # Download TinyLlama from HuggingFace
   # https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
   # Save to: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
   ```

3. **Create virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # macOS/Linux
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment**
   ```bash
   Copy-Item .env.example .env  # Windows PowerShell
   # cp .env.example .env  # macOS/Linux
   ```

6. **Run the chatbot**
   ```bash
   # CLI mode
   python cli_chat.py
   
   # Web server mode
   python run_local_chatbot.py
   ```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t local-chatbot .
```

### Run Docker Container

```bash
# CLI mode
docker run -it --rm local-chatbot python cli_chat.py

# Web server mode
docker run -p 8000:8000 --rm local-chatbot
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  chatbot:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    environment:
      - MODEL_PATH=models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
      - LAZY_LOAD_MODEL=true
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
```

---

## GitHub Actions CI/CD

### CI Workflow

The CI workflow (`.github/workflows/ci.yml`) runs on every push and pull request:

- Checks out code
- Sets up Python 3.12
- Installs dependencies
- Tests CLI imports
- Tests lazy loading initialization
- Tests fast path (FAQ) responses
- Tests model path handling

### Deployment Workflow

The deployment workflow (`.github/workflows/deploy.yml`) runs on main branch pushes:

- **Build and Test**: Runs all CI tests
- **Build and Push**: Builds Docker image and pushes to GitHub Container Registry

### Required Secrets

For Docker deployment, configure these secrets in GitHub repository settings:

- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

### Manual Trigger

To manually trigger deployment:

```bash
git push origin main
```

---

## Environment Variables

### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | Path to GGUF model file |
| `KNOWLEDGE_BASE_PATH` | `data/knowledge_base` | Path to knowledge base directory |

### Model Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `N_CTX` | `2048` | Context window size |
| `N_THREADS` | `0` | Number of threads (0 = auto) |
| `N_GPU_LAYERS` | `-1` | GPU layers (-1 = all, 0 = CPU only) |
| `N_BATCH` | `512` | Batch size |
| `TEMPERATURE` | `0.1` | Sampling temperature (0-2) |
| `MAX_TOKENS` | `512` | Maximum output tokens |

### Retrieval Parameters

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_LIMIT` | `3` | Maximum chunks to retrieve |

### Fast Paths

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_FAQ` | `true` | Enable FAQ fast path |
| `USE_ZERO_TOKEN` | `true` | Enable zero-token greeting responses |

### Optimization

| Variable | Default | Description |
|----------|---------|-------------|
| `ONE_LINER_MODE` | `false` | Enable one-liner responses |
| `LAZY_LOAD_MODEL` | `true` | Enable lazy model loading |

### Server

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

---

## Troubleshooting

### Docker Build Fails

**Problem**: Docker build fails with dependency errors

**Solution**: Ensure base image has required system dependencies:

```dockerfile
RUN apt-get update && apt-get install -y gcc g++
```

### CI Tests Fail

**Problem**: CI tests fail on import errors

**Solution**: Check that all required dependencies are in `requirements.txt` and Python version is 3.12

### Microsoft Visual C++ Build Tools Error

**Problem**: Installation fails with "Microsoft Visual C++ 14.0 or greater is required"

**Solution**: This error occurs if you try to install `requirements-cloud.txt` on Windows without build tools. For the local chatbot, use `requirements.txt` instead (core dependencies only). If you need cloud app dependencies, install Microsoft C++ Build Tools from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Model Not Found

**Problem**: Chatbot shows "Model not found" error

**Solution**: 
- Ensure model file exists in `models/` directory
- Check `MODEL_PATH` in `.env` matches actual file
- For Docker, mount models volume: `-v ./models:/app/models`

### Slow Initialization

**Problem**: Chatbot takes long to start

**Solution**: Ensure `LAZY_LOAD_MODEL=true` in `.env` for instant startup

### Port Already in Use

**Problem**: Web server fails to start with "Address already in use"

**Solution**: Change `PORT` in `.env` or stop conflicting service

### Memory Issues

**Problem**: Chatbot runs out of memory

**Solution**:
- Reduce `N_CTX` (e.g., from 2048 to 1024)
- Reduce `N_BATCH` (e.g., from 512 to 256)
- Use smaller model (e.g., Q4_K_M instead of Q8_0)
- Disable GPU layers: `N_GPU_LAYERS=0`

---

## Performance Tips

1. **Enable Lazy Loading**: Set `LAZY_LOAD_MODEL=true` for instant startup
2. **Use GPU**: Set `N_GPU_LAYERS=-1` if GPU available
3. **Tune Batch Size**: Adjust `N_BATCH` based on CPU cores
4. **Reduce Context**: Lower `N_CTX` if memory constrained
5. **Use Fast Paths**: Keep `USE_FAQ=true` and `USE_ZERO_TOKEN=true`

---

## Security Considerations

- **Do not commit secrets**: Keep API keys and sensitive data out of `.env`
- **Use firewall**: Restrict access to web server in production
- **Update dependencies**: Regularly update dependencies for security patches
- **Monitor logs**: Watch for unusual activity in server logs

---

## Support

For issues or questions:
- Open an issue on GitHub: https://github.com/tvu990034-create/chatbot/issues
- Check documentation: [README.md](README.md)
- Review bug reports: [BUG_SUMMARY.md](BUG_SUMMARY.md)
