FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p models data/knowledge_base

# Expose port 8000
EXPOSE 8000

# Set local chatbot environment variables
ENV MODEL_PATH=models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
ENV KNOWLEDGE_BASE_PATH=data/knowledge_base
ENV N_CTX=2048
ENV N_THREADS=0
ENV N_GPU_LAYERS=-1
ENV N_BATCH=512
ENV TEMPERATURE=0.1
ENV MAX_TOKENS=512
ENV CHUNK_LIMIT=3
ENV USE_FAQ=true
ENV USE_ZERO_TOKEN=true
ENV ONE_LINER_MODE=false
ENV LAZY_LOAD_MODEL=true
ENV HOST=0.0.0.0
ENV PORT=8000

# Run the local chatbot web server
CMD ["python", "run_local_chatbot.py"]
