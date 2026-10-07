#!/bin/bash
# Setup script for macOS/Linux
# This script automates the entire setup process

set -e

echo "========================================"
echo "  Local AI Chatbot - Setup Script"
echo "========================================"
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found"
    echo "Please install Python 3.10-3.12 from https://python.org/downloads/"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Python $PYTHON_VERSION found"
echo ""

# Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv .venv
    echo "Virtual environment created"
fi
echo ""

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source .venv/bin/activate
echo "Virtual environment activated"
echo ""

# Install dependencies
echo "[4/6] Installing dependencies..."
pip install -r requirements.txt
echo "Dependencies installed"
echo ""

# Create directories
echo "[5/6] Creating directories..."
mkdir -p models data/knowledge_base
echo "Directories created"
echo ""

# Copy .env file
echo "[6/6] Configuring environment..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo ".env file created from .env.example"
else
    echo ".env file already exists, skipping..."
fi
echo ""

# Check if model exists
if [ ! -f "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" ]; then
    echo "========================================"
    echo "  Model file not found!"
    echo "========================================"
    echo ""
    echo "Please download the model:"
    echo "  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    echo ""
    echo "Save it to: models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    echo ""
    echo "Or run this command:"
    echo "  wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -P models/"
    echo ""
else
    echo "Model file found"
fi

echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "To run the chatbot:"
echo "  python cli_chat.py"
echo ""
echo "Or start the web server:"
echo "  python run_local_chatbot.py"
echo ""
