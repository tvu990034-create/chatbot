@echo off
REM Setup script for Windows
REM This script automates the entire setup process

echo ========================================
echo   Local AI Chatbot - Setup Script
echo ========================================
echo.

REM Check Python version
echo [1/6] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    echo Please install Python 3.10-3.12 from https://python.org/downloads/
    pause
    exit /b 1
)
echo Python found
echo.

REM Create virtual environment
echo [2/6] Creating virtual environment...
if exist .venv (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv .venv
    echo Virtual environment created
)
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call .venv\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Install dependencies
echo [4/6] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies installed
echo.

REM Create directories
echo [5/6] Creating directories...
if not exist models mkdir models
if not exist data mkdir data
if not exist data\knowledge_base mkdir data\knowledge_base
echo Directories created
echo.

REM Copy .env file
echo [6/6] Configuring environment...
if not exist .env (
    copy .env.example .env >nul
    echo .env file created from .env.example
) else (
    echo .env file already exists, skipping...
)
echo.

REM Check if model exists
if not exist models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf (
    echo ========================================
    echo   Model file not found!
    echo ========================================
    echo.
    echo Please download the model:
    echo   https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    echo.
    echo Save it to: models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
    echo.
    echo Or run this command:
    echo   Invoke-WebRequest -Uri "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" -OutFile "models\tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    echo.
) else (
    echo Model file found
)

echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To run the chatbot:
echo   python cli_chat.py
echo.
echo Or start the web server:
echo   python run_local_chatbot.py
echo.
pause
