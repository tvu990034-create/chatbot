@echo off
REM Simple batch file to run benchmark with correct Python environment
cd /d "%~dp0"
.venv\Scripts\python.exe main.py advanced-benchmark %*