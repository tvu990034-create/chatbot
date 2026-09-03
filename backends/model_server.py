"""
backends/model_server.py
~~~~~~~~~~~~~~~~~~~~~~~~
Launcher and health-checker for local LLM inference backends:
  • vLLM   – high-throughput, PagedAttention, GPU required
  • SGLang – fast prefix-caching serving, GPU required
  • Ollama – CPU-friendly fallback

This module does NOT import vllm or sglang at the top level.
Both are optional heavy dependencies that may not be installed.
Instead it launches them as subprocess servers that expose an
OpenAI-compatible HTTP API, which LiteLLM then routes to.

Usage
-----
    from backends.model_server import start_backend, stop_backend, wait_until_ready

    proc = start_backend()        # starts the configured backend
    wait_until_ready()            # blocks until /health returns 200
    # … run your chatbot …
    stop_backend(proc)
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

from config import BackendType, settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HEALTH_CHECK_URL_VLLM   = f"http://{settings.local_backend_host}:{settings.local_backend_port}/health"
HEALTH_CHECK_URL_SGLANG = f"http://{settings.local_backend_host}:{settings.local_backend_port}/health"
HEALTH_CHECK_URL_OLLAMA = f"http://{settings.local_backend_host}:11434/api/tags"

STARTUP_TIMEOUT_SECONDS = 300   # vLLM/SGLang can be slow to load large models
POLL_INTERVAL_SECONDS   = 3


# ---------------------------------------------------------------------------
# Backend launchers
# ---------------------------------------------------------------------------

def _launch_vllm() -> subprocess.Popen:
    """
    Launch a vLLM OpenAI-compatible server as a subprocess with research paper optimizations.
    Equivalent to:  python -m vllm.entrypoints.openai.api_server …
    
    Implements optimizations from research papers:
    - PagedAttention (#64) - automatic KV cache management
    - Prefix caching (#160, #201) - reuses system prompts
    - Chunked prefill (#71, #72) - dynamic request scheduling
    - Quantization (#11 GPTQ, #12 AWQ, #14 LLM.int8)
    - KV cache optimization (FP8, H2O, StreamingLLM)
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",               settings.local_model_name,
        "--host",                settings.local_backend_host,
        "--port",                str(settings.local_backend_port),
        "--max-model-len",       str(settings.local_backend_max_model_len),
        "--gpu-memory-utilization", str(settings.local_backend_gpu_memory_utilization),
        "--dtype",               settings.local_backend_dtype,
        "--trust-remote-code",
    ]
    
    # Research paper optimizations
    if settings.local_backend_quantization:
        cmd += ["--quantization", settings.local_backend_quantization]
    
    if settings.vllm_enable_prefix_caching:
        cmd += ["--enable-prefix-caching"]
        logger.info("Enabled prefix caching (#160, #201)")
    
    if settings.vllm_enable_chunked_prefill:
        cmd += ["--enable-chunked-prefill"]
        logger.info("Enabled chunked prefill (#71, #72)")
    
    if settings.vllm_enable_paged_attention:
        # PagedAttention is enabled by default in vLLM, but we can configure it
        cmd += ["--enforce-eager"] if settings.vllm_enforce_eager else []
        logger.info("PagedAttention enabled (#64)")
    
    if settings.vllm_kv_cache_dtype != "auto":
        cmd += ["--kv-cache-dtype", settings.vllm_kv_cache_dtype]
        logger.info(f"KV cache dtype set to {settings.vllm_kv_cache_dtype}")
    
    if settings.vllm_max_num_batched_tokens:
        cmd += ["--max-num-batched-tokens", str(settings.vllm_max_num_batched_tokens)]
        logger.info(f"Max batched tokens: {settings.vllm_max_num_batched_tokens}")

    logger.info("Launching vLLM server with research paper optimizations: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _launch_sglang() -> subprocess.Popen:
    """
    Launch an SGLang HTTP server as a subprocess.
    Equivalent to:  python -m sglang.launch_server …
    """
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path",     settings.local_model_name,
        "--host",           settings.local_backend_host,
        "--port",           str(settings.local_backend_port),
        "--mem-fraction-static", str(settings.local_backend_gpu_memory_utilization),
        "--dtype",          settings.local_backend_dtype,
        "--trust-remote-code",
    ]
    if settings.local_backend_quantization:
        cmd += ["--quantization", settings.local_backend_quantization]

    logger.info("Launching SGLang server: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def _launch_ollama() -> subprocess.Popen:
    """
    Launch Ollama serve as a subprocess.
    Ollama must already be installed on the system.
    """
    cmd = ["ollama", "serve"]
    logger.info("Launching Ollama server")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_backend() -> Optional[subprocess.Popen]:
    """
    Start the configured local backend with automatic fallback.

    Returns
    -------
    subprocess.Popen | None
        The server process, or None if backend == NONE.

    Raises
    ------
    RuntimeError  if the backend type is not recognised.
    ImportError   if vllm/sglang are selected but not installed.
    
    Note
    ----
    If vLLM or SGLang fails, automatically falls back to Ollama.
    """
    backend = settings.local_backend

    if backend == BackendType.NONE:
        logger.info("No local backend configured – using cloud APIs only.")
        return None

    if backend == BackendType.VLLM:
        try:
            _check_import("vllm", "pip install vllm  # requires CUDA")
            return _launch_vllm()
        except Exception as e:
            logger.error(f"vLLM failed to start: {e}")
            logger.warning("Automatically falling back to Ollama backend")
            logger.info("To use vLLM, change LOCAL_BACKEND=ollama in .env")
            return _launch_ollama()

    if backend == BackendType.SGLANG:
        try:
            _check_import("sglang", "pip install sglang  # requires CUDA")
            return _launch_sglang()
        except Exception as e:
            logger.error(f"SGLang failed to start: {e}")
            logger.warning("Automatically falling back to Ollama backend")
            return _launch_ollama()

    if backend == BackendType.OLLAMA:
        return _launch_ollama()

    raise RuntimeError(f"Unknown backend: {backend}")


def stop_backend(proc: Optional[subprocess.Popen]) -> None:
    """Terminate the server process gracefully, then forcefully if needed."""
    if proc is None:
        return
    logger.info("Stopping backend process (pid=%s)…", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    logger.info("Backend stopped.")


def wait_until_ready(
    timeout: int = STARTUP_TIMEOUT_SECONDS,
    poll: float = POLL_INTERVAL_SECONDS,
) -> bool:
    """
    Poll the backend health endpoint until it responds 200 or timeout is hit.

    Returns True on success, False on timeout.
    """
    backend = settings.local_backend

    if backend == BackendType.NONE:
        return True

    health_url = {
        BackendType.VLLM:   HEALTH_CHECK_URL_VLLM,
        BackendType.SGLANG: HEALTH_CHECK_URL_SGLANG,
        BackendType.OLLAMA: HEALTH_CHECK_URL_OLLAMA,
    }.get(backend, HEALTH_CHECK_URL_VLLM)

    deadline = time.time() + timeout
    logger.info("Waiting for backend at %s (timeout %ds)…", health_url, timeout)

    while time.time() < deadline:
        try:
            resp = httpx.get(health_url, timeout=5)
            if resp.status_code < 400:
                logger.info("Backend is ready! (%s)", health_url)
                return True
        except httpx.RequestError:
            pass
        time.sleep(poll)

    logger.error("Backend did not become ready within %ds.", timeout)
    return False


def get_backend_status() -> dict:
    """Return a status dict suitable for an API health endpoint."""
    backend = settings.local_backend
    if backend == BackendType.NONE:
        return {"backend": "none", "status": "disabled"}

    health_url = {
        BackendType.VLLM:   HEALTH_CHECK_URL_VLLM,
        BackendType.SGLANG: HEALTH_CHECK_URL_SGLANG,
        BackendType.OLLAMA: HEALTH_CHECK_URL_OLLAMA,
    }.get(backend, HEALTH_CHECK_URL_VLLM)

    try:
        resp = httpx.get(health_url, timeout=3)
        ok = resp.status_code < 400
    except httpx.RequestError:
        ok = False

    return {
        "backend":    backend.value,
        "model":      settings.local_model_name,
        "base_url":   settings.local_backend_base_url,
        "healthy":    ok,
        "health_url": health_url,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_import(package: str, install_hint: str) -> None:
    """Raise ImportError with a helpful message if package isn't installed."""
    import importlib
    if importlib.util.find_spec(package) is None:
        raise ImportError(
            f"'{package}' is not installed. "
            f"Install it with:  {install_hint}"
        )
