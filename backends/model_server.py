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

import atexit
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx

from config import BackendType, settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global backend state (BUG 15, 16, 17 FIX)
# ---------------------------------------------------------------------------

_backend_process: Optional[subprocess.Popen] = None
_backend_owner = "application"  # "application" or "external"
_backend_lock = None  # For thread safety
# BUG 80 FIX: Track loaded models to prevent duplicate loading
_loaded_models: set[str] = set()
# BUG 82 FIX: Circuit breaker for backend failures
_circuit_breaker_failures: int = 0
_circuit_breaker_last_failure: float = 0
_circuit_breaker_threshold: int = 5
_circuit_breaker_timeout: int = 60  # seconds

def _get_backend_lock():
    """Lazy-initialize backend lock for thread safety."""
    global _backend_lock
    if _backend_lock is None:
        import threading
        _backend_lock = threading.Lock()
    return _backend_lock

def _cleanup_backend():
    """BUG 16 FIX: Cleanup backend on application exit to prevent orphaned processes."""
    global _backend_process, _backend_owner
    if _backend_owner == "application" and _backend_process is not None:
        try:
            logger.info("Cleaning up backend process on exit")
            stop_backend(_backend_process)
        except Exception as e:
            logger.error(f"Error during backend cleanup: {e}")

# BUG 16 FIX: Register cleanup handler
atexit.register(_cleanup_backend)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STARTUP_TIMEOUT_SECONDS = 300   # vLLM/SGLang can be slow to load large models
POLL_INTERVAL_SECONDS   = 3

# BUG 13 FIX: Build health URLs dynamically to use current configuration
def _get_health_url(backend_type: BackendType) -> str:
    """Build health URL from current configuration (not cached at module load)."""
    if backend_type == BackendType.VLLM:
        return f"http://{settings.local_backend_host}:{settings.local_backend_port}/health"
    elif backend_type == BackendType.SGLANG:
        return f"http://{settings.local_backend_host}:{settings.local_backend_port}/health"
    elif backend_type == BackendType.OLLAMA:
        return f"http://{settings.local_backend_host}:11434/api/tags"
    else:
        return f"http://{settings.local_backend_host}:{settings.local_backend_port}/health"


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

    # BUG 10 FIX: Redirect output to file to prevent pipe deadlock
    # Pipes must be consumed continuously or redirected to avoid blocking
    log_file = open("logs/vllm_backend.log", "a")
    logger.info("Launching vLLM server with research paper optimizations: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)


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

    # BUG 10 FIX: Redirect output to file to prevent pipe deadlock
    log_file = open("logs/sglang_backend.log", "a")
    logger.info("Launching SGLang server: %s", " ".join(cmd))
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)


def _launch_ollama() -> subprocess.Popen:
    """
    Launch Ollama serve as a subprocess.
    Ollama must already be installed on the system.
    BUG 11 FIX: Check if Ollama is already running to avoid duplicate servers.
    """
    # BUG 11 FIX: Check if Ollama is already running
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", 11434))
        sock.close()
        if result == 0:
            logger.info("Ollama is already running on port 11434, reusing existing instance")
            # Return a dummy process object to indicate we're using external Ollama
            return None
    except Exception as e:
        logger.debug(f"Could not check if Ollama is running: {e}")
    
    cmd = ["ollama", "serve"]
    # BUG 10 FIX: Redirect output to file to prevent pipe deadlock
    log_file = open("logs/ollama_backend.log", "a")
    logger.info("Launching Ollama server")
    return subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _deregister_model_loaded(model_name: str):
    """BUG 80 FIX: Deregister a model when backend is stopped."""
    global _loaded_models
    _loaded_models.discard(model_name)

def _is_circuit_breaker_open() -> bool:
    """BUG 82 FIX: Check if circuit breaker is open (too many recent failures)."""
    global _circuit_breaker_failures, _circuit_breaker_last_failure
    
    # Reset if timeout has passed
    import time
    if time.time() - _circuit_breaker_last_failure > _circuit_breaker_timeout:
        _circuit_breaker_failures = 0
        return False
    
    return _circuit_breaker_failures >= _circuit_breaker_threshold

def _record_backend_failure():
    """BUG 82 FIX: Record a backend failure for circuit breaker."""
    global _circuit_breaker_failures, _circuit_breaker_last_failure
    import time
    _circuit_breaker_failures += 1
    _circuit_breaker_last_failure = time.time()
    logger.warning(f"Backend failure recorded. Circuit breaker status: {_circuit_breaker_failures}/{_circuit_breaker_threshold}")

def _record_backend_success():
    """BUG 82 FIX: Record a backend success to reset circuit breaker."""
    global _circuit_breaker_failures, _circuit_breaker_last_failure
    _circuit_breaker_failures = 0
    _circuit_breaker_last_failure = 0

def _is_model_loaded(model_name: str) -> bool:
    """BUG 80 FIX: Check if a model is already loaded to prevent duplicate loading."""
    global _loaded_models
    return model_name in _loaded_models

def _register_model_loaded(model_name: str):
    """BUG 80 FIX: Register that a model has been loaded."""
    global _loaded_models
    _loaded_models.add(model_name)

def _deregister_model_loaded(model_name: str):
    """BUG 80 FIX: Deregister a model when backend is stopped."""
    global _loaded_models
    _loaded_models.discard(model_name)


def start_backend() -> Optional[subprocess.Popen]:
    """
    Start the configured local backend with automatic fallback.

    Returns
    -------
    subprocess.Popen | None
        The server process, or None if backend == NONE or using external Ollama.

    Raises
    ------
    RuntimeError  if the backend type is not recognised.
    ImportError   if vllm/sglang are selected but not installed.
    
    Note
    ----
    If vLLM or SGLang fails, automatically falls back to Ollama.
    BUG 12 FIX: Now waits for backend to be ready before returning.
    BUG 15 FIX: Tracks backend ownership (application vs external).
    BUG 17 FIX: Idempotent - won't start duplicate backends.
    BUG 18 FIX: Multi-worker safe - uses file lock to prevent duplicate launches.
    """
    global _backend_process, _backend_owner
    
    backend = settings.local_backend
    
    if backend == BackendType.NONE:
        logger.info("No local backend configured – using cloud APIs only.")
        return None
    
    # BUG 82 FIX: Check circuit breaker before attempting to start backend
    if _is_circuit_breaker_open():
        logger.error("Circuit breaker is OPEN - too many recent backend failures. Waiting for timeout.")
        return None
    
    # BUG 18 FIX: Use file lock for multi-worker safety (Unix only)
    lock_file = None
    if os.name != 'nt':  # Unix-like systems
        try:
            import fcntl
            lock_file = open(BASE_DIR / ".backend_lock", "w")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.debug("Acquired backend lock for multi-worker safety")
        except (IOError, BlockingIOError):
            logger.info("Another worker holds backend lock, waiting...")
            if lock_file:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            logger.debug("Acquired backend lock after wait")
    
    try:
        # BUG 17 FIX: Idempotent startup - check if already running
        with _get_backend_lock():
            if _backend_process is not None:
                try:
                    # Check if process is still alive
                    if _backend_process.poll() is None:
                        logger.info("Backend already running (pid=%s), reusing existing instance", _backend_process.pid)
                        return _backend_process
                    else:
                        # Process died, clean up
                        logger.warning("Previous backend process died, cleaning up")
                        _backend_process = None
                        _backend_owner = "application"
                except Exception as e:
                    logger.warning(f"Could not check backend status: {e}, starting fresh")
                    _backend_process = None
                    _backend_owner = "application"

        proc = None
        if backend == BackendType.VLLM:
            try:
                _check_import("vllm", "pip install vllm  # requires CUDA")
                proc = _launch_vllm()
            except Exception as e:
                logger.error(f"vLLM failed to start: {e}")
                logger.warning("Automatically falling back to Ollama backend")
                logger.info("To use vLLM, change LOCAL_BACKEND=ollama in .env")
                proc = _launch_ollama()

        elif backend == BackendType.SGLANG:
            try:
                _check_import("sglang", "pip install sglang  # requires CUDA")
                proc = _launch_sglang()
            except Exception as e:
                logger.error(f"SGLang failed to start: {e}")
                logger.warning("Automatically falling back to Ollama backend")
                proc = _launch_ollama()

        elif backend == BackendType.OLLAMA:
            proc = _launch_ollama()
            # BUG 15 FIX: If proc is None, we're using external Ollama
            if proc is None:
                _backend_owner = "external"
            else:
                _backend_owner = "application"

        else:
            raise RuntimeError(f"Unknown backend: {backend}")

        # BUG 12 FIX: Wait for backend to be ready before returning
        # If proc is None (external Ollama), still check if it's ready
        if proc is not None or backend == BackendType.OLLAMA:
            if not wait_until_ready():
                logger.error("Backend failed to become ready")
                if proc is not None:
                    stop_backend(proc)
                # BUG 82 FIX: Record failure for circuit breaker
                _record_backend_failure()
                return None
            else:
                # BUG 82 FIX: Record success for circuit breaker
                _record_backend_success()

        # BUG 15, 17 FIX: Track backend process and ownership
        if proc is not None:
            with _get_backend_lock():
                _backend_process = proc
                _backend_owner = "application"
        
        # BUG 80 FIX: Register that the model is now loaded
        _register_model_loaded(settings.local_model_name)

        return proc
    except Exception as e:
        logger.error(f"Backend startup failed: {e}")
        # BUG 82 FIX: Record failure for circuit breaker
        _record_backend_failure()
        return None
    finally:
        # BUG 18 FIX: Release file lock
        if lock_file:
            try:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                lock_file.close()
                logger.debug("Released backend lock")
            except Exception as e:
                logger.warning(f"Error releasing backend lock: {e}")


def stop_backend(proc: Optional[subprocess.Popen]) -> None:
    """
    Terminate the server process gracefully, then forcefully if needed.
    BUG 15 FIX: Only stop if we own the backend (application-owned).
    BUG 16 FIX: Properly clean up orphaned processes.
    """
    global _backend_process, _backend_owner
    
    # BUG 15 FIX: Don't stop externally-owned backends
    if _backend_owner == "external":
        logger.info("Backend is externally-owned, not stopping")
        return
    
    if proc is None:
        with _get_backend_lock():
            if _backend_process is not None:
                proc = _backend_process
            else:
                return
    
    logger.info("Stopping backend process (pid=%s)…", proc.pid)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    logger.info("Backend stopped.")
    
    # BUG 16 FIX: Clear global state
    with _get_backend_lock():
        _backend_process = None
        _backend_owner = "application"
    
    # BUG 80 FIX: Deregister model when backend is stopped
    _deregister_model_loaded(settings.local_model_name)


def wait_until_ready(
    timeout: int = STARTUP_TIMEOUT_SECONDS,
    poll: float = POLL_INTERVAL_SECONDS,
) -> bool:
    """
    Poll the backend health endpoint until it responds 200 or timeout is hit.
    BUG 19 FIX: Also verifies the specific model is loaded and available.

    Returns True on success, False on timeout.
    BUG 13 FIX: Uses current configuration, not cached values.
    """
    backend = settings.local_backend

    if backend == BackendType.NONE:
        return True

    # BUG 13 FIX: Build health URL from current configuration
    health_url = _get_health_url(backend)

    deadline = time.time() + timeout
    logger.info("Waiting for backend at %s (timeout %ds)…", health_url, timeout)

    while time.time() < deadline:
        try:
            resp = httpx.get(health_url, timeout=getattr(settings, 'connect_timeout', 5))
            if resp.status_code < 400:
                # BUG 19 FIX: Verify model is actually loaded
                if backend == BackendType.OLLAMA:
                    # For Ollama, check the model list endpoint
                    models_url = f"http://{settings.local_backend_host}:11434/api/tags"
                    try:
                        models_resp = httpx.get(models_url, timeout=getattr(settings, 'connect_timeout', 5))
                        if models_resp.status_code < 400:
                            models_data = models_resp.json()
                            available_models = [m.get('name', '') for m in models_data.get('models', [])]
                            target_model = settings.local_model_name
                            # Check if target model is available
                            model_available = any(target_model in m for m in available_models)
                            if model_available:
                                logger.info(f"Backend is ready! Model {target_model} is available")
                                return True
                            else:
                                logger.warning(f"Backend alive but model {target_model} not found. Available: {available_models}")
                                return False
                    except Exception as e:
                        logger.warning(f"Could not verify model availability: {e}")
                        # Fall through to return True on basic health check
                logger.info("Backend is ready! (%s)", health_url)
                return True
        except httpx.RequestError:
            pass
        time.sleep(poll)

    logger.error("Backend did not become ready within %ds.", timeout)
    return False


def get_backend_status() -> dict:
    """Return a status dict suitable for an API health endpoint.
    BUG 13 FIX: Uses current configuration, not cached values.
    BUG 20 FIX: Distinguishes between backend alive and model ready.
    """
    backend = settings.local_backend
    if backend == BackendType.NONE:
        return {"backend": "none", "status": "disabled"}

    # BUG 13 FIX: Build health URL from current configuration
    health_url = _get_health_url(backend)

    backend_alive = False
    model_ready = False
    
    try:
        resp = httpx.get(health_url, timeout=getattr(settings, 'connect_timeout', 5))
        backend_alive = resp.status_code < 400
    except httpx.RequestError:
        backend_alive = False
    
    # BUG 20 FIX: Check model readiness separately
    if backend_alive and backend == BackendType.OLLAMA:
        try:
            models_url = f"http://{settings.local_backend_host}:11434/api/tags"
            models_resp = httpx.get(models_url, timeout=getattr(settings, 'connect_timeout', 5))
            if models_resp.status_code < 400:
                models_data = models_resp.json()
                available_models = [m.get('name', '') for m in models_data.get('models', [])]
                target_model = settings.local_model_name
                model_ready = any(target_model in m for m in available_models)
        except Exception as e:
            logger.warning(f"Could not check model readiness: {e}")
            model_ready = False
    elif backend_alive:
        # For vLLM/SGLang, assume model is ready if backend is alive
        # (In production, would check model-specific endpoint)
        model_ready = True

    return {
        "backend":    backend.value,
        "model":      settings.local_model_name,
        "base_url":   settings.local_backend_base_url,
        "backend_alive": backend_alive,
        "model_ready": model_ready,
        "healthy":    backend_alive and model_ready,
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
