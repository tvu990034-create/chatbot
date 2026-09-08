"""
Simplified LiteLLM Gateway
"""

import asyncio
import logging
import time
from typing import List, Dict, Any

import litellm
from litellm import completion
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
)

from config import settings
from .simple_cache import get_cache

logger = logging.getLogger(__name__)

# Configure litellm
litellm.set_verbose = False

# Initialize components
_cache = get_cache()


def chat(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    api_base: str = None,
    use_cache: bool = True,
    speed_mode: bool = False,
    system_prompt: str = None,
) -> tuple[str, bool, str]:
    """
    Simple chat function with caching.
    Returns (response, cache_hit, actual_model) tuple.
    Applies generation policy (top_p/top_k) and speed mode to reach inference.
    """
    # Inject system_prompt as first message if provided and not already present.
    if system_prompt:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages
    if model is None:
        model = settings.default_model
    
    # Ensure model has provider prefix for litellm
    if not model.startswith("ollama/"):
        model = f"ollama/{model}"
    
    if max_tokens is None:
        max_tokens = settings.litellm_max_tokens
    
    if temperature is None:
        temperature = settings.litellm_temperature
    
    query = messages[-1]['content'] if messages else ""
    
    # Speed mode override: cap generation budget BEFORE building the cache
    # context so the read and write cache keys match the parameters actually
    # used to generate the response.  Otherwise a 128-token/0.3-temp
    # speed-mode answer gets cached under the caller's full-budget key and is
    # later served to a non-speed request as a truncated baseline reply.
    if speed_mode:
        temperature = min(temperature, 0.3)
        max_tokens = min(max_tokens or 128, 128)
    
    # Build cache context for proper cache key generation
    cache_context = {
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'messages': messages,
        # Bug #16 FIX: Include RAG version to prevent stale cached answers
        'rag_version': '1.0',  # Should be updated when RAG index changes
        'rag_enabled': False,  # Gateway doesn't use RAG by default
    }
    
    # Check cache first
    if use_cache:
        cached = _cache.get(query, context=cache_context)
        if cached:
            logger.debug(f"Cache hit for query: {query[:40]}...")
            return cached, True, model
    
    # API call
    from gateway.opt_core import adaptive_generation_timeout
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": adaptive_generation_timeout(
            max_tokens or settings.litellm_max_tokens,
            getattr(settings, "generation_timeout", 15),
        ),
    }

    # Apply generation policy (top_p/top_k) unless in speed mode
    if not speed_mode:
        try:
            from gateway.opt_core import resolve_generation_policy, estimate_tokens

            class _Analysis:
                is_math = any(w in query.lower() for w in (
                    "calculate", "solve", "equation", "+", "-", "*", "/",
                ))
                is_coding = False
                is_complex = len(query) > 100
                needs_reasoning = any(w in query.lower() for w in (
                    "why", "how", "explain", "reason",
                ))
                expected_response_length = "short" if len(query) < 30 else "medium"
                query_text = query

            policy = resolve_generation_policy(
                _Analysis(),
                base={"temperature": temperature} if temperature is not None else {},
                configured_max_tokens=max_tokens or settings.litellm_max_tokens,
                model_name=model,
            )
            temperature = policy.temperature if temperature is None else temperature
            max_tokens = max_tokens or policy.max_tokens
            kwargs["top_p"] = policy.top_p
            kwargs["top_k"] = policy.top_k
        except Exception:
            logger.warning("Generation policy application failed, using raw params")
    
    if api_base:
        kwargs["api_base"] = api_base
    else:
        kwargs["api_base"] = "http://localhost:11434"
    
    try:
        start_time = time.time()
        response = completion(**kwargs)
        elapsed = time.time() - start_time
        
        reply = response.choices[0].message.content if response.choices else ""
        
        # BUG 3 FIX: Get actual model used from response
        actual_model = response.model if hasattr(response, 'model') else model
        
        logger.debug(f"chat() {elapsed:.2f}s | model={actual_model}")
        
        # Cache the response
        if use_cache and reply:
            _cache.set(query, reply, context=cache_context)
        
        return reply, False, actual_model
        
    except (APIConnectionError, RateLimitError, ServiceUnavailableError) as e:
        logger.error(f"API error: {e}")
        raise


async def achat(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    api_base: str = None,
    use_cache: bool = True,
    use_router: bool = False,
    system_prompt: str = None,
    speed_mode: bool = False,
) -> tuple[str, bool, str]:
    """
    Async chat function with caching.
    Returns (response, cache_hit, actual_model) tuple.
    """
    # Run the sync chat function in a thread pool.
    # Forward system_prompt so it is injected into messages by chat().
    loop = asyncio.get_running_loop()
    reply, cache_hit, actual_model = await loop.run_in_executor(
        None,
        chat,
        messages,
        model,
        temperature,
        max_tokens,
        api_base,
        use_cache,
        speed_mode,
        system_prompt,
    )
    return reply, cache_hit, actual_model


async def achat_stream(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = None,
    max_tokens: int = None,
    api_base: str = None,
    use_cache: bool = True,
    use_router: bool = False,
    system_prompt: str = None,
    speed_mode: bool = False,
):
    """
    Async streaming chat function with caching.
    Yields response chunks.
    """
    # Inject system_prompt as first message if provided and not already present.
    if system_prompt:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages

    # Resolve effective parameters FIRST so the cache-read context and the
    # cache-write context are byte-identical (else the stream cache never hits).
    if model is None:
        model = settings.default_model
    if not model.startswith("ollama/"):
        model = f"ollama/{model}"
    if max_tokens is None:
        max_tokens = settings.litellm_max_tokens
    if temperature is None:
        temperature = settings.litellm_temperature
    if speed_mode:
        temperature = min(temperature, 0.3)
        max_tokens = min(max_tokens or 128, 128)

    cache_context = {
        'model': model,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'messages': messages,
        'rag_version': '1.0',
        'rag_enabled': False,
    }

    # BUG 2 FIX: Check cache first if use_cache is enabled
    if use_cache:
        query = messages[-1]['content'] if messages else ""
        cached = _cache.get(query, context=cache_context)
        if cached:
            # Yield the cached response as a single chunk
            yield cached
            return

    from gateway.opt_core import adaptive_generation_timeout
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": adaptive_generation_timeout(
            max_tokens,
            getattr(settings, "generation_timeout", 15),
        ),
        "stream": True,
    }
    if api_base:
        kwargs["api_base"] = api_base
    else:
        kwargs["api_base"] = "http://localhost:11434"

    # The sync completion() stream is blocking; drive it from a worker thread
    # and hand chunks to the event loop via call_soon_threadsafe so one slow
    # stream never stalls the whole server.
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _SENTINEL = object()
    full_response = ""
    success = False

    def _stream_worker() -> None:
        nonlocal full_response, success
        try:
            response = completion(**kwargs)
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    loop.call_soon_threadsafe(queue.put_nowait, content)
            success = True
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)
        except BaseException as e:  # re-raise on the event loop so the caller sees it
            logger.error(f"Error in streaming: {e}")
            success = False
            loop.call_soon_threadsafe(queue.put_nowait, e)
        loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    worker_task = asyncio.ensure_future(asyncio.to_thread(_stream_worker))
    try:
        while True:
            item = await queue.get()
            if item is _SENTINEL:
                break
            if isinstance(item, BaseException):
                raise item
            yield item
        await worker_task
    finally:
        if not worker_task.done():
            worker_task.cancel()

    # BUG 46 FIX: Only cache the complete response after successful completion
    if use_cache and success and full_response:
        query = messages[-1]['content'] if messages else ""
        _cache.set(query, full_response, context=cache_context)


def get_optimization_stats() -> Dict[str, Any]:
    """Return optimization stats."""
    cache_stats = _cache.stats()
    
    return {
        "optimizations": {
            "response_caching": cache_stats,
        },
        "total_optimizations": 1,
    }
