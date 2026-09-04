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
) -> tuple[str, bool, str]:
    """
    Simple chat function with caching.
    BUG 3 FIX: Returns (response, cache_hit, actual_model) tuple.
    """
    if model is None:
        model = "ollama/tinyllama:latest"
    
    # Ensure model has provider prefix for litellm
    if not model.startswith("ollama/"):
        model = f"ollama/{model}"
    
    if max_tokens is None:
        max_tokens = settings.litellm_max_tokens
    
    if temperature is None:
        temperature = settings.litellm_temperature
    
    query = messages[-1]['content'] if messages else ""
    
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
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 15,
    }
    
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
) -> tuple[str, bool, str]:
    """
    Async chat function with caching.
    BUG 3 FIX: Returns (response, cache_hit, actual_model) tuple.
    """
    # Run the sync chat function in a thread pool
    loop = asyncio.get_event_loop()
    reply, cache_hit, actual_model = await loop.run_in_executor(
        None,
        chat,
        messages,
        model,
        temperature,
        max_tokens,
        api_base,
        use_cache,
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
):
    """
    Async streaming chat function with caching.
    BUG 2 FIX: Now honors use_cache parameter.
    BUG 46 FIX: Only caches after successful completion.
    Yields response chunks.
    """
    # BUG 2 FIX: Check cache first if use_cache is enabled
    if use_cache:
        query = messages[-1]['content'] if messages else ""
        
        # Build cache context for proper cache key generation
        cache_context = {
            'model': model or "ollama/tinyllama:latest",
            'temperature': temperature,
            'max_tokens': max_tokens,
            'messages': messages,
            'rag_version': '1.0',
            'rag_enabled': False,
        }
        
        cached = _cache.get(query, context=cache_context)
        if cached:
            # Yield the cached response as a single chunk
            yield cached
            return
    
    # If not cached, stream the response
    if model is None:
        model = "ollama/tinyllama:latest"
    
    # Ensure model has provider prefix for litellm
    if not model.startswith("ollama/"):
        model = f"ollama/{model}"
    
    if max_tokens is None:
        max_tokens = settings.litellm_max_tokens
    
    if temperature is None:
        temperature = settings.litellm_temperature
    
    # API call with streaming
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": 15,
        "stream": True,
    }
    
    if api_base:
        kwargs["api_base"] = api_base
    else:
        kwargs["api_base"] = "http://localhost:11434"
    
    try:
        response = completion(**kwargs)
        full_response = ""
        success = False
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content
        
        # BUG 46 FIX: Only cache after successful completion
        success = True
        
    except (APIConnectionError, RateLimitError, ServiceUnavailableError) as e:
        logger.error(f"API error in streaming: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in streaming: {e}")
        success = False
        raise
    finally:
        # BUG 46 FIX: Only cache the complete response after successful completion
        if use_cache and success and full_response:
            query = messages[-1]['content'] if messages else ""
            cache_context = {
                'model': model,
                'temperature': temperature,
                'max_tokens': max_tokens,
                'messages': messages,
                'rag_version': '1.0',
                'rag_enabled': False,
            }
            _cache.set(query, full_response, context=cache_context)


def get_optimization_stats() -> Dict[str, Any]:
    """Return optimization stats."""
    cache_stats = {"cached_responses": len(_cache.cache)}
    
    return {
        "optimizations": {
            "response_caching": cache_stats,
        },
        "total_optimizations": 1,
    }
