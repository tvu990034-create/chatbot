"""
Simplified LiteLLM Gateway
"""

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
) -> str:
    """
    Simple chat function with caching.
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
    
    # Check cache first
    if use_cache:
        cached = _cache.get(query)
        if cached:
            logger.debug(f"Cache hit for query: {query[:40]}...")
            return cached
    
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
        
        logger.debug(f"chat() {elapsed:.2f}s | model={model}")
        
        # Cache the response
        if use_cache and reply:
            _cache.set(query, reply)
        
        return reply
        
    except (APIConnectionError, RateLimitError, ServiceUnavailableError) as e:
        logger.error(f"API error: {e}")
        raise


def get_optimization_stats() -> Dict[str, Any]:
    """Return optimization stats."""
    cache_stats = {"cached_responses": len(_cache.cache)}
    
    return {
        "optimizations": {
            "response_caching": cache_stats,
        },
        "total_optimizations": 1,
    }
