"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

"""
FastCloudEngine - Ultra-fast cloud inference engine
Maximizes tokens-per-second and minimizes time-to-first-token
"""
import os
import hashlib
import time
import logging
from typing import Tuple, Dict, Optional, AsyncIterator
from dataclasses import dataclass, field

import httpx
import openai

from ..identity import ConversationState

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FastCloudMetrics:
    """Metrics for FastCloudEngine API calls"""
    model_used: str
    latency_ms: float
    tokens_generated: int
    fallback_triggered: bool
    cache_hit: bool = False
    error: Optional[str] = None


class FastCloudEngine:
    """Ultra-fast cloud inference engine with connection pooling, caching, and fallback"""
    
    def __init__(self, cfg: dict):
        # Model configuration
        self.default_model = cfg.get('default_model', os.environ.get('FASTCLOUD_DEFAULT_MODEL', 'gpt-4.1-nano'))
        self.fallback_model = cfg.get('fallback_model', os.environ.get('FASTCLOUD_FALLBACK_MODEL', 'gpt-3.5-turbo'))
        self.timeout_sec = cfg.get('timeout_sec', float(os.environ.get('FASTCLOUD_TIMEOUT_SEC', '8.0')))
        self.temperature = cfg.get('temperature', 0.0)
        
        # Connection pooling configuration
        max_connections = cfg.get('max_connections', int(os.environ.get('FASTCLOUD_MAX_CONNECTIONS', '20')))
        keepalive_sec = cfg.get('keepalive_sec', int(os.environ.get('FASTCLOUD_KEEPALIVE_SEC', '30')))
        
        # Create httpx client with connection pooling
        limits = httpx.Limits(
            max_connections=max_connections,
            max_keepalive_connections=max_connections,
            keepalive_expiry=keepalive_sec
        )
        self.http_client = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(self.timeout_sec),
            http2=True,  # Enable HTTP/2 for better performance
        )
        
        # OpenAI client with custom httpx transport
        api_key = cfg.get('api_key', os.environ.get('OPENAI_API_KEY'))
        base_url = cfg.get('base_url', None)
        
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=self.http_client,
        )
        
        # Prompt cache
        self.prompt_cache: Dict[str, str] = {}
        
        # Metrics tracking
        self.metrics_history: list[FastCloudMetrics] = []
        
        logger.info(f"FastCloudEngine initialized with default_model={self.default_model}, fallback_model={self.fallback_model}")
    
    def _compute_prompt_hash(self, prompt: str) -> str:
        """Compute SHA-256 hash of prompt for caching"""
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    
    def _get_adaptive_max_tokens(self, query: str) -> int:
        """Compute adaptive max_tokens based on query length"""
        query_words = len(query.split())
        return max(16, min(256, query_words * 3))
    
    async def _generate_with_fallback(
        self, 
        model: str, 
        messages: list, 
        max_tokens: int,
        seed: int,
        stream: bool = False
    ) -> Tuple[str, bool, Optional[str]]:
        """Generate with automatic fallback on rate limit or timeout"""
        try:
            if stream:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    stream=True,
                )
                # Collect streaming response
                full_response = ""
                async for chunk in response:
                    if chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                return full_response, False, None
            else:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    stream=False,
                )
                return response.choices[0].message.content.strip(), False, None
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit for model {model}: {e}")
            return "", True, f"Rate limit: {str(e)}"
        except openai.APITimeoutError as e:
            logger.warning(f"Timeout for model {model}: {e}")
            return "", True, f"Timeout: {str(e)}"
        except Exception as e:
            logger.error(f"Error generating with model {model}: {e}")
            return "", True, str(e)
    
    async def generate(
        self, 
        state: ConversationState, 
        user_input: str, 
        seed: int, 
        prompt_override: str = None,
        enable_reasoning: bool = False
    ) -> Tuple[str, ConversationState, Dict]:
        """Generate response with ultra-fast optimizations"""
        start_time = time.perf_counter()
        
        # Build prompt
        if prompt_override:
            prompt = prompt_override
        else:
            history_text = '\n'.join(f"{t.role}: {t.content}" for t in state.history)
            prompt = f"system: You are a fast assistant.\n{history_text}\nuser: {user_input}\nassistant:"
        
        # Compute prompt hash for caching
        prompt_hash = self._compute_prompt_hash(prompt)
        
        # Check prompt cache
        cache_key = f"{self.default_model}:{prompt_hash}"
        if cache_key in self.prompt_cache:
            logger.info(f"Prompt cache hit for hash {prompt_hash[:8]}...")
            cached_response = self.prompt_cache[cache_key]
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            metrics = FastCloudMetrics(
                model_used=self.default_model,
                latency_ms=latency_ms,
                tokens_generated=len(cached_response.split()),
                fallback_triggered=False,
                cache_hit=True
            )
            self.metrics_history.append(metrics)
            
            new_state = ConversationState(
                history=list(state.history),
                token_ids=list(state.token_ids),
                kv_cache=None,
            )
            new_state.append('user', user_input)
            new_state.append('assistant', cached_response)
            
            return cached_response, new_state, {
                'prompt_tokens': 0,
                'generated_tokens': len(cached_response.split()),
                'speculative_enabled': 0,
                'kv_pruned_tokens': 0,
                'tome_merged_tokens': 0,
            }
        
        # Adaptive max_tokens
        max_tokens = self._get_adaptive_max_tokens(user_input)
        
        # Build messages for OpenAI
        messages = [{'role': 'user', 'content': prompt}]
        
        # Try primary model with streaming
        response, fallback_triggered, error = await self._generate_with_fallback(
            self.default_model, messages, max_tokens, seed, stream=True
        )
        
        # Fallback to secondary model if needed
        if fallback_triggered and self.fallback_model:
            logger.info(f"Falling back to {self.fallback_model}")
            response, fallback_triggered, error = await self._generate_with_fallback(
                self.fallback_model, messages, max_tokens, seed, stream=True
            )
        
        # If still failed, return fast fallback message
        if fallback_triggered or not response:
            response = "I'm thinking... please ask again"
            logger.warning(f"Generation failed, using fallback message")
        
        # Cache the response
        self.prompt_cache[cache_key] = response
        
        # Calculate metrics
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics = FastCloudMetrics(
            model_used=self.fallback_model if fallback_triggered else self.default_model,
            latency_ms=latency_ms,
            tokens_generated=len(response.split()),
            fallback_triggered=fallback_triggered,
            cache_hit=False,
            error=error
        )
        self.metrics_history.append(metrics)
        
        logger.info(f"FastCloudEngine: model={metrics.model_used}, latency={latency_ms:.2f}ms, tokens={metrics.tokens_generated}, fallback={fallback_triggered}")
        
        # Update conversation state
        new_state = ConversationState(
            history=list(state.history),
            token_ids=list(state.token_ids),
            kv_cache=None,
        )
        new_state.append('user', user_input)
        new_state.append('assistant', response)
        
        return response, new_state, {
            'prompt_tokens': 0,
            'generated_tokens': len(response.split()),
            'speculative_enabled': 0,
            'kv_pruned_tokens': 0,
            'tome_merged_tokens': 0,
        }
    
    async def generate_stream(
        self,
        state: ConversationState,
        user_input: str,
        seed: int,
        prompt_override: str = None
    ) -> AsyncIterator[str]:
        """Generate response with streaming for minimal TTFB"""
        start_time = time.perf_counter()
        
        # Build prompt
        if prompt_override:
            prompt = prompt_override
        else:
            history_text = '\n'.join(f"{t.role}: {t.content}" for t in state.history)
            prompt = f"system: You are a fast assistant.\n{history_text}\nuser: {user_input}\nassistant:"
        
        # Compute prompt hash for caching
        prompt_hash = self._compute_prompt_hash(prompt)
        cache_key = f"{self.default_model}:{prompt_hash}"
        
        # Check prompt cache
        if cache_key in self.prompt_cache:
            logger.info(f"Prompt cache hit for hash {prompt_hash[:8]}...")
            cached_response = self.prompt_cache[cache_key]
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            metrics = FastCloudMetrics(
                model_used=self.default_model,
                latency_ms=latency_ms,
                tokens_generated=len(cached_response.split()),
                fallback_triggered=False,
                cache_hit=True
            )
            self.metrics_history.append(metrics)
            
            # Yield cached response as stream
            for token in cached_response.split():
                yield token + " "
            return
        
        # Adaptive max_tokens
        max_tokens = self._get_adaptive_max_tokens(user_input)
        
        # Build messages for OpenAI
        messages = [{'role': 'user', 'content': prompt}]
        
        fallback_triggered = False
        error = None
        full_response = ""
        
        # Try primary model with streaming
        try:
            stream = await self.client.chat.completions.create(
                model=self.default_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
                seed=seed,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    full_response += token
                    yield token
                    
        except (openai.RateLimitError, openai.APITimeoutError) as e:
            logger.warning(f"Primary model failed: {e}")
            fallback_triggered = True
            error = str(e)
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            fallback_triggered = True
            error = str(e)
        
        # Fallback to secondary model if needed
        if fallback_triggered and self.fallback_model:
            logger.info(f"Falling back to {self.fallback_model}")
            try:
                stream = await self.client.chat.completions.create(
                    model=self.fallback_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                    stream=True,
                )
                
                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        full_response += token
                        yield token
                        
                fallback_triggered = False
                error = None
            except Exception as e:
                logger.error(f"Fallback model also failed: {e}")
                error = str(e)
        
        # If still failed, yield fast fallback message
        if not full_response:
            fallback_msg = "I'm thinking... please ask again"
            logger.warning(f"Generation failed, using fallback message")
            for token in fallback_msg.split():
                yield token + " "
            full_response = fallback_msg
        
        # Cache the response
        self.prompt_cache[cache_key] = full_response
        
        # Calculate metrics
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics = FastCloudMetrics(
            model_used=self.fallback_model if fallback_triggered else self.default_model,
            latency_ms=latency_ms,
            tokens_generated=len(full_response.split()),
            fallback_triggered=fallback_triggered,
            cache_hit=False,
            error=error
        )
        self.metrics_history.append(metrics)
        
        logger.info(f"FastCloudEngine stream: model={metrics.model_used}, latency={latency_ms:.2f}ms, tokens={metrics.tokens_generated}, fallback={fallback_triggered}")
    
    def get_metrics(self) -> list[FastCloudMetrics]:
        """Get all recorded metrics"""
        return self.metrics_history
    
    def clear_metrics(self) -> None:
        """Clear metrics history"""
        self.metrics_history.clear()
    
    async def close(self) -> None:
        """Close the HTTP client"""
        await self.http_client.aclose()
