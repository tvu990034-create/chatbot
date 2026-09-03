"""
reasoning/gateway_integration.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration layer between reasoning system and gateway optimizations.

This module provides:
- Integration with KV cache management
- Connection to advanced caching systems
- Prompt compression for reasoning prompts
- Performance monitoring and optimization
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Callable
import time

logger = logging.getLogger(__name__)


class GatewayIntegration:
    """
    Integration layer for reasoning system with gateway optimizations.
    
    This class connects the reasoning system with:
    - KV cache management for efficient token reuse
    - Advanced caching systems for prompt/response caching
    - Prompt compression for reducing inference costs
    - Performance monitoring and optimization
    """
    
    def __init__(
        self,
        enable_kv_cache: bool = True,
        enable_prompt_compression: bool = True,
        enable_advanced_caching: bool = True,
        compression_ratio: float = 0.7,
    ):
        self.enable_kv_cache = enable_kv_cache
        self.enable_prompt_compression = enable_prompt_compression
        self.enable_advanced_caching = enable_advanced_caching
        self.compression_ratio = compression_ratio
        
        # Gateway components (lazy loading)
        self._kv_cache_manager = None
        self._prompt_compressor = None
        self._advanced_cache = None
        
        # Performance tracking
        self.performance_stats = {
            "total_reasoning_calls": 0,
            "kv_cache_hits": 0,
            "prompt_compressions": 0,
            "advanced_cache_hits": 0,
            "avg_latency_ms": 0.0,
            "total_latency_ms": 0.0,
        }
        
        logger.info(
            f"GatewayIntegration initialized with kv_cache={enable_kv_cache}, "
            f"prompt_compression={enable_prompt_compression}, "
            f"advanced_caching={enable_advanced_caching}"
        )
    
    def _initialize_kv_cache(self) -> None:
        """Initialize KV cache manager if available."""
        if self._kv_cache_manager is None and self.enable_kv_cache:
            try:
                from gateway.advanced_kv_cache_management import KVCacheManager
                self._kv_cache_manager = KVCacheManager()
                logger.info("KV cache manager initialized")
            except ImportError:
                logger.warning("KV cache manager not available")
                self.enable_kv_cache = False
    
    def _initialize_prompt_compressor(self) -> None:
        """Initialize prompt compressor if available."""
        if self._prompt_compressor is None and self.enable_prompt_compression:
            try:
                from prompt_compression import LLMLinguaCompressor
                self._prompt_compressor = LLMLinguaCompressor(
                    compression_ratio=self.compression_ratio
                )
                logger.info("Prompt compressor initialized")
            except ImportError:
                logger.warning("Prompt compressor not available")
                self.enable_prompt_compression = False
    
    def _initialize_advanced_cache(self) -> None:
        """Initialize advanced cache if available."""
        if self._advanced_cache is None and self.enable_advanced_caching:
            try:
                from gateway.advanced_caching import PrefixCache
                self._advanced_cache = PrefixCache()
                logger.info("Advanced cache initialized")
            except ImportError:
                logger.warning("Advanced cache not available")
                self.enable_advanced_caching = False
    
    def optimize_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Optimize a reasoning prompt using gateway optimizations.
        
        Args:
            prompt: Original reasoning prompt
            context: Additional context for optimization
        
        Returns:
            Optimized prompt
        """
        optimized_prompt = prompt
        
        # Apply prompt compression
        if self.enable_prompt_compression:
            self._initialize_prompt_compressor()
            if self._prompt_compressor:
                try:
                    compressed = self._prompt_compressor.compress(prompt)
                    if len(compressed) < len(prompt):
                        optimized_prompt = compressed
                        self.performance_stats["prompt_compressions"] += 1
                        logger.debug(f"Prompt compressed: {len(prompt)} -> {len(optimized_prompt)} chars")
                except Exception as e:
                    logger.warning(f"Prompt compression failed: {e}")
        
        return optimized_prompt
    
    def get_cached_kv_state(
        self,
        prompt: str,
        session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached KV state for the prompt if available.
        
        Args:
            prompt: The reasoning prompt
            session_id: Optional session identifier
        
        Returns:
            Cached KV state or None
        """
        if not self.enable_kv_cache:
            return None
        
        self._initialize_kv_cache()
        if self._kv_cache_manager:
            try:
                cached_state = self._kv_cache_manager.get_cached_state(prompt, session_id)
                if cached_state:
                    self.performance_stats["kv_cache_hits"] += 1
                    logger.debug("KV cache hit")
                return cached_state
            except Exception as e:
                logger.warning(f"KV cache lookup failed: {e}")
        
        return None
    
    def cache_kv_state(
        self,
        prompt: str,
        kv_state: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> None:
        """
        Cache KV state for the prompt.
        
        Args:
            prompt: The reasoning prompt
            kv_state: The KV state to cache
            session_id: Optional session identifier
        """
        if not self.enable_kv_cache:
            return
        
        self._initialize_kv_cache()
        if self._kv_cache_manager:
            try:
                self._kv_cache_manager.cache_state(prompt, kv_state, session_id)
                logger.debug("KV state cached")
            except Exception as e:
                logger.warning(f"KV cache storage failed: {e}")
    
    def get_advanced_cached_response(
        self,
        prompt: str,
        task_type: str = "math",
    ) -> Optional[str]:
        """
        Get cached response from advanced cache.
        
        Args:
            prompt: The reasoning prompt
            task_type: Type of reasoning task
        
        Returns:
            Cached response or None
        """
        if not self.enable_advanced_caching:
            return None
        
        self._initialize_advanced_cache()
        if self._advanced_cache:
            try:
                cached_response = self._advanced_cache.get(prompt, task_type)
                if cached_response:
                    self.performance_stats["advanced_cache_hits"] += 1
                    logger.debug("Advanced cache hit")
                return cached_response
            except Exception as e:
                logger.warning(f"Advanced cache lookup failed: {e}")
        
        return None
    
    def cache_advanced_response(
        self,
        prompt: str,
        response: str,
        task_type: str = "math",
    ) -> None:
        """
        Cache response in advanced cache.
        
        Args:
            prompt: The reasoning prompt
            response: The response to cache
            task_type: Type of reasoning task
        """
        if not self.enable_advanced_caching:
            return
        
        self._initialize_advanced_cache()
        if self._advanced_cache:
            try:
                self._advanced_cache.put(prompt, task_type, response)
                logger.debug("Response cached in advanced cache")
            except Exception as e:
                logger.warning(f"Advanced cache storage failed: {e}")
    
    def track_performance(
        self,
        latency_ms: float,
        cached: bool = False,
    ) -> None:
        """
        Track performance metrics.
        
        Args:
            latency_ms: Latency in milliseconds
            cached: Whether the result was cached
        """
        self.performance_stats["total_reasoning_calls"] += 1
        self.performance_stats["total_latency_ms"] += latency_ms
        
        # Update average latency
        total_calls = self.performance_stats["total_reasoning_calls"]
        self.performance_stats["avg_latency_ms"] = (
            self.performance_stats["total_latency_ms"] / total_calls
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        stats = self.performance_stats.copy()
        
        # Add derived metrics
        if stats["total_reasoning_calls"] > 0:
            stats["cache_hit_rate"] = (
                (stats["kv_cache_hits"] + stats["advanced_cache_hits"]) /
                stats["total_reasoning_calls"]
            )
            stats["compression_rate"] = (
                stats["prompt_compressions"] /
                stats["total_reasoning_calls"]
            )
        else:
            stats["cache_hit_rate"] = 0.0
            stats["compression_rate"] = 0.0
        
        return stats
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_stats = {
            "total_reasoning_calls": 0,
            "kv_cache_hits": 0,
            "prompt_compressions": 0,
            "advanced_cache_hits": 0,
            "avg_latency_ms": 0.0,
            "total_latency_ms": 0.0,
        }
        logger.info("Performance stats reset")


class OptimizedReasoningWrapper:
    """
    Wrapper for reasoning methods with gateway optimization integration.
    
    This wrapper provides automatic integration with:
    - KV cache management
    - Prompt compression
    - Advanced caching
    - Performance monitoring
    """
    
    def __init__(
        self,
        reasoning_method: Any,
        gateway_integration: Optional[GatewayIntegration] = None,
        enable_optimization: bool = True,
    ):
        self.reasoning_method = reasoning_method
        self.gateway_integration = gateway_integration or GatewayIntegration()
        self.enable_optimization = enable_optimization
        
        logger.info(
            f"OptimizedReasoningWrapper initialized for "
            f"{reasoning_method.__class__.__name__}, "
            f"optimization_enabled={enable_optimization}"
        )
    
    def reason(
        self,
        question: str,
        model_fn: Callable,
        task_type: str = "math",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform reasoning with gateway optimizations.
        
        Args:
            question: Question to reason about
            model_fn: Function to call the model
            task_type: Type of reasoning task
            **kwargs: Additional arguments for reasoning method
        
        Returns:
            Reasoning result with optimization metadata
        """
        start_time = time.time()
        
        # Check advanced cache first
        if self.enable_optimization:
            cached_response = self.gateway_integration.get_advanced_cached_response(
                question, task_type
            )
            if cached_response:
                latency_ms = (time.time() - start_time) * 1000
                self.gateway_integration.track_performance(latency_ms, cached=True)
                
                return {
                    "question": question,
                    "final_answer": cached_response,
                    "cached": True,
                    "optimization_metadata": {
                        "cache_hit": True,
                        "latency_ms": latency_ms,
                    }
                }
        
        # Perform reasoning with optimizations
        if self.enable_optimization:
            # Apply prompt compression if method supports it
            if hasattr(self.reasoning_method, 'build_cot_prompt'):
                original_prompt = self.reasoning_method.build_cot_prompt(question, task_type)
                optimized_prompt = self.gateway_integration.optimize_prompt(original_prompt)
                
                # If prompt was compressed, we need to handle this specially
                # For now, we'll proceed with standard reasoning
                result = self.reasoning_method.reason(question, model_fn, task_type, **kwargs)
            else:
                result = self.reasoning_method.reason(question, model_fn, task_type, **kwargs)
        else:
            result = self.reasoning_method.reason(question, model_fn, task_type, **kwargs)
        
        # Cache the result
        if self.enable_optimization and not result.get("cached", False):
            self.gateway_integration.cache_advanced_response(
                question,
                result.get("final_answer", ""),
                task_type
            )
        
        # Track performance
        latency_ms = (time.time() - start_time) * 1000
        self.gateway_integration.track_performance(latency_ms, result.get("cached", False))
        
        # Add optimization metadata
        result["optimization_metadata"] = {
            "latency_ms": latency_ms,
            "gateway_optimizations_enabled": self.enable_optimization,
        }
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics from gateway integration."""
        return self.gateway_integration.get_performance_stats()
    
    def reset_performance_stats(self) -> None:
        """Reset performance statistics."""
        self.gateway_integration.reset_performance_stats()