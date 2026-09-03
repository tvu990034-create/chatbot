"""
reasoning/reasoning_manager.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unified reasoning manager that integrates working reasoning frameworks.

This module provides a unified interface for using different reasoning techniques:
|- Chain-of-Thought (CoT)
|- Reflexion
|- Self-Refine
|- Self-Consistency (meta-framework)
|- ReAct
|- Default (existing LangGraph agent)

The manager automatically selects the appropriate framework based on configuration
and problem complexity.
"""

from __future__ import annotations

import logging
from typing import Any, Callable
from enum import Enum

from config import settings

logger = logging.getLogger(__name__)


class ReasoningFramework(str, Enum):
    """Available reasoning frameworks."""
    DEFAULT = "default"
    REFLEXION = "reflexion"
    SELF_REFINE = "self_refine"
    SELF_CONSISTENCY = "self_consistency"
    REACT = "react"


class ReasoningManager:
    """
    Unified manager for working reasoning frameworks.
    
    This manager provides a consistent interface for using different reasoning
    techniques while automatically handling framework selection, configuration,
    and fallback mechanisms.
    """
    
    def __init__(self):
        self.framework = ReasoningFramework(settings.reasoning_framework)
        self._reflexion_agent = None
        self._self_refine_agent = None
        self._self_consistency = None
        self._react_agent = None
        self._default_agent = None
        
        # Initialize selected framework
        self._initialize_framework()
    
    def _initialize_framework(self) -> None:
        """Initialize the selected reasoning framework."""
        if self.framework == ReasoningFramework.REFLEXION and settings.reflexion_enabled:
            try:
                from reasoning.reflexion import ReflexionConfig, get_reflexion_agent
                config = ReflexionConfig(
                    max_reflections=settings.reflexion_max_attempts,
                    max_episodes=settings.reflexion_max_episodes,
                    reflection_temperature=settings.reflexion_temperature,
                    episode_memory_file=settings.reflexion_memory_file,
                    enable_persistent_memory=True
                )
                self._reflexion_agent = get_reflexion_agent(config)
                logger.info("Reflexion reasoning initialized")
            except Exception as exc:
                logger.warning(f"Failed to initialize Reflexion: {exc}, falling back to default")
                self.framework = ReasoningFramework.DEFAULT
        
        elif self.framework == ReasoningFramework.SELF_REFINE and settings.self_refine_enabled:
            try:
                from reasoning.self_refine import SelfRefineConfig, get_self_refine_agent
                config = SelfRefineConfig(
                    max_iterations=settings.self_refine_max_iterations,
                    min_improvement=settings.self_refine_min_improvement,
                    temperature=settings.self_refine_temperature
                )
                self._self_refine_agent = get_self_refine_agent(config)
                logger.info("Self-Refine reasoning initialized")
            except Exception as exc:
                logger.warning(f"Failed to initialize Self-Refine: {exc}, falling back to default")
                self.framework = ReasoningFramework.DEFAULT
        
        elif self.framework == ReasoningFramework.SELF_CONSISTENCY and settings.self_consistency_enabled:
            try:
                from reasoning.self_consistency import SelfConsistencyConfig, get_self_consistency
                config = SelfConsistencyConfig(
                    num_samples=settings.self_consistency_num_samples,
                    temperature=settings.self_consistency_temperature,
                    aggregation_strategy=settings.self_consistency_aggregation_strategy,
                    answer_normalization=settings.self_consistency_answer_normalization,
                    min_agreement_threshold=settings.self_consistency_min_agreement_threshold
                )
                self._self_consistency = get_self_consistency(config)
                logger.info("Self-Consistency reasoning initialized")
            except Exception as exc:
                logger.warning(f"Failed to initialize Self-Consistency: {exc}, falling back to default")
                self.framework = ReasoningFramework.DEFAULT
        
        elif self.framework == ReasoningFramework.REACT and settings.react_enabled:
            try:
                from reasoning.react import ReActConfig, get_react_agent
                config = ReActConfig(
                    max_steps=settings.react_max_steps,
                    max_iterations=settings.react_max_iterations,
                    temperature=settings.react_temperature,
                    allow_self_correction=settings.react_allow_self_correction,
                    verbose_thoughts=settings.react_verbose_thoughts
                )
                self._react_agent = get_react_agent(config)
                logger.info("ReAct reasoning initialized")
            except Exception as exc:
                logger.warning(f"Failed to initialize ReAct: {exc}, falling back to default")
                self.framework = ReasoningFramework.DEFAULT
        
        else:
            # Use default LangGraph agent
            logger.info("Using default LangGraph agent for reasoning")
    
    def solve(self, problem: str, context: str = "", **kwargs) -> tuple[str, dict[str, Any]]:
        """
        Solve a problem using the configured reasoning framework.
        
        Args:
            problem: The problem to solve
            context: Additional context for the problem
            **kwargs: Additional framework-specific parameters
            
        Returns:
            Tuple of (solution, metadata)
        """
        logger.info(f"Solving problem with {self.framework.value} framework")
        
        try:
            if self.framework == ReasoningFramework.REFLEXION and self._reflexion_agent:
                return self._solve_with_reflexion(problem, context, **kwargs)
            elif self.framework == ReasoningFramework.SELF_REFINE and self._self_refine_agent:
                return self._solve_with_self_refine(problem, context, **kwargs)
            elif self.framework == ReasoningFramework.SELF_CONSISTENCY and self._self_consistency:
                return self._solve_with_self_consistency(problem, context, **kwargs)
            elif self.framework == ReasoningFramework.REACT and self._react_agent:
                return self._solve_with_react(problem, context, **kwargs)
            else:
                return self._solve_with_default(problem, context, **kwargs)
        except Exception as exc:
            logger.exception(f"Reasoning failed with {self.framework.value}: {exc}")
            # Fallback to default
            return self._solve_with_default(problem, context, **kwargs)
    
    def _solve_with_reflexion(self, problem: str, context: str, **kwargs) -> tuple[str, dict[str, Any]]:
        """Solve using Reflexion with timeout."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        
        timeout = kwargs.get('timeout', settings.reflexion_timeout)
        
        def solve_reflexion():
            full_problem = f"{context}\n\n{problem}" if context else problem
            # Use the correct method name
            if hasattr(self._reflexion_agent, 'reflect'):
                return self._reflexion_agent.reflect(full_problem)
            elif hasattr(self._reflexion_agent, 'solve'):
                return self._reflexion_agent.solve(full_problem)
            else:
                # Fallback to basic reasoning
                from gateway.litellm_gateway import chat
                messages = [{"role": "user", "content": full_problem}]
                return chat(messages)
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(solve_reflexion)
                solution = future.result(timeout=timeout)
            
            metadata = {
                "framework": "reflexion",
                "reflections": getattr(self._reflexion_agent, 'reflection_count', 0)
            }
            
            return solution, metadata
            
        except FutureTimeoutError:
            logger.warning(f"Reflexion reasoning timed out after {timeout}s")
            return f"Error: Reflexion reasoning timed out after {timeout}s", {"framework": "reflexion", "error": "timeout"}
    
    def _solve_with_self_refine(self, problem: str, context: str, **kwargs) -> tuple[str, dict[str, Any]]:
        """Solve using Self-Refine with timeout."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        
        timeout = kwargs.get('timeout', settings.self_refine_timeout)
        
        def solve_self_refine():
            full_problem = f"{context}\n\n{problem}" if context else problem
            # Use the correct method name
            if hasattr(self._self_refine_agent, 'refine'):
                return self._self_refine_agent.refine(full_problem)
            elif hasattr(self._self_refine_agent, 'solve'):
                return self._self_refine_agent.solve(full_problem)
            else:
                # Fallback to basic reasoning
                from gateway.litellm_gateway import chat
                messages = [{"role": "user", "content": full_problem}]
                return chat(messages)
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(solve_self_refine)
                solution = future.result(timeout=timeout)
            
            metadata = {
                "framework": "self_refine",
                "iterations": getattr(self._self_refine_agent, 'iteration_count', 0)
            }
            
            return solution, metadata
            
        except FutureTimeoutError:
            logger.warning(f"Self-Refine reasoning timed out after {timeout}s")
            return f"Error: Self-Refine reasoning timed out after {timeout}s", {"framework": "self_refine", "error": "timeout"}
    
    def _solve_with_self_consistency(self, problem: str, context: str, **kwargs) -> tuple[str, dict[str, Any]]:
        """Solve using Self-Consistency with timeout."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        
        timeout = kwargs.get('timeout', settings.self_consistency_timeout)
        
        def solve_self_consistency():
            full_problem = f"{context}\n\n{problem}" if context else problem
            # Use the correct method name
            if hasattr(self._self_consistency, 'solve'):
                return self._self_consistency.solve(full_problem)
            else:
                # Fallback to basic reasoning
                from gateway.litellm_gateway import chat
                messages = [{"role": "user", "content": full_problem}]
                return chat(messages)
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(solve_self_consistency)
                solution = future.result(timeout=timeout)
            
            metadata = {
                "framework": "self_consistency",
                "samples": getattr(self._self_consistency, 'sample_count', 0),
                "agreement": getattr(self._self_consistency, 'agreement_score', 0.0)
            }
            
            return solution, metadata
            
        except FutureTimeoutError:
            logger.warning(f"Self-Consistency reasoning timed out after {timeout}s")
            return f"Error: Self-Consistency reasoning timed out after {timeout}s", {"framework": "self_consistency", "error": "timeout"}
    
    def _solve_with_react(self, problem: str, context: str, **kwargs) -> tuple[str, dict[str, Any]]:
        """Solve using ReAct with timeout."""
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
        
        timeout = kwargs.get('timeout', settings.react_timeout)
        
        def solve_react():
            full_problem = f"{context}\n\n{problem}" if context else problem
            # Use the correct method name
            if hasattr(self._react_agent, 'run'):
                return self._react_agent.run(full_problem)
            elif hasattr(self._react_agent, 'solve'):
                return self._react_agent.solve(full_problem)
            else:
                # Fallback to basic reasoning
                from gateway.litellm_gateway import chat
                messages = [{"role": "user", "content": full_problem}]
                return chat(messages)
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(solve_react)
                solution = future.result(timeout=timeout)
            
            metadata = {
                "framework": "react",
                "steps": getattr(self._react_agent, 'step_count', 0)
            }
            
            return solution, metadata
            
        except FutureTimeoutError:
            logger.warning(f"ReAct reasoning timed out after {timeout}s")
            return f"Error: ReAct reasoning timed out after {timeout}s", {"framework": "react", "error": "timeout"}
    
    def _solve_with_default(self, problem: str, context: str, **kwargs) -> tuple[str, dict[str, Any]]:
        """Solve using default method with fallback to direct LLM call."""
        full_problem = f"{context}\n\n{problem}" if context else problem
        
        try:
            # Try LangGraph agent first
            from agents.langgraph_agent import chat as agent_chat
            solution = agent_chat(full_problem)
            method = "langgraph_agent"
        except Exception as e:
            logger.warning(f"LangGraph agent failed: {e}, using direct LLM call")
            # Fallback to direct LLM call
            from gateway.litellm_gateway import chat
            messages = [{"role": "user", "content": full_problem}]
            solution = chat(messages)
            method = "direct_llm"
        
        metadata = {
            "framework": "default",
            "method": method
        }
        
        return solution, metadata
