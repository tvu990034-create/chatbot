"""
reasoning/__init__.py
~~~~~~~~~~~~~~~~~~~~~
Working reasoning framework integration.

This module provides unified access to working reasoning techniques:
- Chain-of-Thought (CoT): Basic step-by-step reasoning
- Reflexion: Self-reflection and error correction
- Self-Refine: Iterative output improvement
- Self-Consistency: Multi-sampling with majority voting
- ReAct: Reasoning + Acting framework
- Efficient Reasoning: Optimized multi-strategy reasoning
- Reasoning Cache: Caching for reasoning results
- Gateway Integration: Integration with LiteLLM gateway
"""

from reasoning.chain_of_thought import ChainOfThought
from reasoning.reflexion import (
    ReflexionAgent,
    ReflexionConfig,
    get_reflexion_agent
)
from reasoning.self_refine import (
    SelfRefineAgent,
    SelfRefineConfig,
    get_self_refine_agent
)
from reasoning.self_consistency import (
    SelfConsistency,
    SelfConsistencyConfig,
    get_self_consistency
)
from reasoning.react import (
    ReActAgent,
    ReActConfig,
    get_react_agent
)
from reasoning.efficient_reasoning import (
    OptimizedBenchmark,
    EfficientReasoner
)
from reasoning.reasoning_cache import (
    ReasoningCache,
    CachedReasoningWrapper
)
from reasoning.gateway_integration import GatewayIntegration

__all__ = [
    # Chain-of-Thought
    "ChainOfThought",
    
    # Reflexion
    "ReflexionAgent",
    "ReflexionConfig",
    "get_reflexion_agent",
    
    # Self-Refine
    "SelfRefineAgent",
    "SelfRefineConfig",
    "get_self_refine_agent",
    
    # Self-Consistency
    "SelfConsistency",
    "SelfConsistencyConfig",
    "get_self_consistency",
    
    # ReAct
    "ReActAgent",
    "ReActConfig",
    "get_react_agent",
    
    # Efficient Reasoning
    "OptimizedBenchmark",
    "EfficientReasoner",
    
    # Reasoning Cache
    "ReasoningCache",
    "CachedReasoningWrapper",
    
    # Gateway Integration
    "GatewayIntegration",
]
