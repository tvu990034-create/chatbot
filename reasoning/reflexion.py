"""
reasoning/reflexion.py
~~~~~~~~~~~~~~~~~~~~~~~
Reflexion: Self-reflection framework for language agents.

Based on: "Reflexion: Language Agents with Verbal Reinforcement Learning"
https://arxiv.org/abs/2303.11366
https://github.com/noahshinn024/reflexion

Reflexion enables agents to improve their performance through:
1. Self-reflection on past failures
2. Maintaining episodic memory of reflections
3. Using reflections to inform future attempts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from datetime import datetime
import json

from config import settings

logger = logging.getLogger(__name__)


class ReflectionType(str, Enum):
    """Types of reflections."""
    ERROR_ANALYSIS = "error_analysis"  # Analyze what went wrong
    IMPROVEMENT = "improvement"  # Suggest improvements
    STRATEGY = "strategy"  # Suggest alternative strategies
    KNOWLEDGE_GAP = "knowledge_gap"  # Identify missing knowledge


@dataclass
class Reflection:
    """A single reflection on past performance."""
    content: str
    reflection_type: ReflectionType
    timestamp: datetime = field(default_factory=datetime.now)
    context: str = ""
    usefulness_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """A complete episode (attempt + reflection)."""
    problem: str
    attempt: str
    success: bool
    reflection: Reflection | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflexionConfig:
    """Configuration for Reflexion framework."""
    max_reflections: int = 5
    max_episodes: int = 20
    reflection_types: list[ReflectionType] = field(default_factory=lambda: [
        ReflectionType.ERROR_ANALYSIS,
        ReflectionType.IMPROVEMENT,
        ReflectionType.STRATEGY
    ])
    reflection_temperature: float = 0.7
    episode_memory_file: str = "data/reflexion_episodes.json"
    enable_persistent_memory: bool = True


class ReflexionMemory:
    """Episodic memory for storing and retrieving reflections."""
    
    def __init__(self, config: ReflexionConfig):
        self.config = config
        self.episodes: list[Episode] = []
        self.reflections: list[Reflection] = []
        
        if config.enable_persistent_memory:
            self._load_memory()
    
    def _load_memory(self) -> None:
        """Load episodes from persistent storage."""
        try:
            import os
            if os.path.exists(self.config.episode_memory_file):
                with open(self.config.episode_memory_file, 'r') as f:
                    data = json.load(f)
                    
                    # Load episodes with validation
                    episodes_data = data.get("episodes", [])
                    self.episodes = []
                    for ep_data in episodes_data:
                        try:
                            # Handle reflection data separately
                            reflection_data = ep_data.get("reflection")
                            if reflection_data:
                                ep_data["reflection"] = Reflection(**reflection_data)
                            # Convert timestamp string back to datetime
                            if "timestamp" in ep_data and isinstance(ep_data["timestamp"], str):
                                from datetime import datetime
                                ep_data["timestamp"] = datetime.fromisoformat(ep_data["timestamp"])
                            self.episodes.append(Episode(**ep_data))
                        except Exception as ep_exc:
                            logger.warning(f"Failed to load episode: {ep_exc}")
                    
                    # Load reflections with validation
                    reflections_data = data.get("reflections", [])
                    self.reflections = []
                    for ref_data in reflections_data:
                        try:
                            # Convert timestamp string back to datetime
                            if "timestamp" in ref_data and isinstance(ref_data["timestamp"], str):
                                from datetime import datetime
                                ref_data["timestamp"] = datetime.fromisoformat(ref_data["timestamp"])
                            self.reflections.append(Reflection(**ref_data))
                        except Exception as ref_exc:
                            logger.warning(f"Failed to load reflection: {ref_exc}")
                    
                    logger.info(f"Loaded {len(self.episodes)} episodes and {len(self.reflections)} reflections")
        except Exception as exc:
            logger.error(f"Failed to load reflexion memory: {exc}", exc_info=True)
            self.episodes = []
            self.reflections = []
    
    def _save_memory(self) -> None:
        """Save episodes to persistent storage."""
        if not self.config.enable_persistent_memory:
            return
        
        try:
            import os
            os.makedirs(os.path.dirname(self.config.episode_memory_file), exist_ok=True)
            
            data = {
                "episodes": [
                    {
                        "problem": ep.problem,
                        "attempt": ep.attempt,
                        "success": ep.success,
                        "reflection": ep.reflection.__dict__ if ep.reflection else None,
                        "timestamp": ep.timestamp.isoformat(),
                        "metadata": ep.metadata
                    }
                    for ep in self.episodes
                ],
                "reflections": [
                    {
                        "content": ref.content,
                        "reflection_type": ref.reflection_type,
                        "timestamp": ref.timestamp.isoformat(),
                        "context": ref.context,
                        "usefulness_score": ref.usefulness_score,
                        "metadata": ref.metadata
                    }
                    for ref in self.reflections
                ]
            }
            
            with open(self.config.episode_memory_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as exc:
            logger.warning(f"Failed to save reflexion memory: {exc}")
    
    def add_episode(self, episode: Episode) -> None:
        """Add an episode to memory."""
        self.episodes.append(episode)
        if episode.reflection:
            self.reflections.append(episode.reflection)
        
        # Trim if exceeding max
        if len(self.episodes) > self.config.max_episodes:
            self.episodes = self.episodes[-self.config.max_episodes:]
        
        self._save_memory()
    
    def get_relevant_reflections(self, problem: str, top_k: int = 3) -> list[Reflection]:
        """Get most relevant reflections for a given problem."""
        if not self.reflections:
            return []
        
        # Simple relevance scoring based on keyword overlap
        problem_words = set(problem.lower().split())
        scored_reflections = []
        
        for reflection in self.reflections:
            context_words = set(reflection.context.lower().split())
            overlap = len(problem_words & context_words)
            scored_reflections.append((reflection, overlap))
        
        # Sort by relevance and usefulness
        scored_reflections.sort(
            key=lambda x: (x[1], x[0].usefulness_score),
            reverse=True
        )
        
        return [ref for ref, _ in scored_reflections[:top_k]]
    
    def get_failure_patterns(self) -> list[str]:
        """Extract common failure patterns from episodes."""
        failures = [ep for ep in self.episodes if not ep.success]
        patterns = []
        
        for failure in failures[-10:]:  # Last 10 failures
            if failure.reflection:
                patterns.append(failure.reflection.content)
        
        return patterns


class ReflexionAgent:
    """
    Reflexion agent with self-reflection capabilities.
    
    The agent improves over time by:
    1. Attempting to solve problems
    2. Reflecting on failures
    3. Using reflections to guide future attempts
    """
    
    def __init__(self, config: ReflexionConfig | None = None):
        self.config = config or ReflexionConfig()
        self.memory = ReflexionMemory(self.config)
        self.llm_caller = self._get_llm_caller()
    
    def _get_llm_caller(self) -> Callable:
        """Get the LLM caller function."""
        try:
            from gateway.litellm_gateway import chat
            return chat
        except ImportError:
            logger.warning("LiteLLM gateway not available, using fallback")
            return self._fallback_llm
    
    def _fallback_llm(self, messages: list[dict], **kwargs) -> str:
        """Fallback LLM caller when gateway is unavailable."""
        logger.error("LLM gateway unavailable, returning mock response")
        return "Mock response - LLM gateway not configured"
    
    def solve_with_reflection(self, problem: str, max_attempts: int = 3) -> tuple[str, bool, list[Episode]]:
        """
        Solve a problem with reflexion (multiple attempts + self-reflection).
        
        Args:
            problem: The problem to solve
            max_attempts: Maximum number of attempts
            
        Returns:
            Tuple of (solution, success, episodes)
        """
        logger.info(f"Starting reflexion for problem: {problem[:100]}...")
        
        episodes = []
        current_problem = problem
        
        for attempt_num in range(max_attempts):
            # Get relevant reflections for this attempt
            if attempt_num > 0:
                relevant_reflections = self.memory.get_relevant_reflections(problem)
                reflection_context = self._format_reflections(relevant_reflections)
                current_problem = f"{problem}\n\nPast reflections to consider:\n{reflection_context}"
            else:
                relevant_reflections = []
            
            # Generate attempt
            attempt = self._generate_attempt(current_problem, attempt_num)
            
            # Evaluate success
            success = self._evaluate_success(problem, attempt)
            
            # Create episode
            episode = Episode(
                problem=problem,
                attempt=attempt,
                success=success,
                timestamp=datetime.now(),
                metadata={"attempt_number": attempt_num + 1}
            )
            
            if not success and attempt_num < max_attempts - 1:
                # Generate reflection on failure
                reflection = self._generate_reflection(problem, attempt, relevant_reflections)
                episode.reflection = reflection
                
                logger.info(f"Attempt {attempt_num + 1} failed. Reflection: {reflection.content[:100]}...")
            else:
                logger.info(f"Attempt {attempt_num + 1} {'succeeded' if success else 'failed (final)'}")
            
            episodes.append(episode)
            self.memory.add_episode(episode)
            
            if success:
                break
        
        # Return best attempt
        best_episode = max(episodes, key=lambda ep: len(ep.attempt))
        return best_episode.attempt, best_episode.success, episodes
    
    def _generate_attempt(self, problem: str, attempt_num: int) -> str:
        """Generate an attempt at solving the problem."""
        prompt = f"""Solve the following problem. Provide a clear, step-by-step solution.

Problem: {problem}

{"This is your first attempt. Provide your best solution." if attempt_num == 0 else f"This is attempt #{attempt_num + 1}. Use any insights from past reflections to improve your approach."}

Solution:"""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_caller(
                messages,
                temperature=self.config.reflection_temperature,
                max_tokens=1000
            )
            return response.strip()
        except Exception as exc:
            logger.warning(f"Failed to generate attempt: {exc}")
            return f"Failed to generate solution: {exc}"
    
    def _evaluate_success(self, problem: str, attempt: str) -> bool:
        """Evaluate whether the attempt successfully solves the problem."""
        prompt = f"""Evaluate whether the following solution successfully solves the problem.

Problem: {problem}

Solution: {attempt}

Provide a binary judgment: SUCCESS or FAILURE. Then briefly explain your reasoning."""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_caller(
                messages,
                temperature=0.1,
                max_tokens=100
            )
            
            # Check for success indicators
            response_lower = response.lower()
            return "success" in response_lower and "failure" not in response_lower
            
        except Exception as exc:
            logger.warning(f"Failed to evaluate success: {exc}")
            return True  # Default to success if evaluation fails
    
    def _generate_reflection(self, problem: str, attempt: str, past_reflections: list[Reflection]) -> Reflection:
        """Generate a reflection on a failed attempt."""
        # Select reflection type
        if past_reflections:
            # Use different reflection type than most recent
            recent_types = [r.reflection_type for r in past_reflections[-3:]]
            available_types = [t for t in self.config.reflection_types if t not in recent_types]
            reflection_type = available_types[0] if available_types else self.config.reflection_types[0]
        else:
            reflection_type = self.config.reflection_types[0]
        
        # Build reflection prompt based on type
        if reflection_type == ReflectionType.ERROR_ANALYSIS:
            prompt = self._build_error_analysis_prompt(problem, attempt)
        elif reflection_type == ReflectionType.IMPROVEMENT:
            prompt = self._build_improvement_prompt(problem, attempt)
        else:  # STRATEGY
            prompt = self._build_strategy_prompt(problem, attempt)
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_caller(
                messages,
                temperature=self.config.reflection_temperature,
                max_tokens=300
            )
            
            return Reflection(
                content=response.strip(),
                reflection_type=reflection_type,
                context=problem,
                metadata={"attempt_length": len(attempt)}
            )
            
        except Exception as exc:
            logger.error(f"Failed to generate reflection: {exc}", exc_info=True)
            # Use heuristic fallback
            return Reflection(
                content=self._heuristic_reflection_content(problem, attempt, reflection_type),
                reflection_type=reflection_type,
                context=problem,
                metadata={"heuristic": True}
            )
    
    def _heuristic_reflection_content(self, problem: str, attempt: str, reflection_type: ReflectionType) -> str:
        """Generate heuristic reflection content when LLM fails."""
        try:
            if reflection_type == ReflectionType.ERROR_ANALYSIS:
                if len(attempt) < 50:
                    return "The solution was too brief and may have missed important details or reasoning steps."
                elif "uncertain" in attempt.lower() or "maybe" in attempt.lower():
                    return "The solution showed uncertainty without clear justification for the chosen approach."
                else:
                    return "Review the solution for potential logical gaps or calculation errors."
            elif reflection_type == ReflectionType.IMPROVEMENT:
                return "Consider breaking down the problem into smaller steps and verifying each step independently."
            else:  # STRATEGY
                return "Try alternative approaches such as working backwards or using analogies to similar problems."
        except Exception as exc:
            logger.warning(f"Heuristic reflection content failed: {exc}")
            return "Review the approach carefully and consider alternative methods."
    
    def _build_error_analysis_prompt(self, problem: str, attempt: str) -> str:
        """Build prompt for error analysis reflection."""
        return f"""Analyze why the following solution failed to solve the problem.

Problem: {problem}

Failed Solution: {attempt}

Identify the specific errors, misconceptions, or missing steps in the solution. Provide concrete recommendations for what should be done differently."""
    
    def _build_improvement_prompt(self, problem: str, attempt: str) -> str:
        """Build prompt for improvement reflection."""
        return f"""Suggest specific improvements to the following solution.

Problem: {problem}

Current Solution: {attempt}

What specific changes, additions, or modifications would improve this solution? Focus on actionable improvements."""
    
    def _build_strategy_prompt(self, problem: str, attempt: str) -> str:
        """Build prompt for strategy reflection."""
        return f"""Suggest an alternative strategy for solving this problem.

Problem: {problem}

Current Approach: {attempt}

What different approach or strategy might work better? Explain the alternative strategy and why it might be more effective."""
    
    def _format_reflections(self, reflections: list[Reflection]) -> str:
        """Format reflections for inclusion in prompts."""
        if not reflections:
            return "No past reflections available."
        
        formatted = []
        for i, reflection in enumerate(reflections, 1):
            formatted.append(f"{i}. [{reflection.reflection_type.value}]: {reflection.content}")
        
        return "\n".join(formatted)
    
    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics."""
        if not self.memory.episodes:
            return {"total_episodes": 0}
        
        total = len(self.memory.episodes)
        successful = sum(1 for ep in self.memory.episodes if ep.success)
        
        return {
            "total_episodes": total,
            "successful_episodes": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "total_reflections": len(self.memory.reflections),
            "reflection_types": {
                rt.value: sum(1 for r in self.memory.reflections if r.reflection_type == rt)
                for rt in ReflectionType
            }
        }


def get_reflexion_agent(config: ReflexionConfig | None = None) -> ReflexionAgent:
    """Get a configured Reflexion agent."""
    return ReflexionAgent(config or ReflexionConfig())