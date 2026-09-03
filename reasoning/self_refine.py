"""
reasoning/self_refine.py
~~~~~~~~~~~~~~~~~~~~~~~~
Self-Refine: Language Models Can Self-Correct for Generation Quality.

Based on: "Self-Refine: Language Models Can Self-Correct for Generation Quality"
https://arxiv.org/abs/2303.17651
https://github.com/madaan/self-refine

Self-Refine enables LLMs to iteratively improve their outputs through
self-evaluation and refinement, without requiring external feedback.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable
from datetime import datetime

from config import settings

logger = logging.getLogger(__name__)


class RefinementCriterion(str, Enum):
    """Criteria for refinement."""
    CLARITY = "clarity"  # Improve clarity
    ACCURACY = "accuracy"  # Improve accuracy
    COMPLETENESS = "completeness"  # Improve completeness
    COHERENCE = "coherence"  # Improve coherence
    CONCISENESS = "conciseness"  # Improve conciseness
    OVERALL = "overall"  # Overall quality


@dataclass
class RefinementIteration:
    """A single iteration in the self-refinement process."""
    iteration: int
    output: str
    feedback: str
    score: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfRefineConfig:
    """Configuration for Self-Refine framework."""
    max_iterations: int = 3
    min_improvement: float = 0.1
    criteria: list[RefinementCriterion] = field(default_factory=lambda: [
        RefinementCriterion.CLARITY,
        RefinementCriterion.ACCURACY,
        RefinementCriterion.COMPLETENESS
    ])
    temperature: float = 0.7
    evaluation_temperature: float = 0.3
    stop_threshold: float = 0.95
    enable_history: bool = True


class SelfRefineAgent:
    """
    Self-Refine agent with iterative improvement capabilities.
    
    The agent improves outputs through:
    1. Initial generation
    2. Self-evaluation based on criteria
    3. Feedback generation
    4. Refinement based on feedback
    5. Repeat until convergence or max iterations
    """
    
    def __init__(self, config: SelfRefineConfig | None = None):
        self.config = config or SelfRefineConfig()
        self.llm_caller = self._get_llm_caller()
        self.history: list[RefinementIteration] = []
    
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
    
    def refine(self, prompt: str, context: str = "") -> tuple[str, list[RefinementIteration]]:
        """
        Generate and iteratively refine an output.
        
        Args:
            prompt: The input prompt
            context: Additional context for the task
            
        Returns:
            Tuple of (final_output, iterations)
        """
        logger.info(f"Starting Self-Refine for prompt: {prompt[:100]}...")
        
        self.history = []
        
        # Initial generation
        current_output = self._generate_initial(prompt, context)
        current_score = self._evaluate_output(current_output, prompt, context)
        
        iteration = RefinementIteration(
            iteration=0,
            output=current_output,
            feedback="Initial generation",
            score=current_score
        )
        self.history.append(iteration)
        
        logger.info(f"Iteration 0: Score = {current_score:.3f}")
        
        # Iterative refinement
        for i in range(1, self.config.max_iterations):
            # Generate feedback
            feedback = self._generate_feedback(current_output, prompt, context)
            
            # Refine based on feedback
            refined_output = self._refine_output(current_output, feedback, prompt, context)
            
            # Evaluate refined output
            refined_score = self._evaluate_output(refined_output, prompt, context)
            
            iteration = RefinementIteration(
                iteration=i,
                output=refined_output,
                feedback=feedback,
                score=refined_score
            )
            self.history.append(iteration)
            
            logger.info(f"Iteration {i}: Score = {refined_score:.3f} (improvement: {refined_score - current_score:+.3f})")
            
            # Check convergence
            if refined_score >= self.config.stop_threshold:
                logger.info(f"Reached stop threshold {self.config.stop_threshold}")
                break
            
            # Check minimum improvement
            if refined_score - current_score < self.config.min_improvement:
                logger.info(f"Improvement below minimum threshold {self.config.min_improvement}")
                break
            
            # Update for next iteration
            current_output = refined_output
            current_score = refined_score
        
        # Return best output
        best_iteration = max(self.history, key=lambda x: x.score)
        logger.info(f"Self-Refine complete. Best score: {best_iteration.score:.3f} at iteration {best_iteration.iteration}")
        
        return best_iteration.output, self.history
    
    def _generate_initial(self, prompt: str, context: str) -> str:
        """Generate the initial output."""
        full_prompt = f"{context}\n\n{prompt}" if context else prompt
        
        try:
            messages = [{"role": "user", "content": full_prompt}]
            response = self.llm_caller(
                messages,
                temperature=self.config.temperature,
                max_tokens=1000
            )
            return response.strip()
        except Exception as exc:
            logger.warning(f"Failed to generate initial output: {exc}")
            return f"Error: {exc}"
    
    def _evaluate_output(self, output: str, prompt: str, context: str) -> float:
        """Evaluate the quality of an output."""
        criteria_text = ", ".join([c.value for c in self.config.criteria])
        
        evaluation_prompt = f"""Evaluate the following output on a scale of 0.0 to 1.0 based on these criteria: {criteria_text}.

Prompt: {prompt}
Context: {context}

Output:
{output}

Provide only a numerical rating (0.0 to 1.0)."""
        
        try:
            messages = [{"role": "user", "content": evaluation_prompt}]
            response = self.llm_caller(
                messages,
                temperature=self.config.evaluation_temperature,
                max_tokens=10
            )
            
            # Extract numerical value
            import re
            match = re.search(r'([0-9.]+)', response)
            if match:
                value = float(match.group(1))
                return min(max(value, 0.0), 1.0)
            
        except Exception as exc:
            logger.warning(f"Failed to evaluate output: {exc}")
        
        return 0.5
    
    def _generate_feedback(self, output: str, prompt: str, context: str) -> str:
        """Generate feedback for improvement."""
        criteria_text = ", ".join([c.value for c in self.config.criteria])
        
        feedback_prompt = f"""Provide constructive feedback to improve the following output based on these criteria: {criteria_text}.

Prompt: {prompt}
Context: {context}

Current output:
{output}

Identify specific areas for improvement and provide actionable suggestions."""
        
        try:
            messages = [{"role": "user", "content": feedback_prompt}]
            response = self.llm_caller(
                messages,
                temperature=self.config.temperature,
                max_tokens=300
            )
            return response.strip()
        except Exception as exc:
            logger.warning(f"Failed to generate feedback: {exc}")
            return "No specific feedback available."
    
    def _refine_output(self, output: str, feedback: str, prompt: str, context: str) -> str:
        """Refine the output based on feedback."""
        refine_prompt = f"""Improve the following output based on the provided feedback.

Prompt: {prompt}
Context: {context}

Current output:
{output}

Feedback:
{feedback}

Provide an improved version of the output that addresses the feedback."""
        
        try:
            messages = [{"role": "user", "content": refine_prompt}]
            response = self.llm_caller(
                messages,
                temperature=self.config.temperature,
                max_tokens=1000
            )
            return response.strip()
        except Exception as exc:
            logger.warning(f"Failed to refine output: {exc}")
            return output  # Return original if refinement fails
    
    def get_improvement_stats(self) -> dict[str, Any]:
        """Get statistics about the refinement process."""
        if not self.history:
            return {"total_iterations": 0}
        
        scores = [iter.score for iter in self.history]
        improvements = [scores[i] - scores[i-1] for i in range(1, len(scores))]
        
        return {
            "total_iterations": len(self.history),
            "initial_score": scores[0],
            "final_score": scores[-1],
            "best_score": max(scores),
            "total_improvement": scores[-1] - scores[0],
            "average_improvement": sum(improvements) / len(improvements) if improvements else 0.0,
            "best_iteration": max(self.history, key=lambda x: x.score).iteration
        }


def get_self_refine_agent(config: SelfRefineConfig | None = None) -> SelfRefineAgent:
    """Get a configured Self-Refine agent."""
    return SelfRefineAgent(config or SelfRefineConfig())