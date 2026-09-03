"""
reasoning/self_consistency.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Self-Consistency: Improves reasoning through multiple sampling and majority voting.

Based on: "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
https://arxiv.org/abs/2203.11171
https://github.com/ethanjperez/self_consistency

Self-Consistency improves reasoning reliability by:
1. Sampling multiple reasoning paths
2. Using majority voting to select the most consistent answer
3. Reducing stochastic errors in complex reasoning tasks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from collections import Counter
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import settings

logger = logging.getLogger(__name__)


class AggregationStrategy(str, Enum):
    """Strategies for aggregating multiple solutions."""
    MAJORITY_VOTE = "majority_vote"  # Most common answer
    WEIGHTED_VOTE = "weighted_vote"  # Vote with confidence weights
    BEST_SCORE = "best_score"  # Solution with highest score
    CONSENSUS = "consensus"  # Most similar solutions
    AVERAGE = "average"  # Average of numeric answers


class AnswerNormalization(str, Enum):
    """Methods for normalizing answers before comparison."""
    EXACT = "exact"  # Exact string match
    CASE_INSENSITIVE = "case_insensitive"  # Ignore case
    WHITESPACE = "whitespace"  # Normalize whitespace
    NUMERIC = "numeric"  # Extract and compare numbers
    FUZZY = "fuzzy"  # Fuzzy string matching


@dataclass
class SampledSolution:
    """A single sampled solution with metadata."""
    solution: str
    reasoning: str = ""
    confidence: float = 0.0
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfConsistencyConfig:
    """Configuration for Self-Consistency framework."""
    num_samples: int = 5
    temperature: float = 0.7
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MAJORITY_VOTE
    answer_normalization: AnswerNormalization = AnswerNormalization.WHITESPACE
    min_agreement_threshold: float = 0.6
    enable_confidence_scoring: bool = True
    max_reasoning_length: int = 1000
    enable_parallel: bool = True
    max_workers: int = 4


class SelfConsistency:
    """
    Self-Consistency framework for improving reasoning reliability.
    
    This meta-framework can wrap other reasoning frameworks to:
    1. Generate multiple independent solutions
    2. Aggregate solutions using various strategies
    3. Provide confidence estimates
    4. Reduce stochastic errors
    """
    
    def __init__(self, config: SelfConsistencyConfig | None = None):
        self.config = config or SelfConsistencyConfig()
        self.llm_caller = self._get_llm_caller()
        self.solutions: list[SampledSolution] = []
    
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
    
    def solve(self, problem: str, context: str = "", solver: Optional[Callable] = None) -> tuple[str, dict[str, Any]]:
        """
        Solve a problem using Self-Consistency.
        
        Args:
            problem: The problem to solve
            context: Additional context
            solver: Optional solver function. If None, uses direct LLM calls.
            
        Returns:
            Tuple of (aggregated_solution, metadata)
        """
        logger.info(f"Starting Self-Consistency with {self.config.num_samples} samples")
        
        self.solutions = []
        
        # Sample multiple solutions
        if self.config.enable_parallel and self.config.num_samples > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                for i in range(self.config.num_samples):
                    logger.info(f"Submitting solution {i+1}/{self.config.num_samples}")
                    
                    if solver:
                        future = executor.submit(
                            self._solve_with_solver_parallel, 
                            problem, context, solver, i
                        )
                    else:
                        future = executor.submit(
                            self._solve_directly_parallel, 
                            problem, context, i
                        )
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    solution, reasoning = future.result()
                    self.solutions.append(SampledSolution(
                        solution=solution,
                        reasoning=reasoning,
                        confidence=0.0
                    ))
        else:
            # Sequential processing
            for i in range(self.config.num_samples):
                logger.info(f"Sampling solution {i+1}/{self.config.num_samples}")
                
                if solver:
                    # Use provided solver (e.g., ToT, ReAct, etc.)
                    solution, reasoning = self._solve_with_solver(problem, context, solver)
                else:
                    # Use direct LLM generation
                    solution, reasoning = self._solve_directly(problem, context)
                
                self.solutions.append(SampledSolution(
                    solution=solution,
                    reasoning=reasoning,
                    confidence=0.0
                ))
        
        # Score solutions if enabled
        if self.config.enable_confidence_scoring:
            self._score_solutions(problem, context)
        
        # Aggregate solutions
        aggregated_solution, metadata = self._aggregate_solutions(problem, context)
        
        logger.info(f"Self-Consistency complete. Agreement: {metadata.get('agreement_ratio', 0):.2f}")
        
        return aggregated_solution, metadata
    
    def _solve_with_solver(self, problem: str, context: str, solver: Callable) -> tuple[str, str]:
        """Solve using a provided solver function."""
        try:
            result = solver(problem, context)
            
            # Handle different return formats
            if isinstance(result, tuple):
                if len(result) == 2:
                    solution, metadata = result
                    reasoning = metadata.get("reasoning", "")
                else:
                    solution = result[0]
                    reasoning = ""
            else:
                solution = result
                reasoning = ""
            
            return solution, reasoning
            
        except Exception as exc:
            logger.warning(f"Solver failed: {exc}")
            return f"Error: {exc}", ""
    
    def _solve_directly(self, problem: str, context: str) -> tuple[str, str]:
        """Solve directly using LLM."""
        prompt = f"{context}\n\n{problem}" if context else problem
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_caller(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_reasoning_length
            )
            
            # Extract solution and reasoning
            solution = response.strip()
            reasoning = solution  # For direct generation, reasoning is the solution
            
            return solution, reasoning
            
        except Exception as exc:
            logger.warning(f"Direct LLM generation failed: {exc}")
            return f"Error: {exc}", ""
    
    def _solve_with_solver_parallel(self, problem: str, context: str, solver: Callable, sample_id: int) -> tuple[str, str]:
        """Parallel wrapper for solver-based solving."""
        try:
            result = solver(problem, context)
            
            # Handle different return formats
            if isinstance(result, tuple):
                if len(result) == 2:
                    solution, metadata = result
                    reasoning = metadata.get("reasoning", "")
                else:
                    solution = result[0]
                    reasoning = ""
            else:
                solution = result
                reasoning = ""
            
            logger.info(f"Completed solution {sample_id + 1}")
            return solution, reasoning
            
        except Exception as exc:
            logger.warning(f"Parallel solver failed for sample {sample_id}: {exc}")
            return f"Error: {exc}", ""
    
    def _solve_directly_parallel(self, problem: str, context: str, sample_id: int) -> tuple[str, str]:
        """Parallel wrapper for direct LLM solving."""
        prompt = f"{context}\n\n{problem}" if context else problem
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_caller(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_reasoning_length
            )
            
            # Extract solution and reasoning
            solution = response.strip()
            reasoning = solution  # For direct generation, reasoning is the solution
            
            logger.info(f"Completed solution {sample_id + 1}")
            return solution, reasoning
            
        except Exception as exc:
            logger.warning(f"Parallel direct LLM generation failed for sample {sample_id}: {exc}")
            return f"Error: {exc}", ""
    
    def _score_solutions(self, problem: str, context: str) -> None:
        """Score each solution for confidence."""
        for sol in self.solutions:
            try:
                score = self._evaluate_solution(sol.solution, problem, context)
                sol.score = score
                sol.confidence = score
            except Exception as exc:
                logger.warning(f"Failed to score solution: {exc}")
                sol.score = 0.5
                sol.confidence = 0.5
    
    def _evaluate_solution(self, solution: str, problem: str, context: str) -> float:
        """Evaluate the quality of a solution."""
        prompt = f"""Rate the following solution on a scale of 0.0 to 1.0 for correctness and quality.

Problem: {problem}
Context: {context}

Solution:
{solution}

Provide only a numerical rating (0.0 to 1.0)."""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm_caller(
                messages,
                temperature=0.1,
                max_tokens=10
            )
            
            # Extract numerical value
            match = re.search(r'([0-9.]+)', response)
            if match:
                value = float(match.group(1))
                return min(max(value, 0.0), 1.0)
            else:
                logger.warning(f"No numerical value found in evaluation response: {response}")
                return self._heuristic_evaluation(solution, problem)
            
        except Exception as exc:
            logger.error(f"Failed to evaluate solution: {exc}", exc_info=True)
            return self._heuristic_evaluation(solution, problem)
    
    def _heuristic_evaluation(self, solution: str, problem: str) -> float:
        """Heuristic fallback evaluation when LLM fails."""
        try:
            # Check for solution quality indicators
            quality_indicators = ['step', 'because', 'therefore', 'thus', 'answer', 'result']
            solution_lower = solution.lower()
            
            # Score based on length and structure
            length_score = min(len(solution) / 100, 1.0)
            structure_score = sum(1 for ind in quality_indicators if ind in solution_lower) / len(quality_indicators)
            
            # Check if solution directly addresses problem
            problem_words = set(problem.lower().split())
            solution_words = set(solution_lower.split())
            relevance_score = len(problem_words & solution_words) / max(len(problem_words), 1)
            
            final_score = 0.3 + (length_score * 0.3) + (structure_score * 0.2) + (relevance_score * 0.2)
            return max(0.0, min(1.0, final_score))
            
        except Exception as exc:
            logger.warning(f"Heuristic evaluation failed: {exc}")
            return 0.5  # Ultimate fallback
    
    def _aggregate_solutions(self, problem: str, context: str) -> tuple[str, dict[str, Any]]:
        """Aggregate multiple solutions into a final answer."""
        if self.config.aggregation_strategy == AggregationStrategy.MAJORITY_VOTE:
            return self._majority_vote_aggregation(problem, context)
        elif self.config.aggregation_strategy == AggregationStrategy.WEIGHTED_VOTE:
            return self._weighted_vote_aggregation(problem, context)
        elif self.config.aggregation_strategy == AggregationStrategy.BEST_SCORE:
            return self._best_score_aggregation(problem, context)
        elif self.config.aggregation_strategy == AggregationStrategy.CONSENSUS:
            return self._consensus_aggregation(problem, context)
        else:  # AVERAGE
            return self._average_aggregation(problem, context)
    
    def _normalize_answer(self, answer: str) -> str:
        """Normalize an answer for comparison."""
        if self.config.answer_normalization == AnswerNormalization.EXACT:
            return answer
        elif self.config.answer_normalization == AnswerNormalization.CASE_INSENSITIVE:
            return answer.lower()
        elif self.config.answer_normalization == AnswerNormalization.WHITESPACE:
            return " ".join(answer.split())
        elif self.config.answer_normalization == AnswerNormalization.NUMERIC:
            # Extract numbers
            numbers = re.findall(r'[-+]?\d*\.?\d+', answer)
            return str(float(numbers[0])) if numbers else answer
        else:  # FUZZY
            return " ".join(answer.split()).lower()
    
    def _majority_vote_aggregation(self, problem: str, context: str) -> tuple[str, dict[str, Any]]:
        """Aggregate using majority voting."""
        if not self.solutions:
            return "", {"aggregation_strategy": "majority_vote", "num_samples": 0, "agreement_ratio": 0.0}

        normalized_solutions = [self._normalize_answer(sol.solution) for sol in self.solutions]
        counts = Counter(normalized_solutions)
        
        # Get most common
        most_common = counts.most_common(1)[0]
        best_normalized, count = most_common
        
        # Find original solution with this normalized form
        best_solution = self.solutions[0].solution
        for sol in self.solutions:
            if self._normalize_answer(sol.solution) == best_normalized:
                best_solution = sol.solution
                break
        
        agreement_ratio = count / len(self.solutions)
        
        # Check if agreement threshold is met
        if agreement_ratio < self.config.min_agreement_threshold:
            logger.warning(f"Agreement ratio {agreement_ratio:.2f} below threshold {self.config.min_agreement_threshold}")
            # Fallback: use best score solution if confidence scoring is enabled
            if self.config.enable_confidence_scoring:
                best_solution = max(self.solutions, key=lambda s: s.score).solution
                logger.info("Using best score solution as fallback")
            else:
                # Fallback: use the solution with highest confidence
                best_solution = max(self.solutions, key=lambda s: s.confidence).solution
                logger.info("Using highest confidence solution as fallback")
        
        metadata = {
            "aggregation_strategy": "majority_vote",
            "agreement_ratio": agreement_ratio,
            "num_samples": len(self.solutions),
            "vote_distribution": dict(counts),
            "confidence": agreement_ratio,
            "threshold_met": agreement_ratio >= self.config.min_agreement_threshold,
            "fallback_used": agreement_ratio < self.config.min_agreement_threshold
        }
        
        return best_solution, metadata
    
    def _weighted_vote_aggregation(self, problem: str, context: str) -> tuple[str, dict[str, Any]]:
        """Aggregate using weighted voting based on scores."""
        if not self.solutions:
            return "", {"aggregation_strategy": "weighted_vote", "num_samples": 0, "agreement_ratio": 0.0}

        weighted_counts: dict[str, float] = {}
        
        for sol in self.solutions:
            normalized = self._normalize_answer(sol.solution)
            weight = sol.confidence if self.config.enable_confidence_scoring else 1.0
            weighted_counts[normalized] = weighted_counts.get(normalized, 0) + weight
        
        # Get highest weighted
        best_normalized = max(weighted_counts, key=weighted_counts.get)
        best_weight = weighted_counts[best_normalized]
        total_weight = sum(weighted_counts.values())
        
        # Find original solution
        best_solution = self.solutions[0].solution
        for sol in self.solutions:
            if self._normalize_answer(sol.solution) == best_normalized:
                best_solution = sol.solution
                break
        
        metadata = {
            "aggregation_strategy": "weighted_vote",
            "agreement_ratio": best_weight / total_weight if total_weight > 0 else 0,
            "num_samples": len(self.solutions),
            "weighted_distribution": weighted_counts,
            "confidence": best_weight / total_weight if total_weight > 0 else 0
        }
        
        return best_solution, metadata
    
    def _best_score_aggregation(self, problem: str, context: str) -> tuple[str, dict[str, Any]]:
        """Aggregate by selecting the highest-scoring solution."""
        if not self.solutions:
            return "", {"aggregation_strategy": "best_score", "num_samples": 0, "confidence": 0.0}

        best_solution = max(self.solutions, key=lambda s: s.score)
        
        metadata = {
            "aggregation_strategy": "best_score",
            "best_score": best_solution.score,
            "num_samples": len(self.solutions),
            "confidence": best_solution.score
        }
        
        return best_solution.solution, metadata
    
    def _consensus_aggregation(self, problem: str, context: str) -> tuple[str, dict[str, Any]]:
        """Aggregate by finding the most similar solutions."""
        # Simple implementation: use majority vote as consensus
        return self._majority_vote_aggregation(problem, context)
    
    def _average_aggregation(self, problem: str, context: str) -> tuple[str, dict[str, Any]]:
        """Aggregate by averaging numeric answers."""
        numeric_answers = []
        
        for sol in self.solutions:
            # Try to extract numeric answer
            numbers = re.findall(r'[-+]?\d*\.?\d+', sol.solution)
            if numbers:
                numeric_answers.append(float(numbers[0]))
        
        if numeric_answers:
            avg = sum(numeric_answers) / len(numeric_answers)
            best_solution = str(avg)
        else:
            # Fallback to majority vote
            return self._majority_vote_aggregation(problem, context)
        
        metadata = {
            "aggregation_strategy": "average",
            "numeric_values": numeric_answers,
            "average": avg,
            "num_samples": len(self.solutions),
            "confidence": 1.0
        }
        
        return best_solution, metadata
    
    def get_solution_diversity(self) -> dict[str, Any]:
        """Get statistics about solution diversity."""
        if not self.solutions:
            return {"total_solutions": 0}
        
        normalized_solutions = [self._normalize_answer(sol.solution) for sol in self.solutions]
        unique_solutions = len(set(normalized_solutions))
        
        return {
            "total_solutions": len(self.solutions),
            "unique_solutions": unique_solutions,
            "diversity_ratio": unique_solutions / len(self.solutions),
            "average_confidence": sum(sol.confidence for sol in self.solutions) / len(self.solutions)
        }


def get_self_consistency(config: SelfConsistencyConfig | None = None) -> SelfConsistency:
    """Get a configured Self-Consistency instance."""
    return SelfConsistency(config or SelfConsistencyConfig())