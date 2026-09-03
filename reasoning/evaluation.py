"""
reasoning/evaluation.py
~~~~~~~~~~~~~~~~~~~~~~
Reasoning evaluation framework for assessing reasoning quality.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Callable
import re
from collections import Counter

logger = logging.getLogger(__name__)


class ReasoningEvaluator:
    """
    Evaluator for reasoning quality and correctness.
    
    Provides comprehensive evaluation metrics for different reasoning methods:
    - Answer correctness
    - Reasoning quality
    - Consistency metrics
    - Step validation
    """
    
    def __init__(self):
        self.evaluation_methods = {
            "exact_match": self._exact_match,
            "numerical_match": self._numerical_match,
            "semantic_similarity": self._semantic_similarity,
            "reasoning_coherence": self._reasoning_coherence,
            "step_relevance": self._step_relevance,
        }
        
        logger.info("ReasoningEvaluator initialized")
    
    def _exact_match(
        self,
        predicted: str,
        ground_truth: str,
    ) -> float:
        """Exact match evaluation."""
        return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0
    
    def _numerical_match(
        self,
        predicted: str,
        ground_truth: str,
    ) -> float:
        """Numerical match evaluation (extracts numbers)."""
        pred_numbers = re.findall(r'-?\d+\.?\d*', predicted)
        true_numbers = re.findall(r'-?\d+\.?\d*', ground_truth)
        
        if not pred_numbers or not true_numbers:
            return 0.0
        
        # Compare first numbers (simplified)
        try:
            return 1.0 if float(pred_numbers[0]) == float(true_numbers[0]) else 0.0
        except ValueError:
            return 0.0
    
    def _semantic_similarity(
        self,
        predicted: str,
        ground_truth: str,
    ) -> float:
        """Semantic similarity using word overlap."""
        pred_words = set(predicted.lower().split())
        true_words = set(ground_truth.lower().split())
        
        if not pred_words or not true_words:
            return 0.0
        
        intersection = len(pred_words & true_words)
        union = len(pred_words | true_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _reasoning_coherence(
        self,
        reasoning_steps: List[str],
    ) -> float:
        """Evaluate coherence of reasoning steps."""
        if len(reasoning_steps) < 2:
            return 1.0
        
        # Check for logical progression indicators
        progression_words = ["then", "next", "after", "following", "so", "therefore", "thus"]
        has_progression = sum(
            1 for step in reasoning_steps
            if any(word in step.lower() for word in progression_words)
        )
        
        coherence_score = has_progression / (len(reasoning_steps) - 1)
        
        return min(coherence_score, 1.0)
    
    def _step_relevance(
        self,
        reasoning_steps: List[str],
        question: str,
    ) -> float:
        """Evaluate relevance of reasoning steps to the question."""
        if not reasoning_steps:
            return 0.0
        
        question_words = set(question.lower().split())
        relevance_scores = []
        
        for step in reasoning_steps:
            step_words = set(step.lower().split())
            overlap = len(question_words & step_words)
            relevance = overlap / len(question_words) if question_words else 0.0
            relevance_scores.append(min(relevance, 1.0))
        
        return sum(relevance_scores) / len(relevance_scores)
    
    def evaluate(
        self,
        reasoning_result: Dict,
        ground_truth: str,
        evaluation_methods: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate reasoning result against ground truth.
        
        Args:
            reasoning_result: Result from reasoning method
            ground_truth: Correct answer
            evaluation_methods: List of evaluation methods to use
        
        Returns:
            Dictionary of evaluation scores
        """
        if evaluation_methods is None:
            evaluation_methods = ["exact_match", "numerical_match", "semantic_similarity"]
        
        predicted = reasoning_result.get("final_answer", "")
        reasoning_steps = reasoning_result.get("reasoning_steps", [])
        
        scores = {}
        
        for method in evaluation_methods:
            if method in self.evaluation_methods:
                if method in ["exact_match", "numerical_match", "semantic_similarity"]:
                    scores[method] = self.evaluation_methods[method](predicted, ground_truth)
                elif method == "reasoning_coherence":
                    scores[method] = self.evaluation_methods[method](reasoning_steps)
                elif method == "step_relevance":
                    scores[method] = self.evaluation_methods[method](reasoning_steps, reasoning_result.get("question", ""))
        
        return scores
    
    def batch_evaluate(
        self,
        reasoning_results: List[Dict],
        ground_truths: List[str],
        evaluation_methods: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate multiple reasoning results.
        
        Args:
            reasoning_results: List of reasoning results
            ground_truths: List of ground truth answers
            evaluation_methods: List of evaluation methods to use
        
        Returns:
            Dictionary of average scores
        """
        if len(reasoning_results) != len(ground_truths):
            logger.warning("Mismatch between results and ground truths")
            return {}
        
        all_scores = {method: [] for method in (evaluation_methods or ["exact_match"])}
        
        for result, truth in zip(reasoning_results, ground_truths):
            scores = self.evaluate(result, truth, evaluation_methods)
            for method, score in scores.items():
                all_scores[method].append(score)
        
        # Compute averages
        avg_scores = {}
        for method, scores_list in all_scores.items():
            if scores_list:
                avg_scores[method] = sum(scores_list) / len(scores_list)
        
        return avg_scores
    
    def compare_methods(
        self,
        method_results: Dict[str, List[Dict]],
        ground_truths: List[str],
        evaluation_methods: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare different reasoning methods.
        
        Args:
            method_results: Dictionary mapping method names to result lists
            ground_truths: List of ground truth answers
            evaluation_methods: List of evaluation methods to use
        
        Returns:
            Dictionary mapping method names to their scores
        """
        comparison = {}
        
        for method_name, results in method_results.items():
            scores = self.batch_evaluate(results, ground_truths, evaluation_methods)
            comparison[method_name] = scores
        
        return comparison
    
    def generate_report(
        self,
        comparison: Dict[str, Dict[str, float]],
    ) -> str:
        """
        Generate a comparison report.
        
        Args:
            comparison: Comparison results from compare_methods
        
        Returns:
            Formatted report string
        """
        report = "Reasoning Method Comparison Report\n"
        report += "=" * 50 + "\n\n"
        
        for method, scores in comparison.items():
            report += f"{method}:\n"
            for metric, score in scores.items():
                report += f"  {metric}: {score:.4f}\n"
            report += "\n"
        
        # Find best method for each metric
        report += "Best Methods by Metric:\n"
        for metric in comparison[list(comparison.keys())[0]].keys():
            best_method = max(
                comparison.items(),
                key=lambda x: x[1].get(metric, 0.0)
            )
            report += f"  {metric}: {best_method[0]} ({best_method[1][metric]:.4f})\n"
        
        return report