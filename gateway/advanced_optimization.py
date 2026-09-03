"""
Advanced Optimization Tools - Lightweight Calculations
Implements practical advanced optimization techniques without complex infrastructure
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import math


class TokenCostOptimizer:
    """
    Lightweight token-cost sensitive decoding simulation.
    Simulates cost-aware token selection without model integration.
    """
    
    def __init__(self, lambda_cost: float = 0.3):
        self.lambda_cost = lambda_cost
    
    def token_cost_sensitive_decode(self, log_probs: List[float], token_costs: List[float]) -> int:
        """
        Select token maximizing log-probability minus cost.
        
        Args:
            log_probs: List of log-probabilities for next token
            token_costs: List of costs per token
        
        Returns:
            Index of chosen token
        """
        adjusted = [log_probs[i] - self.lambda_cost * token_costs[i] for i in range(len(log_probs))]
        return int(np.argmax(adjusted))
    
    def calculate_cost_adjusted_probs(self, log_probs: List[float], token_costs: List[float]) -> List[float]:
        """
        Calculate cost-adjusted probabilities.
        
        Args:
            log_probs: List of log-probabilities
            token_costs: List of costs per token
        
        Returns:
            Adjusted probabilities
        """
        log_probs = np.array(log_probs)
        token_costs = np.array(token_costs)
        adjusted = log_probs - self.lambda_cost * token_costs
        # Softmax to get probabilities
        adjusted = adjusted - np.max(adjusted)
        probs = np.exp(adjusted)
        return (probs / probs.sum()).tolist()


class UtilityCalculator:
    """
    Lightweight utility calculations for model comparison.
    Implements response time utility and latency-normalized scoring.
    """
    
    def __init__(self):
        pass
    
    def response_time_utility(self, accuracy: float, latency: float, beta: float = 0.3) -> float:
        """
        Calculate Cobb-Douglas utility balancing accuracy and latency.
        
        Args:
            accuracy: Accuracy between 0 and 1
            latency: Response time in milliseconds
            beta: Weight on log-latency (higher = penalize latency more)
        
        Returns:
            Utility value
        """
        if accuracy <= 0 or latency <= 0:
            return float('-inf')
        return math.log(accuracy) - beta * math.log(latency)
    
    def latency_normalized_score(self, score: float, latency: float, epsilon: float = 1e-6) -> float:
        """
        Calculate latency-normalized score.
        
        Args:
            score: Benchmark score (e.g., accuracy)
            latency: Average latency in milliseconds
            epsilon: Small constant to avoid division by zero
        
        Returns:
            Normalized score
        """
        return score / (latency + epsilon)
    
    def compare_models(self, models: List[Dict[str, float]], beta: float = 0.3) -> Dict[str, Any]:
        """
        Compare multiple models using utility functions.
        
        Args:
            models: List of model dicts with 'accuracy' and 'latency'
            beta: Weight for utility calculation
        
        Returns:
            Comparison results
        """
        results = []
        for i, model in enumerate(models):
            utility = self.response_time_utility(model['accuracy'], model['latency'], beta)
            norm_score = self.latency_normalized_score(model['accuracy'], model['latency'])
            results.append({
                'model_index': i,
                'utility': utility,
                'normalized_score': norm_score
            })
        
        # Sort by utility
        results.sort(key=lambda x: x['utility'], reverse=True)
        
        return {
            'rankings': results,
            'best_model': results[0]['model_index'] if results else None
        }


class EnsembleVoting:
    """
    Lightweight ensemble voting with confidence weighting.
    Implements confidence-weighted majority voting.
    """
    
    def __init__(self):
        pass
    
    def confidence_weighted_majority(self, predictions: List[Any], weights: List[float]) -> Any:
        """
        Weighted majority voting based on confidence.
        
        Args:
            predictions: List of predicted answers
            weights: List of confidence weights
        
        Returns:
            Answer with highest weighted vote
        """
        votes = {}
        for y, w in zip(predictions, weights):
            votes[y] = votes.get(y, 0.0) + w
        return max(votes, key=votes.get)
    
    def simple_majority(self, predictions: List[Any]) -> Any:
        """
        Simple majority voting (unweighted).
        
        Args:
            predictions: List of predicted answers
        
        Returns:
            Most common answer
        """
        from collections import Counter
        return Counter(predictions).most_common(1)[0][0]
    
    def weighted_consensus(self, predictions: List[Any], weights: List[float], threshold: float = 0.5) -> Optional[Any]:
        """
        Weighted consensus with threshold.
        
        Args:
            predictions: List of predicted answers
            weights: List of confidence weights
            threshold: Minimum weight threshold for consensus
        
        Returns:
        Consensus answer if available, None otherwise
        """
        votes = {}
        for y, w in zip(predictions, weights):
            votes[y] = votes.get(y, 0.0) + w
        
        # Check if any answer meets threshold
        for answer, weight in votes.items():
            if weight >= threshold:
                return answer
        
        return None


class SpeedupCalculator:
    """
    Lightweight speedup calculations for various optimization techniques.
    """
    
    def __init__(self):
        pass
    
    def retrieval_speedup(self, T_gen: float, T_ret: float, r: float) -> float:
        """
        Calculate speedup from retrieval augmentation.
        
        Args:
            T_gen: Full generation time without retrieval (ms)
            T_ret: Retrieval/cache lookup time (ms)
            r: Fraction of generation still needed (0 to 1)
        
        Returns:
            Speedup factor
        """
        return T_gen / (T_ret + r * T_gen)
    
    def optimal_pipeline_stages(self, workload: float, overhead_per_stage: float) -> int:
        """
        Calculate optimal number of pipeline stages.
        
        Args:
            workload: Total computational work
            overhead_per_stage: Synchronization overhead per stage
        
        Returns:
            Optimal number of stages
        """
        p = math.sqrt(workload / overhead_per_stage)
        return max(1, round(p))
    
    def effective_speedup(self, original_time: float, optimized_time: float) -> float:
        """
        Calculate effective speedup from optimization.
        
        Args:
            original_time: Original execution time
            optimized_time: Optimized execution time
        
        Returns:
            Speedup factor
        """
        if optimized_time <= 0:
            return float('inf')
        return original_time / optimized_time


class AdvancedOptimizationSuite:
    """
    Combined advanced optimization suite.
    Integrates token cost optimization, utility calculation, ensemble voting, and speedup analysis.
    """
    
    def __init__(self):
        self.token_optimizer = TokenCostOptimizer()
        self.utility_calc = UtilityCalculator()
        self.ensemble_voting = EnsembleVoting()
        self.speedup_calc = SpeedupCalculator()
    
    def optimize_token_selection(self, log_probs: List[float], token_costs: List[float]) -> int:
        """Select best token considering cost."""
        return self.token_optimizer.token_cost_sensitive_decode(log_probs, token_costs)
    
    def calculate_model_utility(self, accuracy: float, latency: float, beta: float = 0.3) -> float:
        """Calculate utility for model comparison."""
        return self.utility_calc.response_time_utility(accuracy, latency, beta)
    
    def ensemble_vote(self, predictions: List[Any], weights: List[float]) -> Any:
        """Perform weighted ensemble voting."""
        return self.ensemble_voting.confidence_weighted_majority(predictions, weights)
    
    def calculate_retrieval_speedup(self, T_gen: float, T_ret: float, r: float) -> float:
        """Calculate retrieval augmentation speedup."""
        return self.speedup_calc.retrieval_speedup(T_gen, T_ret, r)
    
    def calculate_pipeline_stages(self, workload: float, overhead: float) -> int:
        """Calculate optimal pipeline stages."""
        return self.speedup_calc.optimal_pipeline_stages(workload, overhead)
    
    def get_advanced_stats(self) -> Dict[str, Any]:
        """Return advanced optimization statistics."""
        return {
            "token_cost_optimization": {
                "enabled": True,
                "lambda_cost": self.token_optimizer.lambda_cost
            },
            "utility_calculation": {
                "enabled": True
            },
            "ensemble_voting": {
                "enabled": True
            },
            "speedup_analysis": {
                "enabled": True
            }
        }