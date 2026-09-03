"""
Meta-Reasoning Tools - Lightweight Decision Making
Implements practical meta-reasoning techniques without complex training
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable


class DebateStoppingCriterion:
    """
    Lightweight debate stopping criterion based on confidence vs cost.
    Implements optimal stopping for debate termination.
    """
    
    def __init__(self, cost_per_turn: float = 0.01, threshold: float = 0.01):
        self.cost_per_turn = cost_per_turn
        self.threshold = threshold
    
    def should_stop(self, confidence: float, expected_gain_func: Callable[[], float]) -> bool:
        """
        Determine if debate should stop based on confidence vs cost.
        
        Args:
            confidence: Current probability of correct answer
            expected_gain_func: Function that estimates gain in confidence from one turn
        
        Returns:
            True if should stop, False otherwise
        """
        stop_value = confidence  # Value of stopping = current confidence
        continue_value = confidence + expected_gain_func() - self.cost_per_turn
        return continue_value <= stop_value + self.threshold
    
    def optimal_stopping_time(self, confidence_history: List[float], cost_per_turn: float = 0.01) -> int:
        """
        Find optimal stopping time given confidence history.
        
        Args:
            confidence_history: List of confidence values over time
            cost_per_turn: Cost of one more turn
        
        Returns:
            Optimal stopping index
        """
        best_time = 0
        best_value = confidence_history[0]
        
        for i, conf in enumerate(confidence_history):
            # Simple: stop when marginal gain < cost
            if i > 0:
                marginal_gain = conf - confidence_history[i-1]
                if marginal_gain < cost_per_turn:
                    return i - 1
            
            if conf > best_value:
                best_value = conf
                best_time = i
        
        return best_time


class ComputationalBudgetOptimizer:
    """
    Lightweight knapsack-based computational budget optimization.
    Selects reasoning steps within a fixed computational budget.
    """
    
    def __init__(self):
        pass
    
    def knapsack_select(self, weights: List[float], values: List[float], capacity: float) -> tuple:
        """
        Select items using 0-1 knapsack with continuous relaxation.
        
        Args:
            weights: Computational costs of items
            values: Contribution values of items
            capacity: Total computational budget
        
        Returns:
            Tuple of (total_value, selected_indices)
        """
        n = len(weights)
        if n == 0:
            return 0.0, []
        
        # Simple greedy approach (approximate but fast)
        # Sort by value/weight ratio
        items = list(zip(range(n), weights, values))
        items.sort(key=lambda x: x[2]/x[1] if x[1] > 0 else 0, reverse=True)
        
        selected = []
        total_weight = 0.0
        total_value = 0.0
        
        for idx, weight, value in items:
            if total_weight + weight <= capacity:
                selected.append(idx)
                total_weight += weight
                total_value += value
        
        return total_value, selected
    
    def optimize_reasoning_steps(self, steps: List[Dict[str, float]], budget: float) -> List[Dict[str, float]]:
        """
        Optimize selection of reasoning steps within budget.
        
        Args:
            steps: List of step dicts with 'cost' and 'value' keys
            budget: Total computational budget
        
        Returns:
            Selected steps
        """
        weights = [step['cost'] for step in steps]
        values = [step['value'] for step in steps]
        
        _, selected_indices = self.knapsack_select(weights, values, budget)
        return [steps[i] for i in selected_indices]


class MetaReasoningPolicy:
    """
    Lightweight meta-reasoning policy for tool/action selection.
    Implements Q-table based action selection.
    """
    
    def __init__(self):
        self.Q_table = {}  # (state, action) -> value
    
    def get_value(self, state: int, action: str) -> float:
        """Get Q-value for state-action pair."""
        return self.Q_table.get((state, action), 0.0)
    
    def set_value(self, state: int, action: str, value: float):
        """Set Q-value for state-action pair."""
        self.Q_table[(state, action)] = value
    
    def select_action(self, state: int, available_actions: List[str]) -> str:
        """
        Select best action based on Q-values.
        
        Args:
            state: Current reasoning state (discrete index)
            available_actions: List of available actions
        
        Returns:
            Best action
        """
        if not available_actions:
            return None
        
        best_action = max(available_actions, key=lambda a: self.get_value(state, a))
        return best_action
    
    def update_value(self, state: int, action: str, reward: float, learning_rate: float = 0.1):
        """
        Update Q-value using simple reward signal.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            learning_rate: Learning rate for update
        """
        current_value = self.get_value(state, action)
        new_value = current_value + learning_rate * (reward - current_value)
        self.set_value(state, action, new_value)
    
    def initialize_default_policy(self):
        """Initialize with reasonable default values."""
        # State 0: Initial question
        self.set_value(0, 'retrieve', 0.8)
        self.set_value(0, 'reason', 0.7)
        self.set_value(0, 'cot', 0.6)
        
        # State 1: Partial answer
        self.set_value(1, 'calculate', 0.9)
        self.set_value(1, 'debate', 0.6)
        self.set_value(1, 'rag', 0.7)
        
        # State 2: High confidence
        self.set_value(2, 'answer', 0.95)
        self.set_value(2, 'verify', 0.8)


class MetaReasoningSuite:
    """
    Combined meta-reasoning suite.
    Integrates debate stopping, budget optimization, and action selection.
    """
    
    def __init__(self):
        self.debate_stopping = DebateStoppingCriterion()
        self.budget_optimizer = ComputationalBudgetOptimizer()
        self.meta_policy = MetaReasoningPolicy()
        self.meta_policy.initialize_default_policy()
    
    def should_stop_debate(self, confidence: float, expected_gain: float) -> bool:
        """Determine if debate should stop."""
        return self.debate_stopping.should_stop(confidence, lambda: expected_gain)
    
    def optimize_computational_budget(self, steps: List[Dict[str, float]], budget: float) -> List[Dict[str, float]]:
        """Optimize reasoning steps within budget."""
        return self.budget_optimizer.optimize_reasoning_steps(steps, budget)
    
    def select_reasoning_action(self, state: int, available_actions: List[str]) -> str:
        """Select best reasoning action using meta policy."""
        return self.meta_policy.select_action(state, available_actions)
    
    def update_action_value(self, state: int, action: str, reward: float):
        """Update action value based on reward."""
        self.meta_policy.update_value(state, action, reward)
    
    def get_meta_stats(self) -> Dict[str, Any]:
        """Return meta-reasoning statistics."""
        return {
            "debate_stopping": {
                "cost_per_turn": self.debate_stopping.cost_per_turn,
                "threshold": self.debate_stopping.threshold
            },
            "budget_optimization": {
                "enabled": True
            },
            "meta_policy": {
                "q_table_size": len(self.meta_policy.Q_table)
            }
        }


# Utility functions for practical usage
def create_reasoning_steps(num_steps: int = 5) -> List[Dict[str, float]]:
    """
    Create dummy reasoning steps for testing.
    In practice, these would be actual reasoning operations.
    """
    steps = []
    for i in range(num_steps):
        steps.append({
            'name': f'step_{i}',
            'cost': 1.0 + np.random.randn() * 0.5,
            'value': 0.8 + np.random.randn() * 0.1
        })
    return steps


def estimate_confidence_gain(current_confidence: float, history: List[float]) -> float:
    """
    Estimate expected confidence gain from one more reasoning step.
    Based on historical improvement patterns.
    """
    if len(history) < 2:
        return 0.02  # Default small gain
    
    # Calculate average recent improvement
    recent_gains = [history[i] - history[i-1] for i in range(1, len(history))]
    avg_gain = sum(recent_gains[-3:]) / min(3, len(recent_gains))
    return max(0.01, avg_gain)  # At least 1% gain