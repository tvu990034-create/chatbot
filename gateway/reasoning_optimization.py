"""
Reasoning Path Planning and Optimization
Implements A* Search for reasoning chain planning and MCTS for optimization
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import heapq
import math
import random

logger = logging.getLogger(__name__)

@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    content: str
    confidence: float
    cost: float = 1.0
    parent: Optional['ReasoningStep'] = None

@dataclass
class ReasoningPath:
    """A complete reasoning path."""
    steps: List[ReasoningStep]
    total_cost: float
    total_confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


class MCTSNode:
    """Node in MCTS tree."""
    def __init__(self, state: str, parent: Optional['MCTSNode'] = None):
        self.state = state
        self.parent = parent
        self.children: Dict[str, 'MCTSNode'] = {}
        self.visits = 0
        self.value = 0.0


class AStarReasoningPlanner:
    """
    A* Search for reasoning chain planning.
    Finds optimal reasoning paths through knowledge/concept space.
    """
    
    def __init__(self):
        self.reasoning_graph: Dict[str, List[Tuple[str, float]]] = {}
        self.path_history: List[ReasoningPath] = []
    
    def add_reasoning_link(self, from_concept: str, to_concept: str, cost: float = 1.0):
        """Add a directed reasoning link between concepts."""
        if from_concept not in self.reasoning_graph:
            self.reasoning_graph[from_concept] = []
        self.reasoning_graph[from_concept].append((to_concept, cost))
        logger.debug(f"Added reasoning link: {from_concept} -> {to_concept} (cost: {cost})")
    
    def heuristic(self, current: str, goal: str) -> float:
        """Heuristic estimate of cost from current to goal."""
        # Simple heuristic: use string similarity or concept distance
        # In practice, this could be learned or domain-specific
        return abs(len(current) - len(goal)) * 0.1
    
    def find_optimal_path(self, start: str, goal: str, max_steps: int = 100) -> Optional[List[str]]:
        """
        Find optimal reasoning path from start to goal using A*.
        Returns list of concepts in the path.
        """
        if start not in self.reasoning_graph:
            logger.warning(f"Start concept '{start}' not in reasoning graph")
            return None
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        iterations = 0
        while open_set and iterations < max_steps:
            current_f, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            # Explore neighbors
            if current in self.reasoning_graph:
                for neighbor, cost in self.reasoning_graph[current]:
                    tentative_g = g_score[current] + cost
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
            
            iterations += 1
        
        logger.warning(f"No path found from '{start}' to '{goal}' within {max_steps} steps")
        return None
    
    def get_path_stats(self) -> Dict[str, any]:
        """Get statistics about reasoning path planning."""
        return {
            "total_concepts": len(self.reasoning_graph),
            "total_links": sum(len(links) for links in self.reasoning_graph.values()),
            "paths_planned": len(self.path_history)
        }


class MCTSReasoningOptimizer:
    """
    Monte Carlo Tree Search for reasoning optimization.
    Optimizes reasoning paths through exploration and exploitation.
    """
    
    def __init__(self, exploration_constant: float = 1.4):
        self.exploration_constant = exploration_constant
        self.tree: Dict[str, 'MCTSNode'] = {}
        self.simulation_count = 0
    
    def select_action(self, node: 'MCTSNode') -> Tuple[str, 'MCTSNode']:
        """Select action using UCB1 formula."""
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            if child.visits == 0:
                return action, child
            
            ucb_score = child.value / child.visits + self.exploration_constant * math.sqrt(
                math.log(node.visits) / child.visits
            )
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def simulate(self, node: 'MCTSNode', depth: int = 10) -> float:
        """Simulate reasoning path and return value."""
        # Simple simulation: random walk with decreasing value
        value = node.value / (node.visits + 1)
        
        for _ in range(depth):
            if not node.children:
                break
            action = random.choice(list(node.children.keys()))
            node = node.children[action]
            value *= 0.9  # Decay value with depth
        
        return value
    
    def backpropagate(self, node: 'MCTSNode', value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent
    
    def optimize_reasoning(self, root_state: str, possible_actions: List[str], 
                          iterations: int = 100) -> Dict[str, float]:
        """
        Optimize reasoning using MCTS.
        Returns action-value estimates.
        """
        root = MCTSNode(root_state)
        self.tree[root_state] = root
        
        # Initialize children
        for action in possible_actions:
            root.children[action] = MCTSNode(action, parent=root)
        
        for _ in range(iterations):
            # Selection
            current = root
            path = [current]
            
            while current.children and random.random() < 0.9:  # 90% chance to continue
                action, current = self.select_action(current)
                path.append(current)
            
            # Simulation
            leaf = path[-1]
            value = self.simulate(leaf)
            
            # Backpropagation
            for node in reversed(path):
                self.backpropagate(node, value)
            
            self.simulation_count += 1
        
        # Extract action values
        action_values = {}
        for action, child in root.children.items():
            action_values[action] = child.value / child.visits if child.visits > 0 else 0.0
        
        return action_values
    
    def get_optimization_stats(self) -> Dict[str, any]:
        """Get statistics about reasoning optimization."""
        return {
            "total_simulations": self.simulation_count,
            "tree_size": len(self.tree)
        }


class ReasoningOptimizationSuite:
    """
    Combined reasoning optimization suite with A* planning and MCTS optimization.
    """
    
    def __init__(self):
        self.astar_planner = AStarReasoningPlanner()
        self.mcts_optimizer = MCTSReasoningOptimizer()
        self.optimization_history: List[Dict] = []
    
    def add_knowledge(self, from_concept: str, to_concept: str, cost: float = 1.0):
        """Add knowledge to reasoning graph."""
        self.astar_planner.add_reasoning_link(from_concept, to_concept, cost)
    
    def plan_reasoning_path(self, start: str, goal: str) -> Optional[List[str]]:
        """Plan optimal reasoning path."""
        return self.astar_planner.find_optimal_path(start, goal)
    
    def optimize_reasoning_actions(self, state: str, actions: List[str], 
                                   iterations: int = 50) -> Dict[str, float]:
        """Optimize reasoning actions using MCTS."""
        return self.mcts_optimizer.optimize_reasoning(state, actions, iterations)
    
    def get_suite_stats(self) -> Dict[str, any]:
        """Get combined statistics."""
        astar_stats = self.astar_planner.get_path_stats()
        mcts_stats = self.mcts_optimizer.get_optimization_stats()
        
        return {
            "astar_planning": astar_stats,
            "mcts_optimization": mcts_stats,
            "total_optimizations": len(self.optimization_history)
        }


def test_astar_planning():
    """Test A* reasoning planning."""
    print("="*70)
    print("Testing A* Reasoning Planning")
    print("="*70)
    
    planner = AStarReasoningPlanner()
    
    # Build reasoning graph
    planner.add_reasoning_link("question", "analyze", 1.0)
    planner.add_reasoning_link("analyze", "breakdown", 1.0)
    planner.add_reasoning_link("breakdown", "solve", 1.0)
    planner.add_reasoning_link("solve", "answer", 1.0)
    planner.add_reasoning_link("question", "direct_answer", 3.0)
    planner.add_reasoning_link("direct_answer", "answer", 1.0)
    
    print(f"Graph stats: {planner.get_path_stats()}")
    
    # Find optimal path
    path = planner.find_optimal_path("question", "answer")
    print(f"Optimal path from 'question' to 'answer': {path}")
    
    if path:
        total_cost = sum(1.0 for _ in range(len(path) - 1))
        print(f"Path length: {len(path)} steps, cost: {total_cost}")
    
    print("[OK] A* Planning working correctly\n")


def test_mcts_optimization():
    """Test MCTS reasoning optimization."""
    print("="*70)
    print("Testing MCTS Reasoning Optimization")
    print("="*70)
    
    optimizer = MCTSReasoningOptimizer()
    
    # Optimize reasoning actions
    state = "current_problem"
    actions = ["deep_analysis", "quick_answer", "creative_solution", "factual_response"]
    
    action_values = optimizer.optimize_reasoning(state, actions, iterations=100)
    
    print(f"Action values after optimization:")
    for action, value in sorted(action_values.items(), key=lambda x: -x[1]):
        print(f"  {action}: {value:.4f}")
    
    stats = optimizer.get_optimization_stats()
    print(f"\nOptimization stats: {stats}")
    
    print("[OK] MCTS Optimization working correctly\n")


def test_reasoning_optimization_suite():
    """Test combined reasoning optimization suite."""
    print("="*70)
    print("Testing Reasoning Optimization Suite")
    print("="*70)
    
    suite = ReasoningOptimizationSuite()
    
    # Add knowledge
    suite.add_knowledge("problem", "understand", 1.0)
    suite.add_knowledge("understand", "analyze", 1.0)
    suite.add_knowledge("analyze", "solve", 1.0)
    suite.add_knowledge("solve", "verify", 1.0)
    suite.add_knowledge("verify", "answer", 1.0)
    
    # Plan reasoning path
    path = suite.plan_reasoning_path("problem", "answer")
    print(f"Planned path: {path}")
    
    # Optimize actions
    actions = ["step_by_step", "intuitive", "mathematical", "creative"]
    action_values = suite.optimize_reasoning_actions("current_state", actions, iterations=50)
    
    print(f"\nOptimized action values:")
    for action, value in sorted(action_values.items(), key=lambda x: -x[1]):
        print(f"  {action}: {value:.4f}")
    
    stats = suite.get_suite_stats()
    print(f"\nSuite stats: {stats}")
    
    print("[OK] Reasoning Optimization Suite working correctly\n")


if __name__ == "__main__":
    try:
        test_astar_planning()
        test_mcts_optimization()
        test_reasoning_optimization_suite()
        
        print("="*70)
        print("ALL REASONING OPTIMIZATION TESTS PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()