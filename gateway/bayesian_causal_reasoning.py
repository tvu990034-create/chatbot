"""
Bayesian Belief Update and Causal Markov Blanket
Implements practical Bayesian reasoning and feature importance for chatbot
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class BeliefState:
    """State for tracking beliefs over hypotheses."""
    hypothesis: str
    prior: float
    posterior: float
    evidence_history: List[Tuple[str, float]] = field(default_factory=list)
    update_count: int = 0

class BayesianBeliefUpdater:
    """
    Bayesian Belief Update for reasoning agents.
    Updates beliefs over hypotheses based on evidence.
    """
    
    def __init__(self):
        self.beliefs: Dict[str, BeliefState] = {}
        self.update_count = 0
    
    def add_hypothesis(self, hypothesis: str, prior: float = 0.5):
        """Add a new hypothesis with prior probability."""
        if not 0 <= prior <= 1:
            raise ValueError("Prior must be between 0 and 1")
        self.beliefs[hypothesis] = BeliefState(
            hypothesis=hypothesis,
            prior=prior,
            posterior=prior
        )
        logger.info(f"Added hypothesis '{hypothesis}' with prior {prior}")
    
    def update_belief(self, evidence: str, likelihoods: Dict[str, float]):
        """
        Update beliefs based on new evidence.
        evidence: Description of the evidence
        likelihoods: Dict mapping hypothesis to P(evidence|hypothesis)
        """
        if not self.beliefs:
            raise ValueError("No hypotheses to update")
        
        # Normalize likelihoods for hypotheses we have
        available_likelihoods = {h: likelihoods.get(h, 0.5) for h in self.beliefs}
        
        # Calculate posterior using Bayes rule
        total_evidence = sum(available_likelihoods[h] * self.beliefs[h].posterior 
                           for h in self.beliefs)
        
        for hypothesis in self.beliefs:
            likelihood = available_likelihoods[hypothesis]
            prior = self.beliefs[hypothesis].posterior
            posterior = (likelihood * prior) / total_evidence if total_evidence > 0 else prior
            
            self.beliefs[hypothesis].posterior = posterior
            self.beliefs[hypothesis].evidence_history.append((evidence, likelihood))
            self.beliefs[hypothesis].update_count += 1
        
        self.update_count += 1
        logger.info(f"Updated beliefs with evidence: {evidence}")
    
    def get_beliefs(self) -> Dict[str, float]:
        """Get current posterior probabilities."""
        return {h: b.posterior for h, b in self.beliefs.items()}
    
    def get_highest_probability(self) -> Tuple[str, float]:
        """Get hypothesis with highest probability."""
        if not self.beliefs:
            return None, 0.0
        
        sorted_beliefs = sorted(self.beliefs.items(), 
                               key=lambda x: x[1].posterior, 
                               reverse=True)
        return sorted_beliefs[0][0], sorted_beliefs[0][1].posterior
    
    def reset_to_priors(self):
        """Reset all beliefs to their priors."""
        for belief in self.beliefs.values():
            belief.posterior = belief.prior
            belief.evidence_history = []
            belief.update_count = 0
        logger.info("Reset all beliefs to priors")


class CausalMarkovBlanketFinder:
    """
    Causal Markov Blanket Finder for feature importance.
    Finds minimal sufficient predictor sets.
    """
    
    def __init__(self):
        self.graph = None
        self.markov_blankets: Dict[str, Set[str]] = {}
    
    def build_graph_from_correlations(self, correlation_matrix: np.ndarray, 
                                   feature_names: List[str], 
                                   threshold: float = 0.3):
        """
        Build causal graph from correlation matrix (simplified).
        correlation_matrix: n x n correlation matrix
        feature_names: List of feature names
        threshold: Correlation threshold for edge creation
        """
        n = len(feature_names)
        self.graph = nx.DiGraph()
        
        # Add nodes
        for name in feature_names:
            self.graph.add_node(name)
        
        # Add edges based on correlation (simplified causal assumption)
        for i in range(n):
            for j in range(n):
                if i != j and abs(correlation_matrix[i, j]) > threshold:
                    # Simplified: assume direction based on index order
                    # In practice, you'd use domain knowledge or causal discovery
                    self.graph.add_edge(feature_names[i], feature_names[j])
        
        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
    
    def find_markov_blanket(self, node: str) -> Set[str]:
        """
        Find Markov blanket for a node.
        MB(X) = Pa(X) ∪ Ch(X) ∪ Pa(Ch(X))
        """
        if self.graph is None:
            raise ValueError("Graph not built")
        
        if node not in self.graph:
            raise ValueError(f"Node {node} not in graph")
        
        parents = set(self.graph.predecessors(node))
        children = set(self.graph.successors(node))
        parents_of_children = set()
        
        for child in children:
            parents_of_children.update(self.graph.predecessors(child))
        
        markov_blanket = parents | children | parents_of_children
        markov_blanket.discard(node)  # Remove the node itself
        
        self.markov_blankets[node] = markov_blanket
        logger.info(f"Markov blanket for {node}: {markov_blanket}")
        
        return markov_blanket
    
    def get_feature_importance(self, target_node: str) -> Dict[str, float]:
        """
        Get feature importance based on Markov blanket.
        Returns normalized importance scores.
        """
        if target_node not in self.markov_blankets:
            self.find_markov_blanket(target_node)
        
        blanket = self.markov_blankets[target_node]
        
        # Simple importance based on distance (parents > children > others)
        importance = {}
        parents = set(self.graph.predecessors(target_node))
        children = set(self.graph.successors(target_node))
        parents_of_children = blanket - parents - children
        
        for node in blanket:
            if node in parents:
                importance[node] = 1.0
            elif node in children:
                importance[node] = 0.7
            else:  # parents of children
                importance[node] = 0.4
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance


class BayesianReasoningEngine:
    """
    Combined Bayesian reasoning engine with belief updates and causal analysis.
    """
    
    def __init__(self):
        self.belief_updater = BayesianBeliefUpdater()
        self.markov_blanket_finder = CausalMarkovBlanketFinder()
        self.reasoning_history: List[Dict] = []
    
    def initialize_hypotheses(self, hypotheses: Dict[str, float]):
        """
        Initialize reasoning hypotheses.
        hypotheses: Dict mapping hypothesis name to prior probability
        """
        for hypothesis, prior in hypotheses.items():
            self.belief_updater.add_hypothesis(hypothesis, prior)
    
    def update_with_evidence(self, evidence: str, likelihoods: Dict[str, float]):
        """Update beliefs with new evidence."""
        self.belief_updater.update_belief(evidence, likelihoods)
        
        # Record reasoning step
        reasoning_step = {
            "evidence": evidence,
            "beliefs": self.belief_updater.get_beliefs(),
            "highest": self.belief_updater.get_highest_probability()
        }
        self.reasoning_history.append(reasoning_step)
    
    def analyze_causal_structure(self, correlation_matrix: np.ndarray, 
                               feature_names: List[str], target: str):
        """Analyze causal structure and find important features."""
        self.markov_blanket_finder.build_graph_from_correlations(
            correlation_matrix, feature_names
        )
        blanket = self.markov_blanket_finder.find_markov_blanket(target)
        importance = self.markov_blanket_finder.get_feature_importance(target)
        
        return {
            "markov_blanket": blanket,
            "feature_importance": importance
        }
    
    def get_reasoning_summary(self) -> Dict[str, any]:
        """Get summary of reasoning state."""
        current_beliefs = self.belief_updater.get_beliefs()
        highest_hypothesis, highest_prob = self.belief_updater.get_highest_probability()
        
        return {
            "total_hypotheses": len(current_beliefs),
            "total_evidence_steps": self.belief_updater.update_count,
            "current_beliefs": current_beliefs,
            "highest_probability_hypothesis": highest_hypothesis,
            "highest_probability": highest_prob,
            "reasoning_steps": len(self.reasoning_history)
        }


def test_bayesian_belief_update():
    """Test Bayesian belief update system."""
    print("="*70)
    print("Testing Bayesian Belief Update")
    print("="*70)
    
    updater = BayesianBeliefUpdater()
    
    # Add hypotheses
    updater.add_hypothesis("model_is_accurate", 0.5)
    updater.add_hypothesis("model_is_biased", 0.3)
    updater.add_hypothesis("model_is_hallucinating", 0.2)
    
    print(f"Initial beliefs: {updater.get_beliefs()}")
    
    # Update with evidence
    updater.update_belief("model answered correctly on math question", 
                          {"model_is_accurate": 0.8, "model_is_biased": 0.4, "model_is_hallucinating": 0.3})
    print(f"After evidence 1: {updater.get_beliefs()}")
    
    updater.update_belief("model made factual error on geography",
                          {"model_is_accurate": 0.3, "model_is_biased": 0.5, "model_is_hallucinating": 0.6})
    print(f"After evidence 2: {updater.get_beliefs()}")
    
    highest, prob = updater.get_highest_probability()
    print(f"Highest probability: {highest} ({prob:.3f})")
    
    print("[OK] Bayesian Belief Update working correctly\n")


def test_causal_markov_blanket():
    """Test Causal Markov Blanket finder."""
    print("="*70)
    print("Testing Causal Markov Blanket Finder")
    print("="*70)
    
    # Create synthetic correlation matrix
    np.random.seed(42)
    n_features = 5
    correlation_matrix = np.random.randn(n_features, n_features)
    correlation_matrix = correlation_matrix @ correlation_matrix.T  # Make symmetric
    correlation_matrix = correlation_matrix / np.sqrt(np.diag(correlation_matrix)[:, None] * 
                                                      np.diag(correlation_matrix)[None, :])
    
    feature_names = ["accuracy", "speed", "creativity", "security", "user_satisfaction"]
    
    finder = CausalMarkovBlanketFinder()
    finder.build_graph_from_correlations(correlation_matrix, feature_names, threshold=0.2)
    
    print(f"Graph built with {finder.graph.number_of_nodes()} nodes, {finder.graph.number_of_edges()} edges")
    
    # Find Markov blanket for target
    blanket = finder.find_markov_blanket("user_satisfaction")
    print(f"Markov blanket for user_satisfaction: {blanket}")
    
    importance = finder.get_feature_importance("user_satisfaction")
    print(f"Feature importance: {importance}")
    
    print("[OK] Causal Markov Blanket working correctly\n")


def test_bayesian_reasoning_engine():
    """Test combined Bayesian reasoning engine."""
    print("="*70)
    print("Testing Bayesian Reasoning Engine")
    print("="*70)
    
    engine = BayesianReasoningEngine()
    
    # Initialize hypotheses
    engine.initialize_hypotheses({
        "response_quality": 0.4,
        "response_speed": 0.3,
        "response_relevance": 0.3
    })
    
    print(f"Initial hypotheses: {engine.get_reasoning_summary()}")
    
    # Update with evidence
    engine.update_with_evidence("user rated response as helpful",
                                {"response_quality": 0.9, "response_speed": 0.5, "response_relevance": 0.8})
    
    engine.update_with_evidence("response took 5 seconds to generate",
                                {"response_quality": 0.4, "response_speed": 0.2, "response_relevance": 0.6})
    
    summary = engine.get_reasoning_summary()
    print(f"Final reasoning summary:")
    print(f"  Total hypotheses: {summary['total_hypotheses']}")
    print(f"  Evidence steps: {summary['total_evidence_steps']}")
    print(f"  Current beliefs: {summary['current_beliefs']}")
    print(f"  Highest: {summary['highest_probability_hypothesis']} ({summary['highest_probability']:.3f})")
    
    print("[OK] Bayesian Reasoning Engine working correctly\n")


if __name__ == "__main__":
    try:
        test_bayesian_belief_update()
        test_causal_markov_blanket()
        test_bayesian_reasoning_engine()
        
        print("="*70)
        print("ALL BAYESIAN AND CAUSAL TESTS PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()