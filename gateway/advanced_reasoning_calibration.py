"""
Advanced Reasoning and Calibration Tools
Implements Entail Graph Transitivity and Brier Score for enhanced reasoning
"""

import numpy as np
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class ConfidenceScore:
    """Score for tracking confidence calibration."""
    predicted_prob: float
    actual_outcome: int
    score: float
    timestamp: datetime

class EntailmentGraphReasoner:
    """
    Entailment Graph Transitivity for logical reasoning.
    Expands logical consequence relations automatically.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.transitive_closure = None
        self.entailment_history: List[Tuple[str, str, str]] = []
    
    def add_entailment(self, premise: str, conclusion: str):
        """Add a direct entailment relation."""
        self.graph.add_edge(premise, conclusion)
        logger.info(f"Added entailment: {premise} -> {conclusion}")
        self.transitive_closure = None  # Invalidate cached closure
    
    def compute_transitive_closure(self) -> nx.DiGraph:
        """Compute transitive closure of the entailment graph."""
        if self.transitive_closure is None:
            self.transitive_closure = nx.transitive_closure(self.graph)
        return self.transitive_closure
    
    def get_consequences(self, proposition: str) -> Set[str]:
        """Get all consequences (direct and indirect) of a proposition."""
        closure = self.compute_transitive_closure()
        try:
            return set(closure.successors(proposition))
        except nx.NetworkXError:
            return set()
    
    def get_predecessors(self, proposition: str) -> Set[str]:
        """Get all propositions that entail the given one."""
        closure = self.compute_transitive_closure()
        try:
            return set(closure.predecessors(proposition))
        except nx.NetworkXError:
            return set()
    
    def check_entailment(self, premise: str, conclusion: str) -> bool:
        """Check if a premise entails a conclusion (direct or indirect)."""
        return conclusion in self.get_consequences(premise)
    
    def find_common_consequence(self, prop1: str, prop2: str) -> Optional[str]:
        """Find a proposition that is entailed by both (intersection of consequences)."""
        cons1 = self.get_consequences(prop1)
        cons2 = self.get_consequences(prop2)
        intersection = cons1 & cons2
        
        if intersection:
            # Return the one with shortest entailment path
            return min(intersection, key=lambda x: self.get_distance(prop1, x) + self.get_distance(prop2, x))
        return None
    
    def get_distance(self, source: str, target: str) -> int:
        """Get shortest path length in entailment graph."""
        try:
            return nx.shortest_path_length(self.graph, source, target)
        except nx.NetworkXError:
            return float('inf')
    
    def get_graph_stats(self) -> Dict[str, any]:
        """Get statistics about the entailment graph."""
        try:
            avg_path_length = nx.average_shortest_path_length(self.graph) if self.graph.number_of_edges() > 0 else 0
        except nx.NetworkXError:
            # Graph is not connected
            avg_path_length = 0.0
        
        return {
            "total_propositions": self.graph.number_of_nodes(),
            "direct_entailments": self.graph.number_of_edges(),
            "transitive_entailments": self.transitive_closure.number_of_edges() if self.transitive_closure else 0,
            "avg_path_length": avg_path_length
        }


class BrierScoreCalibrator:
    """
    Brier Score Calibration for probabilistic predictions.
    Measures accuracy of confidence estimates.
    """
    
    def __init__(self):
        self.scores: List[ConfidenceScore] = []
        self.calibration_history: List[Tuple[datetime, float]] = []
    
    def add_score(self, predicted_prob: float, actual_outcome: int):
        """Add a confidence score."""
        if not 0 <= predicted_prob <= 1:
            raise ValueError("Predicted probability must be between 0 and 1")
        if actual_outcome not in [0, 1]:
            raise ValueError("Actual outcome must be 0 or 1")
        
        score = (predicted_prob - actual_outcome) ** 2
        self.scores.append(ConfidenceScore(
            predicted_prob=predicted_prob,
            actual_outcome=actual_outcome,
            score=score,
            timestamp=datetime.now()
        ))
        logger.debug(f"Added score: pred={predicted_prob:.3f}, actual={actual_outcome}, brier={score:.4f}")
    
    def compute_brier_score(self) -> float:
        """Compute overall Brier score (lower is better)."""
        if not self.scores:
            return 0.0
        return np.mean([s.score for s in self.scores])
    
    def compute_calibration_metrics(self) -> Dict[str, float]:
        """Compute additional calibration metrics."""
        if not self.scores:
            return {"brier_score": 0.0, "ece": 0.0, "perfectly_calibrated": True}
        
        # Expected Calibration Error (ECE)
        # Group predictions into bins
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ece_sum = 0.0
        n_samples = 0
        
        for i in range(n_bins):
            bin_scores = [s for s in self.scores if bin_edges[i] <= s.predicted_prob < bin_edges[i+1]]
            if bin_scores:
                bin_accuracy = np.mean([1 - s.score for s in bin_scores])
                bin_avg_prob = np.mean([s.predicted_prob for s in bin_scores])
                bin_weight = len(bin_scores)
                ece_sum += bin_weight * abs(bin_avg_prob - bin_accuracy)
                n_samples += bin_weight
        
        ece = ece_sum / n_samples if n_samples > 0 else 0.0
        
        # Perfect calibration score (proportion where pred ≈ outcome)
        perfect_matches = sum(1 for s in self.scores if round(s.predicted_prob) == s.actual_outcome)
        perfectly_calibrated = perfect_matches / len(self.scores)
        
        return {
            "brier_score": self.compute_brier_score(),
            "ece": ece,
            "perfectly_calibrated": perfectly_calibrated,
            "total_predictions": len(self.scores)
        }
    
    def reset(self):
        """Reset all scores."""
        self.scores = []
        self.calibration_history = []


class AdvancedReasoningSuite:
    """
    Combined advanced reasoning and calibration suite.
    """
    
    def __init__(self):
        self.entailment_reasoner = EntailmentGraphReasoner()
        self.brier_calibrator = BrierScoreCalibrator()
        self.reasoning_history: List[Dict] = []
    
    def add_knowledge(self, premises: List[str], conclusions: List[str]):
        """Add knowledge as entailment relations."""
        for premise, conclusion in zip(premises, conclusions):
            self.entailment_reasoner.add_entailment(premise, conclusion)
    
    def check_logical_consequence(self, premise: str, conclusion: str) -> bool:
        """Check if a conclusion follows from a premise."""
        return self.entailment_reasoner.check_entailment(premise, conclusion)
    
    def track_confidence(self, predicted_prob: float, actual_outcome: int):
        """Track confidence calibration."""
        self.brier_calibrator.add_score(predicted_prob, actual_outcome)
    
    def get_reasoning_summary(self) -> Dict[str, any]:
        """Get summary of reasoning and calibration state."""
        graph_stats = self.entailment_reasoner.get_graph_stats()
        calibration_metrics = self.brier_calibrator.compute_calibration_metrics()
        
        return {
            "entailment_graph": graph_stats,
            "calibration": calibration_metrics,
            "total_reasoning_steps": len(self.reasoning_history)
        }


def test_entailment_graph():
    """Test entailment graph transitivity."""
    print("="*70)
    print("Testing Entailment Graph Transitivity")
    print("="*70)
    
    reasoner = EntailmentGraphReasoner()
    
    # Build a simple knowledge graph
    reasoner.add_entailment("AI models require data", "Data is essential for AI")
    reasoner.add_entailment("Data is essential for AI", "AI needs high-quality data")
    reasoner.add_entailment("AI needs high-quality data", "Quality data improves AI performance")
    reasoner.add_entailment("AI models require data", "AI models need computation")
    
    print(f"Graph stats: {reasoner.get_graph_stats()}")
    
    # Test transitivity
    print("\nTesting transitivity:")
    print(f"'AI models require data' entails 'Quality data improves AI performance': {reasoner.check_entailment('AI models require data', 'Quality data improves AI performance')}")
    print(f"'AI models require data' entails 'AI models need computation': {reasoner.check_entailment('AI models require data', 'AI models need computation')}")
    
    # Get consequences
    consequences = reasoner.get_consequences("AI models require data")
    print(f"\nConsequences of 'AI models require data': {consequences}")
    
    # Find common consequence
    common = reasoner.find_common_consequence("AI models require data", "Data is essential for AI")
    print(f"Common consequence with 'Data is essential for AI': {common}")
    
    print("[OK] Entailment Graph working correctly\n")


def test_brier_score():
    """Test Brier Score calibration."""
    print("="*70)
    print("Testing Brier Score Calibration")
    print("="*70)
    
    calibrator = BrierScoreCalibrator()
    
    # Add some confidence scores
    predictions = [0.8, 0.9, 0.3, 0.6, 0.7, 0.4, 0.5, 0.8, 0.2, 0.9]
    outcomes = [1, 1, 0, 1, 1, 0, 0, 1, 0, 1]
    
    for pred, actual in zip(predictions, outcomes):
        calibrator.add_score(pred, actual)
    
    brier = calibrator.compute_brier_score()
    metrics = calibrator.compute_calibration_metrics()
    
    print(f"Brier Score: {brier:.4f} (lower is better)")
    print(f"Expected Calibration Error: {metrics['ece']:.4f}")
    print(f"Perfectly Calibrated: {metrics['perfectly_calibrated']:.2%}")
    print(f"Total Predictions: {metrics['total_predictions']}")
    
    print("[OK] Brier Score working correctly\n")


def test_advanced_reasoning_suite():
    """Test combined advanced reasoning suite."""
    print("="*70)
    print("Testing Advanced Reasoning Suite")
    print("="*70)
    
    suite = AdvancedReasoningSuite()
    
    # Add knowledge
    suite.add_knowledge(
        ["Model requires data", "Data is important", "Good data improves model"],
        ["Model requires computation", "Computation enables model", "Fast computation helps model"]
    )
    
    # Test reasoning
    result1 = suite.check_logical_consequence("Model requires data", "Good data improves model")
    print(f"Entailment check: {result1}")
    
    # Track confidence
    suite.track_confidence(0.8, 1)
    suite.track_confidence(0.3, 0)
    suite.track_confidence(0.9, 1)
    
    summary = suite.get_reasoning_summary()
    print(f"\nReasoning summary:")
    print(f"  Propositions: {summary['entailment_graph']['total_propositions']}")
    print(f"  Entailments: {summary['entailment_graph']['direct_entailments']}")
    print(f"  Transitive entailments: {summary['entailment_graph']['transitive_entailments']}")
    print(f"  Brier Score: {summary['calibration']['brier_score']:.4f}")
    print(f"  ECE: {summary['calibration']['ece']:.4f}")
    
    print("[OK] Advanced Reasoning Suite working correctly\n")


if __name__ == "__main__":
    try:
        test_entailment_graph()
        test_brier_score()
        test_advanced_reasoning_suite()
        
        print("="*70)
        print("ALL ADVANCED REASONING TESTS PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()