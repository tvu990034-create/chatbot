"""
Calibration Metrics - Simple Diagnostic Tools
Implements lightweight calibration diagnostics without complex overhead
"""

import numpy as np
from typing import List, Tuple, Dict


class CalibrationMetrics:
    """Lightweight calibration diagnostics for confidence assessment."""
    
    def __init__(self):
        self.confidence_history = []
        self.correctness_history = []
    
    def add_prediction(self, confidence: float, is_correct: bool):
        """Add a prediction with confidence and correctness."""
        self.confidence_history.append(confidence)
        self.correctness_history.append(1 if is_correct else 0)
    
    def mutual_information(self, bins: int = 10) -> float:
        """
        Compute mutual information between confidence and correctness.
        Measures how informative confidence is about correctness.
        
        Higher MI = confidence is more informative
        """
        if len(self.confidence_history) < 10:
            return 0.0
        
        conf = np.array(self.confidence_history)
        correct = np.array(self.correctness_history)
        
        # Discretize confidence into bins
        c_binned = np.digitize(conf, np.linspace(0, 1, bins + 1))
        
        # Compute joint distribution
        joint_counts = np.zeros((bins, 2))
        for c_bin, corr in zip(c_binned, correct):
            if 1 <= c_bin <= bins:
                joint_counts[c_bin - 1, corr] += 1
        
        joint = joint_counts / joint_counts.sum()
        
        # Compute marginal distributions
        p_c = joint.sum(axis=1)
        p_a = joint.sum(axis=0)
        
        # Compute mutual information
        mi = 0.0
        for i in range(bins):
            for j in range(2):
                if joint[i, j] > 0 and p_c[i] > 0 and p_a[j] > 0:
                    mi += joint[i, j] * np.log2(joint[i, j] / (p_c[i] * p_a[j]))
        
        return mi
    
    def conditional_entropy(self, bins: int = 10) -> float:
        """
        Compute conditional entropy H(A|C) - uncertainty about correctness given confidence.
        Lower values = confidence is more predictive of correctness.
        """
        if len(self.confidence_history) < 10:
            return 0.0
        
        conf = np.array(self.confidence_history)
        correct = np.array(self.correctness_history)
        
        c_binned = np.digitize(conf, np.linspace(0, 1, bins + 1))
        
        H = 0.0
        for b in range(1, bins + 1):
            mask = c_binned == b
            if mask.sum() == 0:
                continue
            
            p = correct[mask].mean()
            if p > 0 and p < 1:
                h = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
                H += (mask.sum() / len(conf)) * h
        
        return H
    
    def ideal_conditional_entropy(self, bins: int = 10) -> float:
        """
        Compute ideal conditional entropy for perfectly calibrated model.
        Used as a baseline for comparison.
        """
        if len(self.confidence_history) < 10:
            return 0.0
        
        conf = np.array(self.confidence_history)
        c_binned = np.digitize(conf, np.linspace(0, 1, bins + 1))
        
        H = 0.0
        for b in range(1, bins + 1):
            mask = c_binned == b
            if mask.sum() == 0:
                continue
            
            # Use bin center as ideal confidence
            c = (b - 0.5) / bins
            if c > 0 and c < 1:
                h = -c * np.log2(c) - (1 - c) * np.log2(1 - c)
                H += (mask.sum() / len(conf)) * h
        
        return H
    
    def calibration_score(self, bins: int = 10) -> float:
        """
        Compute a simple calibration score based on conditional entropy difference.
        Lower score = better calibration.
        """
        empirical_ce = self.conditional_entropy(bins)
        ideal_ce = self.ideal_conditional_entropy(bins)
        return abs(empirical_ce - ideal_ce)
    
    def confidence_stats(self) -> Dict:
        """Get basic confidence statistics."""
        if not self.confidence_history:
            return {}
        
        conf = np.array(self.confidence_history)
        correct = np.array(self.correctness_history)
        
        return {
            "num_predictions": len(conf),
            "mean_confidence": float(conf.mean()),
            "std_confidence": float(conf.std()),
            "accuracy": float(correct.mean()),
            "confidence_accuracy_correlation": float(np.corrcoef(conf, correct)[0, 1]) if len(conf) > 1 else 0.0
        }
    
    def reset(self):
        """Reset prediction history."""
        self.confidence_history = []
        self.correctness_history = []
    
    def get_diagnostic_report(self) -> str:
        """Generate a human-readable diagnostic report."""
        stats = self.confidence_stats()
        mi = self.mutual_information()
        ce = self.conditional_entropy()
        ideal_ce = self.ideal_conditional_entropy()
        cal_score = self.calibration_score()
        
        report = f"""
Calibration Diagnostic Report
============================
Predictions: {stats.get('num_predictions', 0)}
Accuracy: {stats.get('accuracy', 0):.1%}
Mean Confidence: {stats.get('mean_confidence', 0):.2f}
Confidence-Accuracy Correlation: {stats.get('confidence_accuracy_correlation', 0):.3f}

Mutual Information: {mi:.4f} nats
  (Higher = confidence is more informative)

Conditional Entropy: {ce:.4f} bits
Ideal Conditional Entropy: {ideal_ce:.4f} bits
Calibration Score: {cal_score:.4f}
  (Lower = better calibration)

Assessment:
"""
        if cal_score < 0.1:
            report += "  EXCELLENT - Confidence is well-calibrated"
        elif cal_score < 0.2:
            report += "  GOOD - Confidence is reasonably calibrated"
        elif cal_score < 0.3:
            report += "  FAIR - Some miscalibration detected"
        else:
            report += "  POOR - Significant miscalibration detected"
        
        if mi > 0.1:
            report += "\n  Confidence is informative about correctness"
        else:
            report += "\n  Confidence has low informativeness"
        
        return report


# Simple utility for confidence estimation from response quality
def estimate_confidence_from_response(response: str, query_type: str = "general") -> float:
    """
    Estimate confidence based on response characteristics.
    Lightweight heuristic without complex analysis.
    
    Args:
        response: Model response text
        query_type: Type of query (general, factual, creative, reasoning)
    
    Returns:
        Estimated confidence score [0, 1]
    """
    if not response:
        return 0.0
    
    response_lower = response.lower()
    
    # Uncertainty indicators
    uncertainty_terms = [
        "i'm not sure", "i don't know", "uncertain", "possibly", "might be",
        "probably", "maybe", "unclear", "ambiguous", "depends"
    ]
    
    # Confidence indicators
    confidence_terms = [
        "definitely", "certainly", "absolutely", "clearly", "precisely",
        "exactly", "without doubt", "conclusively"
    ]
    
    uncertainty_count = sum(1 for term in uncertainty_terms if term in response_lower)
    confidence_count = sum(1 for term in confidence_terms if term in response_lower)
    
    # Base confidence by query type
    base_confidence = {
        "factual": 0.8,
        "general": 0.7,
        "creative": 0.6,
        "reasoning": 0.65
    }.get(query_type, 0.7)
    
    # Adjust based on indicators
    adjustment = (confidence_count - uncertainty_count) * 0.1
    confidence = max(0.1, min(0.95, base_confidence + adjustment))
    
    return confidence