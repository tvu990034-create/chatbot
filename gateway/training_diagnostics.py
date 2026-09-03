"""
Training Diagnostic Metrics - Lightweight Assessment Tools
Implements diagnostic metrics that don't require model training
"""

import numpy as np
from typing import List, Dict, Any, Optional
import math


class CalibrationScoring:
    """
    Lightweight calibration scoring metrics.
    Implements Brier score and other proper scoring rules without training.
    """
    
    def __init__(self):
        pass
    
    def brier_score(self, predicted_probs: List[float], actual_correctness: List[int]) -> float:
        """
        Calculate Brier score for probabilistic predictions.
        
        Args:
            predicted_probs: List of predicted probabilities (confidence in correct answer)
            actual_correctness: List of actual correctness (0 or 1)
        
        Returns:
            Brier score (lower is better)
        """
        if len(predicted_probs) != len(actual_correctness):
            raise ValueError("Length of predictions and labels must match")
        
        n = len(predicted_probs)
        brier = 0.0
        for i in range(n):
            c = predicted_probs[i]
            e = actual_correctness[i]
            brier += (c - e) ** 2
        
        return brier / n
    
    def expected_calibration_error(self, predicted_probs: List[float], actual_correctness: List[int], bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            predicted_probs: List of predicted probabilities
            actual_correctness: List of actual correctness (0 or 1)
            bins: Number of bins for ECE calculation
        
        Returns:
            ECE score (lower is better)
        """
        if len(predicted_probs) != len(actual_correctness):
            raise ValueError("Length of predictions and labels must match")
        
        # Bin predictions by confidence
        bin_indices = np.digitize(predicted_probs, np.linspace(0, 1, bins + 1))
        
        ece = 0.0
        total_samples = len(predicted_probs)
        
        for b in range(1, bins + 1):
            mask = bin_indices == b
            if mask.sum() == 0:
                continue
            
            # Average confidence in this bin
            avg_conf = np.array(predicted_probs)[mask].mean()
            # Average accuracy in this bin
            avg_acc = np.array(actual_correctness)[mask].mean()
            # Weight by number of samples
            weight = mask.sum() / total_samples
            
            ece += weight * abs(avg_conf - avg_acc)
        
        return ece
    
    def log_loss(self, predicted_probs: List[float], actual_correctness: List[int]) -> float:
        """
        Calculate log loss (cross-entropy) for probabilistic predictions.
        
        Args:
            predicted_probs: List of predicted probabilities
            actual_correctness: List of actual correctness (0 or 1)
        
        Returns:
            Log loss (lower is better)
        """
        if len(predicted_probs) != len(actual_correctness):
            raise ValueError("Length of predictions and labels must match")
        
        log_loss = 0.0
        for i in range(len(predicted_probs)):
            c = predicted_probs[i]
            e = actual_correctness[i]
            # Avoid log(0)
            c = max(1e-15, min(1 - 1e-15, c))
            if e == 1:
                log_loss += -math.log(c)
            else:
                log_loss += -math.log(1 - c)
        
        return log_loss / len(predicted_probs)
    
    def calibration_summary(self, predicted_probs: List[float], actual_correctness: List[int]) -> Dict[str, Any]:
        """
        Generate comprehensive calibration summary.
        
        Args:
            predicted_probs: List of predicted probabilities
            actual_correctness: List of actual correctness (0 or 1)
        
        Returns:
            Summary of calibration metrics
        """
        return {
            "brier_score": self.brier_score(predicted_probs, actual_correctness),
            "expected_calibration_error": self.expected_calibration_error(predicted_probs, actual_correctness),
            "log_loss": self.log_loss(predicted_probs, actual_correctness),
            "num_predictions": len(predicted_probs),
            "mean_confidence": np.mean(predicted_probs),
            "mean_accuracy": np.mean(actual_correctness)
        }


class ModelQualityAssessment:
    """
    Lightweight model quality assessment without training.
    Provides diagnostic insights into model performance.
    """
    
    def __init__(self):
        self.calibration_scoring = CalibrationScoring()
    
    def assess_model_quality(self, predictions: List[str], correct_answers: List[str], 
                            confidences: List[float]) -> Dict[str, Any]:
        """
        Assess model quality across multiple dimensions.
        
        Args:
            predictions: List of model predictions
            correct_answers: List of correct answers
            confidences: List of confidence scores
        
        Returns:
            Quality assessment summary
        """
        # Convert predictions to correctness
        actual_correctness = []
        for pred, correct in zip(predictions, correct_answers):
            actual_correctness.append(1 if pred.lower() == correct.lower() else 0)
        
        # Calculate calibration metrics
        calibration_summary = self.calibration_scoring.calibration_summary(confidences, actual_correctness)
        
        # Calculate accuracy
        accuracy = sum(actual_correctness) / len(actual_correctness) if actual_correctness else 0.0
        
        # Calculate confidence-accuracy correlation
        if len(confidences) > 1:
            correlation = np.corrcoef(confidences, actual_correctness)[0, 1]
        else:
            correlation = 0.0
        
        return {
            "accuracy": accuracy,
            "calibration": calibration_summary,
            "confidence_accuracy_correlation": correlation,
            "num_predictions": len(predictions)
        }
    
    def compare_configurations(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare different model configurations.
        
        Args:
            results: List of configuration results with 'accuracy' and 'calibration'
        
        Returns:
            Comparison ranking
        """
        rankings = []
        for i, result in enumerate(results):
            score = result['accuracy'] - result['calibration']['brier_score']
            rankings.append({
                'config_index': i,
                'score': score,
                'accuracy': result['accuracy'],
                'brier_score': result['calibration']['brier_score']
            })
        
        rankings.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'rankings': rankings,
            'best_config': rankings[0]['config_index'] if rankings else None
        }


class TrainingDiagnosticsSuite:
    """
    Combined training diagnostics suite.
    Provides assessment tools without requiring actual training.
    """
    
    def __init__(self):
        self.calibration_scoring = CalibrationScoring()
        self.quality_assessment = ModelQualityAssessment()
    
    def assess_calibration(self, predicted_probs: List[float], actual_correctness: List[int]) -> Dict[str, Any]:
        """Assess model calibration."""
        return self.calibration_scoring.calibration_summary(predicted_probs, actual_correctness)
    
    def assess_quality(self, predictions: List[str], correct_answers: List[str], confidences: List[float]) -> Dict[str, Any]:
        """Assess overall model quality."""
        return self.quality_assessment.assess_model_quality(predictions, correct_answers, confidences)
    
    def get_diagnostics_stats(self) -> Dict[str, Any]:
        """Return diagnostics statistics."""
        return {
            "calibration_scoring": {"enabled": True},
            "quality_assessment": {"enabled": True}
        }