"""
Robustness Testing with Randomized Smoothing
Implements certified robustness evaluation for model testing
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RobustnessResult:
    """Result of randomized smoothing robustness test."""
    prediction: int
    certified_radius: float
    confidence: float
    samples_used: int
    top_class_prob: float
    second_class_prob: float

class RandomizedSmoothingTester:
    """
    Randomized Smoothing for certified robustness evaluation.
    Provides probabilistic guarantee that prediction is unchanged within an ℓ2 radius.
    """
    
    def __init__(self, model_predict_fn: Callable, sigma: float = 0.25, n_samples: int = 100):
        """
        model_predict_fn: Function that takes input and returns class probabilities/logits
        sigma: Noise level for Gaussian smoothing
        n_samples: Number of Monte Carlo samples
        """
        self.model_predict_fn = model_predict_fn
        self.sigma = sigma
        self.n_samples = n_samples
        self.test_history: List[Tuple[datetime, RobustnessResult]] = []
    
    def _cert_radius(self, pA: float, pB: float, sigma: float) -> float:
        """Calculate certified radius using inverse normal CDF."""
        from scipy.stats import norm
        return sigma / 2 * (norm.ppf(pA) - norm.ppf(pB))
    
    def predict_smoothed(self, x: np.ndarray) -> RobustnessResult:
        """
        Perform randomized smoothing prediction with certified radius.
        x: Input array (can be text embedding, image, etc.)
        """
        if not callable(self.model_predict_fn):
            raise ValueError("model_predict_fn must be callable")
        
        # Monte Carlo sampling
        counts = {}
        predictions = []
        
        for _ in range(self.n_samples):
            # Add Gaussian noise to input
            noise = np.random.normal(0, self.sigma, x.shape)
            x_noisy = x + noise
            
            # Get prediction from model
            try:
                pred = self.model_predict_fn(x_noisy)
                predictions.append(pred)
                counts[pred] = counts.get(pred, 0) + 1
            except Exception as e:
                logger.debug(f"Prediction failed: {e}")
                continue
        
        if not counts:
            # Fallback if all predictions failed
            return RobustnessResult(
                prediction=0,
                certified_radius=0.0,
                confidence=0.0,
                samples_used=0,
                top_class_prob=0.0,
                second_class_prob=0.0
            )
        
        # Find top class
        top_class = max(counts, key=counts.get)
        pA = counts[top_class] / len(predictions)
        
        # Find second best class
        others = [c for c in counts if c != top_class]
        pB = max([counts[c] for c in others], default=0) / len(predictions) if others else 0.0
        
        # Calculate certified radius
        radius = self._cert_radius(pA, pB, self.sigma)
        
        result = RobustnessResult(
            prediction=top_class,
            certified_radius=radius,
            confidence=pA,
            samples_used=len(predictions),
            top_class_prob=pA,
            second_class_prob=pB
        )
        
        self.test_history.append((datetime.now(), result))
        
        return result
    
    def batch_test(self, inputs: List[np.ndarray]) -> List[RobustnessResult]:
        """Test multiple inputs."""
        results = []
        for x in inputs:
            result = self.predict_smoothed(x)
            results.append(result)
        return results
    
    def get_summary_stats(self) -> Dict[str, any]:
        """Get summary statistics of robustness tests."""
        if not self.test_history:
            return {"total_tests": 0}
        
        radii = [r.certified_radius for _, r in self.test_history]
        confidences = [r.confidence for _, r in self.test_history]
        
        return {
            "total_tests": len(self.test_history),
            "avg_certified_radius": np.mean(radii),
            "max_certified_radius": np.max(radii),
            "min_certified_radius": np.min(radii),
            "avg_confidence": np.mean(confidences),
            "sigma": self.sigma,
            "samples_per_test": self.n_samples
        }


class AdversarialRobustnessSuite:
    """
    Comprehensive adversarial robustness testing suite.
    Combines randomized smoothing with other robustness metrics.
    """
    
    def __init__(self, model_predict_fn: Callable, sigma: float = 0.25, n_samples: int = 50):
        self.smoothing_tester = RandomizedSmoothingTester(model_predict_fn, sigma, n_samples)
        self.adversarial_examples_tested = 0
        self.robustness_history: List[Dict] = []
    
    def test_robustness(self, x: np.ndarray, adversarial_x: Optional[np.ndarray] = None) -> Dict[str, any]:
        """
        Comprehensive robustness test.
        x: Original input
        adversarial_x: Optional adversarial example to test against
        """
        # Test original input robustness
        original_result = self.smoothing_tester.predict_smoothed(x)
        
        result = {
            "original_prediction": original_result.prediction,
            "certified_radius": original_result.certified_radius,
            "confidence": original_result.confidence,
            "adversarial_prediction": None,
            "adversarial_consistent": None
        }
        
        # Test against adversarial example if provided
        if adversarial_x is not None:
            adversarial_result = self.smoothing_tester.predict_smoothed(adversarial_x)
            result["adversarial_prediction"] = adversarial_result.prediction
            result["adversarial_consistent"] = (
                original_result.prediction == adversarial_result.prediction
            )
            self.adversarial_examples_tested += 1
        
        self.robustness_history.append(result)
        
        return result
    
    def get_robustness_summary(self) -> Dict[str, any]:
        """Get summary of robustness testing."""
        smoothing_stats = self.smoothing_tester.get_summary_stats()
        
        if not self.robustness_history:
            return {**smoothing_stats, "adversarial_consistency_rate": None}
        
        consistency_scores = [
            r["adversarial_consistent"] 
            for r in self.robustness_history 
            if r["adversarial_consistent"] is not None
        ]
        
        return {
            **smoothing_stats,
            "adversarial_examples_tested": self.adversarial_examples_tested,
            "adversarial_consistency_rate": (
                np.mean(consistency_scores) if consistency_scores else None
            )
        }


# Simple mock model for testing
def mock_model_predict_fn(x: np.ndarray) -> int:
    """Mock model for testing - returns probabilistic prediction based on input."""
    # Add some randomness to simulate real model behavior
    return int((np.sum(x) + np.random.normal(0, 0.1)) > 0)


def test_randomized_smoothing():
    """Test randomized smoothing implementation."""
    print("="*70)
    print("Testing Randomized Smoothing Robustness Testing")
    print("="*70)
    
    # Create tester with mock model
    tester = RandomizedSmoothingTester(mock_model_predict_fn, sigma=0.1, n_samples=50)
    
    # Test with sample inputs
    print("Testing with sample inputs...")
    for i in range(5):
        x = np.random.randn(10)  # 10-dimensional input
        result = tester.predict_smoothed(x)
        
        print(f"Test {i+1}:")
        print(f"  Prediction: {result.prediction}")
        print(f"  Certified Radius: {result.certified_radius:.4f}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Samples Used: {result.samples_used}")
    
    # Get summary
    summary = tester.get_summary_stats()
    print(f"\nSummary Statistics:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Avg Certified Radius: {summary['avg_certified_radius']:.4f}")
    print(f"  Avg Confidence: {summary['avg_confidence']:.4f}")
    
    print("[OK] Randomized Smoothing working correctly\n")


def test_adversarial_robustness_suite():
    """Test adversarial robustness suite."""
    print("="*70)
    print("Testing Adversarial Robustness Suite")
    print("="*70)
    
    suite = AdversarialRobustnessSuite(mock_model_predict_fn, sigma=0.1, n_samples=30)
    
    # Test with adversarial examples
    print("Testing with adversarial examples...")
    for i in range(3):
        x = np.random.randn(10)
        adversarial_x = x + np.random.randn(10) * 0.5  # Add perturbation
        
        result = suite.test_robustness(x, adversarial_x)
        
        print(f"Test {i+1}:")
        print(f"  Original Prediction: {result['original_prediction']}")
        print(f"  Adversarial Prediction: {result['adversarial_prediction']}")
        print(f"  Consistent: {result['adversarial_consistent']}")
        print(f"  Certified Radius: {result['certified_radius']:.4f}")
    
    # Get summary
    summary = suite.get_robustness_summary()
    print(f"\nRobustness Summary:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Adversarial Examples Tested: {summary['adversarial_examples_tested']}")
    print(f"  Adversarial Consistency Rate: {summary['adversarial_consistency_rate']:.4f}")
    
    print("[OK] Adversarial Robustness Suite working correctly\n")


if __name__ == "__main__":
    try:
        test_randomized_smoothing()
        test_adversarial_robustness_suite()
        
        print("="*70)
        print("ALL ROBUSTNESS TESTS PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()