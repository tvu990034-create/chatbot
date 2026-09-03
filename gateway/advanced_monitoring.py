"""
Advanced Monitoring and Tracking for Enhanced Gateway
Implements Kalman Filter, Calibration Drift, and Change-Point Detection
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class CapabilityState:
    """State for tracking model capability across evaluations."""
    true_capability: float = 0.5  # Estimated true capability
    uncertainty: float = 0.1  # Uncertainty in estimate
    update_count: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)  # (time, capability)

class EvaluationKalmanFilter:
    """
    Kalman Filter for tracking true model capability across evaluation rounds.
    Denoises benchmark scores to get better capability estimates.
    """
    
    def __init__(self, initial_capability: float = 0.5, process_noise: float = 0.01, measurement_noise: float = 0.1):
        # State: [capability]
        self.x = np.array([initial_capability])
        
        # State covariance
        self.P = np.array([[0.1]])
        
        # State transition (identity - capability doesn't change much)
        self.F = np.array([[1.0]])
        
        # Measurement matrix (we observe capability directly)
        self.H = np.array([[1.0]])
        
        # Process noise (how much capability can change)
        self.Q = np.array([[process_noise]])
        
        # Measurement noise (how noisy our benchmarks are)
        self.R = np.array([[measurement_noise]])
        
        self.update_count = 0
    
    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    
    def update(self, observed_score: float) -> np.ndarray:
        """Update with new benchmark score."""
        self.predict()
        
        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ (np.array([observed_score]) - self.H @ self.x)
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        
        self.update_count += 1
        return self.x
    
    def get_filtered_estimate(self) -> Tuple[float, float]:
        """Get filtered capability estimate and uncertainty."""
        return float(self.x[0]), float(np.sqrt(self.P[0, 0]))


class OnlineCalibrationDrift:
    """
    Online Calibration Drift monitoring.
    Tracks calibration error and triggers recalibration when needed.
    """
    
    def __init__(self, initial_error: float = 0.1, threshold: float = 0.3, learning_rate: float = 0.1):
        self.C_t = initial_error  # Current calibration error
        self.threshold = threshold
        self.eta = learning_rate
        self.history: List[Tuple[datetime, float]] = []
        self.recalibration_count = 0
    
    def update(self, observed_accuracy: float, expected_accuracy: float, sample_size: int = 10) -> bool:
        """
        Update calibration drift estimate.
        Returns True if recalibration is needed.
        """
        # Estimate current calibration discrepancy
        E = abs(observed_accuracy - expected_accuracy)
        
        # Estimate variance (simplified)
        V = E * (1 - E)  # Bernoulli variance approximation
        
        # Online update
        dt = 1.0
        n_t = sample_size
        dW = np.random.normal(0, np.sqrt(dt))
        
        dC = self.eta * (E - self.C_t) * dt + np.sqrt(V / n_t) * dW
        self.C_t = self.C_t + dC
        
        # Keep in reasonable bounds
        self.C_t = max(0.0, min(1.0, self.C_t))
        
        self.history.append((datetime.now(), self.C_t))
        
        # Check if recalibration needed
        if self.C_t > self.threshold:
            self.recalibration_count += 1
            logger.warning(f"Calibration drift detected: {self.C_t:.3f} > {self.threshold:.3f}")
            return True
        
        return False
    
    def get_current_error(self) -> float:
        """Get current calibration error estimate."""
        return self.C_t


class BayesianChangePointDetector:
    """
    Bayesian Online Change-Point Detection for benchmark scores.
    Detects when model capability has a structural break.
    """
    
    def __init__(self, hazard_rate: float = 0.05, max_run_length: int = 50):
        self.hazard = hazard_rate
        self.max_run_length = max_run_length
        self.r_probs = np.array([1.0])  # P(r_t | data)
        self.x_history: List[float] = []
        self.change_points: List[Tuple[int, float]] = []
        self.score_means: List[float] = []
        self.score_stds: List[float] = []
    
    def _get_likelihood(self, obs: float, run_length: int) -> float:
        """Gaussian likelihood with mean/std estimated from recent history."""
        if run_length == 0:
            # New regime - use overall mean/std
            if len(self.x_history) > 0:
                mean = np.mean(self.x_history)
                std = max(0.01, np.std(self.x_history))
            else:
                mean = 0.5
                std = 0.1
        else:
            # Recent history - use last run_length observations
            recent = self.x_history[-run_length:] if run_length <= len(self.x_history) else self.x_history
            mean = np.mean(recent) if len(recent) > 0 else 0.5
            std = max(0.01, np.std(recent)) if len(recent) > 1 else 0.1
        
        # Gaussian likelihood
        return np.exp(-0.5 * ((obs - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    
    def update(self, score: float) -> Tuple[bool, float]:
        """
        Update with new benchmark score.
        Returns (change_detected, probability_of_recent_change).
        """
        self.x_history.append(score)
        
        # Limit history to prevent memory issues
        if len(self.x_history) > self.max_run_length * 2:
            self.x_history = self.x_history[-self.max_run_length * 2:]
        
        # Predict step: expand run lengths
        new_r_probs = np.zeros(len(self.r_probs) + 1)
        for r, p in enumerate(self.r_probs):
            # Survival
            new_r_probs[r + 1] += p * (1 - self.hazard)
            # Change
            new_r_probs[0] += p * self.hazard
        
        # Limit run length
        if len(new_r_probs) > self.max_run_length:
            new_r_probs = new_r_probs[:self.max_run_length]
            new_r_probs[0] += new_r_probs.sum() - new_r_probs[:self.max_run_length].sum()
        
        # Update step: multiply by likelihood
        likelihoods = np.array([
            self._get_likelihood(score, r) for r in range(len(new_r_probs))
        ])
        new_r_probs *= likelihoods
        new_r_probs = new_r_probs / (new_r_probs.sum() + 1e-10)
        
        self.r_probs = new_r_probs
        
        # Probability of recent change (r <= 3)
        recent_change_prob = sum(self.r_probs[:min(4, len(self.r_probs))])
        
        # Detect change point
        change_detected = recent_change_prob > 0.7
        if change_detected:
            self.change_points.append((len(self.x_history), score))
            logger.warning(f"Change point detected at step {len(self.x_history)} with prob {recent_change_prob:.3f}")
        
        return change_detected, recent_change_prob
    
    def get_most_likely_run_length(self) -> int:
        """Get the most likely current run length."""
        return int(np.argmax(self.r_probs))


class AdvancedMonitoringSuite:
    """
    Combined monitoring suite using all three techniques.
    Provides comprehensive tracking and alerting.
    """
    
    def __init__(self):
        self.kalman = EvaluationKalmanFilter()
        self.calibration_drift = OnlineCalibrationDrift()
        self.change_detector = BayesianChangePointDetector()
        
        self.evaluation_count = 0
        self.last_filtered_capability = 0.5
    
    def update(self, observed_score: float, expected_score: float = None) -> Dict[str, any]:
        """
        Update all monitors with new evaluation result.
        Returns comprehensive status.
        """
        self.evaluation_count += 1
        
        # Update Kalman filter
        filtered_capability, capability_uncertainty = self.kalman.get_filtered_estimate()
        self.kalman.update(observed_score)
        new_filtered_capability, new_uncertainty = self.kalman.get_filtered_estimate()
        
        # Update calibration drift
        expected = expected_score if expected_score is not None else filtered_capability
        needs_recalibration = self.calibration_drift.update(observed_score, expected)
        
        # Update change point detector
        change_detected, change_prob = self.change_detector.update(observed_score)
        
        self.last_filtered_capability = new_filtered_capability
        
        return {
            "evaluation_count": self.evaluation_count,
            "observed_score": observed_score,
            "filtered_capability": new_filtered_capability,
            "capability_uncertainty": new_uncertainty,
            "calibration_error": self.calibration_drift.get_current_error(),
            "needs_recalibration": needs_recalibration,
            "change_detected": change_detected,
            "change_probability": change_prob,
            "most_likely_run_length": self.change_detector.get_most_likely_run_length()
        }
    
    def get_status(self) -> Dict[str, any]:
        """Get current monitoring status."""
        filtered_capability, uncertainty = self.kalman.get_filtered_estimate()
        
        return {
            "total_evaluations": self.evaluation_count,
            "filtered_capability": filtered_capability,
            "capability_uncertainty": uncertainty,
            "calibration_error": self.calibration_drift.get_current_error(),
            "recalibration_count": self.calibration_drift.recalibration_count,
            "change_points_detected": len(self.change_detector.change_points),
            "current_run_length": self.change_detector.get_most_likely_run_length()
        }


# Global monitoring instance
_monitoring_suite = AdvancedMonitoringSuite()

def update_monitoring(observed_score: float, expected_score: float = None) -> Dict[str, any]:
    """Update global monitoring with new evaluation result."""
    return _monitoring_suite.update(observed_score, expected_score)

def get_monitoring_status() -> Dict[str, any]:
    """Get current monitoring status."""
    return _monitoring_suite.get_status()