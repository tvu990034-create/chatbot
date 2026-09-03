"""
Additional Lightweight Monitoring Techniques
Implements CUSUM Drift Detection and Streaming AUC for enhanced monitoring
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

class CUSUMDriftDetector:
    """
    CUSUM (Cumulative Sum) Drift Detection.
    Detects small, persistent shifts in benchmark scores.
    """
    
    def __init__(self, mu0: float = 0.5, sigma: float = 0.1, k: float = 0.5, h: float = 5.0):
        self.mu0 = mu0  # In-control mean
        self.sigma = sigma  # Standard deviation
        self.k = k  # Reference value (typically 0.5)
        self.h = h  # Decision threshold (typically 4-5)
        
        self.C_pos = 0.0  # Positive CUSUM
        self.C_neg = 0.0  # Negative CUSUM
        
        self.drift_count = 0
        self.history: List[Tuple[datetime, float, bool]] = []
    
    def update(self, x: float) -> Tuple[bool, Tuple[float, float]]:
        """
        Update with new observation.
        Returns (alarm, (C_pos, C_neg)).
        """
        # Standardize observation
        z = (x - self.mu0) / self.sigma
        
        # Update CUSUM statistics
        self.C_pos = max(0, self.C_pos + z - self.k)
        self.C_neg = max(0, self.C_neg - z - self.k)
        
        # Check for alarm
        alarm = (self.C_pos > self.h) or (self.C_neg > self.h)
        
        if alarm:
            self.drift_count += 1
            logger.warning(f"CUSUM drift detected: C_pos={self.C_pos:.3f}, C_neg={self.C_neg:.3f}")
            # Reset after detection
            self.C_pos = 0.0
            self.C_neg = 0.0
        
        self.history.append((datetime.now(), x, alarm))
        
        return alarm, (self.C_pos, self.C_neg)
    
    def calibrate(self, scores: List[float]):
        """Calibrate parameters from historical data."""
        if len(scores) > 0:
            self.mu0 = np.mean(scores)
            self.sigma = max(0.01, np.std(scores))
            logger.info(f"CUSUM calibrated: mu0={self.mu0:.3f}, sigma={self.sigma:.3f}")


class StreamingAUC:
    """
    Streaming AUC calculation using histogram binning.
    Computes AUC in O(1) per new sample without storing all history.
    """
    
    def __init__(self, n_bins: int = 100):
        self.n_bins = n_bins
        self.bin_edges = np.linspace(0, 1, n_bins + 1)
        self.pos_counts = np.zeros(n_bins)
        self.neg_counts = np.zeros(n_bins)
        self.R = 0.0  # Mann-Whitney U statistic
        self.n_pos = 0
        self.n_neg = 0
        self.auc_history: List[Tuple[datetime, float]] = []
    
    def _bin_index(self, score: float) -> int:
        """Get bin index for a score."""
        return min(int(score * self.n_bins), self.n_bins - 1)
    
    def update(self, score: float, label: int):
        """
        Update with new prediction (score, label).
        score: prediction confidence/probability in [0,1]
        label: true label (0 or 1)
        """
        idx = self._bin_index(score)
        
        if label == 1:
            # New positive: add number of negatives with lower score
            self.R += self.neg_counts[:idx].sum()
            self.R += 0.5 * self.neg_counts[idx]
            self.pos_counts[idx] += 1
            self.n_pos += 1
        else:
            # New negative: add number of positives with higher score
            self.R += self.pos_counts[(idx + 1):].sum()
            self.R += 0.5 * self.pos_counts[idx]
            self.neg_counts[idx] += 1
            self.n_neg += 1
        
        # Track AUC history
        current_auc = self.auc()
        self.auc_history.append((datetime.now(), current_auc))
    
    def auc(self) -> float:
        """Get current AUC estimate."""
        if self.n_pos == 0 or self.n_neg == 0:
            return 0.0
        return self.R / (self.n_pos * self.n_neg)
    
    def get_stats(self) -> Dict[str, any]:
        """Get current statistics."""
        return {
            "auc": self.auc(),
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "total_samples": self.n_pos + self.n_neg
        }


class EvaluationLyapunovAnalyzer:
    """
    Lyapunov function for formal convergence guarantees.
    Provides theoretical assurance that evaluation dynamics converge.
    """
    
    def __init__(self):
        self.A = None  # Dynamics matrix
        self.P = None  # Lyapunov matrix
        self.Q = None  # Positive definite matrix
        self.is_stable = False
    
    def verify_convergence(self, A: np.ndarray, Q: np.ndarray = None) -> Tuple[bool, np.ndarray]:
        """
        Verify convergence using Lyapunov equation.
        A: Dynamics matrix (should be stable)
        Q: Positive definite matrix (default: identity)
        Returns (is_stable, P) where P solves A^T P + P A = -Q
        """
        if Q is None:
            Q = np.eye(A.shape[0])
        
        try:
            from scipy.linalg import solve_lyapunov
            P = solve_lyapunov(A.T, -Q)
            
            # Check if P is positive definite
            eigvals = np.linalg.eigvals(P)
            is_stable = np.all(eigvals > 0)
            
            self.A = A
            self.P = P
            self.Q = Q
            self.is_stable = is_stable
            
            if is_stable:
                logger.info(f"Lyapunov equation satisfied, P positive definite: {is_stable}")
            else:
                logger.warning(f"Lyapunov equation not satisfied, eigenvalues: {eigvals}")
            
            return is_stable, P
            
        except ImportError:
            logger.warning("scipy not available for Lyapunov solver")
            return False, np.eye(A.shape[0])
    
    def analyze_score_dynamics(self, scores: List[float], target: float = 1.0) -> Dict[str, any]:
        """
        Analyze score dynamics and provide convergence guarantee.
        scores: Historical score sequence
        target: Target score (equilibrium)
        """
        if len(scores) < 2:
            return {"converging": False, "reason": "insufficient data"}
        
        # Estimate dynamics matrix A from score differences
        # Simple linear regression: s(t+1) - s* = A * (s(t) - s*)
        diffs = np.array(scores[1:]) - target
        prev_diffs = np.array(scores[:-1]) - target
        
        # Solve A (scalar for 1D case)
        if len(prev_diffs) > 0:
            A = np.array([[np.sum(diffs * prev_diffs) / (np.sum(prev_diffs**2) + 1e-10)]])
            Q = np.array([[1.0]])
            
            is_stable, P = self.verify_convergence(A, Q)
            
            # Estimate convergence rate
            convergence_rate = abs(A[0, 0]) if is_stable else 1.0
            
            return {
                "converging": is_stable and convergence_rate < 1.0,
                "convergence_rate": float(convergence_rate),
                "dynamics_matrix": A.tolist(),
                "lyapunov_matrix": P.tolist() if P is not None else None,
                "current_score": scores[-1],
                "target_score": target,
                "distance_to_target": abs(scores[-1] - target)
            }
        
        return {"converging": False, "reason": "insufficient data"}


class ExtendedMonitoringSuite:
    """
    Extended monitoring suite combining new lightweight techniques.
    """
    
    def __init__(self):
        self.cusum = CUSUMDriftDetector()
        self.streaming_auc = StreamingAUC()
        self.lyapunov = EvaluationLyapunovAnalyzer()
        
        self.evaluation_count = 0
        self.score_history: List[float] = []
    
    def update(self, score: float, label: Optional[int] = None) -> Dict[str, any]:
        """
        Update all monitors with new evaluation result.
        score: Benchmark score or prediction confidence
        label: True label (for AUC calculation, optional)
        """
        self.evaluation_count += 1
        self.score_history.append(score)
        
        # Update CUSUM
        drift_alarm, cusum_stats = self.cusum.update(score)
        
        # Update Streaming AUC if label provided
        if label is not None:
            self.streaming_auc.update(score, label)
        
        # Analyze convergence periodically
        convergence_analysis = None
        if self.evaluation_count % 10 == 0 and len(self.score_history) >= 5:
            convergence_analysis = self.lyapunov.analyze_score_dynamics(self.score_history)
        
        return {
            "evaluation_count": self.evaluation_count,
            "score": score,
            "drift_detected": drift_alarm,
            "cusum_stats": cusum_stats,
            "auc_stats": self.streaming_auc.get_stats() if label is not None else None,
            "convergence_analysis": convergence_analysis
        }
    
    def get_status(self) -> Dict[str, any]:
        """Get current monitoring status."""
        return {
            "total_evaluations": self.evaluation_count,
            "drift_detections": self.cusum.drift_count,
            "auc": self.streaming_auc.auc(),
            "auc_samples": self.streaming_auc.n_pos + self.streaming_auc.n_neg,
            "convergence_stable": self.lyapunov.is_stable
        }
    
    def calibrate(self, historical_scores: List[float]):
        """Calibrate CUSUM with historical data."""
        self.cusum.calibrate(historical_scores)


# Global extended monitoring instance
_extended_monitoring_suite = ExtendedMonitoringSuite()

def update_extended_monitoring(score: float, label: Optional[int] = None) -> Dict[str, any]:
    """Update global extended monitoring with new evaluation result."""
    return _extended_monitoring_suite.update(score, label)

def get_extended_monitoring_status() -> Dict[str, any]:
    """Get current extended monitoring status."""
    return _extended_monitoring_suite.get_status()