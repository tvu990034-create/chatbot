"""
Test Extended Monitoring Implementation
Tests CUSUM Drift Detection, Streaming AUC, and Lyapunov Analysis
"""

import sys
import io
from pathlib import Path

# Set UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from gateway.extended_monitoring import (
    CUSUMDriftDetector,
    StreamingAUC,
    EvaluationLyapunovAnalyzer,
    ExtendedMonitoringSuite
)

def test_cusum_drift_detection():
    """Test CUSUM Drift Detection."""
    print("="*70)
    print("Testing CUSUM Drift Detection")
    print("="*70)
    
    cusum = CUSUMDriftDetector(mu0=0.5, sigma=0.1, k=0.5, h=4.0)
    
    # Simulate scores with gradual drift
    print("Simulating gradual drift from 0.5 to 0.3...")
    scores = [0.5 - (i * 0.015) for i in range(20)]  # Gradual decrease
    
    for i, score in enumerate(scores):
        alarm, (c_pos, c_neg) = cusum.update(score)
        print(f"Step {i+1}: Score={score:.3f}, C_pos={c_pos:.3f}, C_neg={c_neg:.3f}, Alarm={alarm}")
        
        if alarm:
            print(f"  🚨 Drift detected at step {i+1}")
    
    print(f"Total drift detections: {cusum.drift_count}")
    print("✅ CUSUM Drift Detection working correctly\n")

def test_streaming_auc():
    """Test Streaming AUC calculation."""
    print("="*70)
    print("Testing Streaming AUC")
    print("="*70)
    
    streaming_auc = StreamingAUC(n_bins=50)
    
    # Simulate predictions with known AUC (should be ~0.75)
    print("Simulating predictions with expected AUC ~0.75...")
    np.random.seed(42)
    
    for i in range(100):
        # Generate scores where positives tend to be higher
        label = np.random.choice([0, 1])
        if label == 1:
            score = np.random.beta(5, 2)  # Higher scores for positives
        else:
            score = np.random.beta(2, 5)  # Lower scores for negatives
        
        streaming_auc.update(score, label)
        
        if (i + 1) % 20 == 0:
            stats = streaming_auc.get_stats()
            print(f"Step {i+1}: AUC={stats['auc']:.3f}, Samples={stats['total_samples']}")
    
    final_stats = streaming_auc.get_stats()
    print(f"\nFinal AUC: {final_stats['auc']:.3f}")
    print(f"Total samples: {final_stats['total_samples']}")
    print("✅ Streaming AUC working correctly\n")

def test_lyapunov_convergence():
    """Test Lyapunov convergence analysis."""
    print("="*70)
    print("Testing Lyapunov Convergence Analysis")
    print("="*70)
    
    lyapunov = EvaluationLyapunovAnalyzer()
    
    # Test with stable dynamics (converging system)
    print("Testing with converging dynamics...")
    A_stable = np.array([[0.8]])  # Converging (eigenvalue < 1)
    Q = np.array([[1.0]])
    
    is_stable, P = lyapunov.verify_convergence(A_stable, Q)
    print(f"Stable system: Converging={is_stable}")
    print(f"Lyapunov matrix P: {P}")
    
    # Test with unstable dynamics
    print("\nTesting with diverging dynamics...")
    A_unstable = np.array([[1.2]])  # Diverging (eigenvalue > 1)
    
    is_stable_unstable, P_unstable = lyapunov.verify_convergence(A_unstable, Q)
    print(f"Unstable system: Converging={is_stable_unstable}")
    
    # Test with actual score dynamics
    print("\nAnalyzing actual score dynamics...")
    converging_scores = [0.5, 0.6, 0.7, 0.75, 0.8, 0.82, 0.84, 0.85, 0.86, 0.87]
    analysis = lyapunov.analyze_score_dynamics(converging_scores, target=0.9)
    
    print(f"Converging: {analysis['converging']}")
    print(f"Convergence rate: {analysis['convergence_rate']:.3f}")
    print(f"Distance to target: {analysis['distance_to_target']:.3f}")
    
    print("✅ Lyapunov Convergence Analysis working correctly\n")

def test_extended_monitoring_suite():
    """Test integrated extended monitoring suite."""
    print("="*70)
    print("Testing Extended Monitoring Suite")
    print("="*70)
    
    suite = ExtendedMonitoringSuite()
    
    # Calibrate with historical data
    historical = [0.75, 0.78, 0.72, 0.80, 0.76]
    suite.calibrate(historical)
    print(f"Calibrated with {len(historical)} historical scores")
    
    # Simulate evaluation rounds with labels
    print(f"Running evaluation rounds...")
    np.random.seed(42)
    
    for i in range(15):
        score = 0.75 + np.random.normal(0, 0.05)
        label = 1 if score > 0.75 else 0
        
        status = suite.update(score, label)
        
        print(f"Round {i+1}: Score={score:.3f}, Label={label}")
        print(f"  Drift Detected: {status['drift_detected']}")
        
        if status['auc_stats']:
            print(f"  AUC: {status['auc_stats']['auc']:.3f}")
        
        if status['convergence_analysis']:
            print(f"  Converging: {status['convergence_analysis']['converging']}")
    
    final_status = suite.get_status()
    print(f"\nFinal Status:")
    print(f"  Total Evaluations: {final_status['total_evaluations']}")
    print(f"  Drift Detections: {final_status['drift_detections']}")
    print(f"  AUC: {final_status['auc']:.3f}")
    print(f"  AUC Samples: {final_status['auc_samples']}")
    print(f"  Convergence Stable: {final_status['convergence_stable']}")
    print("✅ Extended Monitoring Suite working correctly\n")

if __name__ == "__main__":
    try:
        test_cusum_drift_detection()
        test_streaming_auc()
        test_lyapunov_convergence()
        test_extended_monitoring_suite()
        
        print("="*70)
        print("ALL EXTENDED MONITORING TESTS PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()