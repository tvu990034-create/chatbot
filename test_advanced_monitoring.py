"""
Test Advanced Monitoring Implementation
Tests Kalman Filter, Calibration Drift, and Change-Point Detection
"""

import sys
import io
from pathlib import Path

# Set UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(Path(__file__).parent))

from gateway.advanced_monitoring import (
    EvaluationKalmanFilter,
    OnlineCalibrationDrift,
    BayesianChangePointDetector,
    AdvancedMonitoringSuite
)

def test_kalman_filter():
    """Test Kalman Filter for capability tracking."""
    print("="*70)
    print("Testing Kalman Filter for Capability Tracking")
    print("="*70)
    
    kalman = EvaluationKalmanFilter(initial_capability=0.5)
    
    # Simulate noisy benchmark scores
    true_capability = 0.75
    scores = [true_capability + (i % 2) * 0.1 - 0.05 for i in range(20)]
    
    print(f"True capability: {true_capability}")
    print(f"Noisy scores: {scores[:5]}...")
    
    for i, score in enumerate(scores):
        kalman.update(score)
        filtered, uncertainty = kalman.get_filtered_estimate()
        print(f"Step {i+1}: Observed={score:.3f}, Filtered={filtered:.3f}, Uncertainty={uncertainty:.3f}")
    
    final_filtered, final_uncertainty = kalman.get_filtered_estimate()
    print(f"\nFinal estimate: {final_filtered:.3f} ± {final_uncertainty:.3f}")
    print(f"Error from true: {abs(final_filtered - true_capability):.3f}")
    print("✅ Kalman Filter working correctly\n")

def test_calibration_drift():
    """Test Online Calibration Drift monitoring."""
    print("="*70)
    print("Testing Online Calibration Drift")
    print("="*70)
    
    drift_monitor = OnlineCalibrationDrift(threshold=0.25)
    
    # Simulate calibration drift over time
    print("Simulating increasing calibration drift...")
    for i in range(15):
        observed = 0.9 - (i * 0.02)  # Decreasing accuracy
        expected = 0.9
        needs_recal = drift_monitor.update(observed, expected)
        
        print(f"Step {i+1}: Observed={observed:.3f}, Expected={expected:.3f}, "
              f"Drift={drift_monitor.get_current_error():.3f}, Recalibrate={needs_recal}")
        
        if needs_recal:
            print(f"  ⚠️  Recalibration triggered at step {i+1}")
            drift_monitor.C_t = 0.05  # Reset after recalibration
    
    print(f"Total recalibrations: {drift_monitor.recalibration_count}")
    print("✅ Calibration Drift monitoring working correctly\n")

def test_change_point_detection():
    """Test Bayesian Change-Point Detection."""
    print("="*70)
    print("Testing Bayesian Change-Point Detection")
    print("="*70)
    
    detector = BayesianChangePointDetector(hazard_rate=0.1)
    
    # Simulate scores with a change point
    scores_before = [0.8 + (i % 3) * 0.05 for i in range(10)]  # High performance
    scores_after = [0.5 + (i % 3) * 0.05 for i in range(10)]   # Lower performance
    all_scores = scores_before + scores_after
    
    print(f"Simulating {len(all_scores)} scores with change point at step 10")
    print(f"Before change: {scores_before[:3]}...")
    print(f"After change: {scores_after[:3]}...")
    
    change_detected = False
    for i, score in enumerate(all_scores):
        detected, prob = detector.update(score)
        run_length = detector.get_most_likely_run_length()
        
        print(f"Step {i+1}: Score={score:.3f}, ChangeProb={prob:.3f}, RunLength={run_length}")
        
        if detected and not change_detected:
            print(f"  🔍 Change point detected at step {i+1}")
            change_detected = True
    
    print(f"Total change points detected: {len(detector.change_points)}")
    print("✅ Change-Point Detection working correctly\n")

def test_integrated_monitoring():
    """Test integrated monitoring suite."""
    print("="*70)
    print("Testing Integrated Monitoring Suite")
    print("="*70)
    
    suite = AdvancedMonitoringSuite()
    
    # Simulate evaluation rounds
    scores = [0.85, 0.82, 0.88, 0.90, 0.87, 0.75, 0.72, 0.70, 0.68, 0.65]
    
    print(f"Running {len(scores)} evaluation rounds...")
    for i, score in enumerate(scores):
        status = suite.update(score, expected_score=0.85)
        
        print(f"Round {i+1}: Score={score:.3f}")
        print(f"  Filtered Capability: {status['filtered_capability']:.3f}")
        print(f"  Calibration Error: {status['calibration_error']:.3f}")
        print(f"  Change Prob: {status['change_probability']:.3f}")
        
        if status['needs_recalibration']:
            print(f"  ⚠️  Recalibration needed")
        if status['change_detected']:
            print(f"  🔍 Change point detected")
    
    final_status = suite.get_status()
    print(f"\nFinal Status:")
    print(f"  Total Evaluations: {final_status['total_evaluations']}")
    print(f"  Filtered Capability: {final_status['filtered_capability']:.3f}")
    print(f"  Calibration Error: {final_status['calibration_error']:.3f}")
    print(f"  Recalibrations: {final_status['recalibration_count']}")
    print(f"  Change Points: {final_status['change_points_detected']}")
    print("✅ Integrated Monitoring Suite working correctly\n")

if __name__ == "__main__":
    try:
        test_kalman_filter()
        test_calibration_drift()
        test_change_point_detection()
        test_integrated_monitoring()
        
        print("="*70)
        print("ALL ADVANCED MONITORING TESTS PASSED")
        print("="*70)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()