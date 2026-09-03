"""
Quick GPQA-only benchmark test
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from targeted_26_benchmarks import run_benchmark, BENCHMARK_CONFIGS

def test_gpqa_only():
    """Test GPQA benchmark with the fixed fallback data."""
    print("="*60)
    print("GPQA-ONLY BENCHMARK TEST")
    print("="*60)
    print()
    
    # Run only GPQA
    configs = {"GPQA": BENCHMARK_CONFIGS["GPQA"]}
    results = run_benchmark(configs)
    
    print()
    print("="*60)
    print("GPQA TEST COMPLETE")
    print("="*60)
    
    # Print GPQA result
    if "GPQA" in results:
        result = results["GPQA"]
        print(f"GPQA Accuracy: {result['accuracy']:.1%}")
        print(f"GPQA Correct: {result['correct']}/{result['total']}")
        print(f"GPQA Duration: {result['duration']:.1f}s")

if __name__ == "__main__":
    test_gpqa_only()