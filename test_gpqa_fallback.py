"""
Test GPQA fallback data detection
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gateway.fixed_enhanced_gateway import get_fixed_enhanced_gateway

def test_gpqa_fallback_detection():
    """Test that GPQA fallback data is correctly detected."""
    gateway = get_fixed_enhanced_gateway()
    
    # Test the actual fallback data format
    gpqa_fallback = "GPQA Q1: This is a graduate-level expert question requiring specialized domain knowledge. What is the correct answer? A) Basic B) Advanced C) Expert D) Specialized"
    
    print("GPQA Fallback Detection Test:")
    print(f"GPQA fallback detected as GPQA: {gateway._is_gpqa_task(gpqa_fallback)}")
    print(f"GPQA fallback detected as academic: {gateway._is_academic_task(gpqa_fallback)}")
    print()
    
    # Test enhancement
    enhanced = gateway._enhance_gpqa_prompt(gpqa_fallback)
    print("GPQA Fallback Enhancement:")
    print(f"Original: {gpqa_fallback}")
    print(f"Enhanced: {enhanced}")
    print()

if __name__ == "__main__":
    print("="*60)
    print("GPQA FALLBACK DATA TEST")
    print("="*60)
    print()
    
    test_gpqa_fallback_detection()
    
    print("="*60)
    print("TEST COMPLETE")
    print("="*60)