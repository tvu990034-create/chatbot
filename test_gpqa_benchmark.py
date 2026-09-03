"""
Quick GPQA-only benchmark test with the fixed question format
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from targeted_26_benchmarks import generate_fallback

def test_gpqa_only():
    """Test GPQA benchmark with the fixed fallback data."""
    print("="*60)
    print("GPQA-ONLY BENCHMARK TEST")
    print("="*60)
    print()
    
    # Generate fallback data directly
    from targeted_26_benchmarks import generate_fallback
    items = generate_fallback("GPQA", "multiple_choice", 3)
    
    print("GPQA Fallback Questions:")
    for i, item in enumerate(items):
        print(f"Q{i+1}: {item['question']}")
        print(f"Expected Answer: {item['answer']}")
        print()
    
    print("="*60)
    print("Testing GPQA Enhancement Detection")
    print("="*60)
    
    from gateway.fixed_enhanced_gateway import get_fixed_enhanced_gateway
    gateway = get_fixed_enhanced_gateway()
    
    for item in items:
        question = item['question']
        is_gpqa = gateway._is_gpqa_task(question)
        print(f"GPQA detection: {is_gpqa}")
        
        if is_gpqa:
            enhanced = gateway._enhance_gpqa_prompt(question)
            print(f"Enhancement applied: YES")
        else:
            print(f"Enhancement applied: NO")
        print()
    
    print("="*60)

if __name__ == "__main__":
    test_gpqa_only()