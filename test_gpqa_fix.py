"""
Test GPQA fix for the academic enhancement issue
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gateway.fixed_enhanced_gateway import get_fixed_enhanced_gateway

def test_gpqa_detection():
    """Test that GPQA is correctly detected and gets specialized enhancement."""
    gateway = get_fixed_enhanced_gateway()
    
    gpqa_query = "This is a GPQA graduate-level expert question with specialized domain knowledge"
    academic_query = "This is a standard academic multiple choice question: A) option1 B) option2 C) option3 D) option4"
    
    print("GPQA Detection Test:")
    print(f"GPQA query detected as GPQA: {gateway._is_gpqa_task(gpqa_query)}")
    print(f"GPQA query detected as academic: {gateway._is_academic_task(gpqa_query)}")
    print(f"Academic query detected as academic: {gateway._is_academic_task(academic_query)}")
    print(f"Academic query detected as GPQA: {gateway._is_gpqa_task(academic_query)}")
    print()
    
    # Test enhancements
    gpqa_enhanced = gateway._enhance_gpqa_prompt(gpqa_query)
    academic_enhanced = gateway._enhance_academic_knowledge_prompt(academic_query)
    
    print("GPQA Enhancement:")
    print(f"Original: {gpqa_query}")
    print(f"Enhanced: {gpqa_enhanced}")
    print()
    
    print("Academic Enhancement:")
    print(f"Original: {academic_query}")
    print(f"Enhanced: {academic_enhanced}")
    print()

if __name__ == "__main__":
    print("="*60)
    print("GPQA FIX TEST")
    print("="*60)
    print()
    
    test_gpqa_detection()
    
    print("="*60)
    print("TEST COMPLETE")
    print("="*60)