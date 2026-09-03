"""
Test what the model actually responds to GPQA fallback questions
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gateway.fixed_enhanced_gateway import get_fixed_enhanced_gateway

def test_gpqa_model_response():
    """Test the actual model response to GPQA fallback question."""
    gateway = get_fixed_enhanced_gateway()
    
    # Use the exact GPQA fallback question (updated clearer format)
    gpqa_question = "GPQA Q1: For graduate-level expert questions requiring specialized domain knowledge, what level of expertise is typically expected? A) Basic B) Advanced C) Expert D) Beginner"
    
    print("="*60)
    print("GPQA MODEL RESPONSE TEST")
    print("="*60)
    print()
    print(f"Question: {gpqa_question}")
    print()
    
    # Get model response
    messages = [{"role": "user", "content": gpqa_question}]
    response = gateway.chat(messages=messages, use_cache=False)
    
    print(f"Model Response: {response}")
    print()
    
    # Test answer checking logic (NEW LOGIC)
    import re
    response_lower = str(response).lower()
    correct_answer = "c"
    
    # New smarter multiple choice answer extraction
    answer_patterns = [
        r'(?:answer|correct|choice|option)[^\w]*([a-d])',
        r'([a-d])\s*[\)\.]',  # A) or A.
        r'is\s+([a-d])\s*$'  # answer is C
    ]
    
    print("Testing new answer extraction logic:")
    for pattern in answer_patterns:
        matches = re.findall(pattern, response_lower, re.IGNORECASE)
        if matches:
            print(f"Pattern matched: {pattern}")
            print(f"Matched letter: {matches[0].lower()}")
            print(f"Is correct: {matches[0].lower() == correct_answer}")
            break
    else:
        # Fallback: count occurrences of each letter and pick the most common
        letter_counts = {}
        for letter in response_lower:
            if letter in ['a', 'b', 'c', 'd']:
                letter_counts[letter] = letter_counts.get(letter, 0) + 1
        
        if letter_counts:
            most_common = max(letter_counts, key=letter_counts.get)
            print(f"Letter counts: {letter_counts}")
            print(f"Most common letter: {most_common}")
            print(f"Is correct: {most_common == correct_answer}")
        else:
            print(f"Final fallback: {correct_answer in response_lower}")
    
    print()
    print("="*60)

if __name__ == "__main__":
    test_gpqa_model_response()