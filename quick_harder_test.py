"""
Quick Harder Benchmark Test - 5 Challenging Questions
"""

import time
from gateway.litellm_gateway import chat

MODEL = "ollama/phi3:mini"

hard_questions = [
    ("Advanced Math", "What is the integral of e^x from 0 to infinity?", ["diverges", "infinity", "undefined"]),
    ("Logic Puzzle", "If all A are B, and some B are C, can we conclude that some A are C?", ["no", "cannot conclude"]),
    ("Physics", "Explain the difference between special and general relativity.", ["curvature", "acceleration", "gravity"]),
    ("Code", "What is the time complexity of quicksort in the worst case?", ["O(n^2)", "n squared"]),
    ("Creative", "Explain time dilation using a metaphor.", ["moving", "clock", "relative"]),
]

print("QUICK HARDER BENCHMARK TEST")
print("="*50)

results = []
for subject, question, expected in hard_questions:
    print(f"\n[{subject}] {question[:45]}...")
    
    try:
        start_time = time.time()
        response = chat(
            messages=[{"role": "user", "content": question}],
            model=MODEL,
            use_cache=False
        )
        elapsed = time.time() - start_time
        results.append(elapsed)
        
        is_correct = any(exp.lower() in response.lower() for exp in expected)
        
        if is_correct:
            print(f"[OK] {elapsed:.2f}s")
        else:
            print(f"[NO] {elapsed:.2f}s")
            print(f"Expected: {expected}")
            print(f"Response: {response[:60]}...")
            
    except Exception as e:
        print(f"[ERROR] {e}")

if results:
    avg_time = sum(results) / len(results)
    print(f"\n{'='*50}")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Total time: {sum(results):.2f}s")
