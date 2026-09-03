"""
Test adaptive temperature with industry benchmarks
"""

import time
from gateway.litellm_gateway import chat
from gateway.adaptive_temperature import record_response_feedback

MODEL = "ollama/phi3:mini"

# Simple heuristic: if response is non-empty and reasonable length, count as success
def is_good_response(response, query):
    return len(response) > 10 and len(response) < 2000

benchmarks = {
    "MMLU": [
        ("Mathematics", "What is the derivative of x^2?", ["2x", "2*x"]),
        ("Computer Science", "What is the time complexity of binary search?", ["log n", "O(log n)", "logarithmic"]),
        ("History", "Who was the first US President?", ["Washington", "George Washington"]),
    ],
    "GSM8K": [
        ("Basic Arithmetic", "Janet has 16 eggs, eats 2, uses 4. How many left?", ["10", "ten"]),
        ("Multi-step", "Tom has 5 apples, buys 3, gives 2. How many left?", ["6", "six"]),
        ("Percentage", "What is 20% of 150?", ["30", "thirty"]),
    ],
}

print("ADAPTIVE TEMPERATURE TEST")
print("="*50)

results = []
from gateway.adaptive_temperature import get_adaptive_temperature
adaptive = get_adaptive_temperature()

for benchmark_name, questions in benchmarks.items():
    print(f"\n{benchmark_name} BENCHMARK")
    print("-"*40)
    
    for subject, question, expected in questions:
        print(f"[{subject}] {question[:45]}...")
        print(f"Current temp: {adaptive.get_temperature():.3f}")
        
        try:
            start_time = time.time()
            response = chat(
                messages=[{"role": "user", "content": question}],
                model=MODEL,
                use_cache=False
            )
            elapsed = time.time() - start_time
            
            is_correct = any(exp.lower() in response.lower() for exp in expected)
            is_good = is_good_response(response, question)
            
            # Record feedback
            record_response_feedback(is_correct and is_good)
            
            print(f"[{'OK' if is_correct else 'NO'}] {elapsed:.2f}s | New temp: {adaptive.get_temperature():.3f}")
            results.append(elapsed)
            
        except Exception as e:
            print(f"[ERROR] {e}")
            record_response_feedback(False)

if results:
    avg_time = sum(results) / len(results)
    print(f"\n{'='*50}")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Final temperature: {adaptive.get_temperature():.3f}")
