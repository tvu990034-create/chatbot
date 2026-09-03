"""
Test the streamlined gateway to verify it works and is fast
"""

import time
from gateway.litellm_gateway import chat

MODEL = "ollama/phi3:mini"

test_queries = [
    "What is 2+2?",
    "Who is the president of the United States?",
    "Explain gravity",
    "What is the capital of France?",
    "Calculate 15 * 7",
]

print("STREAMLINED GATEWAY SPEED TEST")
print("="*50)

results = []
for i, query in enumerate(test_queries, 1):
    print(f"\n[{i}] {query}")
    
    try:
        start_time = time.time()
        response = chat(
            messages=[{"role": "user", "content": query}],
            model=MODEL,
            use_cache=False  # Test raw speed
        )
        elapsed = time.time() - start_time
        
        results.append(elapsed)
        print(f"[OK] {elapsed:.2f}s")
        print(f"Response: {response[:60]}...")
        
    except Exception as e:
        print(f"[ERROR] {e}")

if results:
    avg_time = sum(results) / len(results)
    print(f"\n{'='*50}")
    print(f"Average time: {avg_time:.2f}s")
    print(f"Min time: {min(results):.2f}s")
    print(f"Max time: {max(results):.2f}s")
