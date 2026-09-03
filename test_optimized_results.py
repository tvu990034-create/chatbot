"""
Test optimized gateway with simple questions
"""

from gateway.litellm_gateway import chat, get_optimization_stats

test_questions = [
    ("Mathematical", "What is 5 + 3?", "8"),
    ("Factual", "What is the capital of France?", "Paris"),
    ("Mathematical", "Solve: 2x + 5 = 15. What is x?", "5"),
    ("Factual", "Who was the first US President?", "Washington"),
    ("Reasoning", "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "5 minutes"),
]

print("Testing optimized gateway with Phase 1 improvements...")
print("="*60)

correct = 0
total = len(test_questions)

for category, question, expected in test_questions:
    print(f"\n[{category}] {question}")
    
    try:
        response = chat(messages=[{"role": "user", "content": question}], use_cache=False)
        
        if expected.lower() in response.lower():
            correct += 1
            print(f"[OK] Response contains: {expected}")
        else:
            print(f"[NO] Expected: {expected}")
            print(f"Response: {response[:150]}...")
        
    except Exception as e:
        print(f"[ERROR] {e}")

print("\n" + "="*60)
print(f"Results: {correct}/{total} ({correct/total*100:.1f}%)")

# Get optimization stats
try:
    stats = get_optimization_stats()
    print("\nOptimization Status:")
    print(f"Optimizations: {stats.get('optimizations', {})}")
except Exception as e:
    print(f"Could not get optimization stats: {e}")
