"""
Quick comprehensive test with optimized gateway
"""

from gateway.litellm_gateway import chat, get_optimization_stats

test_questions = [
    ("Basic - Math", "What is 5 + 3?", "8"),
    ("Basic - Geography", "What is the capital of France?", "Paris"),
    ("Intermediate - Math", "What is 15% of 200?", "30"),
    ("Hard - Math", "Solve: 2x + 5 = 15. What is x?", "5"),
    ("Expert - Math", "Prove that the square root of 2 is irrational using contradiction", "contradiction"),
    ("Reasoning", "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "5 minutes"),
    ("Factual", "Who was the first US President?", "Washington"),
    ("Creative", "Write a short story about a robot discovering emotions", "robot"),
    ("Adversarial", "A plane crashes on the US-Canada border. Where do they bury the survivors?", "survivors"),
]

print("QUICK COMPREHENSIVE TEST WITH OPTIMIZED GATEWAY")
print("="*60)

correct = 0
total = len(test_questions)
results_by_category = {}

for category, question, expected in test_questions:
    print(f"\n[{category}] {question[:50]}...")
    
    try:
        response = chat(messages=[{"role": "user", "content": question}], use_cache=False)
        
        # Check if expected answer is in response
        if expected.lower() in response.lower():
            correct += 1
            print(f"[OK]")
            results_by_category[category] = "OK"
        else:
            print(f"[NO] Expected: {expected}")
            print(f"Response: {response[:100]}...")
            results_by_category[category] = "NO"
        
    except Exception as e:
        print(f"[ERROR] {e}")
        results_by_category[category] = "ERROR"

print("\n" + "="*60)
print(f"Results: {correct}/{total} ({correct/total*100:.1f}%)")

print("\nCategory breakdown:")
for category, result in results_by_category.items():
    print(f"{category}: {result}")

# Get optimization stats
try:
    stats = get_optimization_stats()
    print("\nOptimization Statistics:")
    controller_stats = stats.get('optimizations', {}).get('optimization_controller', {})
    print(f"Total queries processed: {controller_stats.get('total_queries', 0)}")
    print(f"Optimizations applied: {controller_stats.get('optimization_counts', {})}")
    print(f"Performance by domain: {controller_stats.get('performance_by_domain', {})}")
except Exception as e:
    print(f"Could not get optimization stats: {e}")
