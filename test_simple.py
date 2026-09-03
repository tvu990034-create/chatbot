"""
Simplified test with just 5 questions
"""

from benchmark_integrations.simplified_fixed_integration import SimplifiedFixedIntegrationLayer

integration = SimplifiedFixedIntegrationLayer()

test_queries = [
    ("Mathematical", "What is 5 + 3?", "8"),
    ("Factual", "What is the capital of France?", "Paris"),
    ("Mathematical", "Solve: 2x + 5 = 15. What is x?", "5"),
    ("Factual", "Who was the first US President?", "Washington"),
    ("Mathematical", "What is 15% of 200?", "30"),
]

print("Testing optimized gateway with 5 questions...")
print("="*60)

correct = 0
total = len(test_queries)

for category, query, expected in test_queries:
    print(f"\n[{category}] {query}")
    
    try:
        response, enhancement_info = integration.query_with_gateway(query)
        
        # Check if expected answer is in response
        if expected.lower() in response.lower():
            correct += 1
            print(f"[OK] Response contains: {expected}")
        else:
            print(f"[NO] Expected: {expected}")
            print(f"Response: {response[:150]}...")
        
        print(f"Enhancements: {enhancement_info.get('enhancements_applied', [])}")
        
    except Exception as e:
        print(f"[ERROR] {e}")

print("\n" + "="*60)
print(f"Results: {correct}/{total} ({correct/total*100:.1f}%)")
