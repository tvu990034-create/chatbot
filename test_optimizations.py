"""
Quick test of new optimization features
"""

from gateway.litellm_gateway import chat
from gateway.optimization_controller import get_optimization_controller

# Test query classification
controller = get_optimization_controller()

print("Testing query classification and optimization...")
print("="*60)

test_queries = [
    "What is 5 + 3?",
    "Explain the photoelectric effect.",
    "Write a creative story about a robot.",
    "Who was the first US President?",
    "Solve: 2x + 5 = 15. What is x?",
]

for query in test_queries:
    print(f"\nQuery: {query}")
    
    # Analyze query
    analysis = controller.analyze_query(query)
    print(f"Domain: {analysis.get('domain')}")
    print(f"Complexity: {analysis.get('complexity')}")
    print(f"Suggested Temperature: {analysis.get('suggested_temperature')}")
    print(f"Optimizations: {analysis.get('optimizations')}")
    
    # Apply optimizations
    optimized_query, metadata = controller.apply_optimizations(query, analysis)
    print(f"Applied optimizations: {metadata.get('applied_optimizations')}")
    
    # Test gateway call
    try:
        response = chat(messages=[{"role": "user", "content": query}], use_cache=False)
        print(f"Response: {response[:100]}...")
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "="*60)
print("Optimization Statistics:")
stats = controller.get_statistics()
print(f"Total queries: {stats.get('total_queries')}")
print(f"Optimization counts: {stats.get('optimization_counts')}")
print(f"Performance by domain: {stats.get('performance_by_domain')}")
