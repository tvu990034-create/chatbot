"""
Simple test of the optimization controller only
"""

from gateway.optimization_controller import get_optimization_controller

print("Testing optimization controller...")
controller = get_optimization_controller()

test_query = "What is 5 + 3?"
print(f"Query: {test_query}")

analysis = controller.analyze_query(test_query)
print(f"Analysis: {analysis}")

optimized_query, metadata = controller.apply_optimizations(test_query, analysis)
print(f"Optimized query: {optimized_query}")
print(f"Metadata: {metadata}")

print("\nOptimization controller is working!")
