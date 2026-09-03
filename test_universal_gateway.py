"""
Test Universal Gateway with Different Local AI Models
Verifies that optimizations work with any local model
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gateway.universal_enhanced_gateway import get_universal_gateway

def test_model(model_name: str):
    """Test the universal gateway with a specific model."""
    print("="*60)
    print(f"Testing Universal Gateway with: {model_name}")
    print("="*60)
    print()
    
    # Get gateway for this model
    gateway = get_universal_gateway(model_name)
    
    # Test queries for different optimization types
    test_queries = [
        ("Math", "Solve: x + 5 = 10"),
        ("Academic", "What is 2+2? A)3 B)4 C)5 D)6"),
        ("Coding", "Write a function to add two numbers"),
        ("Complex", "Explain why the sky is blue step by step"),
        ("Agent Task", "Plan how to organize a small event")
    ]
    
    results = {}
    
    for query_type, query in test_queries:
        print(f"Testing {query_type}: {query}")
        
        try:
            messages = [{"role": "user", "content": query}]
            response = gateway.chat(messages=messages, use_cache=False)
            
            print(f"Response: {response[:100]}...")
            print(f"Success")
            results[query_type] = True
        except Exception as e:
            print(f"Failed: {e}")
            results[query_type] = False
        
        print()
    
    # Get optimization stats
    stats = gateway.get_optimization_stats()
    print("Optimization Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print()
    print("="*60)
    print(f"Results for {model_name}:")
    print("="*60)
    for query_type, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"{status} {query_type}")
    print()

def main():
    """Test universal gateway with all available models."""
    # Available models from your Ollama
    models = [
        "phi3:mini",      # 2.2 GB - Good balance
        "gemma2:2b",      # 1.6 GB - Lightweight 
        "llama3.2:latest", # 2.0 GB - Capable
        "tinyllama:latest" # 637 MB - Ultra lightweight
    ]
    
    print("Universal Gateway Test Suite")
    print("="*60)
    print("Testing optimization compatibility with different models")
    print("="*60)
    print()
    
    for model in models:
        try:
            test_model(model)
        except Exception as e:
            print(f"Failed to test {model}: {e}")
            print()
    
    print("="*60)
    print("Universal Gateway Test Complete")
    print("="*60)

if __name__ == "__main__":
    main()