import requests
import time
import json

url = 'http://localhost:8000/api/v1/chat'

# Comprehensive test suite
test_queries = [
    ("hello bro", "simple greeting"),
    ("hello bro", "cache hit test"),
    ("what is 2+2", "simple math"),
    ("what is 2+2", "cache hit test"),
    ("Hello, bro!", "punctuation variation"),
    ("Hello Bro", "case variation"),
    ("explain AI", "complex explanation"),
    ("write python code", "coding task"),
    ("solve math problem", "math task"),
    ("the sky is blue", "article variation"),
    ("why sky blue", "rephrased query"),
    ("AI explanation", "keyword variation")
]

print("COMPREHENSIVE PERFORMANCE TEST")
print("=" * 60)

results = []
cache_tests = []
variation_tests = []

for i, (query, test_type) in enumerate(test_queries):
    payload = {
        'messages': [{'role': 'user', 'content': query}],
        'model': 'gemma2:2b',
        'provider': 'ollama',
        'performance_mode': 'balanced'
    }
    
    start = time.time()
    try:
        response = requests.post(url, json=payload, timeout=15)
        duration = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            result = {
                'query': query,
                'test_type': test_type,
                'duration': duration,
                'response_length': len(data['response']),
                'test_number': i + 1
            }
            results.append(result)
            
            if test_type == "cache hit test":
                cache_tests.append(result)
            elif "variation" in test_type:
                variation_tests.append(result)
                
            print(f"Test {i+1}: '{query}' ({test_type}) - {duration:.2f}s - {len(data['response'])} chars")
        else:
            print(f"Test {i+1}: FAILED - {response.status_code}")
    except Exception as e:
        print(f"Test {i+1}: ERROR - {str(e)}")

# Analysis
print("\n" + "=" * 60)
print("PERFORMANCE ANALYSIS")
print("=" * 60)

if cache_tests:
    avg_cache = sum(r['duration'] for r in cache_tests) / len(cache_tests)
    print(f"Cache hit average: {avg_cache:.2f}s")

if variation_tests:
    avg_variation = sum(r['duration'] for r in variation_tests) / len(variation_tests)
    print(f"Variation query average: {avg_variation:.2f}s")

regular_tests = [r for r in results if r['test_type'] not in ["cache hit test"] and "variation" not in r['test_type']]
if regular_tests:
    avg_regular = sum(r['duration'] for r in regular_tests) / len(regular_tests)
    print(f"Regular query average: {avg_regular:.2f}s")

if regular_tests and cache_tests:
    speedup = avg_regular / avg_cache if avg_cache > 0 else 0
    print(f"Cache speedup: {speedup:.1f}x")

print(f"\nTotal tests: {len(results)}")
print(f"Overall average: {sum(r['duration'] for r in results)/len(results):.2f}s")

# Save results
with open('comprehensive_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to comprehensive_test_results.json")