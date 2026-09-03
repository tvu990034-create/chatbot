import requests
import time
import json

url = 'http://localhost:8000/api/v1/chat'

test_queries = [
    "hello bro",
    "hello bro",  # Cache test
    "what is 2+2",
    "what is 2+2",  # Cache test
    "explain AI",
    "write python code",
    "solve math problem"
]

print("IMPROVEMENT TEST - Speed, Intelligence, Cache")
print("=" * 60)

results = []

for i, query in enumerate(test_queries):
    payload = {
        'messages': [{'role': 'user', 'content': query}],
        'model': 'gemma2:2b',
        'provider': 'ollama',
        'performance_mode': 'balanced'
    }
    
    start = time.time()
    response = requests.post(url, json=payload)
    duration = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        results.append({
            'query': query,
            'duration': duration,
            'response_length': len(data['response']),
            'test_number': i + 1
        })
        print(f"Test {i+1}: '{query}' - {duration:.2f}s - {len(data['response'])} chars")
    else:
        print(f"Test {i+1}: FAILED - {response.status_code}")

# Calculate statistics
cache_tests = [r for r in results if r['test_number'] in [2, 4]]  # Repeated queries
other_tests = [r for r in results if r['test_number'] not in [2, 4]]

if cache_tests and other_tests:
    avg_cache = sum(r['duration'] for r in cache_tests) / len(cache_tests)
    avg_other = sum(r['duration'] for r in other_tests) / len(other_tests)
    cache_speedup = avg_other / avg_cache if avg_cache > 0 else 0
    
    print("\n" + "=" * 60)
    print("PERFORMANCE STATISTICS")
    print("=" * 60)
    print(f"Average cache hit time: {avg_cache:.2f}s")
    print(f"Average cache miss time: {avg_other:.2f}s")
    print(f"Cache speedup: {cache_speedup:.1f}x")
    print(f"Total tests: {len(results)}")
    print(f"Overall average: {sum(r['duration'] for r in results)/len(results):.2f}s")
    
    # Save results
    with open('improvement_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to improvement_test_results.json")