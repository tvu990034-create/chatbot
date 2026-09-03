import requests
import time
import json

url = 'http://localhost:8000/api/v1/chat'

# Advanced test suite with more variations
test_queries = [
    ("hello bro", "exact match"),
    ("hello bro", "exact cache hit"),
    ("hello bro", "exact cache hit again"),
    ("hello", "partial match"),
    ("hi bro", "variation 1"),
    ("hey bro", "variation 2"),
    ("hello!", "punctuation only"),
    ("HELLO BRO", "all caps"),
    "What is 2+2?",
    "what is 2+2",
    "2+2", 
    "calculate 2+2",
    "explain artificial intelligence",
    "explain AI",
    "AI explanation",
    "write python function",
    "python code for addition",
    "solve 2x + 4 = 10",
    "math problem 2x + 4 = 10"
]

print("ADVANCED PERFORMANCE TEST")
print("=" * 60)

results = []
exact_cache_hits = []
partial_matches = []
complex_queries = []

for i, query in enumerate(test_queries):
    if isinstance(query, tuple):
        query_text, test_type = query
    else:
        query_text = query
        test_type = "auto-detect"
    
    payload = {
        'messages': [{'role': 'user', 'content': query_text}],
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
                'query': query_text,
                'test_type': test_type,
                'duration': duration,
                'response_length': len(data['response']),
                'test_number': i + 1
            }
            results.append(result)
            
            if test_type == "exact cache hit" or test_type == "exact cache hit again":
                exact_cache_hits.append(result)
            elif "variation" in test_type or "partial" in test_type:
                partial_matches.append(result)
            elif "explain" in query_text.lower() or "write" in query_text.lower() or "solve" in query_text.lower():
                complex_queries.append(result)
                
            print(f"Test {i+1}: '{query_text}' ({test_type}) - {duration:.2f}s - {len(data['response'])} chars")
        else:
            print(f"Test {i+1}: FAILED - {response.status_code}")
    except Exception as e:
        print(f"Test {i+1}: ERROR - {str(e)}")

# Advanced analysis
print("\n" + "=" * 60)
print("ADVANCED PERFORMANCE ANALYSIS")
print("=" * 60)

if exact_cache_hits:
    avg_exact = sum(r['duration'] for r in exact_cache_hits) / len(exact_cache_hits)
    print(f"Exact cache hit average: {avg_exact:.2f}s")

if partial_matches:
    avg_partial = sum(r['duration'] for r in partial_matches) / len(partial_matches)
    print(f"Partial match average: {avg_partial:.2f}s")

if complex_queries:
    avg_complex = sum(r['duration'] for r in complex_queries) / len(complex_queries)
    print(f"Complex query average: {avg_complex:.2f}s")

regular_tests = [r for r in results if r not in exact_cache_hits and r not in partial_matches]
if regular_tests:
    avg_regular = sum(r['duration'] for r in regular_tests) / len(regular_tests)
    print(f"Regular query average: {avg_regular:.2f}s")

if regular_tests and exact_cache_hits:
    speedup = avg_regular / avg_exact if avg_exact > 0 else 0
    print(f"Cache speedup: {speedup:.1f}x")

print(f"\nTotal tests: {len(results)}")
print(f"Overall average: {sum(r['duration'] for r in results)/len(results):.2f}s")

# Cache hit rate analysis
cache_eligible = len([r for r in results if "variation" in r['test_type'] or "partial" in r['test_type'] or "cache" in r['test_type']])
fast_queries = len([r for r in results if r['duration'] < 2.5])
print(f"Cache-eligible queries: {cache_eligible}")
print(f"Fast queries (<2.5s): {fast_queries}")

# Save results
with open('advanced_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to advanced_test_results.json")