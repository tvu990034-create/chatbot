import requests
import time
import json

url = 'http://localhost:8000/api/v1/chat'

# Ultimate test suite - maximum cache challenge
test_queries = [
    # Exact matches (should be instant cache hits)
    ("hello bro", "exact 1"),
    ("hello bro", "exact cache hit"),
    ("hello bro", "exact cache hit again"),
    ("hello bro", "exact cache hit x4"),
    
    # Smart variations (should hit cache with normalization)
    ("hello bro", "exact after variations"),
    ("Hello Bro", "case variation"),
    ("HELLO BRO", "all caps"),
    ("hello!", "punctuation variation"),
    ("hello!?", "more punctuation"),
    
    # Semantic variations (should benefit from stop word removal)
    ("hi there bro", "semantic variation 1"),
    ("hey there bro", "semantic variation 2"),
    ("greetings bro", "semantic variation 3"),
    
    # Math variations (should normalize well)
    ("what is 2+2", "math original"),
    ("calculate 2+2", "math variation 1"),
    ("2 plus 2", "math variation 2"),
    ("what is 2+2?", "math with punctuation"),
    
    # AI variations (should hit cache with normalization)
    ("explain AI", "ai original"),
    ("explain artificial intelligence", "ai expanded"),
    ("AI explanation", "ai rephrased"),
    ("tell me about AI", "ai semantic"),
    
    # Coding variations
    ("write python code", "code original"),
    ("python function", "code variation 1"),
    ("create python function", "code variation 2"),
]

print("ULTIMATE CACHE CHALLENGE TEST")
print("=" * 60)

results = []
exact_hits = []
smart_hits = []
semantic_hits = []

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
            
            if "exact cache hit" in test_type:
                exact_hits.append(result)
            elif "variation" in test_type and "semantic" not in test_type:
                smart_hits.append(result)
            elif "semantic" in test_type:
                semantic_hits.append(result)
                
            cache_status = "CACHE HIT" if duration < 2.5 else "CACHE MISS"
            print(f"Test {i+1}: '{query}' ({test_type}) - {duration:.2f}s - {cache_status}")
        else:
            print(f"Test {i+1}: FAILED - {response.status_code}")
    except Exception as e:
        print(f"Test {i+1}: ERROR - {str(e)}")

# Ultimate analysis
print("\n" + "=" * 60)
print("ULTIMATE CACHE PERFORMANCE ANALYSIS")
print("=" * 60)

if exact_hits:
    avg_exact = sum(r['duration'] for r in exact_hits) / len(exact_hits)
    print(f"Exact cache hits: {len(exact_hits)} - Average: {avg_exact:.2f}s")

if smart_hits:
    avg_smart = sum(r['duration'] for r in smart_hits) / len(smart_hits)
    cache_smart_hits = len([r for r in smart_hits if r['duration'] < 2.5])
    print(f"Smart variations: {len(smart_hits)} - Average: {avg_smart:.2f}s - Cache hits: {cache_smart_hits}")

if semantic_hits:
    avg_semantic = sum(r['duration'] for r in semantic_hits) / len(semantic_hits)
    cache_semantic_hits = len([r for r in semantic_hits if r['duration'] < 2.5])
    print(f"Semantic variations: {len(semantic_hits)} - Average: {avg_semantic:.2f}s - Cache hits: {cache_semantic_hits}")

overall_cache_hits = len([r for r in results if r['duration'] < 2.5])
print(f"\nTotal cache hits: {overall_cache_hits}/{len(results)} ({overall_cache_hits/len(results)*100:.1f}%)")
print(f"Overall average: {sum(r['duration'] for r in results)/len(results):.2f}s")

# Save results
with open('ultimate_test_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to ultimate_test_results.json")