"""
Quick Benchmark: Baseline vs Optimized
Fast comparison with fewer queries
"""

import time
import json
import requests

API_URL = "http://localhost:8080"

# Simplified test queries
TEST_QUERIES = [
    "What is 2+2?",
    "Capital of France?",
    "Hello world",
]

def make_request(query: str, use_cache: bool, performance_mode: str) -> dict:
    """Make a chat request"""
    payload = {
        "messages": [{"role": "user", "content": query}],
        "model": "ollama/phi3:mini",
        "provider": "ollama",
        "use_cache": use_cache,
        "performance_mode": performance_mode
    }
    
    start = time.time()
    response = requests.post(f"{API_URL}/api/v1/chat", json=payload)
    elapsed = time.time() - start
    data = response.json()
    
    return {
        "duration": elapsed,
        "cache_hit": data.get("cache_hit", False),
        "response": data.get("response", "")
    }

def run_benchmark():
    """Run quick benchmark comparison"""
    print("Quick Benchmark Comparison")
    print("=" * 50)
    
    results = {
        "baseline_no_cache": [],
        "cache_enabled": [],
        "speed_mode": [],
    }
    
    # Baseline (no cache, balanced)
    print("\n1. Baseline (no cache, balanced mode):")
    for query in TEST_QUERIES:
        result = make_request(query, use_cache=False, performance_mode="balanced")
        results["baseline_no_cache"].append(result["duration"])
        print(f"   {query[:20]}...: {result['duration']:.2f}s")
    
    # Cache enabled (first pass to populate, second to measure)
    print("\n2. Cache enabled (populating cache...):")
    for query in TEST_QUERIES:
        make_request(query, use_cache=True, performance_mode="balanced")
    
    print("\n3. Cache enabled (measuring cache hits):")
    cache_hits = 0
    for query in TEST_QUERIES:
        result = make_request(query, use_cache=True, performance_mode="balanced")
        results["cache_enabled"].append(result["duration"])
        if result["cache_hit"]:
            cache_hits += 1
        print(f"   {query[:20]}...: {result['duration']:.2f}s (cache: {result['cache_hit']})")
    
    # Speed mode
    print("\n4. Speed mode (no cache):")
    for query in TEST_QUERIES:
        result = make_request(query, use_cache=False, performance_mode="speed")
        results["speed_mode"].append(result["duration"])
        print(f"   {query[:20]}...: {result['duration']:.2f}s")
    
    # Calculate statistics
    baseline_avg = sum(results["baseline_no_cache"]) / len(results["baseline_no_cache"])
    cache_avg = sum(results["cache_enabled"]) / len(results["cache_enabled"])
    speed_avg = sum(results["speed_mode"]) / len(results["speed_mode"])
    
    cache_speedup = baseline_avg / cache_avg
    speed_speedup = baseline_avg / speed_avg
    cache_hit_rate = (cache_hits / len(TEST_QUERIES)) * 100
    
    # Generate report
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"\nBaseline (no cache):     {baseline_avg:.2f}s avg")
    print(f"Cache enabled:          {cache_avg:.2f}s avg")
    print(f"Speed mode:             {speed_avg:.2f}s avg")
    print(f"\nCache Speedup:          {cache_speedup:.2f}x")
    print(f"Speed Mode Speedup:      {speed_speedup:.2f}x")
    print(f"Cache Hit Rate:          {cache_hit_rate:.1f}%")
    
    print("\n" + "=" * 50)
    print("CONCLUSIONS")
    print("=" * 50)
    
    if cache_speedup > 1.5:
        print(f"✅ Cache optimization is EXCELLENT ({cache_speedup:.2f}x speedup)")
    elif cache_speedup > 1.2:
        print(f"✅ Cache optimization is GOOD ({cache_speedup:.2f}x speedup)")
    else:
        print(f"⚠️ Cache optimization needs improvement ({cache_speedup:.2f}x speedup)")
    
    if speed_speedup > 1.2:
        print(f"✅ Speed mode is EFFECTIVE ({speed_speedup:.2f}x speedup)")
    else:
        print(f"⚠️ Speed mode shows modest improvement ({speed_speedup:.2f}x speedup)")
    
    # Save results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report = f"""# Quick Benchmark Report
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Results
- Baseline (no cache): {baseline_avg:.2f}s avg
- Cache enabled: {cache_avg:.2f}s avg  
- Speed mode: {speed_avg:.2f}s avg

## Performance
- Cache Speedup: {cache_speedup:.2f}x
- Speed Mode Speedup: {speed_speedup:.2f}x
- Cache Hit Rate: {cache_hit_rate:.1f}%

## Detailed Results
{json.dumps(results, indent=2)}
"""
    
    filename = f"quick_benchmark_{timestamp}.md"
    with open(filename, "w") as f:
        f.write(report)
    
    print(f"\nReport saved to: {filename}")

if __name__ == "__main__":
    try:
        run_benchmark()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the API server is running on http://localhost:8080")