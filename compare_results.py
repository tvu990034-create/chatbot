import json
import time

# Previous test results (from earlier)
previous_results = {
    "cache_test": {
        "first_query": 2.07,
        "cache_hit": 2.04,
        "intelligence_test": 6.90
    },
    "comprehensive_benchmarks": {
        "average_duration": 9.90,
        "total_benchmarks": 26
    }
}

print("PERFORMANCE COMPARISON")
print("=" * 60)
print("PREVIOUS RESULTS:")
print(f"  Cache miss: {previous_results['cache_test']['first_query']:.2f}s")
print(f"  Cache hit: {previous_results['cache_test']['cache_hit']:.2f}s")
print(f"  Intelligence test: {previous_results['cache_test']['intelligence_test']:.2f}s")
print(f"  Comprehensive avg: {previous_results['comprehensive_benchmarks']['average_duration']:.2f}s")

print("\n" + "=" * 60)
print("OPTIMIZATIONS APPLIED:")
print("  1. Reduced max_tokens for speed (350 -> 120 for gemma2:2b)")
print("  2. Lowered temperature for focus (0.3 -> 0.1)")
print("  3. Normalized cache keys (removed punctuation, case)")
print("  4. Removed performance mode from cache key")
print("  5. Reduced timeout (30s -> 10s)")
print("  6. Minimal prompt enhancements for speed")
print("  7. Optimized top_p sampling (0.7 -> 0.4)")

print("\n" + "=" * 60)
print("EXPECTED IMPROVEMENTS:")
print("  - Speed: 30-50% faster on cache misses")
print("  - Cache: Better hit rate from normalization")
print("  - Intelligence: Slightly reduced due to speed focus")