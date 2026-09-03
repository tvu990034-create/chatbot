"""
Speed and Intelligence Validation Test
Tests the maximum speed and intelligence improvements
"""

import requests
import time
import json
from typing import Dict, List, Any
import sys
import io

# Fix unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

class SpeedIntelligenceValidator:
    """Validate speed and intelligence improvements."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    def test_ultra_fast_responses(self) -> Dict[str, Any]:
        """Test ultra-fast responses for common queries."""
        print("=== Testing Ultra-Fast Responses ===")
        
        common_queries = [
            "hello",
            "hi",
            "yes",
            "2+2",
            "what is ai"
        ]
        
        results = []
        for query in common_queries:
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/chat",
                    json={"messages": [{"role": "user", "content": query}]},
                    timeout=10
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    results.append({
                        "query": query,
                        "duration": duration,
                        "response_length": len(data.get("response", "")),
                        "cache_hit": data.get("cache_hit", False)
                    })
                    print(f"✅ {query}: {duration:.3f}s (cache: {data.get('cache_hit', False)})")
                else:
                    print(f"❌ {query}: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {query}: {str(e)}")
        
        avg_duration = sum(r["duration"] for r in results) / len(results) if results else 0
        cache_hits = sum(1 for r in results if r["cache_hit"])
        
        return {
            "test": "ultra_fast_responses",
            "total_queries": len(common_queries),
            "successful": len(results),
            "avg_duration": avg_duration,
            "cache_hits": cache_hits,
            "cache_hit_rate": cache_hits / len(results) if results else 0,
            "details": results
        }
    
    def test_intelligent_responses(self) -> Dict[str, Any]:
        """Test intelligent responses for complex queries."""
        print("\n=== Testing Intelligent Responses ===")
        
        intelligent_queries = [
            "write a python function to add two numbers",
            "calculate 15 * 23",
            "explain how machine learning works",
            "what is the difference between python and javascript"
        ]
        
        results = []
        for query in intelligent_queries:
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/chat",
                    json={"messages": [{"role": "user", "content": query}]},
                    timeout=10
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get("response", "")
                    results.append({
                        "query": query,
                        "duration": duration,
                        "response_length": len(response_text),
                        "has_code": "```" in response_text or "def " in response_text,
                        "has_structure": len(response_text.split('\n')) > 2
                    })
                    print(f"✅ {query}: {duration:.3f}s (length: {len(response_text)} chars)")
                else:
                    print(f"❌ {query}: HTTP {response.status_code}")
            except Exception as e:
                print(f"❌ {query}: {str(e)}")
        
        avg_duration = sum(r["duration"] for r in results) / len(results) if results else 0
        intelligent_responses = sum(1 for r in results if r["has_code"] or r["has_structure"])
        
        return {
            "test": "intelligent_responses",
            "total_queries": len(intelligent_queries),
            "successful": len(results),
            "avg_duration": avg_duration,
            "intelligent_responses": intelligent_responses,
            "intelligence_rate": intelligent_responses / len(results) if results else 0,
            "details": results
        }
    
    def test_cache_performance(self) -> Dict[str, Any]:
        """Test cache performance with repeated queries."""
        print("\n=== Testing Cache Performance ===")
        
        test_query = "what is artificial intelligence"
        
        # First request (cache miss)
        start_time = time.time()
        response1 = requests.post(
            f"{self.base_url}/api/v1/chat",
            json={"messages": [{"role": "user", "content": test_query}]},
            timeout=10
        )
        duration1 = time.time() - start_time
        
        # Second request (should be cache hit)
        start_time = time.time()
        response2 = requests.post(
            f"{self.base_url}/api/v1/chat",
            json={"messages": [{"role": "user", "content": test_query}]},
            timeout=10
        )
        duration2 = time.time() - start_time
        
        # Third request (should be cache hit)
        start_time = time.time()
        response3 = requests.post(
            f"{self.base_url}/api/v1/chat",
            json={"messages": [{"role": "user", "content": test_query}]},
            timeout=10
        )
        duration3 = time.time() - start_time
        
        data1 = response1.json() if response1.status_code == 200 else {}
        data2 = response2.json() if response2.status_code == 200 else {}
        data3 = response3.json() if response3.status_code == 200 else {}
        
        cache_hits = sum(1 for d in [data1, data2, data3] if d.get("cache_hit", False))
        cache_hit_rate = cache_hits / 3
        
        print(f"First request: {duration1:.3f}s (cache: {data1.get('cache_hit', False)})")
        print(f"Second request: {duration2:.3f}s (cache: {data2.get('cache_hit', False)})")
        print(f"Third request: {duration3:.3f}s (cache: {data3.get('cache_hit', False)})")
        print(f"Cache hit rate: {cache_hit_rate:.1%}")
        
        return {
            "test": "cache_performance",
            "first_request_duration": duration1,
            "second_request_duration": duration2,
            "third_request_duration": duration3,
            "cache_hit_rate": cache_hit_rate,
            "speed_improvement": duration1 / duration2 if duration2 > 0 else 0
        }
    
    def test_concurrent_performance(self) -> Dict[str, Any]:
        """Test concurrent request performance."""
        print("\n=== Testing Concurrent Performance ===")
        
        import concurrent.futures
        
        def make_request(query):
            start_time = time.time()
            try:
                response = requests.post(
                    f"{self.base_url}/api/v1/chat",
                    json={"messages": [{"role": "user", "content": query}]},
                    timeout=10
                )
                duration = time.time() - start_time
                return {
                    "query": query,
                    "duration": duration,
                    "success": response.status_code == 200
                }
            except Exception as e:
                return {
                    "query": query,
                    "duration": time.time() - start_time,
                    "success": False,
                    "error": str(e)
                }
        
        queries = [f"query {i}" for i in range(5)]
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, query) for query in queries]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_duration = time.time() - start_time
        
        successful = sum(1 for r in results if r["success"])
        avg_duration = sum(r["duration"] for r in results) / len(results) if results else 0
        
        print(f"Concurrent requests: {len(queries)}")
        print(f"Successful: {successful}/{len(queries)}")
        print(f"Total duration: {total_duration:.3f}s")
        print(f"Average duration: {avg_duration:.3f}s")
        
        return {
            "test": "concurrent_performance",
            "concurrent_requests": len(queries),
            "successful": successful,
            "total_duration": total_duration,
            "avg_duration": avg_duration,
            "details": results
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all speed and intelligence tests."""
        print("=" * 60)
        print("SPEED AND INTELLIGENCE VALIDATION")
        print("=" * 60)
        
        tests = [
            self.test_ultra_fast_responses,
            self.test_intelligent_responses,
            self.test_cache_performance,
            self.test_concurrent_performance
        ]
        
        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
            except Exception as e:
                print(f"❌ Test failed: {str(e)}")
                results.append({"test": test.__name__, "error": str(e)})
        
        # Overall summary
        print("\n" + "=" * 60)
        print("OVERALL SUMMARY")
        print("=" * 60)
        
        successful_tests = sum(1 for r in results if "error" not in r)
        print(f"Tests passed: {successful_tests}/{len(tests)}")
        
        # Key metrics
        for result in results:
            if "error" not in result:
                test_name = result["test"]
                if test_name == "ultra_fast_responses":
                    print(f"Ultra-fast avg: {result['avg_duration']:.3f}s, cache hit rate: {result['cache_hit_rate']:.1%}")
                elif test_name == "intelligent_responses":
                    print(f"Intelligent avg: {result['avg_duration']:.3f}s, intelligence rate: {result['intelligence_rate']:.1%}")
                elif test_name == "cache_performance":
                    print(f"Cache hit rate: {result['cache_hit_rate']:.1%}, speed improvement: {result['speed_improvement']:.1f}x")
                elif test_name == "concurrent_performance":
                    print(f"Concurrent success: {result['successful']}/{result['concurrent_requests']}, avg: {result['avg_duration']:.3f}s")
        
        return {
            "total_tests": len(tests),
            "successful_tests": successful_tests,
            "results": results,
            "timestamp": time.time()
        }

def main():
    """Main validation test."""
    validator = SpeedIntelligenceValidator()
    results = validator.run_all_tests()
    
    # Save results
    with open("speed_intelligence_validation.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: speed_intelligence_validation.json")

if __name__ == "__main__":
    main()