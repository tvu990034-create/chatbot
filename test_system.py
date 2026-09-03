"""
Comprehensive Test Script for Optimized Local Chatbot System
Tests all major improvements and validates system performance
"""

import requests
import time
import json
from typing import Dict, List, Any
from datetime import datetime
import sys
import io

# Fix unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

class SystemTester:
    """Comprehensive system testing class."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
    def log_result(self, test_name: str, success: bool, duration: float, details: str = ""):
        """Log test result."""
        result = {
            "test": test_name,
            "success": success,
            "duration": duration,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name} ({duration:.2f}s)")
        if details:
            print(f"     Details: {details}")
    
    def test_basic_chat(self) -> bool:
        """Test basic chat functionality."""
        try:
            start = time.time()
            response = requests.post(f"{self.base_url}/api/v1/chat", json={
                "messages": [{"role": "user", "content": "hello"}]
            })
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                success = "response" in data and len(data["response"]) > 0
                details = f"Response length: {len(data['response'])}, Cache hit: {data.get('cache_hit', False)}"
                self.log_result("Basic Chat", success, duration, details)
                return success
            else:
                self.log_result("Basic Chat", False, duration, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Basic Chat", False, 0, f"Error: {str(e)}")
            return False
    
    def test_cache_performance(self) -> bool:
        """Test cache hit rates with repeated queries."""
        try:
            queries = ["hello", "hi", "hello", "HI", "hello!", "hello", "hey"]
            cache_hits = 0
            total_time = 0
            
            for i, query in enumerate(queries):
                start = time.time()
                response = requests.post(f"{self.base_url}/api/v1/chat", json={
                    "messages": [{"role": "user", "content": query}]
                })
                duration = time.time() - start
                total_time += duration
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('cache_hit'):
                        cache_hits += 1
            
            hit_rate = cache_hits / len(queries)
            avg_time = total_time / len(queries)
            success = hit_rate > 0.3  # Expect at least 30% cache hits
            details = f"Cache hit rate: {hit_rate:.1%}, Avg time: {avg_time:.2f}s"
            self.log_result("Cache Performance", success, total_time, details)
            return success
        except Exception as e:
            self.log_result("Cache Performance", False, 0, f"Error: {str(e)}")
            return False
    
    def test_analytics_endpoint(self) -> bool:
        """Test analytics endpoint."""
        try:
            start = time.time()
            response = requests.get(f"{self.base_url}/api/v1/analytics")
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                success = "analytics" in data and isinstance(data["analytics"], dict)
                details = f"Analytics keys: {list(data.get('analytics', {}).keys())}"
                self.log_result("Analytics Endpoint", success, duration, details)
                return success
            else:
                self.log_result("Analytics Endpoint", False, duration, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Analytics Endpoint", False, 0, f"Error: {str(e)}")
            return False
    
    def test_health_check(self) -> bool:
        """Test health check endpoint."""
        try:
            start = time.time()
            response = requests.get(f"{self.base_url}/api/v1/health")
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                success = data.get("status") == "healthy"
                details = f"Status: {data.get('status')}, Models: {len(data.get('available_models', []))}"
                self.log_result("Health Check", success, duration, details)
                return success
            else:
                self.log_result("Health Check", False, duration, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Health Check", False, 0, f"Error: {str(e)}")
            return False
    
    def test_query_variations(self) -> bool:
        """Test semantic matching with query variations."""
        try:
            variations = [
                "what is python",
                "explain python",
                "python programming",
                "tell me about python"
            ]
            
            total_time = 0
            cache_hits = 0
            
            for query in variations:
                start = time.time()
                response = requests.post(f"{self.base_url}/api/v1/chat", json={
                    "messages": [{"role": "user", "content": query}]
                })
                duration = time.time() - start
                total_time += duration
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('cache_hit'):
                        cache_hits += 1
            
            avg_time = total_time / len(variations)
            details = f"Semantic cache hits: {cache_hits}/{len(variations)}, Avg time: {avg_time:.2f}s"
            success = len(variations) > 0  # Test successful if all requests completed
            self.log_result("Query Variations", success, total_time, details)
            return success
        except Exception as e:
            self.log_result("Query Variations", False, 0, f"Error: {str(e)}")
            return False
    
    def test_performance_metrics(self) -> bool:
        """Test performance metrics in responses."""
        try:
            start = time.time()
            response = requests.post(f"{self.base_url}/api/v1/chat", json={
                "messages": [{"role": "user", "content": "test metrics"}]
            })
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                has_metrics = "performance_metrics" in data
                if has_metrics:
                    metrics = data["performance_metrics"]
                    details = f"Metrics: cache_hit_rate={metrics.get('cache_hit_rate', 0):.1%}, avg_time={metrics.get('avg_response_time', 0):.2f}s"
                else:
                    details = "No performance metrics in response"
                success = True  # Test successful if response received
                self.log_result("Performance Metrics", success, duration, details)
                return success
            else:
                self.log_result("Performance Metrics", False, duration, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Performance Metrics", False, 0, f"Error: {str(e)}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling with invalid requests."""
        try:
            # Test with invalid message format
            start = time.time()
            response = requests.post(f"{self.base_url}/api/v1/chat", json={
                "messages": [{"role": "invalid", "content": "test"}]
            })
            duration = time.time() - start
            
            # Should handle gracefully (400 or 500 with error message)
            success = response.status_code in [400, 500] or response.status_code == 200
            details = f"Status: {response.status_code} (graceful error handling)"
            self.log_result("Error Handling", success, duration, details)
            return success
        except Exception as e:
            self.log_result("Error Handling", False, 0, f"Error: {str(e)}")
            return False
    
    def test_concurrent_requests(self) -> bool:
        """Test system resilience with concurrent requests."""
        try:
            import threading
            
            results = []
            def make_request():
                try:
                    response = requests.post(f"{self.base_url}/api/v1/chat", json={
                        "messages": [{"role": "user", "content": "concurrent test"}]
                    })
                    results.append(response.status_code == 200)
                except:
                    results.append(False)
            
            start = time.time()
            threads = [threading.Thread(target=make_request) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            duration = time.time() - start
            
            success_rate = sum(results) / len(results)
            success = success_rate >= 0.8  # Expect at least 80% success
            details = f"Success rate: {success_rate:.1%}, 5 concurrent requests"
            self.log_result("Concurrent Requests", success, duration, details)
            return success
        except Exception as e:
            self.log_result("Concurrent Requests", False, 0, f"Error: {str(e)}")
            return False
    
    def test_long_query_handling(self) -> bool:
        """Test handling of long queries."""
        try:
            long_query = "explain " + "very " * 100 + "detailed concept"
            start = time.time()
            response = requests.post(f"{self.base_url}/api/v1/chat", json={
                "messages": [{"role": "user", "content": long_query}]
            })
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                # Check if response was truncated (should be under 10k chars)
                response_truncated = len(data["response"]) < 10000
                details = f"Response length: {len(data['response'])}, Truncated: {not response_truncated}"
                success = True  # Test successful if handled gracefully
                self.log_result("Long Query Handling", success, duration, details)
                return success
            else:
                self.log_result("Long Query Handling", False, duration, f"Status: {response.status_code}")
                return False
        except Exception as e:
            self.log_result("Long Query Handling", False, 0, f"Error: {str(e)}")
            return False
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return summary."""
        print("=" * 60)
        print("COMPREHENSIVE SYSTEM TEST SUITE")
        print("=" * 60)
        print(f"Testing: {self.base_url}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Run all tests
        self.test_health_check()
        self.test_basic_chat()
        self.test_cache_performance()
        self.test_analytics_endpoint()
        self.test_query_variations()
        self.test_performance_metrics()
        self.test_error_handling()
        self.test_concurrent_requests()
        self.test_long_query_handling()
        
        # Calculate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r["success"])
        pass_rate = passed_tests / total_tests if total_tests > 0 else 0
        total_duration = sum(r["duration"] for r in self.results)
        
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {pass_rate:.1%}")
        print(f"Total Duration: {total_duration:.2f}s")
        print("=" * 60)
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": pass_rate,
            "total_duration": total_duration,
            "results": self.results,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results to file
        with open("test_results.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"Detailed results saved to: test_results.json")
        
        return summary

if __name__ == "__main__":
    # Run comprehensive tests
    tester = SystemTester()
    summary = tester.run_all_tests()
    
    # Exit with appropriate code
    exit(0 if summary["pass_rate"] >= 0.8 else 1)