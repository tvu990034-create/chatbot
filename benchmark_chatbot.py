"""
Custom Benchmark Script for Chatbot API
Tests latency, throughput, and concurrency performance
"""
import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@dataclass
class BenchmarkResult:
    """Stores individual request benchmark results"""
    request_id: int
    success: bool
    latency_ms: float
    tokens_generated: int
    time_to_first_token_ms: float
    error_message: str = ""


@dataclass
class BenchmarkSummary:
    """Summary of benchmark results"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_tokens_per_second: float
    avg_time_to_first_token_ms: float
    requests_per_second: float
    total_duration_seconds: float


class ChatbotBenchmark:
    """Benchmark client for chatbot API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.test_questions = [
            "What is machine learning?",
            "How does KV cache work?",
            "Explain quantization in AI",
            "What is attention mechanism?",
            "Difference between CNN and RNN?",
            "What is transfer learning?",
            "How does gradient descent work?",
            "What is overfitting?",
            "Explain neural network layers",
            "What is backpropagation?"
        ]
    
    async def single_request(self, session: aiohttp.ClientSession, 
                           message: str, session_id: str, 
                           request_id: int) -> BenchmarkResult:
        """Perform a single chat request and measure performance"""
        start_time = time.time()
        ttft = None
        tokens_generated = 0
        
        try:
            payload = {
                "message": message,
                "session_id": session_id,
                "enable_reasoning": True
            }
            
            async with session.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=self.timeout
            ) as response:
                if response.status != 200:
                    return BenchmarkResult(
                        request_id=request_id,
                        success=False,
                        latency_ms=0,
                        tokens_generated=0,
                        time_to_first_token_ms=0,
                        error_message=f"HTTP {response.status}"
                    )
                
                # Process streaming response
                first_token_time = None
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data: "):
                        data = json.loads(line_str[6:])
                        
                        # Record time to first token
                        if first_token_time is None and data.get("type") == "token":
                            first_token_time = time.time()
                            ttft = (first_token_time - start_time) * 1000
                        
                        # Count tokens
                        if data.get("type") == "token":
                            tokens_generated += 1
                        
                        # Check for completion
                        if data.get("type") == "done":
                            break
                
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                return BenchmarkResult(
                    request_id=request_id,
                    success=True,
                    latency_ms=latency_ms,
                    tokens_generated=tokens_generated,
                    time_to_first_token_ms=ttft or 0,
                    error_message=""
                )
                
        except asyncio.TimeoutError:
            return BenchmarkResult(
                request_id=request_id,
                success=False,
                latency_ms=0,
                tokens_generated=0,
                time_to_first_token_ms=0,
                error_message="Timeout"
            )
        except Exception as e:
            return BenchmarkResult(
                request_id=request_id,
                success=False,
                latency_ms=0,
                tokens_generated=0,
                time_to_first_token_ms=0,
                error_message=str(e)
            )
    
    async def run_concurrent_benchmark(self, num_concurrent: int, 
                                      num_requests: int) -> BenchmarkSummary:
        """Run benchmark with concurrent requests"""
        print(f"Starting benchmark: {num_concurrent} concurrent users, {num_requests} total requests")
        
        results: List[BenchmarkResult] = []
        start_time = time.time()
        
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            tasks = []
            for i in range(num_requests):
                message = self.test_questions[i % len(self.test_questions)]
                session_id = f"benchmark_user_{i % num_concurrent}"
                task = self.single_request(session, message, session_id, i)
                tasks.append(task)
            
            # Execute all requests concurrently
            results = await asyncio.gather(*tasks)
        
        total_duration = time.time() - start_time
        return self._analyze_results(results, total_duration)
    
    def _analyze_results(self, results: List[BenchmarkResult], 
                        total_duration: float) -> BenchmarkSummary:
        """Analyze benchmark results and create summary"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        if not successful:
            print("No successful requests!")
            return BenchmarkSummary(
                total_requests=len(results),
                successful_requests=0,
                failed_requests=len(failed),
                avg_latency_ms=0,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                avg_tokens_per_second=0,
                avg_time_to_first_token_ms=0,
                requests_per_second=0,
                total_duration_seconds=total_duration
            )
        
        latencies = [r.latency_ms for r in successful]
        ttfts = [r.time_to_first_token_ms for r in successful if r.time_to_first_token_ms > 0]
        
        # Calculate tokens per second for each request
        tokens_per_sec = []
        for r in successful:
            if r.latency_ms > 0:
                tokens_per_sec.append((r.tokens_generated / r.latency_ms) * 1000)
        
        return BenchmarkSummary(
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            avg_latency_ms=statistics.mean(latencies),
            p50_latency_ms=statistics.median(latencies),
            p95_latency_ms=percentile(latencies, 95),
            p99_latency_ms=percentile(latencies, 99),
            avg_tokens_per_second=statistics.mean(tokens_per_sec) if tokens_per_sec else 0,
            avg_time_to_first_token_ms=statistics.mean(ttfts) if ttfts else 0,
            requests_per_second=len(successful) / total_duration,
            total_duration_seconds=total_duration
        )


def percentile(data: List[float], p: int) -> float:
    """Calculate percentile"""
    sorted_data = sorted(data)
    index = int((p / 100) * len(sorted_data))
    return sorted_data[min(index, len(sorted_data) - 1)]


def print_summary(summary: BenchmarkSummary, details: bool = True):
    """Print benchmark summary"""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"Total Requests: {summary.total_requests}")
    print(f"Successful: {summary.successful_requests}")
    print(f"Failed: {summary.failed_requests}")
    print(f"Success Rate: {(summary.successful_requests/summary.total_requests*100):.2f}%")
    print(f"Total Duration: {summary.total_duration_seconds:.2f}s")
    print(f"Requests/Second: {summary.requests_per_second:.2f}")
    print()
    print("Latency Metrics:")
    print(f"  Average: {summary.avg_latency_ms:.2f}ms")
    print(f"  P50: {summary.p50_latency_ms:.2f}ms")
    print(f"  P95: {summary.p95_latency_ms:.2f}ms")
    print(f"  P99: {summary.p99_latency_ms:.2f}ms")
    print()
    print("Performance Metrics:")
    print(f"  Avg Tokens/Second: {summary.avg_tokens_per_second:.2f}")
    print(f"  Avg Time to First Token: {summary.avg_time_to_first_token_ms:.2f}ms")
    print("="*60)


async def main():
    parser = argparse.ArgumentParser(description="Benchmark chatbot API performance")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Chatbot API base URL")
    parser.add_argument("--concurrent", type=int, default=10,
                       help="Number of concurrent users")
    parser.add_argument("--requests", type=int, default=100,
                       help="Total number of requests")
    parser.add_argument("--timeout", type=int, default=120,
                       help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    print(f"Chatbot Benchmark Tool")
    print(f"Target URL: {args.url}")
    print(f"Concurrent Users: {args.concurrent}")
    print(f"Total Requests: {args.requests}")
    print(f"Timeout: {args.timeout}s")
    
    # Check if server is running
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                if resp.status == 200:
                    print("[OK] Server is running and healthy")
                else:
                    print(f"[ERROR] Server health check failed: HTTP {resp.status}")
                    return
    except Exception as e:
        print(f"[ERROR] Cannot connect to server: {e}")
        print("Please start the chatbot server first:")
        print("  uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload")
        return
    
    benchmark = ChatbotBenchmark(base_url=args.url, timeout=args.timeout)
    summary = await benchmark.run_concurrent_benchmark(args.concurrent, args.requests)
    print_summary(summary)
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"benchmark_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "config": {
                "url": args.url,
                "concurrent_users": args.concurrent,
                "total_requests": args.requests,
                "timeout": args.timeout
            },
            "summary": {
                "total_requests": summary.total_requests,
                "successful_requests": summary.successful_requests,
                "failed_requests": summary.failed_requests,
                "avg_latency_ms": summary.avg_latency_ms,
                "p50_latency_ms": summary.p50_latency_ms,
                "p95_latency_ms": summary.p95_latency_ms,
                "p99_latency_ms": summary.p99_latency_ms,
                "avg_tokens_per_second": summary.avg_tokens_per_second,
                "avg_time_to_first_token_ms": summary.avg_time_to_first_token_ms,
                "requests_per_second": summary.requests_per_second,
                "total_duration_seconds": summary.total_duration_seconds
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    asyncio.run(main())