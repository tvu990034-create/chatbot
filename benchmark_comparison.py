"""
Benchmark Comparison: Baseline vs Optimized
Tests the current system with different optimization settings
"""

import time
import asyncio
import json
from typing import Dict, List
import statistics

# Test queries for benchmarking
TEST_QUERIES = [
    "What is the capital of France?",
    "Explain photosynthesis in simple terms",
    "What is 2 + 2?",
    "Who wrote Romeo and Juliet?",
    "What is the largest planet in our solar system?",
    "How do you make a simple cake?",
    "What is the meaning of life?",
    "Explain the concept of gravity",
    "What is Python programming language?",
    "How do you boil an egg?",
]

class BenchmarkRunner:
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
        self.results = {
            "baseline": {"times": [], "cache_hits": 0},
            "cache_enabled": {"times": [], "cache_hits": 0},
            "speed_mode": {"times": [], "cache_hits": 0},
            "quality_mode": {"times": [], "cache_hits": 0},
        }
    
    async def make_request(self, query: str, use_cache: bool, performance_mode: str) -> Dict:
        """Make a chat request and return timing and cache info"""
        import aiohttp
        
        payload = {
            "messages": [{"role": "user", "content": query}],
            "model": "ollama/phi3:mini",
            "provider": "ollama",
            "use_cache": use_cache,
            "performance_mode": performance_mode
        }
        
        start_time = time.time()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/api/v1/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                data = await response.json()
                elapsed = time.time() - start_time
                return {
                    "duration": elapsed,
                    "cache_hit": data.get("cache_hit", False),
                    "response": data.get("response", "")
                }
    
    async def run_baseline(self):
        """Run baseline benchmark (no cache, balanced mode)"""
        print("Running baseline benchmark...")
        for query in TEST_QUERIES:
            result = await self.make_request(query, use_cache=False, performance_mode="balanced")
            self.results["baseline"]["times"].append(result["duration"])
            print(f"  Query: {query[:30]}... - {result['duration']:.2f}s")
    
    async def run_cache_enabled(self):
        """Run cache-enabled benchmark"""
        print("Running cache-enabled benchmark...")
        # First pass to populate cache
        for query in TEST_QUERIES:
            await self.make_request(query, use_cache=True, performance_mode="balanced")
        
        # Second pass to measure cache hits
        for query in TEST_QUERIES:
            result = await self.make_request(query, use_cache=True, performance_mode="balanced")
            self.results["cache_enabled"]["times"].append(result["duration"])
            if result["cache_hit"]:
                self.results["cache_enabled"]["cache_hits"] += 1
            print(f"  Query: {query[:30]}... - {result['duration']:.2f}s (cache: {result['cache_hit']})")
    
    async def run_speed_mode(self):
        """Run speed mode benchmark"""
        print("Running speed mode benchmark...")
        for query in TEST_QUERIES:
            result = await self.make_request(query, use_cache=False, performance_mode="speed")
            self.results["speed_mode"]["times"].append(result["duration"])
            print(f"  Query: {query[:30]}... - {result['duration']:.2f}s")
    
    async def run_quality_mode(self):
        """Run quality mode benchmark"""
        print("Running quality mode benchmark...")
        for query in TEST_QUERIES:
            result = await self.make_request(query, use_cache=False, performance_mode="quality")
            self.results["quality_mode"]["times"].append(result["duration"])
            print(f"  Query: {query[:30]}... - {result['duration']:.2f}s")
    
    def calculate_stats(self, times: List[float]) -> Dict:
        """Calculate statistics for a list of times"""
        if not times:
            return {}
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "std_dev": statistics.stdev(times) if len(times) > 1 else 0
        }
    
    def generate_report(self) -> str:
        """Generate a comprehensive benchmark report"""
        report = []
        report.append("# BENCHMARK COMPARISON REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"API URL: {self.api_url}")
        report.append(f"Test Queries: {len(TEST_QUERIES)}")
        report.append("")
        
        # Calculate statistics for each mode
        stats = {}
        for mode, data in self.results.items():
            stats[mode] = self.calculate_stats(data["times"])
        
        # Baseline stats
        baseline_mean = stats["baseline"]["mean"]
        
        report.append("## PERFORMANCE COMPARISON")
        report.append("")
        report.append("| Mode | Mean Time (s) | Median Time (s) | Min Time (s) | Max Time (s) | Speedup vs Baseline |")
        report.append("|------|---------------|-----------------|--------------|--------------|---------------------|")
        
        for mode in ["baseline", "cache_enabled", "speed_mode", "quality_mode"]:
            s = stats[mode]
            speedup = baseline_mean / s["mean"] if s["mean"] > 0 else 0
            speedup_str = f"{speedup:.2f}x" if mode != "baseline" else "1.00x"
            cache_info = f" (hits: {self.results[mode]['cache_hits']})" if mode == "cache_enabled" else ""
            report.append(f"| {mode.replace('_', ' ').title()}{cache_info} | {s['mean']:.2f} | {s['median']:.2f} | {s['min']:.2f} | {s['max']:.2f} | {speedup_str} |")
        
        report.append("")
        report.append("## KEY FINDINGS")
        report.append("")
        
        # Cache performance
        cache_speedup = baseline_mean / stats["cache_enabled"]["mean"]
        cache_hit_rate = (self.results["cache_enabled"]["cache_hits"] / len(TEST_QUERIES)) * 100
        report.append(f"### Cache Optimization")
        report.append(f"- **Speedup**: {cache_speedup:.2f}x faster than baseline")
        report.append(f"- **Cache Hit Rate**: {cache_hit_rate:.1f}%")
        report.append(f"- **Time Savings**: {baseline_mean - stats['cache_enabled']['mean']:.2f}s per query")
        report.append("")
        
        # Speed mode performance
        speed_speedup = baseline_mean / stats["speed_mode"]["mean"]
        report.append(f"### Speed Mode")
        report.append(f"- **Speedup**: {speed_speedup:.2f}x faster than baseline")
        report.append(f"- **Time Savings**: {baseline_mean - stats['speed_mode']['mean']:.2f}s per query")
        report.append("")
        
        # Quality mode performance
        quality_speedup = baseline_mean / stats["quality_mode"]["mean"]
        report.append(f"### Quality Mode")
        report.append(f"- **Speedup**: {quality_speedup:.2f}x vs baseline")
        report.append(f"- **Trade-off**: Higher quality output, potentially slower")
        report.append("")
        
        # Overall assessment
        report.append("## OVERALL ASSESSMENT")
        report.append("")
        if cache_speedup > 1.5:
            report.append("✅ **Cache optimization is highly effective** (>1.5x speedup)")
        elif cache_speedup > 1.2:
            report.append("✅ **Cache optimization is effective** (>1.2x speedup)")
        else:
            report.append("⚠️ **Cache optimization needs improvement** (<1.2x speedup)")
        
        if speed_speedup > 1.2:
            report.append("✅ **Speed mode provides significant speedup** (>1.2x)")
        else:
            report.append("⚠️ **Speed mode speedup is modest** (<1.2x)")
        
        report.append("")
        report.append("## RECOMMENDATIONS")
        report.append("")
        report.append("1. **Enable cache by default** for frequently asked questions")
        report.append("2. **Use speed mode** for time-sensitive applications")
        report.append("3. **Use quality mode** for complex reasoning tasks")
        report.append("4. **Monitor cache hit rates** to optimize cache strategies")
        
        return "\n".join(report)

async def main():
    """Main benchmark execution"""
    runner = BenchmarkRunner()
    
    print("Starting Benchmark Comparison...")
    print("=" * 60)
    
    # Run all benchmarks
    await runner.run_baseline()
    print()
    await runner.run_cache_enabled()
    print()
    await runner.run_speed_mode()
    print()
    await runner.run_quality_mode()
    
    # Generate and save report
    report = runner.generate_report()
    print("\n" + "=" * 60)
    print(report)
    
    # Save to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_comparison_{timestamp}.md"
    with open(filename, "w") as f:
        f.write(report)
    
    print(f"\nReport saved to: {filename}")

if __name__ == "__main__":
    # Check if aiohttp is available
    try:
        import aiohttp
    except ImportError:
        print("Installing aiohttp...")
        import subprocess
        subprocess.check_call(["pip", "install", "aiohttp"])
        import aiohttp
    
    asyncio.run(main())