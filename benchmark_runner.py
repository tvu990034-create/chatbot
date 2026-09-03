"""
Benchmark Runner for Local Chatbot API
Automated benchmarking with accuracy calculation and data export
"""

import requests
import json
import time
import csv
import sys
import io
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# Fix unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

class BenchmarkRunner:
    """Automated benchmark runner for local chatbot API."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []
        self.errors = []
    
    def run_single_question(
        self, 
        question: str, 
        ground_truth: Optional[str] = None,
        subject: Optional[str] = None,
        model: str = "gemma2:2b",
        use_cache: bool = False,
        performance_mode: str = "baseline"
    ) -> Dict[str, Any]:
        """Run a single benchmark question."""
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/chat",
                json={
                    "messages": [{"role": "user", "content": question}],
                    "model": model,
                    "provider": "ollama",
                    "use_cache": use_cache,
                    "performance_mode": performance_mode
                },
                timeout=30
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                result = {
                    "question": question,
                    "ground_truth": ground_truth,
                    "subject": subject,
                    "model": model,
                    "response": data.get("response", ""),
                    "duration": duration,
                    "cache_hit": data.get("cache_hit", False),
                    "optimizations_applied": data.get("optimizations_applied", 0),
                    "performance_mode": performance_mode,
                    "timestamp": datetime.now().isoformat(),
                    "success": True,
                    "extracted_answer": self._extract_answer(data.get("response", "")),
                    "is_correct": self._check_correct(data.get("response", ""), ground_truth) if ground_truth else None
                }
                
                self.results.append(result)
                return result
            else:
                error = {
                    "question": question,
                    "error": f"HTTP {response.status_code}",
                    "timestamp": datetime.now().isoformat(),
                    "success": False
                }
                self.errors.append(error)
                return error
                
        except Exception as e:
            error = {
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            self.errors.append(error)
            return error
    
    def run_benchmark(
        self, 
        questions: List[Dict[str, str]],
        model: str = "gemma2:2b",
        use_cache: bool = False,
        performance_mode: str = "baseline"
    ) -> Dict[str, Any]:
        """Run a complete benchmark on a list of questions."""
        print(f"Running benchmark with {len(questions)} questions...")
        print(f"Model: {model}, Cache: {use_cache}, Mode: {performance_mode}")
        
        for i, item in enumerate(questions, 1):
            question = item.get("question", "")
            ground_truth = item.get("ground_truth")
            subject = item.get("subject")
            
            print(f"Question {i}/{len(questions)}: {question[:50]}...")
            
            result = self.run_single_question(
                question=question,
                ground_truth=ground_truth,
                subject=subject,
                model=model,
                use_cache=use_cache,
                performance_mode=performance_mode
            )
            
            if result.get("success"):
                print(f"  [OK] Success ({result['duration']:.2f}s)")
            else:
                print(f"  [ERROR] Error: {result.get('error')}")
        
        return self.calculate_statistics()
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """Calculate benchmark statistics."""
        successful = [r for r in self.results if r.get("success")]
        failed = [r for r in self.results if not r.get("success")]
        
        with_truth = [r for r in successful if r.get("ground_truth")]
        correct = [r for r in with_truth if r.get("is_correct")]
        
        total_duration = sum(r.get("duration", 0) for r in successful)
        avg_duration = total_duration / len(successful) if successful else 0
        
        cache_hits = sum(1 for r in successful if r.get("cache_hit"))
        cache_hit_rate = cache_hits / len(successful) if successful else 0
        
        # Calculate by subject
        subject_stats = {}
        for result in successful:
            subject = result.get("subject", "general")
            if subject not in subject_stats:
                subject_stats[subject] = {"total": 0, "correct": 0}
            subject_stats[subject]["total"] += 1
            if result.get("is_correct"):
                subject_stats[subject]["correct"] += 1
        
        for subject in subject_stats:
            subject_stats[subject]["accuracy"] = subject_stats[subject]["correct"] / subject_stats[subject]["total"]
        
        return {
            "total_questions": len(self.results),
            "successful": len(successful),
            "failed": len(failed),
            "accuracy": len(correct) / len(with_truth) if with_truth else None,
            "total_correct": len(correct),
            "total_with_truth": len(with_truth),
            "avg_duration": avg_duration,
            "cache_hit_rate": cache_hit_rate,
            "subject_accuracy": subject_stats,
            "errors": len(self.errors)
        }
    
    def _extract_answer(self, response: str) -> str:
        """Extract answer from response (simple implementation)."""
        # Simple extraction - can be enhanced based on expected format
        # For multiple choice, look for patterns like "A) answer", "B) answer", etc.
        import re
        
        # Try to find multiple choice answer
        mc_pattern = r'([A-D])\)'
        matches = re.findall(mc_pattern, response)
        if matches:
            return matches[0]  # Return the letter
        
        # Otherwise return first few words as extracted answer
        words = response.split()
        return ' '.join(words[:5]) if words else response[:50]
    
    def _check_correct(self, response: str, ground_truth: str) -> bool:
        """Check if response matches ground truth."""
        # Simple exact match - can be enhanced for fuzzy matching
        return ground_truth.lower() in response.lower()
    
    def export_json(self, filename: str = "benchmark_results.json"):
        """Export results to JSON."""
        export_data = {
            "benchmark_info": {
                "timestamp": datetime.now().isoformat(),
                "api_url": self.api_url,
                "total_questions": len(self.results) + len(self.errors)
            },
            "statistics": self.calculate_statistics(),
            "results": self.results,
            "errors": self.errors
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Results exported to {filename}")
    
    def export_csv(self, filename: str = "benchmark_results.csv"):
        """Export results to CSV."""
        if not self.results:
            print("No results to export")
            return
        
        fieldnames = ["question", "ground_truth", "subject", "response", "extracted_answer", 
                     "is_correct", "duration", "cache_hit", "model", "performance_mode", "timestamp"]
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                # Handle None values
                row = {k: (v if v is not None else "") for k, v in result.items() if k in fieldnames}
                writer.writerow(row)
        
        print(f"Results exported to {filename}")
    
    def compare_runs(self, baseline_results: List[Dict], optimized_results: List[Dict]) -> Dict[str, Any]:
        """Compare baseline vs optimized results."""
        baseline_correct = sum(1 for r in baseline_results if r.get("is_correct"))
        optimized_correct = sum(1 for r in optimized_results if r.get("is_correct"))
        
        baseline_duration = sum(r.get("duration", 0) for r in baseline_results)
        optimized_duration = sum(r.get("duration", 0) for r in optimized_results)
        
        return {
            "baseline_accuracy": baseline_correct / len(baseline_results) if baseline_results else 0,
            "optimized_accuracy": optimized_correct / len(optimized_results) if optimized_results else 0,
            "accuracy_improvement": (optimized_correct - baseline_correct) / len(baseline_results) if baseline_results else 0,
            "baseline_avg_duration": baseline_duration / len(baseline_results) if baseline_results else 0,
            "optimized_avg_duration": optimized_duration / len(optimized_results) if optimized_results else 0,
            "speed_improvement": (baseline_duration - optimized_duration) / baseline_duration if baseline_duration > 0 else 0
        }

# Example usage
if __name__ == "__main__":
    runner = BenchmarkRunner()
    
    # Sample questions (MMLU-style)
    sample_questions = [
        {
            "question": "What is 2+2?",
            "ground_truth": "4",
            "subject": "math"
        },
        {
            "question": "What is the capital of France?",
            "ground_truth": "Paris",
            "subject": "geography"
        },
        {
            "question": "Write a Python function to add two numbers",
            "ground_truth": None,  # No ground truth for coding questions
            "subject": "coding"
        }
    ]
    
    # Run benchmark
    stats = runner.run_benchmark(sample_questions, model="gemma2:2b", use_cache=False, performance_mode="baseline")
    
    print("\nBenchmark Statistics:")
    print(json.dumps(stats, indent=2))
    
    # Export results
    runner.export_json()
    runner.export_csv()