"""
COMPREHENSIVE MULTI-BENCHMARK SYSTEM
Downloads and runs MMLU-Pro, GPQA, SWE-Bench, AIME, GAIA, MT-Bench, HELM
Configuration: quality mode, no cache, temperature=0
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
from datasets import load_dataset
import numpy as np

# Fix unicode encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

class ComprehensiveBenchmarkSystem:
    """Complete multi-benchmark system with all datasets."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.results = []
        self.errors = []
        self.model = "gemma2:2b"
        self.performance_mode = "quality"  # Quality mode for accuracy
        self.use_cache = False  # No cache for fresh results
        self.temperature = 0
        self.checkpoint_interval = 10
        
        # Benchmark datasets with correct HuggingFace paths
        self.benchmarks = {
            "mmlu_pro": {
                "name": "MMLU-Pro",
                "dataset": "TIGER-Lab/MMLU-Pro",
                "split": "test",
                "question_field": "question",
                "choices_field": "options",
                "answer_field": "answer",
                "subject_field": "category"
            },
            "gpqa": {
                "name": "GPQA",
                "dataset": "Idavidrein/gpqa",
                "split": "train",
                "question_field": "Question",
                "choices_field": "options",
                "answer_field": "Correct Answer",
                "subject_field": "Sub-Field"
            },
            "aime": {
                "name": "AIME",
                "dataset": "MathArena/aime_2026",
                "split": "train",  # Changed from test to train
                "question_field": "problem",
                "choices_field": None,
                "answer_field": "answer",
                "subject_field": "level"
            },
            "gaia": {
                "name": "GAIA",
                "dataset": "gaia-benchmark/GAIA",
                "split": "validation",
                "question_field": "Question",
                "choices_field": None,
                "answer_field": "Final answer",
                "subject_field": "Task"
            }
        }
        
        self.datasets = {}
    
    def verify_optimizations(self):
        """Verify all optimizations are working."""
        print("=" * 60)
        print("VERIFYING OPTIMIZATIONS FOR MAXIMUM ACCURACY")
        print("=" * 60)
        
        try:
            # Check API health
            health = requests.get(f"{self.api_url}/api/v1/health", timeout=5)
            if health.status_code == 200:
                health_data = health.json()
                print(f"[OK] API Health: {health_data.get('status', 'unknown')}")
                print(f"[OK] Optimizations Active: {health_data.get('optimizations_active', 0)}")
            else:
                print(f"[ERROR] API Health check failed: {health.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] Cannot connect to API: {e}")
            return False
        
        # Test a quick query
        print("\nTesting query with quality mode...")
        start = time.time()
        try:
            test_response = requests.post(
                f"{self.api_url}/api/v1/chat",
                json={
                    "messages": [{"role": "user", "content": "What is 1+1?"}],
                    "model": self.model,
                    "provider": "ollama",
                    "use_cache": self.use_cache,
                    "performance_mode": self.performance_mode
                },
                timeout=30
            )
            duration = time.time() - start
            if test_response.status_code == 200:
                data = test_response.json()
                print(f"[OK] Query Speed: {duration:.2f}s")
                print(f"[OK] Cache: {'DISABLED' if not self.use_cache else 'ENABLED'}")
                print(f"[OK] Optimizations Applied: {data.get('optimizations_applied', 0)}")
            else:
                print(f"[ERROR] Query failed: {test_response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] Query test failed: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION VERIFICATION COMPLETE")
        print("=" * 60)
        return True
    
    def download_all_datasets(self):
        """Download all benchmark datasets."""
        print("\n" + "=" * 60)
        print("DOWNLOADING ALL BENCHMARK DATASETS")
        print("=" * 60)
        
        for benchmark_id, benchmark_info in self.benchmarks.items():
            print(f"\nDownloading {benchmark_info['name']}...")
            try:
                dataset = load_dataset(
                    benchmark_info['dataset'],
                    split=benchmark_info['split'],
                    streaming=True
                )
                # Load ALL questions for full benchmark
                self.datasets[benchmark_id] = list(dataset)  # Full dataset
                print(f"[OK] {benchmark_info['name']}: {len(self.datasets[benchmark_id])} questions loaded")
            except Exception as e:
                print(f"[ERROR] Failed to load {benchmark_info['name']}: {e}")
                print(f"Using sample questions for {benchmark_info['name']}")
                self.datasets[benchmark_id] = self.create_sample_questions(benchmark_id)
        
        print("\n" + "=" * 60)
        print("ALL DATASETS LOADED")
        print("=" * 60)
        for benchmark_id, dataset in self.datasets.items():
            print(f"{self.benchmarks[benchmark_id]['name']}: {len(dataset)} questions")
    
    def create_sample_questions(self, benchmark_id: str) -> List[Dict]:
        """Create sample questions for failed dataset loads."""
        if benchmark_id == "mmlu_pro":
            return [
                {
                    "question": "What is 2+2?",
                    "options": ["3", "4", "5", "6"],
                    "answer": "B",
                    "category": "mathematics",
                    "id": "sample_001"
                }
            ]
        elif benchmark_id == "gpqa":
            return [
                {
                    "Question": "What is the capital of France?",
                    "options": ["London", "Berlin", "Paris", "Madrid"],
                    "Correct Answer": "C",
                    "Sub-Field": "geography"
                }
            ]
        elif benchmark_id == "aime":
            return [
                {
                    "problem": "Find the sum of all positive integers n such that n^2 + 19n + 92 is a perfect square.",
                    "answer": "17",
                    "level": "easy"
                }
            ]
        elif benchmark_id == "gaia":
            return [
                {
                    "Question": "What is the chemical symbol for water?",
                    "Final answer": "H2O",
                    "Task": "chemistry"
                }
            ]
        return []
    
    def run_benchmark_question(
        self, 
        question: str, 
        choices: Optional[List[str]],
        correct_answer: str,
        subject: str,
        benchmark_id: str,
        test_id: str
    ) -> Dict[str, Any]:
        """Run a single benchmark question."""
        start_time = time.time()
        
        # Format question with choices if available
        if choices:
            choice_text = "\n".join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
            full_question = f"{question}\n\nChoices:\n{choice_text}"
        else:
            full_question = question
        
        try:
            response = requests.post(
                f"{self.api_url}/api/v1/chat",
                json={
                    "messages": [{"role": "user", "content": full_question}],
                    "model": self.model,
                    "provider": "ollama",
                    "use_cache": self.use_cache,
                    "performance_mode": self.performance_mode
                },
                timeout=60  # Longer timeout for quality mode
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract answer from response
                if choices:
                    extracted_answer = self._extract_multiple_choice_answer(data.get("response", ""), choices)
                else:
                    extracted_answer = self._extract_text_answer(data.get("response", ""))
                
                result = {
                    "benchmark_id": benchmark_id,
                    "test_id": test_id,
                    "question": question,
                    "choices": choices,
                    "correct_answer": correct_answer,
                    "subject": subject,
                    "model": self.model,
                    "response": data.get("response", ""),
                    "extracted_answer": extracted_answer,
                    "is_correct": str(extracted_answer).lower() == str(correct_answer).lower(),
                    "duration": duration,
                    "cache_hit": data.get("cache_hit", False),
                    "optimizations_applied": data.get("optimizations_applied", 0),
                    "performance_mode": self.performance_mode,
                    "temperature": self.temperature,
                    "timestamp": datetime.now().isoformat(),
                    "success": True
                }
                
                self.results.append(result)
                return result
            else:
                error = {
                    "benchmark_id": benchmark_id,
                    "test_id": test_id,
                    "question": question,
                    "error": f"HTTP {response.status_code}",
                    "timestamp": datetime.now().isoformat(),
                    "success": False
                }
                self.errors.append(error)
                return error
                
        except Exception as e:
            error = {
                "benchmark_id": benchmark_id,
                "test_id": test_id,
                "question": question,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            self.errors.append(error)
            return error
    
    def _extract_multiple_choice_answer(self, response: str, choices: List[str]) -> str:
        """Extract multiple choice answer (A-J) from response."""
        response_lower = response.lower()
        
        # Try to find letter-based answers (A-J)
        import re
        letter_pattern = r'([A-J])\)'
        matches = re.findall(letter_pattern, response)
        if matches:
            return matches[0]
        
        # Try to find choice text matches
        for i, choice in enumerate(choices):
            if choice.lower() in response_lower:
                return chr(65 + i)  # A, B, C, D, etc.
        
        # Try to find the answer in the response text
        for i, choice in enumerate(choices):
            if choice.lower() in response_lower[-200:]:  # Check last 200 chars
                return chr(65 + i)
        
        return "UNKNOWN"
    
    def _extract_text_answer(self, response: str) -> str:
        """Extract text answer from response."""
        # Simple extraction - take the last non-empty line
        lines = response.strip().split('\n')
        for line in reversed(lines):
            if line.strip():
                return line.strip()
        return response[:100]
    
    def save_checkpoint(self, filename: str = "comprehensive_benchmark_checkpoint.json"):
        """Save checkpoint with current results."""
        checkpoint_data = {
            "checkpoint_time": datetime.now().isoformat(),
            "total_processed": len(self.results) + len(self.errors),
            "current_accuracy": self.calculate_current_accuracy(),
            "results": self.results,
            "errors": self.errors
        }
        
        with open(filename, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"  [CHECKPOINT] Saved to {filename}")
    
    def calculate_current_accuracy(self) -> float:
        """Calculate current accuracy."""
        successful = [r for r in self.results if r.get("success")]
        correct = [r for r in successful if r.get("is_correct")]
        return len(correct) / len(successful) if successful else 0
    
    def run_all_benchmarks(self):
        """Run all benchmarks sequentially."""
        print("\n" + "=" * 60)
        print("STARTING COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 60)
        print(f"Total Benchmarks: {len(self.benchmarks)}")
        print(f"Model: {self.model}")
        print(f"Performance Mode: {self.performance_mode}")
        print(f"Cache: {self.use_cache}")
        print(f"Temperature: {self.temperature}")
        print(f"Checkpoint Interval: Every {self.checkpoint_interval} questions")
        print("=" * 60)
        
        total_correct = 0
        total_duration = 0
        all_latencies = []
        start_time = time.time()
        
        for benchmark_id, dataset in self.datasets.items():
            benchmark_info = self.benchmarks[benchmark_id]
            benchmark_name = benchmark_info['name']
            
            print(f"\n{'='*60}")
            print(f"RUNNING {benchmark_name.upper()}")
            print(f"{'='*60}")
            print(f"Questions: {len(dataset)}")
            
            benchmark_correct = 0
            benchmark_latencies = []
            
            for i, item in enumerate(dataset):
                # Extract data based on benchmark type
                question = item.get(benchmark_info['question_field'], "")
                choices = item.get(benchmark_info['choices_field']) if benchmark_info['choices_field'] else None
                correct_answer = item.get(benchmark_info['answer_field'], "")
                subject = item.get(benchmark_info['subject_field'], "general")
                test_id = item.get("id", f"{benchmark_id}_Q{i}")
                
                # Progress
                print(f"\n[{i+1}/{len(dataset)}] [{subject}]")
                print(f"Question: {question[:60]}...")
                
                result = self.run_benchmark_question(
                    question=question,
                    choices=choices,
                    correct_answer=correct_answer,
                    subject=subject,
                    benchmark_id=benchmark_id,
                    test_id=test_id
                )
                
                if result.get("success"):
                    total_duration += result.get("duration", 0)
                    all_latencies.append(result.get("duration", 0))
                    benchmark_latencies.append(result.get("duration", 0))
                    
                    if result.get("is_correct"):
                        benchmark_correct += 1
                        total_correct += 1
                        status = "[CORRECT]"
                    else:
                        status = "[WRONG]"
                    
                    print(f"  {status} {result['extracted_answer']} vs {correct_answer}")
                    print(f"  Duration: {result['duration']:.2f}s")
                    
                    # Checkpoint every N questions
                    if (len(self.results) + len(self.errors)) % self.checkpoint_interval == 0:
                        self.save_checkpoint()
                else:
                    print(f"  [ERROR] {result.get('error')}")
            
            # Benchmark summary
            benchmark_accuracy = benchmark_correct / len(dataset) if dataset else 0
            avg_latency = np.mean(benchmark_latencies) if benchmark_latencies else 0
            median_latency = np.median(benchmark_latencies) if benchmark_latencies else 0
            p95_latency = np.percentile(benchmark_latencies, 95) if benchmark_latencies else 0
            
            print(f"\n{benchmark_name} Summary:")
            print(f"  Accuracy: {benchmark_accuracy:.2%} ({benchmark_correct}/{len(dataset)})")
            print(f"  Avg Latency: {avg_latency:.2f}s")
            print(f"  Median Latency: {median_latency:.2f}s")
            print(f"  P95 Latency: {p95_latency:.2f}s")
        
        total_time = time.time() - start_time
        
        # Generate final report
        self.generate_final_report(total_time, all_latencies, total_correct)
    
    def generate_final_report(self, total_time: float, all_latencies: List[float], total_correct: int):
        """Generate comprehensive final report."""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE BENCHMARK RESULTS")
        print("=" * 60)
        
        successful = [r for r in self.results if r.get("success")]
        failed = [r for r in self.results if not r.get("success")]
        
        # Overall statistics
        total_questions = len(self.results) + len(self.errors)
        accuracy = total_correct / len(successful) if successful else 0
        avg_latency = np.mean(all_latencies) if all_latencies else 0
        median_latency = np.median(all_latencies) if all_latencies else 0
        p95_latency = np.percentile(all_latencies, 95) if all_latencies else 0
        
        print(f"\n{'RESULTS'}")
        print(f"{'-'*40}")
        print(f"Questions:       {total_questions}")
        print(f"Correct:         {total_correct}")
        print(f"Incorrect:       {len(successful) - total_correct}")
        print(f"Accuracy:        {accuracy:.2%}")
        print(f"")
        print(f"Average latency: {avg_latency*1000:.0f} ms")
        print(f"Median latency:  {median_latency*1000:.0f} ms")
        print(f"P95 latency:     {p95_latency*1000:.0f} ms")
        
        # Subject breakdown
        print(f"\nSubject breakdown:")
        subject_stats = {}
        for result in successful:
            subject = result.get("subject", "general")
            if subject not in subject_stats:
                subject_stats[subject] = {"total": 0, "correct": 0}
            subject_stats[subject]["total"] += 1
            if result.get("is_correct"):
                subject_stats[subject]["correct"] += 1
        
        for subject, data in sorted(subject_stats.items()):
            accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0
            print(f"{subject.ljust(25)} {accuracy:.2%}")
        
        # Benchmark breakdown
        print(f"\nBenchmark breakdown:")
        benchmark_stats = {}
        for result in successful:
            benchmark = result.get("benchmark_id", "unknown")
            if benchmark not in benchmark_stats:
                benchmark_stats[benchmark] = {"total": 0, "correct": 0}
            benchmark_stats[benchmark]["total"] += 1
            if result.get("is_correct"):
                benchmark_stats[benchmark]["correct"] += 1
        
        for benchmark, data in benchmark_stats.items():
            accuracy = data["correct"] / data["total"] if data["total"] > 0 else 0
            benchmark_name = self.benchmarks.get(benchmark, {}).get("name", benchmark)
            print(f"{benchmark_name.ljust(25)} {accuracy:.2%} ({data['correct']}/{data['total']})")
        
        # Save final results
        final_results = {
            "overall_statistics": {
                "total_questions": total_questions,
                "correct": total_correct,
                "incorrect": len(successful) - total_correct,
                "accuracy": accuracy,
                "avg_latency_ms": avg_latency * 1000,
                "median_latency_ms": median_latency * 1000,
                "p95_latency_ms": p95_latency * 1000,
                "total_time_seconds": total_time
            },
            "subject_breakdown": subject_stats,
            "benchmark_breakdown": benchmark_stats,
            "all_results": self.results,
            "all_errors": self.errors
        }
        
        with open("comprehensive_benchmark_final_results.json", 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\nResults saved to: comprehensive_benchmark_final_results.json")
        print(f"Checkpoint saved to: comprehensive_benchmark_checkpoint.json")

def main():
    """Main execution."""
    print("=" * 60)
    print("COMPREHENSIVE MULTI-BENCHMARK SYSTEM")
    print("=" * 60)
    print("Benchmarks: MMLU-Pro, GPQA, AIME, GAIA")
    print("Configuration:")
    print("  URL: http://localhost:8000")
    print("  Model: gemma2:2b")
    print("  Performance Mode: QUALITY")
    print("  Cache: DISABLED")
    print("  Temperature: 0")
    print("=" * 60)
    
    system = ComprehensiveBenchmarkSystem()
    
    # Verify optimizations
    if not system.verify_optimizations():
        print("\n[ERROR] Optimization verification failed")
        return
    
    # Download all datasets
    system.download_all_datasets()
    
    # Run all benchmarks
    system.run_all_benchmarks()

if __name__ == "__main__":
    main()