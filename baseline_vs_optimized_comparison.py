"""
Baseline vs Optimized Comparison for phi3:mini
Compares performance between baseline and optimized versions across benchmarks
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ComparisonResult:
    """Result from comparing baseline vs optimized performance."""
    benchmark_name: str
    baseline_accuracy: float
    optimized_accuracy: float
    accuracy_improvement: float
    speed_improvement: float
    significance: str


class BaselineOptimizedComparison:
    """Compare baseline phi3:mini with optimized version."""
    
    def __init__(self):
        self.baseline_results: Dict[str, Dict] = {}
        self.optimized_results: Dict[str, Dict] = {}
        self.comparisons: List[ComparisonResult] = []
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def load_baseline_results(self, results_file: str):
        """Load baseline benchmark results."""
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            for result in data.get("results", []):
                self.baseline_results[result["benchmark"]] = result
            
            logger.info(f"Loaded baseline results from {results_file}")
        except Exception as e:
            logger.error(f"Error loading baseline results: {e}")
    
    def load_optimized_results(self, results_file: str):
        """Load optimized benchmark results."""
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            
            for result in data.get("results", []):
                self.optimized_results[result["benchmark"]] = result
            
            logger.info(f"Loaded optimized results from {results_file}")
        except Exception as e:
            logger.error(f"Error loading optimized results: {e}")
    
    def compare_performance(self) -> List[ComparisonResult]:
        """Compare baseline vs optimized performance."""
        comparisons = []
        
        # Get all benchmark names
        all_benchmarks = set(self.baseline_results.keys()) | set(self.optimized_results.keys())
        
        for benchmark in all_benchmarks:
            baseline = self.baseline_results.get(benchmark)
            optimized = self.optimized_results.get(benchmark)
            
            if baseline and optimized:
                baseline_acc = baseline.get("accuracy", 0.0)
                optimized_acc = optimized.get("accuracy", 0.0)
                improvement = optimized_acc - baseline_acc
                
                baseline_duration = baseline.get("duration", 1.0)
                optimized_duration = optimized.get("duration", 1.0)
                speed_improvement = baseline_duration / optimized_duration if optimized_duration > 0 else 1.0
                
                # Determine significance
                if improvement > 0.05:
                    significance = "SIGNIFICANT"
                elif improvement > 0.01:
                    significance = "MODERATE"
                elif improvement > 0:
                    significance = "MINOR"
                elif improvement > -0.01:
                    significance = "NEUTRAL"
                else:
                    significance = "REGRESSION"
                
                comparison = ComparisonResult(
                    benchmark_name=benchmark,
                    baseline_accuracy=baseline_acc,
                    optimized_accuracy=optimized_acc,
                    accuracy_improvement=improvement,
                    speed_improvement=speed_improvement,
                    significance=significance
                )
                
                comparisons.append(comparison)
        
        self.comparisons = comparisons
        return comparisons
    
    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report."""
        report = []
        report.append("="*80)
        report.append("BASELINE vs OPTIMIZED COMPARISON REPORT - phi3:mini")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        report.append("PERFORMANCE COMPARISON SUMMARY")
        report.append("-"*80)
        report.append(f"{'Benchmark':<20} {'Baseline':<12} {'Optimized':<12} {'Improvement':<12} {'Speed':<12} {'Significance':<12}")
        report.append("-"*80)
        
        for comp in self.comparisons:
            report.append(
                f"{comp.benchmark_name:<20} "
                f"{comp.baseline_accuracy:>10.1%} "
                f"{comp.optimized_accuracy:>10.1%} "
                f"{comp.accuracy_improvement:>9.1%} "
                f"{comp.speed_improvement:>9.1%} "
                f"{comp.significance:<12}"
            )
        
        report.append("")
        
        # Overall statistics
        if self.comparisons:
            avg_baseline_acc = np.mean([c.baseline_accuracy for c in self.comparisons])
            avg_optimized_acc = np.mean([c.optimized_accuracy for c in self.comparisons])
            avg_improvement = np.mean([c.accuracy_improvement for c in self.comparisons])
            avg_speed = np.mean([c.speed_improvement for c in self.comparisons])
            
            report.append("OVERALL STATISTICS")
            report.append("-"*80)
            report.append(f"Average Baseline Accuracy: {avg_baseline_acc:.2%}")
            report.append(f"Average Optimized Accuracy: {avg_optimized_acc:.2%}")
            report.append(f"Average Accuracy Improvement: {avg_improvement:.2%}")
            report.append(f"Average Speed Improvement: {avg_speed:.2%}x")
            
            # Count significant improvements
            significant = sum(1 for c in self.comparisons if c.significance == "SIGNIFICANT")
            moderate = sum(1 for c in self.comparisons if c.significance == "MODERATE")
            minor = sum(1 for c in self.comparisons if c.significance == "MINOR")
            regressions = sum(1 for c in self.comparisons if c.significance == "REGRESSION")
            
            report.append("")
            report.append("SIGNIFICANCE BREAKDOWN")
            report.append("-"*80)
            report.append(f"Significant Improvements: {significant}")
            report.append(f"Moderate Improvements: {moderate}")
            report.append(f"Minor Improvements: {minor}")
            report.append(f"Regressions: {regressions}")
        
        report.append("")
        
        # Key findings
        report.append("KEY FINDINGS")
        report.append("-"*80)
        
        if self.comparisons:
            best_improvement = max(self.comparisons, key=lambda c: c.accuracy_improvement)
            worst_improvement = min(self.comparisons, key=lambda c: c.accuracy_improvement)
            best_speed = max(self.comparisons, key=lambda c: c.speed_improvement)
            
            report.append(f"Best Accuracy Improvement: {best_improvement.benchmark_name} (+{best_improvement.accuracy_improvement:.1%})")
            report.append(f"Worst Performance: {worst_improvement.benchmark_name} ({worst_improvement.accuracy_improvement:+.1%})")
            report.append(f"Best Speed Improvement: {best_speed.benchmark_name} ({best_speed.speed_improvement:.1%}x)")
        
        report.append("")
        
        # Conclusion
        report.append("CONCLUSION")
        report.append("-"*80)
        
        if avg_improvement > 0:
            report.append(f"The optimized phi3:mini shows an average accuracy improvement of {avg_improvement:.2%}")
            report.append(f"across {len(self.comparisons)} benchmarks, with an average speed improvement of {avg_speed:.2%}x.")
            
            if avg_improvement > 0.05:
                report.append("The optimizations are providing SIGNIFICANT performance gains.")
            elif avg_improvement > 0.01:
                report.append("The optimizations are providing MODERATE performance gains.")
            else:
                report.append("The optimizations are providing MINOR performance gains.")
        else:
            report.append("The optimized version shows no significant improvement over baseline.")
            report.append("Further optimization may be needed.")
        
        return "\n".join(report)
    
    def save_comparison(self):
        """Save comparison results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"baseline_vs_optimized_comparison_{timestamp}.json"
        
        comparison_data = {
            "timestamp": timestamp,
            "baseline_results": self.baseline_results,
            "optimized_results": self.optimized_results,
            "comparisons": [
                {
                    "benchmark": c.benchmark_name,
                    "baseline_accuracy": c.baseline_accuracy,
                    "optimized_accuracy": c.optimized_accuracy,
                    "accuracy_improvement": c.accuracy_improvement,
                    "speed_improvement": c.speed_improvement,
                    "significance": c.significance
                }
                for c in self.comparisons
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info(f"Comparison saved to {filename}")
        
        # Also save text report
        report_filename = self.results_dir / f"baseline_vs_optimized_comparison_{timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write(self.generate_comparison_report())
        
        logger.info(f"Report saved to {report_filename}")


def generate_sample_baseline_results():
    """Generate sample baseline results for demonstration."""
    return {
        "timestamp": "20260818_000000",
        "model": "phi3:mini-baseline",
        "results": [
            {
                "benchmark": "MMLU",
                "total_questions": 10,
                "correct_answers": 6,
                "accuracy": 0.6,
                "duration": 50.0,
                "metadata": {}
            },
            {
                "benchmark": "GSM8K",
                "total_questions": 10,
                "correct_answers": 5,
                "accuracy": 0.5,
                "duration": 60.0,
                "metadata": {}
            },
            {
                "benchmark": "HellaSwag",
                "total_questions": 10,
                "correct_answers": 6,
                "accuracy": 0.6,
                "duration": 45.0,
                "metadata": {}
            },
            {
                "benchmark": "ARC",
                "total_questions": 10,
                "correct_answers": 5,
                "accuracy": 0.5,
                "duration": 55.0,
                "metadata": {}
            }
        ]
    }


def generate_sample_optimized_results():
    """Generate sample optimized results for demonstration."""
    return {
        "timestamp": "20260818_000000",
        "model": "phi3:mini-optimized",
        "results": [
            {
                "benchmark": "MMLU",
                "total_questions": 10,
                "correct_answers": 7,
                "accuracy": 0.7,
                "duration": 35.0,
                "metadata": {"optimizations": "16 techniques applied"}
            },
            {
                "benchmark": "GSM8K",
                "total_questions": 10,
                "correct_answers": 6,
                "accuracy": 0.6,
                "duration": 40.0,
                "metadata": {"optimizations": "16 techniques applied"}
            },
            {
                "benchmark": "HellaSwag",
                "total_questions": 10,
                "correct_answers": 7,
                "accuracy": 0.7,
                "duration": 30.0,
                "metadata": {"optimizations": "16 techniques applied"}
            },
            {
                "benchmark": "ARC",
                "total_questions": 10,
                "correct_answers": 6,
                "accuracy": 0.6,
                "duration": 38.0,
                "metadata": {"optimizations": "16 techniques applied"}
            }
        ]
    }


def main():
    """Main function to run baseline vs optimized comparison."""
    # Create comparison instance
    comparison = BaselineOptimizedComparison()
    
    # For demonstration, use sample data
    # In production, load actual benchmark results
    baseline_data = generate_sample_baseline_results()
    optimized_data = generate_sample_optimized_results()
    
    # Process data into the expected format
    for result in baseline_data["results"]:
        comparison.baseline_results[result["benchmark"]] = result
    
    for result in optimized_data["results"]:
        comparison.optimized_results[result["benchmark"]] = result
    
    # Run comparison
    comparisons = comparison.compare_performance()
    
    # Generate and save report
    report = comparison.generate_comparison_report()
    print(report)
    
    comparison.save_comparison()
    
    logger.info("Comparison completed")


if __name__ == "__main__":
    main()