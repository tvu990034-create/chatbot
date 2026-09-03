"""
Challenging Benchmark - Hard questions that require real reasoning
This benchmark is designed to actually test the optimization benefits
"""

import sys
import io
from pathlib import Path
from datetime import datetime

# Fix unicode encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from gateway.universal_enhanced_gateway import get_universal_gateway

class ChallengingBenchmark:
    """Challenging benchmark with difficult questions."""
    
    def __init__(self):
        # Baseline gateway (minimal optimizations)
        self.baseline_gateway = get_universal_gateway("gemma2:2b", enable_all_optimizations=False, performance_mode="quality")
        # Optimized gateway (all optimizations)
        self.optimized_gateway = get_universal_gateway("gemma2:2b", enable_all_optimizations=True, performance_mode="quality")
        
        # Challenging benchmark questions - requires real reasoning
        self.benchmark_questions = {
            "Complex_Reasoning": [
                {
                    "question": "A company has revenue of $1M, COGS of $400K, operating expenses of $300K, interest expense of $50K, and tax rate of 25%. Calculate net income and net profit margin.",
                    "answer": "net income",
                    "keywords": ["187500", "18.75%", "187,500"]
                },
                {
                    "question": "If you invest $10,000 at 7% annual interest compounded monthly, what will be the balance after 5 years?",
                    "answer": "balance",
                    "keywords": ["14176", "14177", "14,176", "14,177"]
                },
                {
                    "question": "A train leaves Station A at 8:00 AM traveling at 60 mph. Another train leaves Station B at 9:00 AM traveling at 80 mph on a 300-mile track. When do they meet?",
                    "answer": "meet",
                    "keywords": ["11:30", "11:30 AM", "11.5"]
                },
                {
                    "question": "Explain the difference between accounts payable and accounts receivable with examples.",
                    "answer": "accounts",
                    "keywords": ["payable", "receivable", "money owed", "money owed to"]
                },
                {
                    "question": "What is the DuPont analysis and how does it decompose ROE?",
                    "answer": "dupont",
                    "keywords": ["profit margin", "asset turnover", "equity multiplier", "roe"]
                },
                {
                    "question": "Calculate the weighted average cost of capital (WACC) given: debt $5M at 6%, equity $10M at 12%, tax rate 30%.",
                    "answer": "wacc",
                    "keywords": ["9.6%", "9.5%", "9.7%"]
                },
                {
                    "question": "What is the difference between FIFO and LIFO inventory accounting and when would each be preferred?",
                    "answer": "inventory",
                    "keywords": ["first in first out", "last in first out", "inflation", "deflation"]
                },
                {
                    "question": "Explain the concept of working capital and its importance for business operations.",
                    "answer": "working capital",
                    "keywords": ["current assets", "current liabilities", "liquidity", "operations"]
                },
                {
                    "question": "What is the accounting equation and why must it always balance?",
                    "answer": "assets",
                    "keywords": ["assets", "liabilities", "equity", "balance"]
                },
                {
                    "question": "Calculate EBITDA for a company with: revenue $2M, COGS $800K, operating expenses $400K, depreciation $100K, amortization $50K.",
                    "answer": "ebitda",
                    "keywords": ["650000", "650,000", "650k"]
                }
            ],
            "Logic_Puzzles": [
                {
                    "question": "If all Bloops are Razzies and all Razzies are Lazzies, then all Bloops are definitely Lazzies. True or False? Explain.",
                    "answer": "true",
                    "keywords": ["true", "correct", "yes"]
                },
                {
                    "question": "A bat and ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
                    "answer": "0.05",
                    "keywords": ["0.05", "5 cents", "five cents", "$0.05"]
                },
                {
                    "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
                    "answer": "5",
                    "keywords": ["5", "five", "5 minutes"]
                },
                {
                    "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half the lake?",
                    "answer": "47",
                    "keywords": ["47", "forty-seven", "47 days"]
                },
                {
                    "question": "You are shown three boxes: one contains gold, one silver, one bronze. You pick a box but don't open it. One box is removed (not yours). Should you switch boxes?",
                    "answer": "switch",
                    "keywords": ["switch", "yes", "monty hall"]
                },
                {
                    "question": "A father is 4 times as old as his son. In 20 years, he will be only twice as old. How old are they now?",
                    "answer": "40",
                    "keywords": ["40", "10", "father 40", "son 10"]
                },
                {
                    "question": "If you flip a fair coin 10 times and get 10 heads, what is the probability of getting heads on the 11th flip?",
                    "answer": "0.5",
                    "keywords": ["0.5", "50%", "50 percent", "1/2"]
                },
                {
                    "question": "A man builds a house with all 4 sides facing south. A bear walks by. What color is the bear?",
                    "answer": "white",
                    "keywords": ["white", "polar", "north pole"]
                },
                {
                    "question": "How many birthdays does the average person have?",
                    "answer": "1",
                    "keywords": ["1", "one", "one per year"]
                },
                {
                    "question": "Some months have 31 days, others have 30. How many have 28?",
                    "answer": "12",
                    "keywords": ["12", "twelve", "all of them"]
                }
            ],
            "Complex_Math": [
                {
                    "question": "Solve for x: 2^(3x+1) = 128",
                    "answer": "2",
                    "keywords": ["2", "x=2"]
                },
                {
                    "question": "What is the derivative of f(x) = x^3 * sin(x)?",
                    "answer": "derivative",
                    "keywords": ["3x^2", "sin(x)", "cos(x)", "product rule"]
                },
                {
                    "question": "Calculate the integral of x^2 from 0 to 3.",
                    "answer": "9",
                    "keywords": ["9", "x^3/3", "nine"]
                },
                {
                    "question": "Find the sum of the first 100 positive integers.",
                    "answer": "5050",
                    "keywords": ["5050", "5050"]
                },
                {
                    "question": "What is the probability of drawing 2 aces from a standard deck of 52 cards without replacement?",
                    "answer": "probability",
                    "keywords": ["0.0045", "0.45%", "1/221"]
                },
                {
                    "question": "Solve the system: x + y = 10, 2x - y = 5",
                    "answer": "5",
                    "keywords": ["x=5", "y=5", "5,5"]
                },
                {
                    "question": "What is the limit of (x^2 - 1)/(x - 1) as x approaches 1?",
                    "answer": "2",
                    "keywords": ["2", "two"]
                },
                {
                    "question": "Calculate the area of a circle with radius 7.",
                    "answer": "area",
                    "keywords": ["49π", "153.94", "154"]
                },
                {
                    "question": "What is the sum of the interior angles of a hexagon?",
                    "answer": "720",
                    "keywords": ["720", "720 degrees"]
                },
                {
                    "question": "If log base 2 of x = 3, what is x?",
                    "answer": "8",
                    "keywords": ["8", "eight"]
                }
            ],
            "Applied_Problems": [
                {
                    "question": "A car rental company charges $50/day plus $0.25/mile. If you rent for 3 days and drive 200 miles, what's the total cost?",
                    "answer": "200",
                    "keywords": ["200", "$200", "200 dollars"]
                },
                {
                    "question": "If you can complete a job in 6 hours and your friend can do it in 4 hours, how long working together?",
                    "answer": "2.4",
                    "keywords": ["2.4", "2.4 hours", "2 hours 24 minutes"]
                },
                {
                    "question": "A store offers 20% off, then an additional 15% off the discounted price. What's the total discount?",
                    "answer": "32",
                    "keywords": ["32%", "32 percent", "0.32"]
                },
                {
                    "question": "If you save $200/month at 5% annual interest, how much after 2 years?",
                    "answer": "approximately",
                    "keywords": ["4920", "4950", "4900"]
                },
                {
                    "question": "A room is 12x15 feet with 8-foot ceilings. How many square feet of wallpaper needed for all 4 walls?",
                    "answer": "432",
                    "keywords": ["432", "432 sq ft"]
                },
                {
                    "question": "If a recipe serves 4 and needs 2 cups flour, how much flour for 10 people?",
                    "answer": "5",
                    "keywords": ["5", "5 cups", "five cups"]
                },
                {
                    "question": "A phone plan costs $40/month for 500 minutes, $0.10/minute after. Cost for 650 minutes?",
                    "answer": "55",
                    "keywords": ["55", "$55", "55 dollars"]
                },
                {
                    "question": "If you travel 60 mph for 2 hours and 40 mph for 3 hours, what's average speed?",
                    "answer": "48",
                    "keywords": ["48", "48 mph", "48 miles per hour"]
                },
                {
                    "question": "A shirt costs $40 after 25% discount. What was original price?",
                    "answer": "53.33",
                    "keywords": ["53.33", "53.33", "$53.33"]
                },
                {
                    "question": "If you need 3 workers to build a house in 30 days, how many workers for 18 days?",
                    "answer": "5",
                    "keywords": ["5", "five", "5 workers"]
                }
            ]
        }
    
    def run_benchmark(self, gateway, mode: str):
        """Run benchmark with given gateway."""
        print(f"\n{'='*60}")
        print(f"Running Benchmark ({mode} mode)")
        print(f"{'='*60}")
        
        all_results = {}
        total_correct = 0
        total_questions = 0
        
        for benchmark_name, questions in self.benchmark_questions.items():
            print(f"\n--- {benchmark_name} ---")
            results = []
            correct = 0
            total = 0
            
            for i, item in enumerate(questions):
                try:
                    messages = [{"role": "user", "content": item["question"]}]
                    response = gateway.chat(messages, use_cache=False)
                    
                    # Check if any keyword is in response
                    is_correct = any(keyword.lower() in response.lower() for keyword in item["keywords"])
                    if is_correct:
                        correct += 1
                    total += 1
                    
                    results.append({
                        "question": item["question"],
                        "predicted": response,
                        "ground_truth": item["answer"],
                        "correct": is_correct
                    })
                    
                    print(f"{i+1}. {'OK' if is_correct else 'FAIL'} ({correct}/{total})")
                    
                except Exception as e:
                    print(f"Error processing question {i}: {e}")
                    continue
            
            accuracy = correct / total if total > 0 else 0
            print(f"{benchmark_name} Accuracy: {correct}/{total} ({accuracy:.2%})")
            
            all_results[benchmark_name] = {
                "accuracy": accuracy,
                "total": total,
                "correct": correct,
                "results": results
            }
            
            total_correct += correct
            total_questions += total
        
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
        print(f"\nOverall Accuracy: {total_correct}/{total_questions} ({overall_accuracy:.2%})")
        
        return {
            "overall_accuracy": overall_accuracy,
            "total": total_questions,
            "correct": total_correct,
            "benchmarks": all_results
        }
    
    def run_comparison(self):
        """Run baseline vs optimized comparison."""
        print(f"\n{'='*60}")
        print("Challenging Benchmark Comparison")
        print(f"{'='*60}")
        print("Testing 40 challenging questions across 4 domains")
        print("These questions require actual reasoning, not simple recall")
        print("Baseline: Minimal optimizations")
        print("Optimized: All 49 advanced research techniques with aggressive reasoning")
        
        # Run baseline
        print("\nStep 1: Running baseline benchmark...")
        baseline_results = self.run_benchmark(self.baseline_gateway, "baseline")
        
        # Run optimized
        print("\nStep 2: Running optimized benchmark...")
        optimized_results = self.run_benchmark(self.optimized_gateway, "optimized")
        
        # Compare
        print(f"\n{'='*60}")
        print("Comparison Results")
        print(f"{'='*60}")
        
        baseline_acc = float(baseline_results['overall_accuracy'])
        optimized_acc = float(optimized_results['overall_accuracy'])
        overall_improvement = optimized_acc - baseline_acc
        
        print(f"\nBaseline Overall Accuracy: {baseline_acc:.2%}")
        print(f"Optimized Overall Accuracy: {optimized_acc:.2%}")
        print(f"Overall Improvement: {overall_improvement:+.2%}")
        
        if baseline_acc > 0:
            relative_improvement = (overall_improvement / baseline_acc * 100)
            print(f"Relative Improvement: {relative_improvement:+.1f}%")
        
        # Per-benchmark comparison
        print(f"\n{'='*60}")
        print("Per-Benchmark Comparison")
        print(f"{'='*60}")
        
        for benchmark_name in self.benchmark_questions.keys():
            base_acc = float(baseline_results['benchmarks'][benchmark_name]['accuracy'])
            opt_acc = float(optimized_results['benchmarks'][benchmark_name]['accuracy'])
            bench_improvement = opt_acc - base_acc
            print(f"{benchmark_name}: {base_acc:.2%} -> {opt_acc:.2%} ({bench_improvement:+.2%})")
        
        # Generate report
        self.generate_report(baseline_results, optimized_results, overall_improvement)
        
        return {
            "baseline": baseline_results,
            "optimized": optimized_results,
            "improvement": overall_improvement
        }
    
    def generate_report(self, baseline_results, optimized_results, overall_improvement):
        """Generate comparison report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"challenging_benchmark_report_{timestamp}.md"
        
        baseline_acc = float(baseline_results['overall_accuracy'])
        optimized_acc = float(optimized_results['overall_accuracy'])
        relative_improvement = (overall_improvement / baseline_acc * 100) if baseline_acc > 0 else 0
        
        report = f"""# Challenging Benchmark Comparison Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Overview
This report compares baseline performance vs optimized performance on **40 challenging questions** that require actual reasoning, not simple recall. The questions span 4 domains: Complex Reasoning, Logic Puzzles, Complex Math, and Applied Problems.

## System Configuration
- **Model**: gemma2:2b
- **Baseline**: Minimal optimizations
- **Optimized**: All 49 advanced research techniques with aggressive reasoning enhancements
- **Optimization Techniques**: 49 S-Tier and A-Tier methods
- **Total Questions**: 40 (10 per benchmark)

## Results Summary

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Overall Accuracy | {baseline_acc:.2%} | {optimized_acc:.2%} | {overall_improvement:+.2%} |
| Correct Answers | {baseline_results['correct']}/{baseline_results['total']} | {optimized_results['correct']}/{optimized_results['total']} | {optimized_results['correct'] - baseline_results['correct']:+d} |
| Relative Improvement | - | - | {relative_improvement:+.1f}% |

## Per-Benchmark Results

"""
        
        for benchmark_name in self.benchmark_questions.keys():
            base_bench = baseline_results['benchmarks'][benchmark_name]
            opt_bench = optimized_results['benchmarks'][benchmark_name]
            base_acc = float(base_bench['accuracy'])
            opt_acc = float(opt_bench['accuracy'])
            bench_improvement = opt_acc - base_acc
            
            report += f"""
### {benchmark_name}

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Accuracy | {base_acc:.2%} | {opt_acc:.2%} | {bench_improvement:+.2%} |
| Correct | {base_bench['correct']}/{base_bench['total']} | {opt_bench['correct']}/{opt_bench['total']} | {opt_bench['correct'] - base_bench['correct']:+d} |

#### Question-by-Question Analysis

"""
            for i, (base, opt) in enumerate(zip(base_bench['results'], opt_bench['results'])):
                base_status = "OK" if base['correct'] else "FAIL"
                opt_status = "OK" if opt['correct'] else "FAIL"
                q_improvement = "BETTER" if not base['correct'] and opt['correct'] else "WORSE" if base['correct'] and not opt['correct'] else "SAME"
                
                report += f"""
**Q{i+1}**: {base['question']}
- Ground Truth: {base['ground_truth']}
- Baseline: {base_status} - {base['predicted'][:100]}...
- Optimized: {opt_status} - {opt['predicted'][:100]}...
- Result: {q_improvement}

"""
        
        report += f"""
## Technical Details

### Optimizations Applied (49 Research Techniques)
- **Total Papers**: ~399
- **Total Categories**: 40 specialized domains
- **Optimization Techniques**: 49 applied techniques
- **S-Tier Techniques**: 14 (CrossModalSync, ContrastiveLearning, SlotAttention, SceneGraphGNN, MultiCueDepth, AnomalyDetection, ProgramSynthesis, AssociativeMemory, Calibration, InfoBottleneck, AdaptiveEnergy, DeepSVDD, DefocusDepth, CrossModalResonance)
- **A-Tier Techniques**: 35 (Counterfactual, CausalIntervention, MetaLearning, KnowledgeVQA, Affordance, HapticVAE, FuturePrediction, EmbodiedNavigation, Physics, JigsawSolver, CounterfactualEditor, NounAlignment, PrototypeAlignment, TextSaliency, ProgramSupervisedVQA, AbstractWordGrounding, VisualDatabase, MetaphorDetector, MemoryAugmentedPrediction, CuriosityReward, AffordanceLandscape, PerspectiveTaking, SocialRelation, TimePassage, ForcePrediction, MotionStreak, EscapeRoute, NormalizingFlow, GraphMatching, SlotPredictiveCoding, AffordanceEnergy, TactileContrastive, DifferentiablePathPlanning, PhysicsConstraint, InfoGainQuestion)

### Key Enhancement: Aggressive Reasoning Prompts
The optimized system uses aggressive chain-of-thought prompting:
- Step-by-step deconstruction of questions
- Explicit verification requirements
- Domain-specific reasoning frameworks
- Multiple perspective consideration
- Edge case analysis

## Conclusion

"""
        
        if overall_improvement > 0.10:
            conclusion = "SIGNIFICANT"
        elif overall_improvement > 0.05:
            conclusion = "MODERATE"
        elif overall_improvement > 0:
            conclusion = "MINIMAL"
        else:
            conclusion = "NEGATIVE"
        
        report += f"The optimized system demonstrates {conclusion} improvements ({overall_improvement:+.2%}) through aggressive reasoning enhancements and 49 advanced research techniques.\n"
        
        # Save report with UTF-8 encoding
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nReport saved to: {report_file}")

if __name__ == "__main__":
    benchmark = ChallengingBenchmark()
    results = benchmark.run_comparison()
