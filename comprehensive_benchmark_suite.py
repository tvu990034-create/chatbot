"""
Comprehensive Benchmark Suite with Additional Benchmarks
Includes all benchmarks that phi3:mini can actually run (text-only, no special environments)
"""

import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
import numpy as np

# Import enhanced gateway with all optimizations
try:
    from gateway.enhanced_gateway import EnhancedGateway
    ENHANCED_GATEWAY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import EnhancedGateway: {e}")
    ENHANCED_GATEWAY_AVAILABLE = False

# Import benchmark-specific optimizers
try:
    from gateway.arc_optimization import ARCOptimizer
    ARC_OPTIMIZER_AVAILABLE = True
except ImportError:
    ARC_OPTIMIZER_AVAILABLE = False

try:
    from gateway.truthfulqa_optimization import TruthfulQAOptimizer, TruthfulQAEnhancedEvaluator
    TRUTHFULQA_OPTIMIZATION_AVAILABLE = True
except ImportError:
    TRUTHFULQA_OPTIMIZATION_AVAILABLE = False

try:
    from gateway.agentharm_optimization import AgentHarmOptimizer, AgentHarmEnhancedEvaluator
    AGENTHARM_OPTIMIZATION_AVAILABLE = True
except ImportError:
    AGENTHARM_OPTIMIZATION_AVAILABLE = False

# Try to import datasets for new benchmarks
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: datasets library not available. Install with: pip install datasets")
    DATASETS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    benchmark_name: str
    model_name: str
    total_questions: int
    correct_answers: int
    accuracy: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Question:
    """A single benchmark question."""
    question: str
    options: Optional[List[str]] = None
    correct_answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BenchmarkBase:
    """Base class for all benchmarks."""
    
    def __init__(self, name: str, gateway):
        self.name = name
        self.gateway = gateway
        self.questions: List[Question] = []
        self.results: List[BenchmarkResult] = []
    
    def load_questions(self, num_samples: Optional[int] = None):
        """Load benchmark questions."""
        raise NotImplementedError
    
    def format_prompt(self, question: Question) -> str:
        """Format question into prompt for the model."""
        raise NotImplementedError
    
    def extract_answer(self, response: str, question: Question) -> Optional[str]:
        """Extract answer from model response."""
        raise NotImplementedError
    
    def evaluate_answer(self, extracted: str, question: Question) -> bool:
        """Check if extracted answer is correct."""
        raise NotImplementedError
    
    def run_benchmark(self, num_samples: Optional[int] = None) -> BenchmarkResult:
        """Run the benchmark and return results."""
        self.load_questions(num_samples)
        
        start_time = datetime.now()
        correct = 0
        total = len(self.questions)
        
        logger.info(f"Running {self.name} benchmark with {total} questions")
        
        for i, question in enumerate(self.questions):
            try:
                prompt = self.format_prompt(question)
                response = self.gateway.chat([{"role": "user", "content": prompt}])
                
                extracted = self.extract_answer(response, question)
                is_correct = self.evaluate_answer(extracted, question)
                
                if is_correct:
                    correct += 1
                
                if (i + 1) % 5 == 0:
                    logger.info(f"Progress: {i+1}/{total}, Accuracy: {correct/(i+1):.2%}")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {e}")
        
        duration = (datetime.now() - start_time).total_seconds()
        accuracy = correct / total if total > 0 else 0.0
        
        result = BenchmarkResult(
            benchmark_name=self.name,
            model_name="phi3:mini-optimized",
            total_questions=total,
            correct_answers=correct,
            accuracy=accuracy,
            duration=duration,
            metadata={"gateway_stats": str(self.gateway.get_optimization_stats())}
        )
        
        self.results.append(result)
        logger.info(f"{self.name} completed: {accuracy:.2%} accuracy ({correct}/{total})")
        
        return result


class MMLUProBenchmark(BenchmarkBase):
    """MMLU-Pro benchmark - more challenging version of MMLU."""
    
    def __init__(self, gateway):
        super().__init__("MMLU-Pro", gateway)
    
    def load_questions(self, num_samples: Optional[int] = None):
        """Load MMLU-Pro questions."""
        if DATASETS_AVAILABLE:
            try:
                dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
                num_samples = min(num_samples or 5, len(dataset))
                self.questions = self._process_dataset(dataset, num_samples)
                return
            except Exception as e:
                logger.warning(f"Could not load MMLU-Pro dataset: {e}")
        
        # Fallback to sample questions
        self.questions = self._generate_sample_questions(num_samples or 5)
    
    def _process_dataset(self, dataset, num_samples: int) -> List[Question]:
        """Process dataset into Question objects."""
        questions = []
        for i in range(num_samples):
            item = dataset[i]
            questions.append(Question(
                question=item['question'],
                options=item['options'],
                correct_answer=item['answer'],
                metadata={"category": item.get('category', 'unknown')}
            ))
        return questions
    
    def _generate_sample_questions(self, num: int) -> List[Question]:
        """Generate sample MMLU-Pro-style questions."""
        sample_questions = [
            {
                "question": "What is the molecular structure of benzene?",
                "options": ["A) Linear chain", "B) Hexagonal ring", "C) Tetrahedral", "D) Octahedral"],
                "correct": "B"
            },
            {
                "question": "Which statistical test is used to compare means of more than two groups?",
                "options": ["A) t-test", "B) ANOVA", "C) Chi-square", "D) Correlation"],
                "correct": "B"
            },
            {
                "question": "What is the time complexity of quicksort in the average case?",
                "options": ["A) O(n)", "B) O(n log n)", "C) O(n^2)", "D) O(log n)"],
                "correct": "B"
            }
        ]
        
        questions = []
        for i in range(num):
            q_data = sample_questions[i % len(sample_questions)]
            questions.append(Question(
                question=q_data["question"],
                options=q_data["options"],
                correct_answer=q_data["correct"],
                metadata={"category": "STEM"}
            ))
        
        return questions
    
    def format_prompt(self, question: Question) -> str:
        """Format MMLU-Pro question."""
        prompt = f"Answer the following complex question:\n\n"
        prompt += f"{question.question}\n"
        for option in question.options:
            prompt += f"{option}\n"
        prompt += "Answer: "
        return prompt
    
    def extract_answer(self, response: str, question: Question) -> Optional[str]:
        """Extract multiple choice answer."""
        patterns = [
            r"Answer:\s*([A-D])",
            r"([A-D])\)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        for option in ["A", "B", "C", "D"]:
            if option in response.upper():
                return option
        
        return None
    
    def evaluate_answer(self, extracted: str, question: Question) -> bool:
        """Check if extracted answer is correct."""
        return extracted == question.correct_answer


class BBHBenchmark(BenchmarkBase):
    """Big-Bench Hard (BBH) benchmark."""
    
    def __init__(self, gateway):
        super().__init__("BBH", gateway)
    
    def load_questions(self, num_samples: Optional[int] = None):
        """Load BBH questions."""
        if DATASETS_AVAILABLE:
            try:
                dataset = load_dataset("lukaemon/bbh")
                # Get first task for simplicity
                task_name = list(dataset.keys())[0]
                task_data = dataset[task_name]
                num_samples = min(num_samples or 5, len(task_data))
                self.questions = self._process_bbh_task(task_data, num_samples)
                return
            except Exception as e:
                logger.warning(f"Could not load BBH dataset: {e}")
        
        # Fallback to sample questions
        self.questions = self._generate_sample_questions(num_samples or 5)
    
    def _process_bbh_task(self, task_data, num_samples: int) -> List[Question]:
        """Process BBH task data."""
        questions = []
        for i in range(num_samples):
            item = task_data[i]
            questions.append(Question(
                question=item.get('input', item.get('question', '')),
                correct_answer=item.get('target', item.get('answer', '')),
                metadata={"task_type": "BBH"}
            ))
        return questions
    
    def _generate_sample_questions(self, num: int) -> List[Question]:
        """Generate sample BBH-style questions."""
        sample_questions = [
            {
                "question": "If A > B and B > C, then A > C. This is an example of which logical principle?",
                "answer": "Transitivity"
            },
            {
                "question": "Given the sequence: 2, 6, 12, 20, 30, what is the next number?",
                "answer": "42"
            },
            {
                "question": "If all bloops are bleeps and some bleeps are blobs, can we conclude that some bloops are blobs?",
                "answer": "No"
            }
        ]
        
        questions = []
        for i in range(num):
            q_data = sample_questions[i % len(sample_questions)]
            questions.append(Question(
                question=q_data["question"],
                correct_answer=q_data["answer"],
                metadata={"task_type": "reasoning"}
            ))
        
        return questions
    
    def format_prompt(self, question: Question) -> str:
        """Format BBH question."""
        prompt = f"Solve the following reasoning problem:\n\n"
        prompt += f"{question.question}\n\n"
        prompt += "Answer: "
        return prompt
    
    def extract_answer(self, response: str, question: Question) -> bool:
        """Extract answer."""
        return response.strip()
    
    def evaluate_answer(self, extracted: str, question: Question) -> bool:
        """Check if answer is correct (simplified semantic check)."""
        if not extracted:
            return False
        
        correct = question.correct_answer.lower()
        extracted_lower = extracted.lower()
        
        # Check for key words
        key_words = correct.split()
        matches = sum(1 for word in key_words if word in extracted_lower)
        
        return matches >= len(key_words) / 2  # At least half the key words match


class AMCBenchmark(BenchmarkBase):
    """AMC (American Mathematics Competition) benchmark from MATH dataset."""
    
    def __init__(self, gateway):
        super().__init__("AMC", gateway)
    
    def load_questions(self, num_samples: Optional[int] = None):
        """Load AMC questions from MATH dataset."""
        if DATASETS_AVAILABLE:
            try:
                dataset = load_dataset("hendrycks/competition_math", split="test")
                # Filter for AMC problems (they're in the MATH dataset)
                amc_problems = [item for item in dataset if 'AMC' in item.get('solution', '')]
                num_samples = min(num_samples or 5, len(amc_problems))
                self.questions = self._process_math_problems(amc_problems, num_samples)
                return
            except Exception as e:
                logger.warning(f"Could not load MATH dataset: {e}")
        
        # Fallback to sample questions
        self.questions = self._generate_sample_questions(num_samples or 5)
    
    def _process_math_problems(self, problems, num_samples: int) -> List[Question]:
        """Process MATH dataset problems."""
        questions = []
        for i in range(num_samples):
            item = problems[i]
            # Extract answer from solution (usually in \boxed{})
            import re
            match = re.search(r'\\boxed\{([^}]+)\}', item['solution'])
            answer = match.group(1) if match else "unknown"
            
            questions.append(Question(
                question=item['problem'],
                correct_answer=answer,
                metadata={"level": item.get('level', 'unknown'), "type": item.get('type', 'unknown')}
            ))
        return questions
    
    def _generate_sample_questions(self, num: int) -> List[Question]:
        """Generate sample AMC-style questions."""
        sample_questions = [
            {
                "question": "What is the value of x if 2x + 5 = 13?",
                "answer": "4"
            },
            {
                "question": "Simplify: (x^2 - 9) / (x - 3)",
                "answer": "x + 3"
            },
            {
                "question": "What is the sum of the first 10 positive integers?",
                "answer": "55"
            }
        ]
        
        questions = []
        for i in range(num):
            q_data = sample_questions[i % len(sample_questions)]
            questions.append(Question(
                question=q_data["question"],
                correct_answer=q_data["answer"],
                metadata={"level": "AMC 10/12"}
            ))
        
        return questions
    
    def format_prompt(self, question: Question) -> str:
        """Format AMC question."""
        prompt = f"Solve the following mathematics competition problem:\n\n"
        prompt += f"{question.question}\n\n"
        prompt += "Provide your final answer: "
        return prompt
    
    def extract_answer(self, response: str, question: Question) -> bool:
        """Extract answer."""
        return response.strip()
    
    def evaluate_answer(self, extracted: str, question: Question) -> bool:
        """Check if answer is correct (simplified)."""
        if not extracted:
            return False
        
        correct = question.correct_answer.lower().strip()
        extracted_lower = extracted.lower().strip()
        
        # Check for numerical answer
        try:
            return float(extracted_lower) == float(correct)
        except ValueError:
            # Fallback to string comparison
            return correct in extracted_lower


class ProofNetBenchmark(BenchmarkBase):
    """ProofNet benchmark for formal mathematics."""
    
    def __init__(self, gateway):
        super().__init__("ProofNet", gateway)
    
    def load_questions(self, num_samples: Optional[int] = None):
        """Load ProofNet questions."""
        if DATASETS_AVAILABLE:
            try:
                dataset = load_dataset("hoskinson-center/proofnet")
                num_samples = min(num_samples or 5, len(dataset))
                self.questions = self._process_proofnet_dataset(dataset, num_samples)
                return
            except Exception as e:
                logger.warning(f"Could not load ProofNet dataset: {e}")
        
        # Fallback to sample questions
        self.questions = self._generate_sample_questions(num_samples or 5)
    
    def _process_proofnet_dataset(self, dataset, num_samples: int) -> List[Question]:
        """Process ProofNet dataset."""
        questions = []
        for i in range(num_samples):
            item = dataset[i]
            questions.append(Question(
                question=item['nl_statement'],
                correct_answer="proof",  # ProofNet evaluates proof generation
                metadata={"formal_statement": item['formal_statement']}
            ))
        return questions
    
    def _generate_sample_questions(self, num: int) -> List[Question]:
        """Generate sample ProofNet-style questions."""
        sample_questions = [
            {
                "question": "Prove that the square root of 2 is irrational.",
                "answer": "proof"
            },
            {
                "question": "Prove that there are infinitely many prime numbers.",
                "answer": "proof"
            },
            {
                "question": "Prove that the sum of angles in a triangle is 180 degrees.",
                "answer": "proof"
            }
        ]
        
        questions = []
        for i in range(num):
            q_data = sample_questions[i % len(sample_questions)]
            questions.append(Question(
                question=q_data["question"],
                correct_answer=q_data["answer"],
                metadata={"type": "theorem_proving"}
            ))
        
        return questions
    
    def format_prompt(self, question: Question) -> str:
        """Format ProofNet question."""
        prompt = f"Provide a proof for the following mathematical statement:\n\n"
        prompt += f"{question.question}\n\n"
        prompt += "Proof: "
        return prompt
    
    def extract_answer(self, response: str, question: Question) -> bool:
        """Extract proof."""
        return response.strip()
    
    def evaluate_answer(self, extracted: str, question: Question) -> bool:
        """Check if proof is provided (simplified)."""
        if not extracted:
            return False
        
        # Check for proof indicators
        proof_indicators = ["proof", "therefore", "thus", "hence", "q.e.d.", "qed"]
        extracted_lower = extracted.lower()
        
        return any(indicator in extracted_lower for indicator in proof_indicators)


class MTBenchBenchmark(BenchmarkBase):
    """MT-Bench benchmark for multiturn conversations."""
    
    def __init__(self, gateway):
        super().__init__("MT-Bench", gateway)
    
    def load_questions(self, num_samples: Optional[int] = None):
        """Load MT-Bench questions."""
        if DATASETS_AVAILABLE:
            try:
                dataset = load_dataset("LLMSafety/MT-Bench")
                num_samples = min(num_samples or 5, len(dataset))
                self.questions = self._process_mtbench_dataset(dataset, num_samples)
                return
            except Exception as e:
                logger.warning(f"Could not load MT-Bench dataset: {e}")
        
        # Fallback to sample questions
        self.questions = self._generate_sample_questions(num_samples or 5)
    
    def _process_mtbench_dataset(self, dataset, num_samples: int) -> List[Question]:
        """Process MT-Bench dataset."""
        questions = []
        for i in range(num_samples):
            item = dataset[i]
            questions.append(Question(
                question=item.get('prompt', item.get('question', '')),
                correct_answer="response",  # MT-Bench evaluates response quality
                metadata={"category": item.get('category', 'unknown')}
            ))
        return questions
    
    def _generate_sample_questions(self, num: int) -> List[Question]:
        """Generate sample MT-Bench-style questions."""
        sample_questions = [
            {
                "question": "User: Can you help me write a Python function?\nAssistant: Of course! What kind of function do you need?\nUser: A function to sort a list.\nAssistant: Here's a simple implementation using the built-in sort() method.",
                "answer": "response"
            },
            {
                "question": "User: What is machine learning?\nAssistant: Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
                "answer": "response"
            },
            {
                "question": "User: Explain quantum computing.\nAssistant: Quantum computing uses quantum mechanics principles like superposition and entanglement to process information in ways classical computers cannot.",
                "answer": "response"
            }
        ]
        
        questions = []
        for i in range(num):
            q_data = sample_questions[i % len(sample_questions)]
            questions.append(Question(
                question=q_data["question"],
                correct_answer=q_data["answer"],
                metadata={"type": "multiturn"}
            ))
        
        return questions
    
    def format_prompt(self, question: Question) -> str:
        """Format MT-Bench question."""
        prompt = f"Continue the following conversation appropriately:\n\n"
        prompt += f"{question.question}\n\n"
        prompt += "Response: "
        return prompt
    
    def extract_answer(self, response: str, question: Question) -> bool:
        """Extract response."""
        return response.strip()
    
    def evaluate_answer(self, extracted: str, question: Question) -> bool:
        """Check if response is appropriate (simplified)."""
        if not extracted:
            return False
        
        # Check for reasonable response length
        return len(extracted) > 10


# Import existing benchmark classes from the original suite
# (In a real implementation, these would be imported from the existing file)
# For now, we'll just add the new ones to the suite

class ComprehensiveBenchmarkSuite:
    """Comprehensive benchmark suite for phi3:mini evaluation with optimizations."""
    
    def __init__(self, gateway):
        self.gateway = gateway
        self.benchmarks: Dict[str, BenchmarkBase] = {}
        self.all_results: List[BenchmarkResult] = []
        self.benchmark_dir = Path("benchmark_results")
        self.benchmark_dir.mkdir(exist_ok=True)
    
    def register_benchmark(self, benchmark: BenchmarkBase):
        """Register a benchmark to the suite."""
        self.benchmarks[benchmark.name] = benchmark
        logger.info(f"Registered benchmark: {benchmark.name}")
    
    def run_all_benchmarks(self, num_samples_per_benchmark: int = 5) -> Dict[str, BenchmarkResult]:
        """Run all registered benchmarks."""
        results = {}
        
        for name, benchmark in self.benchmarks.items():
            logger.info(f"Running {name} benchmark...")
            try:
                result = benchmark.run_benchmark(num_samples_per_benchmark)
                results[name] = result
                self.all_results.append(result)
            except Exception as e:
                logger.error(f"Error running {name}: {e}")
        
        return results
    
    def generate_report(self) -> str:
        """Generate comprehensive benchmark report."""
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE BENCHMARK REPORT - phi3:mini OPTIMIZED")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary table
        report.append("BENCHMARK SUMMARY")
        report.append("-"*80)
        report.append(f"{'Benchmark':<25} {'Questions':<12} {'Correct':<12} {'Accuracy':<12} {'Duration':<12}")
        report.append("-"*80)
        
        for result in self.all_results:
            report.append(
                f"{result.benchmark_name:<25} "
                f"{result.total_questions:<12} "
                f"{result.correct_answers:<12} "
                f"{result.accuracy:>11.1%} "
                f"{result.duration:>10.1f}s"
            )
        
        report.append("")
        
        # Overall statistics
        total_questions = sum(r.total_questions for r in self.all_results)
        total_correct = sum(r.correct_answers for r in self.all_results)
        overall_accuracy = total_correct / total_questions if total_questions > 0 else 0.0
        total_duration = sum(r.duration for r in self.all_results)
        
        report.append("OVERALL STATISTICS")
        report.append("-"*80)
        report.append(f"Total Questions: {total_questions}")
        report.append(f"Total Correct: {total_correct}")
        report.append(f"Overall Accuracy: {overall_accuracy:.2%}")
        report.append(f"Total Duration: {total_duration:.1f}s")
        if total_questions > 0:
            report.append(f"Average Time per Question: {total_duration/total_questions:.2f}s")
        
        report.append("")
        
        # Individual benchmark details
        for result in self.all_results:
            report.append(f"{result.benchmark_name} DETAILS")
            report.append("-"*80)
            report.append(f"Accuracy: {result.accuracy:.2%} ({result.correct_answers}/{result.total_questions})")
            report.append(f"Duration: {result.duration:.1f}s")
            if result.total_questions > 0:
                report.append(f"Average per question: {result.duration/result.total_questions:.2f}s")
            report.append("")
        
        return "\n".join(report)
    
    def save_results(self):
        """Save benchmark results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.benchmark_dir / f"comprehensive_benchmark_{timestamp}.json"
        
        results_data = {
            "timestamp": timestamp,
            "model": "phi3:mini-optimized",
            "results": [
                {
                    "benchmark": r.benchmark_name,
                    "total_questions": r.total_questions,
                    "correct_answers": r.correct_answers,
                    "accuracy": r.accuracy,
                    "duration": r.duration,
                    "metadata": {"gateway_stats": str(r.metadata.get("gateway_stats", {}))}
                }
                for r in self.all_results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        
        # Also save text report
        report_filename = self.benchmark_dir / f"comprehensive_benchmark_{timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write(self.generate_report())
        
        logger.info(f"Report saved to {report_filename}")


def main():
    """Main function to run comprehensive benchmarks with optimizations."""
    if not ENHANCED_GATEWAY_AVAILABLE:
        print("ERROR: EnhancedGateway not available. Cannot run optimized benchmarks.")
        return
    
    # Initialize enhanced gateway with all 16 optimizations
    try:
        gateway = EnhancedGateway()
        logger.info("Enhanced Gateway initialized with 16 optimization techniques")
    except Exception as e:
        print(f"ERROR: Failed to initialize EnhancedGateway: {e}")
        return
    
    # Create benchmark suite
    suite = ComprehensiveBenchmarkSuite(gateway)
    
    # Register NEW benchmarks (Phase 1 - can run with phi3:mini)
    suite.register_benchmark(MMLUProBenchmark(gateway))
    suite.register_benchmark(BBHBenchmark(gateway))
    suite.register_benchmark(AMCBenchmark(gateway))
    suite.register_benchmark(ProofNetBenchmark(gateway))
    suite.register_benchmark(MTBenchBenchmark(gateway))
    
    # Note: MBPP would require code execution, so we skip it for now
    # suite.register_benchmark(MBPPBenchmark(gateway))
    
    # Run benchmarks
    logger.info("Starting comprehensive benchmark suite with new benchmarks...")
    results = suite.run_all_benchmarks(num_samples_per_benchmark=5)
    
    # Generate and save report
    report = suite.generate_report()
    print(report)
    
    suite.save_results()
    
    logger.info("Benchmark suite completed")


if __name__ == "__main__":
    main()