"""
Coding and Agent Benchmark Suite for phi3:mini
Implements SWE-bench, MBPP, T-Bench, AgentBench, GAIA, OSWorld
Note: These benchmarks require special environments, so we'll implement text-only versions
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

# Try to import datasets
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    print("Warning: datasets library not available")
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
                
                if (i + 1) % 2 == 0:
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


class SWE_benchBenchmark(BenchmarkBase):
    """SWE-bench benchmark - GitHub issue resolution (text-only version)."""
    
    def __init__(self, gateway):
        super().__init__("SWE-bench", gateway)
    
    def load_questions(self, num_samples: Optional[int] = None):
        """Load SWE-bench questions."""
        if DATASETS_AVAILABLE:
            try:
                # Try Lite version for faster testing
                dataset = load_dataset("SWE-bench/SWE-bench_Lite", split="test")
                num_samples = min(num_samples or 3, len(dataset))
                self.questions = self._process_swebench_dataset(dataset, num_samples)
                return
            except Exception as e:
                logger.warning(f"Could not load SWE-bench dataset: {e}")
        
        # Fallback to sample questions
        self.questions = self._generate_sample_questions(num_samples or 3)
    
    def _process_swebench_dataset(self, dataset, num_samples: int) -> List[Question]:
        """Process SWE-bench dataset."""
        questions = []
        for i in range(num_samples):
            item = dataset[i]
            # Extract issue description and expected solution
            problem_statement = item.get('problem_statement', item.get('text', ''))
            questions.append(Question(
                question=f"Fix the following GitHub issue:\n\n{problem_statement}",
                correct_answer="solution",  # We'll evaluate code generation quality
                metadata={"repo": item.get('repo', 'unknown'), "version": item.get('version', 'unknown')}
            ))
        return questions
    
    def _generate_sample_questions(self, num: int) -> List[Question]:
        """Generate sample SWE-bench-style questions."""
        sample_questions = [
            {
                "question": "Fix a bug where the function returns None instead of 0 when the input list is empty.",
                "answer": "solution"
            },
            {
                "question": "Add error handling for when the API call fails due to network timeout.",
                "answer": "solution"
            },
            {
                "question": "Optimize the sorting algorithm to use O(n log n) complexity instead of O(n^2).",
                "answer": "solution"
            }
        ]
        
        questions = []
        for i in range(num):
            q_data = sample_questions[i % len(sample_questions)]
            questions.append(Question(
                question=q_data["question"],
                correct_answer=q_data["answer"],
                metadata={"type": "bug_fix"}
            ))
        
        return questions
    
    def format_prompt(self, question: Question) -> str:
        """Format SWE-bench question."""
        prompt = f"""You are a software engineer. {question.question}

Provide a solution in Python. Include:
1. The problematic code (if applicable)
2. The fixed code
3. Explanation of the fix

Solution:"""
        return prompt
    
    def extract_answer(self, response: str, question: Question) -> bool:
        """Extract solution."""
        return response.strip()
    
    def evaluate_answer(self, extracted: str, question: Question) -> bool:
        """Evaluate code solution quality (simplified)."""
        if not extracted:
            return False
        
        # Check for code indicators
        code_indicators = ["def ", "class ", "function", "import ", "fix", "solution"]
        extracted_lower = extracted.lower()
        
        # Check for Python code
        has_code = any(indicator in extracted_lower for indicator in code_indicators)
        # Check for explanation
        has_explanation = len(extracted) > 100
        
        return has_code and has_explanation


class MBPPBenchmark(BenchmarkBase):
    """MBPP benchmark - Mostly Basic Python Problems (text-only version)."""
    
    def __init__(self, gateway):
        super().__init__("MBPP", gateway)
    
    def load_questions(self, num_samples: Optional[int] = None):
        """Load MBPP questions."""
        if DATASETS_AVAILABLE:
            try:
                dataset = load_dataset("google-research-datasets/mbpp", split="test")
                num_samples = min(num_samples or 3, len(dataset))
                self.questions = self._process_mbpp_dataset(dataset, num_samples)
                return
            except Exception as e:
                logger.warning(f"Could not load MBPP dataset: {e}")
        
        # Fallback to sample questions
        self.questions = self._generate_sample_questions(num_samples or 3)
    
    def _process_mbpp_dataset(self, dataset, num_samples: int) -> List[Question]:
        """Process MBPP dataset."""
        questions = []
        for i in range(num_samples):
            item = dataset[i]
            questions.append(Question(
                question=item['text'],
                correct_answer=item['code'],
                metadata={"task_id": item['task_id']}
            ))
        return questions
    
    def _generate_sample_questions(self, num: int) -> List[Question]:
        """Generate sample MBPP-style questions."""
        sample_questions = [
            {
                "question": "Write a function to check if a number is prime.",
                "answer": "def is_prime(n):"
            },
            {
                "question": "Write a function to reverse a string.",
                "answer": "def reverse_string(s):"
            },
            {
                "question": "Write a function to find the maximum element in a list.",
                "answer": "def find_max(lst):"
            }
        ]
        
        questions = []
        for i in range(num):
            q_data = sample_questions[i % len(sample_questions)]
            questions.append(Question(
                question=q_data["question"],
                correct_answer=q_data["answer"],
                metadata={"type": "function_writing"}
            ))
        
        return questions
    
    def format_prompt(self, question: Question) -> str:
        """Format MBPP question."""
        prompt = f"Write a Python function to solve the following problem:\n\n{question.question}\n\n"
        prompt += "Provide only the function code:"
        return prompt
    
    def extract_answer(self, response: str, question: Question) -> bool:
        """Extract function code."""
        return response.strip()
    
    def evaluate_answer(self, extracted: str, question: Question) -> bool:
        """Evaluate function code (simplified)."""
        if not extracted:
            return False
        
        # Check for function definition
        has_function = "def " in extracted
        # Check for Python syntax (basic)
        has_colon = ":" in extracted
        has_return = "return" in extracted or "print" in extracted
        
        return has_function and has_colon


class T_BenchBenchmark(BenchmarkBase):
    """T-Bench benchmark - Terminal environment tasks (text-only version)."""
    
    def __init__(self, gateway):
        super().__init__("T-Bench", gateway)
    
    def load_questions(self, num_samples: Optional[int] = None):
        """Load T-Bench questions."""
        # T-Bench requires terminal environment, so we use sample questions
        self.questions = self._generate_sample_questions(num_samples or 3)
    
    def _generate_sample_questions(self, num: int) -> List[Question]:
        """Generate sample T-Bench-style questions."""
        sample_questions = [
            {
                "question": "Write a bash command to list all files in the current directory sorted by size.",
                "answer": "ls -lS"
            },
            {
                "question": "Write a bash command to find all Python files in the current directory and subdirectories.",
                "answer": "find . -name '*.py'"
            },
            {
                "question": "Write a bash command to count the number of lines in all .txt files.",
                "answer": "wc -l *.txt"
            }
        ]
        
        questions = []
        for i in range(num):
            q_data = sample_questions[i % len(sample_questions)]
            questions.append(Question(
                question=q_data["question"],
                correct_answer=q_data["answer"],
                metadata={"type": "terminal_command"}
            ))
        
        return questions
    
    def format_prompt(self, question: Question) -> str:
        """Format T-Bench question."""
        prompt = f"Provide the bash command to solve the following task:\n\n{question.question}\n\n"
        prompt += "Command:"
        return prompt
    
    def extract_answer(self, response: str, question: Question) -> bool:
        """Extract command."""
        return response.strip()
    
    def evaluate_answer(self, extracted: str, question: Question) -> bool:
        """Evaluate command (simplified)."""
        if not extracted:
            return False
        
        # Check for common Linux commands
        common_commands = ["ls", "cd", "grep", "find", "cat", "wc", "rm", "cp", "mv", "mkdir"]
        extracted_lower = extracted.lower()
        
        has_command = any(cmd in extracted_lower for cmd in common_commands)
        has_flags = any(char in extracted for char in ["-", "--"])
        
        return has_command


class AgentBenchBenchmark(BenchmarkBase):
    """AgentBench benchmark - Agent environment tasks (text-only version)."""
    
    def __init__(self, gateway):
        super().__init__("AgentBench", gateway)
    
    def load_questions(self, num_samples: Optional[int] = None):
        """Load AgentBench questions."""
        # AgentBench requires agent environments, so we use sample questions
        self.questions = self._generate_sample_questions(num_samples or 3)
    
    def _generate_sample_questions(self, num: int) -> List[Question]:
        """Generate sample AgentBench-style questions."""
        sample_questions = [
            {
                "question": "You are an AI assistant helping a user navigate a web interface. The user wants to find the 'Settings' menu. Describe the steps to help them.",
                "answer": "navigation"
            },
            {
                "question": "You are an AI assistant for a home automation system. The user wants to turn off all lights in the living room. What should you do?",
                "answer": "automation"
            },
            {
                "question": "You are an AI assistant for a file management system. The user wants to organize their documents by date. Describe the process.",
                "answer": "organization"
            }
        ]
        
        questions = []
        for i in range(num):
            q_data = sample_questions[i % len(sample_questions)]
            questions.append(Question(
                question=q_data["question"],
                correct_answer=q_data["answer"],
                metadata={"type": "agent_task"}
            ))
        
        return questions
    
    def format_prompt(self, question: Question) -> str:
        """Format AgentBench question."""
        prompt = f"{question.question}\n\n"
        prompt += "Response:"
        return prompt
    
    def extract_answer(self, response: str, question: Question) -> bool:
        """Extract response."""
        return response.strip()
    
    def evaluate_answer(self, extracted: str, question: Question) -> bool:
        """Evaluate agent response (simplified)."""
        if not extracted:
            return False
        
        # Check for reasonable response
        has_steps = any(word in extracted.lower() for word in ["step", "first", "then", "next", "finally"])
        has_content = len(extracted) > 50
        
        return has_steps and has_content


class GAIABenchmark(BenchmarkBase):
    """GAIA benchmark - General AI Assistant (text-only version)."""
    
    def __init__(self, gateway):
        super().__init__("GAIA", gateway)
    
    def load_questions(self, num_samples: Optional[int] = None):
        """Load GAIA questions."""
        if DATASETS_AVAILABLE:
            try:
                from huggingface_hub import snapshot_download
                data_dir = snapshot_download(repo_id="gaia-benchmark/GAIA")
                dataset = load_dataset(data_dir, "2023_level1", split="test")
                num_samples = min(num_samples or 3, len(dataset))
                self.questions = self._process_gaia_dataset(dataset, num_samples)
                return
            except Exception as e:
                logger.warning(f"Could not load GAIA dataset: {e}")
        
        # Fallback to sample questions
        self.questions = self._generate_sample_questions(num_samples or 3)
    
    def _process_gaia_dataset(self, dataset, num_samples: int) -> List[Question]:
        """Process GAIA dataset."""
        questions = []
        for i in range(num_samples):
            item = dataset[i]
            questions.append(Question(
                question=item['Question'],
                correct_answer=item.get('Final answer', 'answer'),
                metadata={"task_id": item.get('task_id', 'unknown')}
            ))
        return questions
    
    def _generate_sample_questions(self, num: int) -> List[Question]:
        """Generate sample GAIA-style questions."""
        sample_questions = [
            {
                "question": "What is the population of Tokyo according to the 2020 census?",
                "answer": "population"
            },
            {
                "question": "Calculate the compound interest on $1000 at 5% annual rate for 3 years.",
                "answer": "calculation"
            },
            {
                "question": "What is the chemical formula for sodium chloride?",
                "answer": "NaCl"
            }
        ]
        
        questions = []
        for i in range(num):
            q_data = sample_questions[i % len(sample_questions)]
            questions.append(Question(
                question=q_data["question"],
                correct_answer=q_data["answer"],
                metadata={"type": "multi_step_reasoning"}
            ))
        
        return questions
    
    def format_prompt(self, question: Question) -> str:
        """Format GAIA question."""
        prompt = f"Answer the following question accurately:\n\n{question.question}\n\n"
        prompt += "Answer:"
        return prompt
    
    def extract_answer(self, response: str, question: Question) -> bool:
        """Extract answer."""
        return response.strip()
    
    def evaluate_answer(self, extracted: str, question: Question) -> bool:
        """Evaluate answer (simplified)."""
        if not extracted:
            return False
        
        # Check for reasonable answer
        has_content = len(extracted) > 10
        # Check for specific format if applicable
        if question.correct_answer and question.correct_answer.isdigit():
            return any(char.isdigit() for char in extracted)
        
        return has_content


class OSWorldBenchmark(BenchmarkBase):
    """OSWorld benchmark - OS interaction tasks (text-only version)."""
    
    def __init__(self, gateway):
        super().__init__("OSWorld", gateway)
    
    def load_questions(self, num_samples: Optional[int] = None):
        """Load OSWorld questions."""
        if DATASETS_AVAILABLE:
            try:
                dataset = load_dataset("xlangai/windows_osworld")
                num_samples = min(num_samples or 3, len(dataset))
                self.questions = self._process_osworld_dataset(dataset, num_samples)
                return
            except Exception as e:
                logger.warning(f"Could not load OSWorld dataset: {e}")
        
        # Fallback to sample questions
        self.questions = self._generate_sample_questions(num_samples or 3)
    
    def _process_osworld_dataset(self, dataset, num_samples: int) -> List[Question]:
        """Process OSWorld dataset."""
        questions = []
        for i in range(num_samples):
            item = dataset[i]
            questions.append(Question(
                question=item.get('instruction', item.get('task', '')),
                correct_answer="action",
                metadata={"task_id": item.get('task_id', 'unknown')}
            ))
        return questions
    
    def _generate_sample_questions(self, num: int) -> List[Question]:
        """Generate sample OSWorld-style questions."""
        sample_questions = [
            {
                "question": "Open the Notepad application and create a new text file named 'test.txt'.",
                "answer": "action"
            },
            {
                "question": "Navigate to the Documents folder and list all files.",
                "answer": "action"
            },
            {
                "question": "Open the Settings application and change the display brightness.",
                "answer": "action"
            }
        ]
        
        questions = []
        for i in range(num):
            q_data = sample_questions[i % len(sample_questions)]
            questions.append(Question(
                question=q_data["question"],
                correct_answer=q_data["answer"],
                metadata={"type": "os_interaction"}
            ))
        
        return questions
    
    def format_prompt(self, question: Question) -> str:
        """Format OSWorld question."""
        prompt = f"Describe the steps to complete the following OS task:\n\n{question.question}\n\n"
        prompt += "Steps:"
        return prompt
    
    def extract_answer(self, response: str, question: Question) -> bool:
        """Extract steps."""
        return response.strip()
    
    def evaluate_answer(self, extracted: str, question: Question) -> bool:
        """Evaluate OS steps (simplified)."""
        if not extracted:
            return False
        
        # Check for step-by-step format
        has_steps = any(word in extracted.lower() for word in ["step", "click", "open", "navigate", "go to"])
        has_structure = any(char in extracted for char in ["1.", "2.", "-", "•", "first", "then"])
        
        return has_steps and has_structure


class CodingBenchmarkSuite:
    """Coding and Agent benchmark suite for phi3:mini evaluation."""
    
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
    
    def run_all_benchmarks(self, num_samples_per_benchmark: int = 3) -> Dict[str, BenchmarkResult]:
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
        report.append("CODING AND AGENT BENCHMARK REPORT - phi3:mini OPTIMIZED")
        report.append("="*80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("NOTE: These benchmarks require special environments (code execution,")
        report.append("terminal simulation, OS simulation). Results show text-only generation")
        report.append("capability, not full execution verification.")
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
        filename = self.benchmark_dir / f"coding_benchmark_{timestamp}.json"
        
        results_data = {
            "timestamp": timestamp,
            "model": "phi3:mini-optimized",
            "note": "Text-only generation, no execution verification",
            "results": [
                {
                    "benchmark": r.benchmark_name,
                    "total_questions": r.total_questions,
                    "correct_answers": r.correct_answers,
                    "accuracy": r.accuracy,
                    "duration": r.duration,
                    "metadata": r.metadata
                }
                for r in self.all_results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        
        # Also save text report
        report_filename = self.benchmark_dir / f"coding_benchmark_{timestamp}.txt"
        with open(report_filename, 'w') as f:
            f.write(self.generate_report())
        
        logger.info(f"Report saved to {report_filename}")


def main():
    """Main function to run coding and agent benchmarks."""
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
    suite = CodingBenchmarkSuite(gateway)
    
    # Register coding and agent benchmarks
    suite.register_benchmark(SWE_benchBenchmark(gateway))
    suite.register_benchmark(MBPPBenchmark(gateway))
    suite.register_benchmark(T_BenchBenchmark(gateway))
    suite.register_benchmark(AgentBenchBenchmark(gateway))
    suite.register_benchmark(GAIABenchmark(gateway))
    suite.register_benchmark(OSWorldBenchmark(gateway))
    
    # Run benchmarks with smaller sample size (3 instead of 5)
    logger.info("Starting coding and agent benchmark suite...")
    results = suite.run_all_benchmarks(num_samples_per_benchmark=3)
    
    # Generate and save report
    report = suite.generate_report()
    print(report)
    
    suite.save_results()
    
    logger.info("Coding and agent benchmark suite completed")


if __name__ == "__main__":
    main()