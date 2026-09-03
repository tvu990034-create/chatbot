"""
Harder Industry Benchmark Test - Challenging Questions
Tests model capabilities with more difficult questions that 100% accuracy is unlikely
"""

import time
from gateway.litellm_gateway import chat

MODEL = "ollama/phi3:mini"

# Harder, more challenging benchmarks
hard_benchmarks = {
    "Advanced Math": [
        ("Calculus", "What is the integral of e^x from 0 to infinity?", ["diverges", "infinity", "undefined"]),
        ("Linear Algebra", "What is the determinant of a 3x3 identity matrix?", ["1", "one"]),
        ("Number Theory", "Is 2^31 - 1 a prime number? Show your work.", ["yes", "prime", "Mersenne"]),
    ],
    "Complex Reasoning": [
        ("Logic Puzzle", "If all A are B, and some B are C, can we conclude that some A are C?", ["no", "cannot conclude"]),
        ("Multi-step", "A car travels at 60 mph for 2 hours, then 40 mph for 3 hours. What is the average speed?", ["48", "48 mph"]),
        ("Counterfactual", "If gravity were twice as strong, how would this affect a pendulum's period?", ["shorter", "decrease", "faster"]),
    ],
    "Advanced Knowledge": [
        ("Physics", "Explain the difference between special and general relativity.", ["curvature", "acceleration", "gravity"]),
        ("Chemistry", "What is the electron configuration of iron (Fe)?", ["[Ar] 3d6 4s2", "2 8 14 2"]),
        ("Biology", "What is the difference between DNA and RNA?", ["thymine", "uracil", "sugar"]),
    ],
    "Code Generation": [
        ("Python", "Write a function to reverse a linked list in Python.", ["def", "prev", "next"]),
        ("Algorithm", "What is the time complexity of quicksort in the worst case?", ["O(n^2)", "n squared"]),
        ("Data Structure", "What is the difference between a stack and a queue?", ["LIFO", "FIFO"]),
    ],
    "Creative & Nuanced": [
        ("Metaphor", "Explain the concept of 'time dilation' using a metaphor.", ["moving", "clock", "relative"]),
        ("Analogy", "How is a neural network like a brain?", ["neurons", "connections", "weights"]),
        ("Synthesis", "Combine quantum mechanics and classical mechanics in one explanation.", ["macro", "micro", "scale"]),
    ],
}

print("HARDER INDUSTRY BENCHMARK TEST")
print("="*60)

results = {}

for benchmark_name, questions in hard_benchmarks.items():
    print(f"\n{benchmark_name} BENCHMARK")
    print("-"*40)
    
    benchmark_correct = 0
    benchmark_total = len(questions)
    benchmark_times = []
    
    for subject, question, expected in questions:
        print(f"[{subject}] {question[:50]}...")
        
        try:
            start_time = time.time()
            response = chat(
                messages=[{"role": "user", "content": question}],
                model=MODEL,
                use_cache=False
            )
            elapsed = time.time() - start_time
            benchmark_times.append(elapsed)
            
            # More flexible matching for harder questions
            is_correct = any(exp.lower() in response.lower() for exp in expected)
            
            if is_correct:
                benchmark_correct += 1
                print(f"[OK] {elapsed:.2f}s")
            else:
                print(f"[NO] Expected: {expected}")
                print(f"Response: {response[:80]}...")
                
        except Exception as e:
            print(f"[ERROR] {e}")
            benchmark_times.append(0)
    
    benchmark_accuracy = (benchmark_correct / benchmark_total) * 100
    avg_time = sum(benchmark_times) / len(benchmark_times) if benchmark_times else 0
    
    results[benchmark_name] = {
        "correct": benchmark_correct,
        "total": benchmark_total,
        "accuracy": benchmark_accuracy,
        "avg_time": avg_time
    }
    
    print(f"{benchmark_name}: {benchmark_accuracy:.1f}% ({benchmark_correct}/{benchmark_total}) - {avg_time:.2f}s avg")

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

total_correct = sum(r["correct"] for r in results.values())
total_questions = sum(r["total"] for r in results.values())
overall_accuracy = (total_correct / total_questions) * 100
total_avg_time = sum(r["avg_time"] for r in results.values()) / len(results)

print(f"Overall: {overall_accuracy:.1f}% ({total_correct}/{total_questions})")
print(f"Average time per question: {total_avg_time:.2f}s")

print("\nBenchmark Results:")
for benchmark_name, result in results.items():
    print(f"{benchmark_name}: {result['accuracy']:.1f}% ({result['correct']}/{result['total']}) - {result['avg_time']:.2f}s")

if overall_accuracy < 100:
    print(f"\n⚠️  Model achieved {overall_accuracy:.1f}% - These benchmarks are challenging the model!")
else:
    print(f"\n✅ Model achieved 100% - Still too easy, need even harder questions!")
