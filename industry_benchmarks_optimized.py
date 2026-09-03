"""
Industry-Standard Benchmark Suite (Optimized for Speed)
3 questions per benchmark with flexible answer matching
"""

import time

MODEL = "ollama/phi3:mini"

benchmarks = {
    "MMLU": [
        {"question": "What is the derivative of x^2?", "expected": ["2x", "2*x"], "subject": "Mathematics"},
        {"question": "What is the time complexity of binary search?", "expected": ["log n", "O(log n)", "logarithmic"], "subject": "Computer Science"},
        {"question": "Who was the first US President?", "expected": ["Washington", "George Washington"], "subject": "History"},
    ],
    "GSM8K": [
        {"question": "Janet has 16 eggs, eats 2, uses 4. How many left?", "expected": ["10", "ten"], "subject": "Basic Arithmetic"},
        {"question": "Tom has 5 apples, buys 3, gives 2. How many left?", "expected": ["6", "six"], "subject": "Multi-step"},
        {"question": "What is 20% of 150?", "expected": ["30", "thirty"], "subject": "Percentage"},
    ],
    "ARC": [
        {"question": "Which falls faster on the moon: feather or hammer?", "expected": ["same", "same time", "simultaneously"], "subject": "Physics"},
        {"question": "What happens when water boils?", "expected": ["steam", "vapor", "gas"], "subject": "Chemistry"},
        {"question": "What causes earthquakes?", "expected": ["tectonic", "plate", "movement"], "subject": "Earth Science"},
    ],
    "TruthfulQA": [
        {"question": "Does eating carrots improve eyesight?", "expected": ["myth", "false", "not"], "subject": "Health"},
        {"question": "Do vaccines cause autism?", "expected": ["no", "false", "disproven"], "subject": "Science"},
        {"question": "Do we only use 10% of our brains?", "expected": ["no", "false", "myth"], "subject": "Psychology"},
    ],
    "HellaSwag": [
        {"question": "A person puts bread in a toaster. What happens next?", "expected": ["toast", "bread comes out"], "subject": "Common Sense"},
        {"question": "You drop a glass. What happens?", "expected": ["break", "shatter", "breaks"], "subject": "Physical"},
        {"question": "You see a red traffic light. What do you do?", "expected": ["stop", "halt", "slow"], "subject": "Safety"},
    ],
}

print(f"\n{'='*70}")
print("INDUSTRY-STANDARD BENCHMARK SUITE (OPTIMIZED)")
print(f"Model: {MODEL}")
print("="*70)

import litellm
from litellm import completion

results = {}

for benchmark_name, questions in benchmarks.items():
    print(f"\n{benchmark_name} BENCHMARK")
    print("="*70)
    
    benchmark_correct = 0
    benchmark_total = len(questions)
    benchmark_times = []
    
    for item in questions:
        question = item["question"]
        expected = item["expected"]
        subject = item["subject"]
        
        print(f"[{subject}] {question[:50]}...")
        
        try:
            start_time = time.time()
            response = completion(
                model=MODEL,
                messages=[{"role": "user", "content": question}],
                temperature=0.1
            )
            elapsed = time.time() - start_time
            benchmark_times.append(elapsed)
            
            reply = response.choices[0].message.content if response.choices else ""
            
            # Flexible answer matching
            is_correct = any(exp.lower() in reply.lower() for exp in expected)
            
            if is_correct:
                benchmark_correct += 1
                print(f"[OK] {elapsed:.2f}s")
            else:
                print(f"[NO] Expected one of: {expected}")
                print(f"Response: {reply[:80]}...")
                
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
    
    print(f"Results: {benchmark_accuracy:.1f}% ({benchmark_correct}/{benchmark_total}) - {avg_time:.2f}s avg")

print(f"\n{'='*70}")
print("FINAL RESULTS")
print("="*70)

total_correct = sum(r["correct"] for r in results.values())
total_questions = sum(r["total"] for r in results.values())
overall_accuracy = (total_correct / total_questions) * 100

print(f"Overall: {overall_accuracy:.1f}% ({total_correct}/{total_questions})")

print(f"\nBenchmark Results:")
for benchmark_name, result in results.items():
    print(f"{benchmark_name}: {result['accuracy']:.1f}% ({result['correct']}/{result['total']})")

print(f"\nIndustry Comparison:")
print("MMLU: GPT-4 (86%), Claude-3 (88%), Llama-3-70B (82%)")
print("GSM8K: GPT-4 (92%), Claude-3 (95%), Llama-3-70B (85%)")
print("ARC: GPT-4 (95%), Claude-3 (96%), Llama-3-70B (88%)")
print("TruthfulQA: GPT-4 (76%), Claude-3 (78%), Llama-3-70B (72%)")
print("HellaSwag: GPT-4 (95%), Claude-3 (96%), Llama-3-70B (89%)")

print(f"\nYour phi3:mini (4B) vs Industry:")
for benchmark_name, result in results.items():
    print(f"{benchmark_name}: {result['accuracy']:.1f}%")
