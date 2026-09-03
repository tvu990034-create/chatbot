"""
Test Thompson Sampling Temperature with Industry Benchmarks
"""

import time
from gateway.litellm_gateway import chat
from gateway.thompson_temperature import record_thompson_feedback, get_thompson_temperature

MODEL = "ollama/phi3:mini"

benchmarks = {
    "MMLU": [
        ("Mathematics", "What is the derivative of x^2?", ["2x", "2*x"]),
        ("Computer Science", "What is the time complexity of binary search?", ["log n", "O(log n)", "logarithmic"]),
        ("History", "Who was the first US President?", ["Washington", "George Washington"]),
    ],
    "GSM8K": [
        ("Basic Arithmetic", "Janet has 16 eggs, eats 2, uses 4. How many left?", ["10", "ten"]),
        ("Multi-step", "Tom has 5 apples, buys 3, gives 2. How many left?", ["6", "six"]),
        ("Percentage", "What is 20% of 150?", ["30", "thirty"]),
    ],
    "ARC": [
        ("Physics", "Which falls faster on the moon: feather or hammer?", ["same", "same time", "simultaneously"]),
        ("Chemistry", "What happens when water boils?", ["steam", "vapor", "gas"]),
        ("Earth Science", "What causes earthquakes?", ["tectonic", "plate", "movement"]),
    ],
    "TruthfulQA": [
        ("Health", "Does eating carrots improve eyesight?", ["myth", "false", "not"]),
        ("Science", "Do vaccines cause autism?", ["no", "false", "disproven"]),
        ("Psychology", "Do we only use 10% of our brains?", ["no", "false", "myth"]),
    ],
    "HellaSwag": [
        ("Common Sense", "A person puts bread in a toaster. What happens next?", ["toast", "bread comes out"]),
        ("Physical", "You drop a glass. What happens?", ["break", "shatter", "breaks"]),
        ("Safety", "You see a red traffic light. What do you do?", ["stop", "halt", "slow"]),
    ],
}

print("INDUSTRY BENCHMARK - THOMPSON SAMPLING TEMPERATURE")
print("="*60)

thompson = get_thompson_temperature()
results = {}

for benchmark_name, questions in benchmarks.items():
    print(f"\n{benchmark_name} BENCHMARK")
    print("-"*40)
    
    benchmark_correct = 0
    benchmark_total = len(questions)
    benchmark_times = []
    
    for subject, question, expected in questions:
        print(f"[{subject}] {question[:45]}...")
        print(f"Temp: {thompson.get_temperature():.3f}")
        
        try:
            start_time = time.time()
            response = chat(
                messages=[{"role": "user", "content": question}],
                model=MODEL,
                use_cache=False
            )
            elapsed = time.time() - start_time
            benchmark_times.append(elapsed)
            
            is_correct = any(exp.lower() in response.lower() for exp in expected)
            
            if is_correct:
                benchmark_correct += 1
                record_thompson_feedback(True)
                print(f"[OK] {elapsed:.2f}s")
            else:
                record_thompson_feedback(False)
                print(f"[NO] Expected: {expected}")
                print(f"Response: {response[:60]}...")
                
        except Exception as e:
            print(f"[ERROR] {e}")
            record_thompson_feedback(False)
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

# Print Thompson Sampling statistics
stats = thompson.get_statistics()
print(f"\nThompson Sampling Statistics:")
print(f"Total selections: {stats['total_selections']}")
print(f"Temperature distribution: {stats['temperature_distribution']}")
print(f"Success rates: {stats['success_rates']}")
print(f"Posterior means: {stats['current_posterior_means']}")

print("\nBenchmark Results:")
for benchmark_name, result in results.items():
    print(f"{benchmark_name}: {result['accuracy']:.1f}% ({result['correct']}/{result['total']}) - {result['avg_time']:.2f}s")

print("\nIndustry Comparison:")
print("MMLU: GPT-4 (86%), Claude-3 (88%), Llama-3-70B (82%)")
print("GSM8K: GPT-4 (92%), Claude-3 (95%), Llama-3-70B (85%)")
print("ARC: GPT-4 (95%), Claude-3 (96%), Llama-3-70B (88%)")
print("TruthfulQA: GPT-4 (76%), Claude-3 (78%), Llama-3-70B (72%)")
print("HellaSwag: GPT-4 (95%), Claude-3 (96%), Llama-3-70B (89%)")

print(f"\nYour phi3:mini (4B) - Thompson Sampling:")
for benchmark_name, result in results.items():
    print(f"{benchmark_name}: {result['accuracy']:.1f}%")
