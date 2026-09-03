"""
Quick Industry Benchmark Test - 1 question per benchmark
"""

import time
import litellm
from litellm import completion

MODEL = "ollama/phi3:mini"

questions = [
    ("MMLU", "What is the derivative of x^2?", ["2x", "2*x"]),
    ("GSM8K", "Janet has 16 eggs, eats 2, uses 4. How many left?", ["10", "ten"]),
    ("ARC", "Which falls faster on the moon: feather or hammer?", ["same", "same time"]),
    ("TruthfulQA", "Does eating carrots improve eyesight?", ["myth", "false", "not"]),
    ("HellaSwag", "A person puts bread in a toaster. What happens next?", ["toast", "bread comes out"]),
]

print("QUICK INDUSTRY BENCHMARK TEST")
print("="*50)

results = {}

for benchmark, question, expected in questions:
    print(f"\n{benchmark}: {question[:40]}...")
    
    try:
        start_time = time.time()
        response = completion(
            model=MODEL,
            messages=[{"role": "user", "content": question}],
            temperature=0.1
        )
        elapsed = time.time() - start_time
        
        reply = response.choices[0].message.content if response.choices else ""
        
        is_correct = any(exp.lower() in reply.lower() for exp in expected)
        
        if is_correct:
            results[benchmark] = {"correct": True, "time": elapsed}
            print(f"[OK] {elapsed:.2f}s")
        else:
            results[benchmark] = {"correct": False, "time": elapsed}
            print(f"[NO] Expected: {expected}")
            print(f"Response: {reply[:80]}...")
            
    except Exception as e:
        results[benchmark] = {"correct": False, "time": 0, "error": str(e)}
        print(f"[ERROR] {e}")

print("\n" + "="*50)
print("RESULTS")

correct = sum(1 for r in results.values() if r.get("correct"))
total = len(results)
accuracy = (correct / total) * 100

print(f"Overall: {accuracy:.1f}% ({correct}/{total})")

for benchmark, result in results.items():
    status = "OK" if result.get("correct") else "NO"
    print(f"{benchmark}: {status} - {result.get('time', 0):.2f}s")

print("\nIndustry Comparison:")
print("MMLU: GPT-4 (86%), Claude-3 (88%), Llama-3-70B (82%)")
print("GSM8K: GPT-4 (92%), Claude-3 (95%), Llama-3-70B (85%)")
print("ARC: GPT-4 (95%), Claude-3 (96%), Llama-3-70B (88%)")
print("TruthfulQA: GPT-4 (76%), Claude-3 (78%), Llama-3-70B (72%)")
print("HellaSwag: GPT-4 (95%), Claude-3 (96%), Llama-3-70B (89%)")

print(f"\nYour phi3:mini (4B): {accuracy:.1f}%")
