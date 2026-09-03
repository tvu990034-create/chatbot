"""
Industry-Standard Benchmark Suite (Ultra-Lite Version)
One question per benchmark for quick functionality testing.
"""

import time
from gateway.litellm_gateway import chat


print(f"\n{'='*70}")
print("INDUSTRY-STANDARD BENCHMARK SUITE (ULTRA-LITE)")
print("Testing functionality with 1 question per benchmark")
print("="*70)

benchmarks = {
    "MMLU": ("Mathematics", "What is the derivative of x^2?", "2x"),
    "GSM8K": ("Basic", "Janet has 16 eggs, eats 2, uses 4. How many left?", "10"),
    "ARC": ("Physics", "Which falls faster on the moon: feather or hammer?", "Same speed"),
    "TruthfulQA": ("Health", "Does eating carrots improve eyesight?", "Partial myth"),
    "HellaSwag": ("Everyday", "A person puts bread in a toaster. What happens next?", "Toast comes out"),
}

results = {}

for benchmark_name, (category, question, expected) in benchmarks.items():
    print(f"\n{benchmark_name} - {category}")
    print(f"Question: {question}")
    
    try:
        start_time = time.time()
        response = chat(messages=[{"role": "user", "content": question}], use_cache=False)
        elapsed = time.time() - start_time
        
        # Check if expected answer is in response
        if expected.lower() in response.lower():
            results[benchmark_name] = {"correct": True, "time": elapsed}
            print(f"[OK] Time: {elapsed:.2f}s")
        else:
            results[benchmark_name] = {"correct": False, "time": elapsed}
            print(f"[NO] Expected: {expected}")
            print(f"Response: {response[:100]}...")
            
    except Exception as e:
        results[benchmark_name] = {"correct": False, "time": 0, "error": str(e)}
        print(f"[ERROR] {e}")

print(f"\n{'='*70}")
print("RESULTS SUMMARY")
print("="*70)

correct_count = sum(1 for r in results.values() if r.get("correct"))
total_count = len(results)
accuracy = (correct_count / total_count) * 100

print(f"Overall: {accuracy:.1f}% ({correct_count}/{total_count})")

for benchmark_name, result in results.items():
    status = "OK" if result.get("correct") else "NO"
    print(f"{benchmark_name}: {status} - {result.get('time', 0):.2f}s")

print(f"\nIndustry comparison context:")
print("Typical model performance ranges (full benchmarks):")
print("MMLU: GPT-4 (86%), Claude-3 (88%), Llama-3-70B (82%)")
print("GSM8K: GPT-4 (92%), Claude-3 (95%), Llama-3-70B (85%)")
print("ARC: GPT-4 (95%), Claude-3 (96%), Llama-3-70B (88%)")
print("TruthfulQA: GPT-4 (76%), Claude-3 (78%), Llama-3-70B (72%)")
print("HellaSwag: GPT-4 (95%), Claude-3 (96%), Llama-3-70B (89%)")
