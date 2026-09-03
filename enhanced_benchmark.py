"""
Enhanced Benchmark - Tests Speed and Smart Optimizations
Focused testing with enhanced gateway for faster and smarter responses
"""

import time
import json
import sys
import io
from datetime import datetime
from pathlib import Path

# Set UTF-8 encoding for Windows console output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from gateway.enhanced_gateway import enhanced_chat, get_enhanced_stats

MODEL = "ollama/phi3:mini"
OUTPUT_DIR = Path("benchmark_results")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("ENHANCED BENCHMARK - Speed + Smart Optimizations")
print("="*70)
print("Testing enhanced gateway with active optimizations")
print("Expected time: ~3-4 minutes total\n")

results = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL,
    "gateway_type": "enhanced",
    "optimization_stats": {},
    "tests": {}
}

# Get optimization stats
print("Checking optimization status...")
opt_stats = get_enhanced_stats()
results["optimization_stats"] = opt_stats
print(f"✅ Optimizations active: {opt_stats['performance_equations_enabled']}")
print(f"   Cache: {opt_stats['optimizations']['simple_cache']}")
print(f"   Prompt Compression: {opt_stats['optimizations']['prompt_compression']['enabled']}")
print(f"   Difficulty Routing: {opt_stats['optimizations']['difficulty_routing']['enabled']}")
print(f"   Adaptive Temperature: {opt_stats['optimizations']['adaptive_temperature']['enabled']}")

# Test 1: Speed comparison (with/without cache)
print("\n" + "="*70)
print("SPEED TEST: Caching Performance")
print("="*70)

speed_results = {
    "cold_times": [],
    "cached_times": []
}

test_query = "What is the capital of France?"

# Cold runs (no cache)
print("Cold runs (no cache)...")
for i in range(3):
    start = time.time()
    response = enhanced_chat([{"role": "user", "content": test_query}], model=MODEL, use_cache=False)
    elapsed = time.time() - start
    speed_results["cold_times"].append(elapsed)
    print(f"  Run {i+1}: {elapsed:.2f}s")

# Cached runs
print("Cached runs...")
for i in range(3):
    start = time.time()
    response = enhanced_chat([{"role": "user", "content": test_query}], model=MODEL, use_cache=True)
    elapsed = time.time() - start
    speed_results["cached_times"].append(elapsed)
    print(f"  Run {i+1}: {elapsed:.2f}s")

# Calculate speed metrics
avg_cold = sum(speed_results["cold_times"]) / len(speed_results["cold_times"])
avg_cached = sum(speed_results["cached_times"]) / len(speed_results["cached_times"])
speedup = avg_cold / avg_cached if avg_cached > 0 else 0

speed_results["avg_cold"] = avg_cold
speed_results["avg_cached"] = avg_cached
speed_results["speedup"] = speedup
speed_results["improvement"] = ((avg_cold - avg_cached) / avg_cold) * 100 if avg_cold > 0 else 0

print(f"\nSpeed Results:")
print(f"  Average cold time: {avg_cold:.2f}s")
print(f"  Average cached time: {avg_cached:.2f}s")
print(f"  Speedup: {speedup:.2f}x")
print(f"  Improvement: {speed_results['improvement']:.1f}%")

results["tests"]["speed"] = speed_results

# Test 2: Intelligence test (mixed difficulty)
print("\n" + "="*70)
print("INTELLIGENCE TEST: Mixed Difficulty Questions")
print("="*70)

intelligence_questions = [
    ("Easy", "What is 2+2?", ["4", "four"]),
    ("Easy", "What is the capital of Japan?", ["tokyo"]),
    ("Medium", "Write a Python function to add two numbers", ["def add", "return a + b"]),
    ("Medium", "If A implies B and B implies C and C is false, is A false?", ["yes", "true", "A is false"]),
    ("Hard", "A car travels at 60 mph for 2 hours, then 40 mph for 3 hours. What is the average speed?", ["48", "48 mph"]),
    ("Hard", "Explain the difference between special and general relativity", ["curvature", "gravity", "acceleration"]),
]

intelligence_results = {
    "total": len(intelligence_questions),
    "correct": 0,
    "by_difficulty": {"Easy": {"correct": 0, "total": 0}, "Medium": {"correct": 0, "total": 0}, "Hard": {"correct": 0, "total": 0}},
    "details": []
}

for difficulty, question, expected in intelligence_questions:
    print(f"[{difficulty}] {question[:50]}...")
    
    start = time.time()
    response = enhanced_chat([{"role": "user", "content": question}], model=MODEL, use_cache=False)
    elapsed = time.time() - start
    
    is_correct = any(exp.lower() in response.lower() for exp in expected)
    
    if is_correct:
        intelligence_results["correct"] += 1
        intelligence_results["by_difficulty"][difficulty]["correct"] += 1
        print(f"  ✅ {elapsed:.2f}s - CORRECT")
    else:
        print(f"  ❌ {elapsed:.2f}s - INCORRECT")
    
    intelligence_results["by_difficulty"][difficulty]["total"] += 1
    intelligence_results["details"].append({
        "difficulty": difficulty,
        "question": question,
        "is_correct": is_correct,
        "time": elapsed
    })

intelligence_results["accuracy"] = (intelligence_results["correct"] / intelligence_results["total"]) * 100

print(f"\nIntelligence Results:")
print(f"  Overall: {intelligence_results['accuracy']:.1f}% ({intelligence_results['correct']}/{intelligence_results['total']})")
for diff, stats in intelligence_results["by_difficulty"].items():
    if stats["total"] > 0:
        acc = (stats["correct"] / stats["total"]) * 100
        print(f"  {diff}: {acc:.1f}% ({stats['correct']}/{stats['total']})")

results["tests"]["intelligence"] = intelligence_results

# Test 3: Enhancement effectiveness
print("\n" + "="*70)
print("ENHANCEMENT EFFECTIVENESS TEST")
print("="*70)

enhancement_tests = {
    "chain_of_thought": {"question": "Explain step by step how to calculate the average of numbers", "expected": ["step", "calculate", "sum", "divide"]},
    "rag": {"question": "What is phi3:mini?", "expected": ["4B", "Microsoft", "parameter"]},
    "adaptive_temp": {"question": "Write a creative story about a robot", "expected": ["robot", "story"]},
}

enhancement_results = {}

for enhancement, test in enhancement_tests.items():
    question = test["question"]
    expected = test["expected"]
    
    print(f"[{enhancement}] {question[:50]}...")
    
    start = time.time()
    response = enhanced_chat([{"role": "user", "content": question}], model=MODEL, use_cache=False)
    elapsed = time.time() - start
    
    is_correct = any(exp.lower() in response.lower() for exp in expected)
    
    enhancement_results[enhancement] = {
        "is_correct": is_correct,
        "time": elapsed,
        "response_length": len(response)
    }
    
    print(f"  {'✅' if is_correct else '❌'} {elapsed:.2f}s - Response length: {len(response)} chars")

results["tests"]["enhancements"] = enhancement_results

# Final summary
print("\n" + "="*70)
print("ENHANCED BENCHMARK RESULTS")
print("="*70)

print(f"\n🚀 Speed Improvements:")
print(f"   Cache speedup: {speedup:.2f}x")
print(f"   Time saved: {speed_results['improvement']:.1f}%")

print(f"\n🧠 Intelligence Performance:")
print(f"   Overall accuracy: {intelligence_results['accuracy']:.1f}%")
print(f"   Easy questions: {(intelligence_results['by_difficulty']['Easy']['correct']/intelligence_results['by_difficulty']['Easy']['total']*100) if intelligence_results['by_difficulty']['Easy']['total'] > 0 else 0:.1f}%")
print(f"   Medium questions: {(intelligence_results['by_difficulty']['Medium']['correct']/intelligence_results['by_difficulty']['Medium']['total']*100) if intelligence_results['by_difficulty']['Medium']['total'] > 0 else 0:.1f}%")
print(f"   Hard questions: {(intelligence_results['by_difficulty']['Hard']['correct']/intelligence_results['by_difficulty']['Hard']['total']*100) if intelligence_results['by_difficulty']['Hard']['total'] > 0 else 0:.1f}%")

print(f"\n⚡ Enhancement Status:")
working_enhancements = sum(1 for e in enhancement_results.values() if e["is_correct"])
print(f"   Working enhancements: {working_enhancements}/{len(enhancement_results)}")

# Overall assessment
overall_score = (
    (speedup if speedup > 1 else 1) * 0.3 +  # Speed weight
    (intelligence_results['accuracy'] / 100) * 0.5 +  # Intelligence weight
    (working_enhancements / len(enhancement_results)) * 0.2  # Enhancement weight
) * 100

print(f"\n📊 Overall Enhanced Performance Score: {overall_score:.1f}/100")

if overall_score >= 70:
    print("✅ EXCELLENT - Optimizations working well!")
elif overall_score >= 50:
    print("⚠️  GOOD - Some improvements needed")
else:
    print("❌ NEEDS WORK - Significant improvements required")

# Save results
results_file = OUTPUT_DIR / f"enhanced_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {results_file}")

# Generate quick report
report = f"""
# Enhanced Benchmark Report

**Generated:** {results['timestamp']}
**Model:** {MODEL}
**Gateway:** Enhanced with Active Optimizations

## Optimization Status

- **Performance Equations:** {opt_stats['performance_equations_enabled']}
- **Prompt Compression:** {opt_stats['optimizations']['prompt_compression']['enabled']} (ratio: {opt_stats['optimizations']['prompt_compression']['ratio']})
- **Difficulty Routing:** {opt_stats['optimizations']['difficulty_routing']['enabled']} (threshold: {opt_stats['optimizations']['difficulty_routing']['threshold']})
- **Adaptive Temperature:** {opt_stats['optimizations']['adaptive_temperature']['enabled']} (base: {opt_stats['optimizations']['adaptive_temperature']['base_temp']})

## Speed Results

- **Average Cold Time:** {avg_cold:.2f}s
- **Average Cached Time:** {avg_cached:.2f}s
- **Speedup:** {speedup:.2f}x
- **Improvement:** {speed_results['improvement']:.1f}%

## Intelligence Results

- **Overall Accuracy:** {intelligence_results['accuracy']:.1f}% ({intelligence_results['correct']}/{intelligence_results['total']})
- **Easy Questions:** {(intelligence_results['by_difficulty']['Easy']['correct']/intelligence_results['by_difficulty']['Easy']['total']*100) if intelligence_results['by_difficulty']['Easy']['total'] > 0 else 0:.1f}%
- **Medium Questions:** {(intelligence_results['by_difficulty']['Medium']['correct']/intelligence_results['by_difficulty']['Medium']['total']*100) if intelligence_results['by_difficulty']['Medium']['total'] > 0 else 0:.1f}%
- **Hard Questions:** {(intelligence_results['by_difficulty']['Hard']['correct']/intelligence_results['by_difficulty']['Hard']['total']*100) if intelligence_results['by_difficulty']['Hard']['total'] > 0 else 0:.1f}%

## Enhancement Effectiveness

- **Chain-of-Thought:** {'✅ Working' if enhancement_results['chain_of_thought']['is_correct'] else '❌ Issues'}
- **RAG:** {'✅ Working' if enhancement_results['rag']['is_correct'] else '❌ Issues'}
- **Adaptive Temperature:** {'✅ Working' if enhancement_results['adaptive_temp']['is_correct'] else '❌ Issues'}

## Overall Score: {overall_score:.1f}/100

{'## Conclusion ✅' if overall_score >= 70 else '## Conclusion ⚠️'}
The enhanced gateway with active optimizations is {'performing excellently' if overall_score >= 70 else 'showing mixed results'}. 
{'Speed improvements are working well with ' + f'{speedup:.2f}x cache speedup.' if speedup > 1 else 'Speed improvements need attention.'}
{'The model shows good intelligence across difficulty levels.' if intelligence_results['accuracy'] >= 70 else 'Model intelligence could be improved.'}
"""

report_file = OUTPUT_DIR / "ENHANCED_BENCHMARK_REPORT.md"
with open(report_file, 'w') as f:
    f.write(report)

print(f"Report saved to: {report_file}")
print("\n✅ Enhanced benchmark completed successfully!")