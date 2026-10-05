#!/usr/bin/env python3
"""Test if real user usage speed matches benchmark speed."""

import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent))

from local_chatbot.config import ChatbotConfig
from local_chatbot.graph import LocalChatGraph
from local_chatbot.optimized_graph import OptimizedChatGraph


# Simulate real user conversation
USER_SCENARIOS = [
    {
        "name": "Greeting",
        "questions": ["hi", "hello", "hey"],
        "expected_source": "zero_token",
        "expected_max_ms": 50,
    },
    {
        "name": "FAQ Questions",
        "questions": [
            "What is the capital of France?",
            "What is BM25?",
            "What is quantization?",
        ],
        "expected_source": "faq",
        "expected_max_ms": 50,
    },
    {
        "name": "Repeated Questions (Cache)",
        "questions": [
            "What is the capital of France?",  # Repeated
            "What is BM25?",  # Repeated
        ],
        "expected_source": "faq",
        "expected_max_ms": 30,  # Should be faster due to cache
    },
    {
        "name": "LLM Questions",
        "questions": [
            "What is attention?",
            "Explain transformers",
        ],
        "expected_source": "llm",
        "expected_max_ms": 80000,  # 80s max (includes model load)
    },
]


def test_chatbot(name, chatbot_class):
    print(f"\n{'='*70}")
    print(f"  TESTING: {name}")
    print(f"{'='*70}\n")

    # Initialize
    print("Initializing...")
    start = time.time()
    cfg = ChatbotConfig()
    chatbot = chatbot_class(cfg)
    init_time = time.time() - start
    print(f"Init time: {init_time:.2f}s\n")

    results = []
    session_id = "user_test"

    for scenario in USER_SCENARIOS:
        print(f"Scenario: {scenario['name']}")
        print("-" * 70)

        for question in scenario["questions"]:
            start = time.time()
            result = chatbot.chat(question, session_id=session_id)
            latency = (time.time() - start) * 1000

            # Check if within expected range
            in_range = latency <= scenario["expected_max_ms"]
            status = "[OK]" if in_range else "[SLOW]"

            print(f"  Q: {question[:50]}")
            print(f"  Source: {result.source} (expected: {scenario['expected_source']})")
            print(f"  Latency: {latency:.2f}ms {status}")
            print(f"  Expected max: {scenario['expected_max_ms']}ms")

            results.append({
                "scenario": scenario["name"],
                "question": question,
                "latency_ms": latency,
                "source": result.source,
                "expected_source": scenario["expected_source"],
                "expected_max_ms": scenario["expected_max_ms"],
                "in_range": in_range,
            })
            print()

    return results


def compare_with_benchmark(name, results):
    print(f"\n{'='*70}")
    print(f"  {name} - COMPARISON WITH BENCHMARK")
    print(f"{'='*70}\n")

    # Calculate stats by scenario
    scenarios = {}
    for r in results:
        if r["scenario"] not in scenarios:
            scenarios[r["scenario"]] = []
        scenarios[r["scenario"]].append(r)

    for scenario_name, scenario_results in scenarios.items():
        latencies = [r["latency_ms"] for r in scenario_results]
        avg_latency = mean(latencies)
        in_range_count = sum(1 for r in scenario_results if r["in_range"])
        total = len(scenario_results)
        pass_rate = (in_range_count / total) * 100

        print(f"{scenario_name}:")
        print(f"  Avg latency: {avg_latency:.2f}ms")
        print(f"  Pass rate: {pass_rate:.0f}% ({in_range_count}/{total})")
        print()

    # Overall stats
    all_latencies = [r["latency_ms"] for r in results]
    overall_avg = mean(all_latencies)
    overall_pass = sum(1 for r in results if r["in_range"])
    overall_pass_rate = (overall_pass / len(results)) * 100

    print(f"Overall:")
    print(f"  Avg latency: {overall_avg:.2f}ms")
    print(f"  Pass rate: {overall_pass_rate:.0f}% ({overall_pass}/{len(results)})")

    if overall_pass_rate >= 80:
        print(f"  Status: [EXCELLENT] User speed matches benchmark")
    elif overall_pass_rate >= 60:
        print(f"  Status: [GOOD] Mostly matches benchmark")
    else:
        print(f"  Status: [POOR] Slower than expected")

    print()


def main():
    print("\n" + "=" * 70)
    print("  USER SPEED TEST - Compare Real Usage vs Benchmark")
    print("=" * 70)
    print("\nThis test simulates real user conversations and compares")
    print("the actual speed with expected benchmark performance.")
    print("=" * 70)

    # Test basic RAG
    basic_results = test_chatbot("Basic RAG (graph.py)", LocalChatGraph)
    compare_with_benchmark("Basic RAG", basic_results)

    # Test optimized
    optimized_results = test_chatbot("Optimized (optimized_graph.py)", OptimizedChatGraph)
    compare_with_benchmark("Optimized", optimized_results)

    # Final comparison
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)

    basic_avg = mean([r["latency_ms"] for r in basic_results])
    optimized_avg = mean([r["latency_ms"] for r in optimized_results])

    print(f"\nBasic RAG avg: {basic_avg:.2f}ms")
    print(f"Optimized avg: {optimized_avg:.2f}ms")
    print(f"Difference: {basic_avg - optimized_avg:.2f}ms")

    if optimized_avg < basic_avg:
        print(f"Optimized is {(basic_avg/optimized_avg):.2f}x faster")
    else:
        print(f"Basic is {(optimized_avg/basic_avg):.2f}x faster")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
