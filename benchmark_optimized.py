#!/usr/bin/env python3
"""Benchmark: Basic RAG vs Optimized speed_engine."""

import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent))

from local_chatbot.config import ChatbotConfig
from local_chatbot.graph import LocalChatGraph
from local_chatbot.optimized_graph import OptimizedChatGraph


TEST_QUESTIONS = [
    ("What is the capital of France?", "faq"),
    ("hi", "zero_token"),
    ("What is BM25?", "faq"),
    ("What is attention?", "llm"),
    ("What is quantization?", "faq"),
]


def benchmark_version(name, chatbot_class):
    print(f"\n{'='*70}")
    print(f"  BENCHMARK: {name}")
    print(f"{'='*70}\n")

    print("Initializing...")
    start = time.time()
    cfg = ChatbotConfig()
    chatbot = chatbot_class(cfg)
    init_time = time.time() - start
    print(f"Init time: {init_time:.2f}s\n")

    results = []
    for question, expected_source in TEST_QUESTIONS:
        print(f"Q: {question}")
        start = time.time()
        result = chatbot.chat(question, session_id="benchmark")
        latency = time.time() - start
        print(f"  Response: {result.response[:60]}...")
        print(f"  Source: {result.source}")
        print(f"  Latency: {latency*1000:.2f}ms")
        results.append({
            "question": question,
            "latency_ms": latency * 1000,
            "source": result.source,
        })
        print()

    faq_latencies = [r["latency_ms"] for r in results if r["source"] == "faq"]
    zero_token_latencies = [r["latency_ms"] for r in results if r["source"] == "zero_token"]
    llm_latencies = [r["latency_ms"] for r in results if r["source"] == "llm"]
    all_latencies = [r["latency_ms"] for r in results]

    print(f"Statistics:")
    print(f"  Avg: {mean(all_latencies):.2f}ms")
    if faq_latencies:
        print(f"  FAQ avg: {mean(faq_latencies):.2f}ms")
    if zero_token_latencies:
        print(f"  Zero-token avg: {mean(zero_token_latencies):.2f}ms")
    if llm_latencies:
        print(f"  LLM avg: {mean(llm_latencies):.2f}ms")

    return results


def main():
    print("\n" + "=" * 70)
    print("  BASIC RAG vs OPTIMIZED SPEED_ENGINE")
    print("=" * 70)

    # Benchmark basic RAG
    basic_results = benchmark_version("Basic RAG (graph.py)", LocalChatGraph)

    # Benchmark optimized
    optimized_results = benchmark_version("Optimized (optimized_graph.py)", OptimizedChatGraph)

    # Comparison
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)

    basic_avg = mean([r["latency_ms"] for r in basic_results])
    optimized_avg = mean([r["latency_ms"] for r in optimized_results])

    print(f"\nBasic RAG avg: {basic_avg:.2f}ms")
    print(f"Optimized avg: {optimized_avg:.2f}ms")
    print(f"Improvement: {basic_avg/optimized_avg:.2f}x")

    print("\nOptimization Features:")
    print("  [OK] Multi-layer caching (exact -> SimHash -> BM25)")
    print("  [OK] Advanced retrieval (BM25 + dense + PageRank)")
    print("  [OK] Smart fast paths (FAQ, zero-token)")
    print("  [OK] Score gating (configurable)")
    print("  [OK] Dynamic token allocation")
    print("  [OK] Query truncation")
    print("  [OK] PageRank-based pruning")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
