#!/usr/bin/env python3
"""Benchmark cache performance - unique vs repeated questions."""

import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).parent))

from local_chatbot.config import ChatbotConfig
from local_chatbot.optimized_graph import OptimizedChatGraph


def main():
    print("\n" + "=" * 70)
    print("  CACHE PERFORMANCE BENCHMARK")
    print("=" * 70)
    print("\nTesting: Unique questions vs Repeated questions")
    print("Expected: Repeated questions should be faster (cache hits)")
    print("=" * 70 + "\n")

    cfg = ChatbotConfig()
    chatbot = OptimizedChatGraph(cfg)

    questions = [
        "What is the capital of France?",
        "What is BM25?",
        "What is quantization?",
    ]

    print("ROUND 1: Unique questions (cache misses)")
    print("-" * 70)
    round1_times = []
    for q in questions:
        start = time.time()
        result = chatbot.chat(q, session_id="cache_test")
        latency = (time.time() - start) * 1000
        round1_times.append(latency)
        print(f"{q[:40]:40s} -> {latency:7.2f}ms (source: {result.source})")

    print("\nROUND 2: Repeated questions (cache hits)")
    print("-" * 70)
    round2_times = []
    for q in questions:
        start = time.time()
        result = chatbot.chat(q, session_id="cache_test")
        latency = (time.time() - start) * 1000
        round2_times.append(latency)
        print(f"{q[:40]:40s} -> {latency:7.2f}ms (source: {result.source})")

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    avg1 = mean(round1_times)
    avg2 = mean(round2_times)
    speedup = avg1 / avg2

    print(f"\nRound 1 (unique):    {avg1:.2f}ms avg")
    print(f"Round 2 (repeated):  {avg2:.2f}ms avg")
    print(f"Cache speedup:       {speedup:.2f}x")

    if speedup > 1.5:
        print("\n[SUCCESS] Cache is working effectively!")
    else:
        print("\n[INFO] Cache not very effective for these questions (likely FAQ fast path)")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
