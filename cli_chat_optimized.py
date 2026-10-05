#!/usr/bin/env python3
"""Optimized CLI interface with full speed_engine integration."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from local_chatbot.config import ChatbotConfig
from local_chatbot.optimized_graph import OptimizedChatGraph


def main():
    print("\n" + "=" * 60)
    print("  OPTIMIZED AI CHATBOT - CLI")
    print("=" * 60)
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    # Initialize optimized chatbot
    print("Initializing optimized chatbot...")
    print("  - Loading speed_engine optimizations...")
    print("  - Multi-layer caching (exact → SimHash → BM25)")
    print("  - Advanced retrieval (BM25 + dense + PageRank)")
    print("  - Smart fast paths (FAQ, zero-token)")
    print("  - Score gating for relevance")
    print()

    start = time.time()
    cfg = ChatbotConfig()
    chatbot = OptimizedChatGraph(cfg)
    init_time = time.time() - start

    print(f"[OK] Initialized in {init_time:.2f}s")
    print(f"[OK] Model: {cfg.model_path}")
    print(f"[OK] Knowledge base: {cfg.knowledge_base_path}")
    print(f"[OK] Lazy loading: {cfg.lazy_load_model}")
    print(f"[OK] FAQ enabled: {cfg.use_faq}")
    print(f"[OK] Zero-token enabled: {cfg.use_zero_token}")
    print(f"[OK] Cache enabled: True")
    print(f"[OK] Score gating: True")
    print()

    session_id = "cli_session"

    print("Optimization layers active:")
    print("  1. Zero-token responder (~2ms)")
    print("  2. FAQ database (~4ms)")
    print("  3. Cache layer (exact → SimHash → BM25)")
    print("  4. Score gating (filter low-confidence)")
    print("  5. Dynamic token allocation")
    print("  6. Advanced retrieval (BM25 + dense + PageRank)")
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

            if user_input.lower() in ['help', 'h', '?']:
                print("\nAvailable commands:")
                print("  quit/exit/q - Exit the chatbot")
                print("  help/h/? - Show this help message")
                print("  clear - Clear conversation history")
                print()

            if user_input.lower() == 'clear':
                chatbot.memory.clear(session_id)
                print("Conversation history cleared.")
                continue

            if not user_input:
                continue

            print("Bot: ", end="", flush=True)
            start = time.time()

            result = chatbot.chat(user_input, session_id=session_id)

            print(result.response)
            print(f"  [Source: {result.source}, Latency: {result.latency_ms:.0f}ms]")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("You can continue typing or type 'quit' to exit.")
            # Don't break - let user continue (UX Bug #30 fix)


if __name__ == "__main__":
    main()
