#!/usr/bin/env python3
"""Simple CLI interface for the local chatbot."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from local_chatbot.config import ChatbotConfig
from local_chatbot.graph import LocalChatGraph


def main():
    print("\n" + "=" * 60)
    print("  LOCAL AI CHATBOT - CLI")
    print("=" * 60)
    print("  Type 'quit' or 'exit' to stop")
    print("=" * 60 + "\n")

    # Initialize chatbot (should be instant with lazy loading)
    print("Initializing chatbot...")
    start = time.time()
    cfg = ChatbotConfig()
    chatbot = LocalChatGraph(cfg)
    init_time = time.time() - start

    print(f"✓ Initialized in {init_time:.2f}s")
    print(f"✓ Model: {cfg.model_path}")
    print(f"✓ Knowledge base: {cfg.knowledge_base_path}")
    print(f"✓ LangGraph: {'enabled' if chatbot._graph else 'disabled'}")
    print(f"✓ Model loaded on first request (lazy loading)")
    print()

    session_id = "cli_session"

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye!")
                break

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
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
