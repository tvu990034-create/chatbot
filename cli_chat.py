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
