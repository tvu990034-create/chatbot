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
    
    # Check if model file exists (Setup Bug #60 fix)
    import os
    if not os.path.exists(cfg.model_path):
        print(f"\nError: Model file not found at {cfg.model_path}")
        print("Please download the model. See README.md for instructions.")
        print("\nQuick download (Windows PowerShell):")
        print("  Invoke-WebRequest -Uri \"https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf\" -OutFile \"models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf\"")
        print("\nQuick download (macOS/Linux):")
        print("  wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -P models/")
        sys.exit(1)
    
    chatbot = LocalChatGraph(cfg)
    init_time = time.time() - start

    print(f"[OK] Initialized in {init_time:.2f}s")
    print(f"[OK] Model: {cfg.model_path}")
    print(f"[OK] Knowledge base: {cfg.knowledge_base_path}")
    print(f"[OK] LangGraph: {'enabled' if chatbot._graph else 'disabled'}")
    print(f"[OK] Model loaded on first request (lazy loading)")
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
        except EOFError:
            # Exit gracefully in non-interactive mode (EOF Error fix)
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("You can continue typing or type 'quit' to exit.")
            # Don't break - let user continue (UX Bug #30 fix)


if __name__ == "__main__":
    main()
