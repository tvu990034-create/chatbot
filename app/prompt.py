"""
Prompt Builders for Cloud AI Chatbot
"""
from typing import Sequence, Dict, List, Optional


class MinimalPromptBuilder:
    """Eq 20: stripped prompt with no boilerplate."""
    def build(self, history: Sequence[Dict[str, str]], user_input: str, retrieved_chunks: Sequence[str]) -> str:
        prompt = ""
        if retrieved_chunks and len(retrieved_chunks) > 0:
            prompt += retrieved_chunks[0] + "\n"
        if history:
            last = history[-1]
            prompt += f"Q: {last['user']}\nA: {last['assistant']}\n"
        prompt += f"Q: {user_input}\nA:"
        return prompt


class CloudPromptBuilder:
    """
    Wraps MinimalPromptBuilder (or any builder) to add cloud‑specific
    optimisations that do not change the AI logic.
    """
    def __init__(
        self,
        system_prompt: str = "",
        use_cache_control: bool = False,
        conciseness_short_queries: bool = False,
        conciseness_max_query_words: int = 5,
        conciseness_instruction: str = "Answer in one sentence.\n",
    ):
        self.system_prompt = system_prompt
        self.use_cache_control = use_cache_control
        self.conciseness_short_queries = conciseness_short_queries
        self.conciseness_max_query_words = conciseness_max_query_words
        self.conciseness_instruction = conciseness_instruction
        self.minimal_builder = MinimalPromptBuilder()

    def build(
        self,
        history: Sequence[Dict[str, str]],
        user_input: str,
        retrieved_chunks: Sequence[str],
    ) -> str:
        """
        Build a prompt that is identical in content to the minimal builder,
        but may include cache_control markers and a conciseness prefix.
        """
        # 1. Start with the same minimal format as before (no boilerplate)
        prompt = self.minimal_builder.build(history, user_input, retrieved_chunks)

        # 2. Optionally inject conciseness instruction (Eq 5 / Eq 46)
        if self.conciseness_short_queries:
            word_count = len(user_input.strip().split())
            if word_count <= self.conciseness_max_query_words:
                prompt = self.conciseness_instruction + prompt

        # 3. Prepend system message with cache_control if needed (Eq 2 / Eq 10)
        if self.use_cache_control and self.system_prompt:
            # OpenAI / Anthropic prompt caching: mark the whole system message as cacheable
            # The format depends on the provider; we use a generic marker that the API
            # can interpret (e.g., for OpenAI you need to send a specific object, but here
            # we inject a comment that the engine adapter will recognise).
            prompt = f"[CACHE_SYSTEM]{self.system_prompt}\n" + prompt
        elif self.system_prompt:
            prompt = f"system: {self.system_prompt}\n" + prompt

        return prompt


class PromptBuilder:
    """Standard prompt builder with full context."""
    def __init__(self, system_prompt: str, max_history_turns: int = 12):
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns

    def build(self, history: Sequence[Dict[str, str]], user_input: str, retrieved_chunks: Sequence[str]) -> str:
        parts = [f"system: {self.system_prompt}\n"]
        if retrieved_chunks:
            ctx = "\n---\n".join(retrieved_chunks)
            parts.append(f"context:\n{ctx}\n")
        for turn in history[-self.max_history_turns:]:
            parts.append(f"user: {turn['user']}\nassistant: {turn['assistant']}\n")
        parts.append(f"user: {user_input}\nassistant:")
        return "".join(parts)
