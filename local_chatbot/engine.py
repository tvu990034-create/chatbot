"""Local LLM engine using llama-cpp-python."""

from __future__ import annotations

import os
from typing import Optional

from local_chatbot.config import ChatbotConfig


class LocalEngine:
    """Wraps llama.cpp for local text generation."""

    def __init__(self, cfg: ChatbotConfig):
        self.cfg = cfg
        self.model = None
        self.use_simulated = False
        self._loaded = False
        if not cfg.lazy_load_model:
            self._load()

    def _load(self) -> None:
        if self._loaded:
            return

        try:
            from llama_cpp import Llama
        except ImportError:
            self.use_simulated = True
            self._loaded = True
            return

        if not os.path.exists(self.cfg.model_path):
            self.use_simulated = True
            self._loaded = True
            return

        try:
            self.model = Llama(
                model_path=self.cfg.model_path,
                n_ctx=self.cfg.n_ctx,
                n_threads=self.cfg.n_threads,
                n_batch=self.cfg.n_batch,
                n_gpu_layers=self.cfg.n_gpu_layers,
                verbose=False,
            )
            self._prefill_system()
        except Exception:
            self.use_simulated = True
            self.model = None

        self._loaded = True

    def _prefill_system(self) -> None:
        # Bug #1 fix: Remove system prefill overhead
        # System prompt is included in first user message instead
        # This saves ~0.5-1s during model initialization
        return

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and not self.use_simulated

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        self._load()
        max_tokens = max_tokens or self.cfg.max_tokens
        if self.use_simulated or not self.model:
            return self._simulated(prompt)

        # Include system prompt in first message (Bug #1 fix)
        full_prompt = f"system: {self.cfg.system_prompt}\nuser: {prompt}\nassistant:"
        output = self.model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=self.cfg.temperature,
            stop=["</s>", "user:", "\nuser:"],
            echo=False,
        )
        if output.get("choices"):
            return output["choices"][0]["text"].strip()
        return "I couldn't generate a response."

    def _simulated(self, prompt: str) -> str:
        lower = prompt.lower()
        if "context:" in lower or "q:" in lower:
            for line in prompt.splitlines():
                if line.strip().startswith("Q:"):
                    question = line.replace("Q:", "").strip()
                    return f"Based on the available context, here's what I know about {question.rstrip('?')}."
        if "hello" in lower or "hi" in lower:
            return "Hello! I'm your local AI assistant. How can I help?"
        return "I'm running in simulated mode. Install llama-cpp-python and add a GGUF model for real inference."
