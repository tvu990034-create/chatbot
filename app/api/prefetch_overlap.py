import asyncio, time
from typing import Dict, List

class PrefetchCache:
    def __init__(self, ttl_seconds: int = 30):
        self.cache: Dict[str, list] = {}
        self.ttl = ttl_seconds

    def get(self, query: str) -> list | None:
        if query in self.cache:
            entry, ts = self.cache[query]
            if time.time() - ts < self.ttl: return entry
            del self.cache[query]
        return None

    def put(self, query: str, chunks: list):
        self.cache[query] = (chunks, time.time())

    def predict(self, answer: str) -> List[str]:
        """Simple template-based predictor (replace with Markov/LLM)."""
        preds = []
        lower = answer.lower()
        if "capital" in lower: preds.append("what is the population of")
        if "weather" in lower: preds.append("what about tomorrow")
        return preds