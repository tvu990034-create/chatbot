import json
import re

def normalize(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    t = text.lower().strip()
    t = re.sub(r'[^\w\s]', '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

class PhraseNormalizer:
    """
    Maps common user paraphrases to a canonical string so that
    the semantic cache can share one answer for many similar queries.
    """
    def __init__(self, map_path: str = "phrase_map.json"):
        self.map = {}
        if map_path:
            try:
                with open(map_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                # raw: dict of canonical -> list of paraphrases
                for canonical, phrases in raw.items():
                    for phrase in phrases:
                        self.map[normalize(phrase)] = normalize(canonical)
            except FileNotFoundError:
                pass

    def canonical(self, query: str) -> str:
        """Return the canonical form of a query, or the normalised query itself if not found."""
        key = normalize(query)
        return self.map.get(key, key)