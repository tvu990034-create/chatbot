import json, re
from typing import Dict, Tuple

class HeuristicClassifier:
    def __init__(self, rules: Dict[str, list] = None):
        self.rules = rules or {
            "greeting": [r"\b(hi|hello|hey|good morning)\b"],
            "code": [r"\b(code|function|def|import)\b"],
            "math": [r"\b(equation|math|solve|calculate)\b"],
            "factual": [r"\b(who|what|when|where|why|how)\b"],
        }
        self.compiled = {cat: [re.compile(p, re.IGNORECASE) for p in pats]
                         for cat, pats in self.rules.items()}

    def classify(self, query: str) -> Tuple[str, float]:
        q = query.lower()
        for cat, patterns in self.compiled.items():
            if any(p.search(q) for p in patterns):
                return cat, 1.0
        return "default", 0.0

class DynamicPromptSelector:
    def __init__(self, classifier, mapping: Dict[str, str], threshold=0.8, enabled=True):
        self.classifier = classifier
        self.mapping = mapping
        self.threshold = threshold
        self.enabled = enabled

    def select(self, query: str) -> str:
        if not self.enabled: return self.mapping.get("default", "You are a helpful assistant.")
        cat, conf = self.classifier.classify(query)
        return self.mapping.get(cat) if conf >= self.threshold else self.mapping.get("default",
                                    "You are a helpful assistant.")

def load_prompt_mapping(path="system_prompts.json") -> Dict[str, str]:
    with open(path) as f:
        mapping = json.load(f)
    mapping.setdefault("default", "You are a helpful assistant.")
    return mapping