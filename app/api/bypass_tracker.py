class BypassTracker:
    def __init__(self):
        self.total = 0
        self.reasons = {"cache": 0, "zero_token": 0, "faq": 0, "score_gate": 0, "frgl": 0}

    def record_bypass(self, reason: str):
        self.total += 1
        if reason in self.reasons:
            self.reasons[reason] += 1

    def record_llm(self):
        self.total += 1

    @property
    def fraction(self) -> float:
        if self.total == 0: return 0.0
        return sum(self.reasons.values()) / self.total
    