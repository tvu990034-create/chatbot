from dataclasses import dataclass

@dataclass
class PricingConfig:
    prompt_cost_per_million: float = 2.50
    completion_cost_per_million: float = 10.00

    def prompt_cost_per_token(self): return self.prompt_cost_per_million / 1_000_000
    def completion_cost_per_token(self): return self.completion_cost_per_million / 1_000_000

class CostTracker:
    def __init__(self, pricing: PricingConfig = PricingConfig()):
        self.pricing = pricing
        self.total_prompt = 0
        self.total_completion = 0
        self.total_cost = 0.0
        self.turns = 0

    def add_turn(self, prompt_tokens: int, completion_tokens: int):
        cost = (self.pricing.prompt_cost_per_token() * prompt_tokens +
                self.pricing.completion_cost_per_token() * completion_tokens)
        self.total_prompt += prompt_tokens
        self.total_completion += completion_tokens
        self.total_cost += cost
        self.turns += 1
        return cost

    def summary(self):
        return {"turns": self.turns, "prompt_tokens": self.total_prompt,
                "completion_tokens": self.total_completion, "total_cost": round(self.total_cost,6)}
    