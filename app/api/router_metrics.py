import math
from dataclasses import dataclass

@dataclass
class RouterMetrics:
    small_requests: int = 0
    large_requests: int = 0
    small_total_time: float = 0.0
    large_total_time: float = 0.0
    router_overhead_total: float = 0.0

    def record(self, used_small: bool, latency: float, overhead: float):
        if used_small:
            self.small_requests += 1
            self.small_total_time += latency
        else:
            self.large_requests += 1
            self.large_total_time += latency
        self.router_overhead_total += overhead

    @property
    def p_observed(self) -> float:
        total = self.small_requests + self.large_requests
        return self.small_requests / total if total else 0.0

    def predicted_speedup(self, t_large: float, t_small: float) -> float:
        p = self.p_observed
        denom = (1-p)*t_large + p*t_small + (self.router_overhead_total /
                 max(1, self.small_requests+self.large_requests))
        return t_large / denom if denom > 0 else 1.0

def predictive_speedup(t_large, t_small, frac_small, overhead=0.0) -> float:
    return t_large / ((1-frac_small)*t_large + frac_small*t_small + overhead)

def required_p_for_target(t_large, t_small, target_speedup, overhead=0.0) -> float:
    num = t_large - overhead - t_large / target_speedup
    denom = t_large - t_small
    if denom == 0: return 1.0
    return max(0.0, min(1.0, num / denom))