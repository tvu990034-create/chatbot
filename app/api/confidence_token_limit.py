import math
from dataclasses import dataclass

@dataclass
class ConfidenceTokenConfig:
    enabled: bool = True
    theta: float = 5.0
    M_max: int = 256
    M_min: int = 10
    score_threshold: float = 0.45

def confidence_to_max_tokens(score: float, cfg: ConfidenceTokenConfig) -> int | None:
    if score < cfg.score_threshold: return None
    M = cfg.M_min + (cfg.M_max - cfg.M_min) * (1 - math.exp(-cfg.theta * score))
    return max(cfg.M_min, int(M))

