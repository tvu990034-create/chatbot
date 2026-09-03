"""
factuality package - Advanced factuality and hallucination reduction capabilities

Based on research papers:
- Contrastive Decoding: https://arxiv.org/abs/2210.15097
- DoLa: https://arxiv.org/abs/2309.03883
- CAD: https://arxiv.org/abs/2305.14703
- ITI: https://arxiv.org/abs/2306.03341
- SelfCheckGPT: https://arxiv.org/abs/2303.08896
- Factual-Nucleus: https://arxiv.org/abs/2106.07447
"""

from factuality.logit_processor import LogitProcessor
from factuality.contrastive_decoding import ContrastiveDecoding
from factuality.dola_decoding import DoLaDecoding
from factuality.cad_decoding import CADDecoding
from factuality.iti_intervention import ITIIntervention
from factuality.selfcheck_gpt import SelfCheckGPT
from factuality.semantic_uncertainty import SemanticUncertainty
from factuality.conformal_prediction import ConformalPrediction
from factuality.logit_lens import LogitLens
from factuality.factual_nucleus import FactualNucleus
from factuality.hallucination_detector import HallucinationDetector

__all__ = [
    "LogitProcessor",
    "ContrastiveDecoding",
    "DoLaDecoding",
    "CADDecoding",
    "ITIIntervention",
    "SelfCheckGPT",
    "SemanticUncertainty",
    "ConformalPrediction",
    "LogitLens",
    "FactualNucleus",
    "HallucinationDetector",
]