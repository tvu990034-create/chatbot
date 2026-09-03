"""
merging package - Advanced model merging capabilities based on research papers

Foundational papers:
- Model Soups: https://arxiv.org/abs/2203.05482
- Git Re-Basin: https://arxiv.org/abs/2209.04836
- Linear Mode Connectivity: https://arxiv.org/abs/1912.05671
- TIES-Merging: https://arxiv.org/abs/2306.01708
- DARE: https://arxiv.org/abs/2311.03099
- Task Arithmetic: https://arxiv.org/abs/2212.04089
- SLERP: https://arxiv.org/abs/2401.02905
- Fisher Merging: https://arxiv.org/abs/2111.09832
- RegMean: https://arxiv.org/abs/2212.09849
- Model Stock: https://arxiv.org/abs/2403.19522
"""

from merging.model_merger import ModelMerger
from merging.ties_merging import TIESMerger
from merging.dare_merging import DAREMerger
from merging.task_arithmetic import TaskArithmetic
from merging.slerp_merging import SLERPMerger
from merging.git_rebasin import GitReBasin
from merging.fisher_merging import FisherMerger
from merging.regmean_merging import RegMeanMerger
from merging.model_stock import ModelStock

__all__ = [
    "ModelMerger",
    "TIESMerger",
    "DAREMerger",
    "TaskArithmetic",
    "SLERPMerger",
    "GitReBasin",
    "FisherMerger",
    "RegMeanMerger",
    "ModelStock",
]