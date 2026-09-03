"""
prompt_compression package - Advanced prompt compression capabilities

Based on research papers:
- LLMLingua: https://arxiv.org/abs/2310.05736
- LongLLMLingua: https://arxiv.org/abs/2310.06839
- Selective Context: https://arxiv.org/abs/2304.12102
- LLMLingua-2: https://arxiv.org/abs/2403.12968
- AutoCompressor: https://arxiv.org/abs/2305.14788
- Gist: https://arxiv.org/abs/2304.08467
"""

from prompt_compression.compressor import PromptCompressor
from prompt_compression.llmlingua import LLMLinguaCompressor
from prompt_compression.selective_context import SelectiveContextCompressor
from prompt_compression.token_pruning import TokenPruningCompressor
from prompt_compression.summarization_compression import SummarizationCompressor
from prompt_compression.recursive_compression import RecursiveCompressor

__all__ = [
    "PromptCompressor",
    "LLMLinguaCompressor",
    "SelectiveContextCompressor",
    "TokenPruningCompressor",
    "SummarizationCompressor",
    "RecursiveCompressor",
]