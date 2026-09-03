"""
prompt_compression/summarization_compression.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Summarization-based prompt compression.

Based on research papers:
- RECOMP: https://arxiv.org/abs/2310.04408
- AutoCompressor: https://arxiv.org/abs/2305.14788
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from prompt_compression.compressor import PromptCompressor

logger = logging.getLogger(__name__)


class SummarizationCompressor(PromptCompressor):
    """
    Summarization-based prompt compressor.
    
    Based on:
    - RECOMP: https://arxiv.org/abs/2310.04408
    - AutoCompressor: https://arxiv.org/abs/2305.14788
    
    Key innovations:
    - Summarizes context before retrieval
    - Abstractive compression
    - Can use soft tokens or text summaries
    - Preserves semantic meaning
    """
    
    def __init__(
        self,
        target_ratio: float = 0.5,
        summary_ratio: float = 0.3,
        preserve_structure: bool = True,
    ):
        super().__init__(target_ratio, preserve_structure)
        self.summary_ratio = summary_ratio
        
        logger.info(
            f"SummarizationCompressor initialized with summary_ratio={summary_ratio}"
        )
    
    def summarize(
        self,
        text: str,
        target_length: int,
    ) -> str:
        """
        Summarize text to target length.
        
        Args:
            text: Input text
            target_length: Target length in characters
        
        Returns:
            Summarized text
        """
        # Simplified summarization
        # In practice, you'd use a summarization model (BART, Pegasus, T5)
        
        sentences = text.split(". ")
        target_sentences = max(1, int(len(sentences) * self.summary_ratio))
        
        summary = ". ".join(sentences[:target_sentences])
        
        if len(summary) > target_length:
            summary = summary[:target_length]
        
        return summary
    
    def compress(
        self,
        prompt: str,
        target_length: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Compress prompt using summarization.
        
        Args:
            prompt: Input prompt
            target_length: Target character count
        
        Returns:
            Dictionary with compressed prompt and statistics
        """
        original_length = len(prompt)
        
        if target_length is None:
            target_length = int(original_length * self.target_ratio)
        
        logger.info(f"Compressing {original_length} characters to {target_length}")
        
        # Summarize prompt
        compressed_prompt = self.summarize(prompt, target_length)
        compressed_length = len(compressed_prompt)
        
        return {
            "original_prompt": prompt,
            "compressed_prompt": compressed_prompt,
            "original_length": original_length,
            "compressed_length": compressed_length,
            "compression_ratio": compressed_length / original_length if original_length > 0 else 0,
            "reduction": 1 - (compressed_length / original_length) if original_length > 0 else 0,
        }