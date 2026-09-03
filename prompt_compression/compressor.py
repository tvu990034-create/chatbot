"""
prompt_compression/compressor.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Base prompt compressor with common functionality.

Based on research papers:
- LLMLingua: https://arxiv.org/abs/2310.05736
- Prompt Compression Survey: https://arxiv.org/abs/2311.08168
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)


class PromptCompressor:
    """
    Base class for prompt compression algorithms.
    
    Provides common functionality for:
    - Token counting and estimation
    - Compression ratio calculation
    - Compression statistics
    - Preservation metrics
    """
    
    def __init__(
        self,
        target_ratio: float = 0.5,
        preserve_structure: bool = True,
    ):
        self.target_ratio = target_ratio
        self.preserve_structure = preserve_structure
        
        logger.info(
            f"PromptCompressor initialized with target_ratio={target_ratio}, "
            f"preserve_structure={preserve_structure}"
        )
    
    def count_tokens(
        self,
        text: str,
    ) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Input text
        
        Returns:
            Number of tokens
        """
        # Simplified token counting
        # In practice, you'd use tiktoken or the model's tokenizer
        return len(text.split())
    
    def compress(
        self,
        prompt: str,
        target_length: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Compress prompt to target length.
        
        Args:
            prompt: Input prompt
            target_length: Target token count (uses ratio if None)
        
        Returns:
            Dictionary with compressed prompt and statistics
        """
        original_length = self.count_tokens(prompt)
        
        if target_length is None:
            target_length = int(original_length * self.target_ratio)
        
        # Base implementation - should be overridden
        compressed_prompt = prompt[:int(len(prompt) * (target_length / original_length))]
        
        compressed_length = self.count_tokens(compressed_prompt)
        
        return {
            "original_prompt": prompt,
            "compressed_prompt": compressed_prompt,
            "original_length": original_length,
            "compressed_length": compressed_length,
            "compression_ratio": compressed_length / original_length if original_length > 0 else 0,
            "reduction": 1 - (compressed_length / original_length) if original_length > 0 else 0,
        }
    
    def get_compression_stats(
        self,
        result: Dict[str, any],
    ) -> Dict[str, any]:
        """
        Get compression statistics.
        
        Args:
            result: Result from compress() method
        
        Returns:
            Statistics dictionary
        """
        return {
            "original_tokens": result["original_length"],
            "compressed_tokens": result["compressed_length"],
            "compression_ratio": result["compression_ratio"],
            "reduction_percentage": result["reduction"] * 100,
            "tokens_saved": result["original_length"] - result["compressed_length"],
        }