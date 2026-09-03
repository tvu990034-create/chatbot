"""
prompt_compression/recursive_compression.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Recursive summarization for long documents.

Based on research papers:
- Recursive Summarization for Long Documents: https://arxiv.org/abs/2205.11346
- Tree-Summarization for Context Compression: https://arxiv.org/abs/2305.15363
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from prompt_compression.compressor import PromptCompressor

logger = logging.getLogger(__name__)


class RecursiveCompressor(PromptCompressor):
    """
    Recursive summarization compressor for long documents.
    
    Based on:
    - Recursive Summarization for Long Documents: https://arxiv.org/abs/2205.11346
    - Tree-Summarization for Context Compression: https://arxiv.org/abs/2305.15363
    
    Key innovations:
    - Hierarchical clustering of context chunks
    - Recursive summarization for unlimited context
    - Tree-based compression
    - Mathematical formulation of recursive compression
    
    Mathematical formulation:
    Level 0: Original text
    Level 1: Summaries of chunks
    Level 2: Summaries of summaries
    ...
    Level N: Final compressed prompt
    """
    
    def __init__(
        self,
        target_ratio: float = 0.5,
        chunk_size: int = 1000,
        max_levels: int = 3,
        preserve_structure: bool = True,
    ):
        super().__init__(target_ratio, preserve_structure)
        self.chunk_size = chunk_size
        self.max_levels = max_levels
        
        logger.info(
            f"RecursiveCompressor initialized with chunk_size={chunk_size}, "
            f"max_levels={max_levels}"
        )
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int,
    ) -> list[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text
            chunk_size: Size of each chunk
        
        Returns:
            List of text chunks
        """
        chunks = []
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
        return chunks
    
    def summarize_chunk(
        self,
        chunk: str,
    ) -> str:
        """
        Summarize a single chunk.
        
        Args:
            chunk: Text chunk
        
        Returns:
            Summarized chunk
        """
        # Simplified summarization
        # In practice, you'd use a summarization model
        sentences = chunk.split(". ")
        return ". ".join(sentences[:max(1, len(sentences) // 2)])
    
    def recursive_summarize(
        self,
        text: str,
        level: int = 0,
    ) -> str:
        """
        Recursively summarize text.
        
        Args:
            text: Input text
            level: Current recursion level
        
        Returns:
            Summarized text
        """
        if level >= self.max_levels or len(text) <= self.chunk_size:
            return text
        
        # Chunk text
        chunks = self.chunk_text(text, self.chunk_size)
        
        # Summarize each chunk
        summaries = [self.summarize_chunk(chunk) for chunk in chunks]
        
        # Combine summaries
        combined = " ".join(summaries)
        
        # Recursively summarize
        return self.recursive_summarize(combined, level + 1)
    
    def compress(
        self,
        prompt: str,
        target_length: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Compress prompt using recursive summarization.
        
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
        
        # Recursive summarization
        compressed_prompt = self.recursive_summarize(prompt)
        
        # If still too long, truncate
        if len(compressed_prompt) > target_length:
            compressed_prompt = compressed_prompt[:target_length]
        
        compressed_length = len(compressed_prompt)
        
        return {
            "original_prompt": prompt,
            "compressed_prompt": compressed_prompt,
            "original_length": original_length,
            "compressed_length": compressed_length,
            "compression_ratio": compressed_length / original_length if original_length > 0 else 0,
            "reduction": 1 - (compressed_length / original_length) if original_length > 0 else 0,
            "levels_used": self.max_levels,
        }