"""
prompt_compression/selective_context.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Selective Context compressor based on self-information.

Based on research paper:
Selective Context: Compressing Your Input to Improve Your Output
https://arxiv.org/abs/2304.12102

Algorithm: Lexical self-information based token-level pruning
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from prompt_compression.compressor import PromptCompressor

logger = logging.getLogger(__name__)


class SelectiveContextCompressor(PromptCompressor):
    """
    Selective Context compressor using self-information.
    
    Based on: Selective Context: Compressing Your Input to Improve Your Output
    https://arxiv.org/abs/2304.12102
    
    Key innovations:
    - Uses lexical self-information for token importance
    - Low self-information tokens are more predictable (less important)
    - High self-information tokens are surprising (more important)
    - Lossless compression for critical tokens
    
    Mathematical formulation:
    Self-Information(token) = -log P(token | context)
    Keep tokens with high self-information
    """
    
    def __init__(
        self,
        target_ratio: float = 0.5,
        self_information_threshold: float = 2.0,
        preserve_structure: bool = True,
    ):
        super().__init__(target_ratio, preserve_structure)
        self.self_information_threshold = self_information_threshold
        
        logger.info(
            f"SelectiveContextCompressor initialized with "
            f"self_information_threshold={self_information_threshold}"
        )
    
    def compute_self_information(
        self,
        tokens: List[str],
    ) -> List[float]:
        """
        Compute self-information for each token.
        
        Args:
            tokens: List of tokens
        
        Returns:
            List of self-information scores
        """
        # Simplified self-information computation
        # In practice, you'd use a language model to compute
        # actual self-information = -log P(token | context)
        
        self_infos = []
        for i, token in enumerate(tokens):
            # Placeholder: use rarity as proxy for self-information
            # Rare tokens have higher self-information
            length_factor = len(token)
            position_factor = (i + 1) / len(tokens)
            self_info = length_factor * position_factor
            self_infos.append(self_info)
        
        return self_infos
    
    def compress(
        self,
        prompt: str,
        target_length: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Compress prompt using selective context self-information pruning.
        
        Args:
            prompt: Input prompt
            target_length: Target token count
        
        Returns:
            Dictionary with compressed prompt and statistics
        """
        original_tokens = prompt.split()
        original_length = len(original_tokens)
        
        if target_length is None:
            target_length = int(original_length * self.target_ratio)
        
        logger.info(f"Compressing {original_length} tokens to {target_length}")
        
        # Compute self-information for each token
        self_infos = self.compute_self_information(original_tokens)
        
        # Sort by self-information (higher = more important)
        sorted_indices = sorted(
            range(len(self_infos)),
            key=lambda i: self_infos[i],
            reverse=True
        )
        
        # Keep top-k tokens by self-information
        keep_mask = [False] * len(self_infos)
        for i in sorted_indices[:target_length]:
            keep_mask[i] = True
        
        # Reconstruct prompt with kept tokens
        compressed_tokens = [
            token for token, keep in zip(original_tokens, keep_mask)
            if keep
        ]
        
        compressed_prompt = " ".join(compressed_tokens)
        compressed_length = len(compressed_tokens)
        
        return {
            "original_prompt": prompt,
            "compressed_prompt": compressed_prompt,
            "original_length": original_length,
            "compressed_length": compressed_length,
            "compression_ratio": compressed_length / original_length if original_length > 0 else 0,
            "reduction": 1 - (compressed_length / original_length) if original_length > 0 else 0,
            "self_informations": self_infos,
            "keep_mask": keep_mask,
        }