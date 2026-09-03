"""
prompt_compression/llmlingua.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
LLMLingua-style perplexity-based prompt compression.

Based on research paper:
LLMLingua: Compressing Prompts for Accelerated Inference
https://arxiv.org/abs/2310.05736

Algorithm: Black-box compression via small LM - token-level perplexity-based budget allocation
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from prompt_compression.compressor import PromptCompressor

logger = logging.getLogger(__name__)


class LLMLinguaCompressor(PromptCompressor):
    """
    LLMLingua-style perplexity-based prompt compressor.
    
    Based on: LLMLingua: Compressing Prompts for Accelerated Inference
    https://arxiv.org/abs/2310.05736
    
    Key innovations:
    - Uses small LM to compute token perplexity
    - Budget allocation based on perplexity scores
    - Low perplexity tokens are more important
    - Black-box compression (no model training required)
    
    Mathematical formulation:
    Perplexity(token) = exp(-log P(token | context))
    Budget(token) ∝ 1 / Perplexity(token)
    """
    
    def __init__(
        self,
        target_ratio: float = 0.5,
        perplexity_threshold: float = 100.0,
        preserve_structure: bool = True,
    ):
        super().__init__(target_ratio, preserve_structure)
        self.perplexity_threshold = perplexity_threshold
        
        logger.info(
            f"LLMLinguaCompressor initialized with perplexity_threshold={perplexity_threshold}"
        )
    
    def compute_perplexity(
        self,
        tokens: List[str],
    ) -> List[float]:
        """
        Compute perplexity for each token.
        
        Args:
            tokens: List of tokens
        
        Returns:
            List of perplexity scores
        """
        # Simplified perplexity computation
        # In practice, you'd use a small LM (e.g., GPT-2 small)
        # to compute actual perplexity scores
        
        perplexities = []
        for i, token in enumerate(tokens):
            # Placeholder: use length and position as proxy for perplexity
            # Shorter, earlier tokens typically have lower perplexity
            position_factor = 1.0 / (i + 1)
            length_factor = 1.0 / (len(token) + 1)
            perplexity = 1.0 / (position_factor * length_factor + 0.1)
            perplexities.append(perplexity)
        
        return perplexities
    
    def allocate_budget(
        self,
        perplexities: List[float],
        target_budget: int,
    ) -> List[bool]:
        """
        Allocate budget based on perplexity scores.
        
        Args:
            perplexities: List of perplexity scores
            target_budget: Target number of tokens to keep
        
        Returns:
            List of booleans indicating which tokens to keep
        """
        # Sort by perplexity (lower = more important)
        sorted_indices = sorted(
            range(len(perplexities)),
            key=lambda i: perplexities[i]
        )
        
        # Keep top-k tokens
        keep = [False] * len(perplexities)
        for i in sorted_indices[:target_budget]:
            keep[i] = True
        
        return keep
    
    def compress(
        self,
        prompt: str,
        target_length: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Compress prompt using LLMLingua-style perplexity-based compression.
        
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
        
        # Compute perplexity for each token
        perplexities = self.compute_perplexity(original_tokens)
        
        # Allocate budget
        keep_mask = self.allocate_budget(perplexities, target_length)
        
        # Reconstruct prompt with kept tokens
        compressed_tokens = [
            token for token, keep in zip(original_tokens, keep_mask)
            if keep
        ]
        
        # Preserve structure if enabled
        if self.preserve_structure:
            # Try to maintain sentence structure
            compressed_prompt = " ".join(compressed_tokens)
        else:
            compressed_prompt = " ".join(compressed_tokens)
        
        compressed_length = len(compressed_tokens)
        
        return {
            "original_prompt": prompt,
            "compressed_prompt": compressed_prompt,
            "original_length": original_length,
            "compressed_length": compressed_length,
            "compression_ratio": compressed_length / original_length if original_length > 0 else 0,
            "reduction": 1 - (compressed_length / original_length) if original_length > 0 else 0,
            "perplexities": perplexities,
            "keep_mask": keep_mask,
        }