"""
factuality/logit_processor.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Base logit processor for factuality improvements.

Based on Hugging Face LogitProcessor API and research on logit manipulation:
- Customizing Text Generation with LogitsProcessor
- The Illustrated GPT-2 (logit visualization)
- Generating with Confidence (uncertainty quantification)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from config import settings

logger = logging.getLogger(__name__)


class LogitProcessor:
    """
    Base class for logit manipulation techniques.
    
    Provides common functionality for:
    - Logit extraction and modification
    - Temperature scaling
    - Top-k/top-p filtering
    - Logit bias application
    """
    
    def __init__(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        
        logger.info(
            f"LogitProcessor initialized with temperature={temperature}, "
            f"top_k={top_k}, top_p={top_p}"
        )
    
    def process_logits(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process logits with base operations.
        
        Args:
            logits: Input logits [batch_size, vocab_size]
            attention_mask: Optional attention mask
        
        Returns:
            Processed logits
        """
        # Apply temperature scaling
        if self.temperature != 1.0:
            logits = logits / self.temperature
        
        # Apply top-k filtering
        if self.top_k is not None:
            logits = self._apply_top_k(logits)
        
        # Apply top-p (nucleus) filtering
        if self.top_p is not None:
            logits = self._apply_top_p(logits)
        
        return logits
    
    def _apply_top_k(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        top_k = min(self.top_k, logits.size(-1))
        
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(logits, top_k)
        
        # Create mask
        mask = torch.zeros_like(logits)
        mask.scatter_(-1, top_k_indices, 1)
        
        # Apply mask (set non-top-k to -inf)
        logits = logits.masked_fill(mask == 0, float('-inf'))
        
        return logits
    
    def _apply_top_p(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply top-p (nucleus) filtering to logits."""
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(
            torch.softmax(sorted_logits, dim=-1),
            dim=-1
        )
        
        # Create mask for tokens within top-p
        mask = cumulative_probs <= self.top_p
        
        # Include at least one token
        mask[..., 0] = True
        
        # Sort mask back to original order
        mask = torch.gather(mask, 1, sorted_indices.argsort(dim=-1))
        
        # Apply mask
        logits = logits.masked_fill(~mask, float('-inf'))
        
        return logits
    
    def apply_logit_bias(
        self,
        logits: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply logit bias to specific tokens.
        
        Args:
            logits: Input logits [batch_size, vocab_size]
            bias: Bias tensor [vocab_size]
        
        Returns:
            Biased logits
        """
        return logits + bias
    
    def apply_logit_mask(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply logit mask (set masked tokens to -inf).
        
        Args:
            logits: Input logits [batch_size, vocab_size]
            mask: Boolean mask [vocab_size]
        
        Returns:
            Masked logits
        """
        return logits.masked_fill(~mask, float('-inf'))
    
    def compute_confidence(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute confidence score from logits.
        
        Based on: Generating with Confidence: Uncertainty Quantification
        https://arxiv.org/abs/2305.19187
        
        Args:
            logits: Input logits [batch_size, vocab_size]
        
        Returns:
            Confidence scores [batch_size]
        """
        # Compute softmax probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Normalize entropy to confidence (higher entropy = lower confidence)
        max_entropy = torch.log(torch.tensor(logits.size(-1), dtype=torch.float))
        confidence = 1.0 - (entropy / max_entropy)
        
        return confidence
    
    def compute_token_uncertainty(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token uncertainty from logits.
        
        Based on: Language Models (Mostly) Know What They Know
        https://arxiv.org/abs/2207.05221
        
        Args:
            logits: Input logits [batch_size, vocab_size]
        
        Returns:
            Uncertainty scores [batch_size]
        """
        # Compute variance of top-k probabilities
        top_k = min(10, logits.size(-1))
        top_k_values, _ = torch.topk(logits, top_k)
        top_k_probs = torch.softmax(top_k_values, dim=-1)
        
        # Compute variance
        mean_prob = top_k_probs.mean(dim=-1, keepdim=True)
        variance = torch.mean((top_k_probs - mean_prob) ** 2, dim=-1)
        
        return variance.mean(dim=-1)
    
    def detect_low_confidence(
        self,
        logits: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        Detect low-confidence generations.
        
        Based on: A Stitch in Time Saves Nine
        https://arxiv.org/abs/2307.03987
        
        Args:
            logits: Input logits [batch_size, vocab_size]
            threshold: Confidence threshold
        
        Returns:
            Boolean mask for low-confidence samples [batch_size]
        """
        confidence = self.compute_confidence(logits)
        return confidence < threshold