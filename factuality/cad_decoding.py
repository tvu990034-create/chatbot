"""
factuality/cad_decoding.py
~~~~~~~~~~~~~~~~~~~~~~~~
Context-Aware Decoding (CAD) implementation based on research paper:

Context-Aware Decoding for Factual Hallucination Reduction (CAD)
https://arxiv.org/abs/2305.14703

Algorithm: Contrasts output with/without context in logit space
- Generates with and without retrieval context
- Contrasts logits to reduce hallucinations
- Math-based logit space manipulation
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from factuality.logit_processor import LogitProcessor

logger = logging.getLogger(__name__)


class CADDecoding(LogitProcessor):
    """
    Context-Aware Decoding implementation for factual hallucination reduction.
    
    Based on: Context-Aware Decoding for Factual Hallucination Reduction (CAD)
    https://arxiv.org/abs/2305.14703
    
    Key innovations:
    - Generates responses with and without context
    - Contrasts logits in logit space
    - Reduces hallucinations by emphasizing factual tokens
    - Particularly effective for retrieval-augmented generation
    
    Mathematical formulation:
    CAD_logits = context_logits - β * (context_logits - no_context_logits)
    where β is the context contrasting coefficient
    """
    
    def __init__(
        self,
        context_coefficient: float = 0.5,
        base_temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        super().__init__(base_temperature, top_k, top_p)
        
        self.context_coefficient = context_coefficient
        
        logger.info(
            f"CADDecoding initialized with context_coefficient={context_coefficient}"
        )
    
    def compute_cad_logits(
        self,
        context_logits: torch.Tensor,
        no_context_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CAD logits by contrasting context and no-context outputs.
        
        Args:
            context_logits: Logits with context [batch_size, vocab_size]
            no_context_logits: Logits without context [batch_size, vocab_size]
        
        Returns:
            CAD-processed logits [batch_size, vocab_size]
        """
        # Ensure shapes match
        if context_logits.shape != no_context_logits.shape:
            raise ValueError(
                f"Context and no-context logits shapes must match: "
                f"{context_logits.shape} vs {no_context_logits.shape}"
            )
        
        # Apply CAD formula
        # CAD = context - β * (context - no_context) = (1-β) * context + β * no_context
        cad_logits = (
            (1 - self.context_coefficient) * context_logits +
            self.context_coefficient * no_context_logits
        )
        
        return cad_logits
    
    def process(
        self,
        context_logits: torch.Tensor,
        no_context_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply CAD processing to context-based logits.
        
        Args:
            context_logits: Logits with context
            no_context_logits: Logits without context
            attention_mask: Optional attention mask
        
        Returns:
            Processed CAD logits
        """
        # Compute CAD logits
        cad_logits = self.compute_cad_logits(context_logits, no_context_logits)
        
        # Apply base processing (temperature, top-k, top-p)
        processed_logits = self.process_logits(cad_logits, attention_mask)
        
        return processed_logits
    
    def process_with_model(
        self,
        model: nn.Module,
        input_ids_with_context: torch.Tensor,
        input_ids_without_context: torch.Tensor,
        attention_mask_with_context: Optional[torch.Tensor] = None,
        attention_mask_without_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply CAD decoding using a single model with different inputs.
        
        Args:
            model: Language model
            input_ids_with_context: Input with retrieval context
            input_ids_without_context: Input without context
            attention_mask_with_context: Attention mask for context input
            attention_mask_without_context: Attention mask for no-context input
        
        Returns:
            Processed CAD logits
        """
        model.eval()
        
        with torch.no_grad():
            # Get context logits
            context_outputs = model(
                input_ids=input_ids_with_context,
                attention_mask=attention_mask_with_context,
            )
            context_logits = context_outputs.logits[:, -1, :]
            
            # Get no-context logits
            no_context_outputs = model(
                input_ids=input_ids_without_context,
                attention_mask=attention_mask_without_context,
            )
            no_context_logits = no_context_outputs.logits[:, -1, :]
        
        # Apply CAD
        return self.process(
            context_logits,
            no_context_logits,
            attention_mask_with_context
        )