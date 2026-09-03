"""
factuality/contrastive_decoding.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Contrastive Decoding implementation based on research paper:

Contrastive Decoding: Open-ended Text Generation as Inference-time Optimization
https://arxiv.org/abs/2210.15097

Algorithm: Subtract amateur logits from expert logits
- Uses "amateur" model (low temperature) vs "expert" model (high temperature)
- Contrasts logits to reduce hallucinations
- Simple yet effective for factuality improvement
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn

from factuality.logit_processor import LogitProcessor

logger = logging.getLogger(__name__)


class ContrastiveDecoding(LogitProcessor):
    """
    Contrastive Decoding implementation for hallucination reduction.
    
    Based on: Contrastive Decoding: Open-ended Text Generation as Inference-time Optimization
    https://arxiv.org/abs/2210.15097
    
    Key innovations:
    - Uses two models: amateur (low temp) and expert (high temp)
    - Subtracts amateur logits from expert logits
    - Reduces hallucinations by contrasting outputs
    - Simple inference-time optimization
    
    Mathematical formulation:
    CD_logits = logits_expert - λ * logits_amateur
    where λ is the contrastive coefficient
    """
    
    def __init__(
        self,
        amateur_temperature: float = 0.7,
        expert_temperature: float = 1.0,
        contrastive_coefficient: float = 0.5,
        base_temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        super().__init__(base_temperature, top_k, top_p)
        
        self.amateur_temperature = amateur_temperature
        self.expert_temperature = expert_temperature
        self.contrastive_coefficient = contrastive_coefficient
        
        logger.info(
            f"ContrastiveDecoding initialized with amateur_temp={amateur_temperature}, "
            f"expert_temp={expert_temperature}, λ={contrastive_coefficient}"
        )
    
    def compute_contrastive_logits(
        self,
        expert_logits: torch.Tensor,
        amateur_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive logits by subtracting amateur from expert.
        
        Args:
            expert_logits: Logits from expert model [batch_size, vocab_size]
            amateur_logits: Logits from amateur model [batch_size, vocab_size]
        
        Returns:
            Contrastive logits [batch_size, vocab_size]
        """
        # Ensure shapes match
        if expert_logits.shape != amateur_logits.shape:
            raise ValueError(
                f"Expert and amateur logits shapes must match: "
                f"{expert_logits.shape} vs {amateur_logits.shape}"
            )
        
        # Apply contrastive decoding formula
        contrastive_logits = expert_logits - self.contrastive_coefficient * amateur_logits
        
        return contrastive_logits
    
    def process_with_model(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate both amateur and expert logits from a single model.
        
        Args:
            model: Language model
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask
        
        Returns:
            Tuple of (expert_logits, amateur_logits)
        """
        model.eval()
        
        with torch.no_grad():
            # Get expert logits (normal temperature)
            expert_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=self.expert_temperature,
            )
            expert_logits = expert_outputs.logits[:, -1, :]
            
            # Get amateur logits (low temperature)
            amateur_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=self.amateur_temperature,
            )
            amateur_logits = amateur_outputs.logits[:, -1, :]
        
        return expert_logits, amateur_logits
    
    def process(
        self,
        expert_logits: torch.Tensor,
        amateur_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply contrastive decoding to logits.
        
        Args:
            expert_logits: Logits from expert model
            amateur_logits: Logits from amateur model
            attention_mask: Optional attention mask
        
        Returns:
            Processed contrastive logits
        """
        # Compute contrastive logits
        contrastive_logits = self.compute_contrastive_logits(
            expert_logits,
            amateur_logits
        )
        
        # Apply base processing (temperature, top-k, top-p)
        processed_logits = self.process_logits(contrastive_logits, attention_mask)
        
        return processed_logits
    
    def process_single_model(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply contrastive decoding using a single model with different temperatures.
        
        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Optional attention mask
        
        Returns:
            Processed contrastive logits
        """
        # Get both amateur and expert logits
        expert_logits, amateur_logits = self.process_with_model(
            model,
            input_ids,
            attention_mask
        )
        
        # Apply contrastive decoding
        return self.process(expert_logits, amateur_logits, attention_mask)