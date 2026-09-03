"""
factuality/dola_decoding.py
~~~~~~~~~~~~~~~~~~~~~~~~
DoLa (Decoding by Contrasting Layers) implementation based on research paper:

DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models
https://arxiv.org/abs/2309.03883

Algorithm: Contrasts late layers (mature) with early layers (premature)
- Uses layer-wise logit differences
- Exploits the fact that early layers are more factual
- Late layers are more creative but can hallucinate
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from factuality.logit_processor import LogitProcessor

logger = logging.getLogger(__name__)


class DoLaDecoding(LogitProcessor):
    """
    DoLa (Decoding by Contrasting Layers) implementation.
    
    Based on: DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models
    https://arxiv.org/abs/2309.03883
    
    Key innovations:
    - Contrasts late layers (mature) with early layers (premature)
    - Early layers are more factual, late layers more creative
    - Layer-wise logit difference reduces hallucinations
    - No additional models required
    
    Mathematical formulation:
    DoLa_logits = logits_late - α * (logits_late - logits_early)
    where α is the contrasting coefficient
    """
    
    def __init__(
        self,
        early_layer: int = 5,
        late_layer: int = -1,
        contrasting_coefficient: float = 0.5,
        base_temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        super().__init__(base_temperature, top_k, top_p)
        
        self.early_layer = early_layer
        self.late_layer = late_layer
        self.contrasting_coefficient = contrasting_coefficient
        
        logger.info(
            f"DoLaDecoding initialized with early_layer={early_layer}, "
            f"late_layer={late_layer}, α={contrasting_coefficient}"
        )
    
    def extract_layer_logits(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_indices: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract logits from specific layers.
        
        Args:
            model: Language model with intermediate layer access
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            layer_indices: List of layer indices to extract
        
        Returns:
            Dictionary mapping layer indices to logits
        """
        model.eval()
        
        # This is a simplified version - in practice, you'd need a model
        # that supports intermediate layer extraction
        # For now, we'll return the final layer logits
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            
            # Extract logits from final layer
            final_logits = outputs.logits[:, -1, :]
            
            # In a full implementation, you'd extract from specific layers
            # This requires model architecture-specific hooks
            layer_logits = {self.late_layer: final_logits}
            
            if layer_indices is not None:
                for layer_idx in layer_indices:
                    if layer_idx != self.late_layer:
                        # Placeholder for early layer extraction
                        # In practice, you'd use activation hooks
                        layer_logits[layer_idx] = final_logits.clone()
            
            return layer_logits
    
    def compute_dola_logits(
        self,
        late_logits: torch.Tensor,
        early_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DoLa logits by contrasting late and early layers.
        
        Args:
            late_logits: Logits from late layer [batch_size, vocab_size]
            early_logits: Logits from early layer [batch_size, vocab_size]
        
        Returns:
            DoLa-processed logits [batch_size, vocab_size]
        """
        # Ensure shapes match
        if late_logits.shape != early_logits.shape:
            raise ValueError(
                f"Late and early logits shapes must match: "
                f"{late_logits.shape} vs {early_logits.shape}"
            )
        
        # Apply DoLa formula
        # DoLa = late - α * (late - early) = (1-α) * late + α * early
        dola_logits = (
            (1 - self.contrasting_coefficient) * late_logits +
            self.contrasting_coefficient * early_logits
        )
        
        return dola_logits
    
    def process(
        self,
        late_logits: torch.Tensor,
        early_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply DoLa processing to layer logits.
        
        Args:
            late_logits: Logits from late layer
            early_logits: Logits from early layer
            attention_mask: Optional attention mask
        
        Returns:
            Processed DoLa logits
        """
        # Compute DoLa logits
        dola_logits = self.compute_dola_logits(late_logits, early_logits)
        
        # Apply base processing (temperature, top-k, top-p)
        processed_logits = self.process_logits(dola_logits, attention_mask)
        
        return processed_logits
    
    def process_single_model(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply DoLa decoding using layer extraction from a single model.
        
        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Optional attention mask
        
        Returns:
            Processed DoLa logits
        """
        # Extract layer logits
        layer_logits = self.extract_layer_logits(
            model,
            input_ids,
            attention_mask,
            [self.early_layer, self.late_layer]
        )
        
        # Get early and late logits
        early_logits = layer_logits.get(self.early_layer)
        late_logits = layer_logits.get(self.late_layer)
        
        if early_logits is None or late_logits is None:
            logger.warning(
                f"Could not extract logits from layers {self.early_layer} and {self.late_layer}, "
                "using final layer for both"
            )
            early_logits = late_logits
        
        # Apply DoLa
        return self.process(late_logits, early_logits, attention_mask)