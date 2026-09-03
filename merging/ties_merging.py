"""
merging/ties_merging.py
~~~~~~~~~~~~~~~~~~~~~~~
TIES-Merging implementation based on research paper:

TIES-Merging: Resolving Interference When Merging Models
https://arxiv.org/abs/2306.01708

Algorithm: Trim, Elect, Sign, Merge
- Trim: Remove small magnitude parameters
- Elect: Keep only top-k parameters by magnitude
- Sign: Use sign-based merging to reduce interference
- Merge: Merge parameters with sign-based voting
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from merging.model_merger import ModelMerger

logger = logging.getLogger(__name__)


class TIESMerger(ModelMerger):
    """
    TIES-Merging implementation for resolving parameter interference.
    
    Based on: TIES-Merging: Resolving Interference When Merging Models
    https://arxiv.org/abs/2306.01708
    
    Key innovations:
    - Trim: Remove parameters below magnitude threshold
    - Elect: Keep only top-k parameters per task
    - Sign: Use sign-based merging to reduce interference
    - Merge: Apply sign-based voting for final weights
    """
    
    def __init__(
        self,
        base_model: Optional[str] = None,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        trim_threshold: float = 0.1,
        k_ratio: float = 0.1,
    ):
        super().__init__(base_model, device, dtype)
        
        self.trim_threshold = trim_threshold
        self.k_ratio = k_ratio
        
        logger.info(
            f"TIESMerger initialized with trim_threshold={trim_threshold}, "
            f"k_ratio={k_ratio}"
        )
    
    def trim_deltas(
        self,
        delta_weights: Dict[str, torch.Tensor],
        threshold: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Trim parameters below magnitude threshold.
        
        Args:
            delta_weights: Dictionary of delta weights
            threshold: Magnitude threshold (uses default if None)
        
        Returns:
            Trimmed delta weights
        """
        threshold = threshold or self.trim_threshold
        
        logger.info(f"Trimming deltas with threshold {threshold}")
        
        trimmed_deltas = {}
        for key, delta in delta_weights.items():
            # Compute magnitude
            magnitude = torch.abs(delta)
            
            # Create mask for parameters above threshold
            mask = magnitude > (magnitude.max() * threshold)
            
            # Apply mask
            trimmed_deltas[key] = delta * mask.float()
        
        return trimmed_deltas
    
    def elect_top_k(
        self,
        delta_weights: Dict[str, torch.Tensor],
        k_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Elect top-k parameters by magnitude.
        
        Args:
            delta_weights: Dictionary of delta weights
            k_ratio: Ratio of top parameters to keep (uses default if None)
        
        Returns:
        Delta weights with only top-k parameters
        """
        k_ratio = k_ratio or self.k_ratio
        
        logger.info(f"Electing top-{k_ratio} parameters")
        
        elected_deltas = {}
        for key, delta in delta_weights.items():
            # Compute magnitude
            magnitude = torch.abs(delta)
            
            # Determine k (number of parameters to keep)
            k = max(1, int(delta.numel() * k_ratio))
            
            # Get top-k indices
            top_k_values, top_k_indices = torch.topk(magnitude.flatten(), k)
            
            # Create mask
            mask = torch.zeros_like(delta)
            mask.flatten()[top_k_indices] = 1.0
            
            # Apply mask
            elected_deltas[key] = delta * mask
        
        return elected_deltas
    
    def resolve_signs(
        self,
        delta_weights_list: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Resolve sign conflicts across multiple task vectors.
        
        Args:
            delta_weights_list: List of delta weight dictionaries
        
        Returns:
            Sign-resolved delta weights
        """
        logger.info("Resolving sign conflicts")
        
        if not delta_weights_list:
            return {}
        
        # Initialize sign matrix
        first_deltas = delta_weights_list[0]
        resolved_deltas = {}
        
        for key in first_deltas:
            # Collect signs from all models
            signs = []
            for deltas in delta_weights_list:
                if key in deltas:
                    signs.append(torch.sign(deltas[key]))
            
            if not signs:
                continue
            
            # Stack signs
            sign_matrix = torch.stack(signs)  # (num_models, *param_shape)
            
            # Vote on final sign (majority vote)
            sign_sum = torch.sum(sign_matrix, dim=0)
            final_sign = torch.sign(sign_sum)
            
            # Apply final sign to first model's deltas
            resolved_deltas[key] = first_deltas[key] * final_sign
        
        return resolved_deltas
    
    def ties_merge(
        self,
        model_names: List[str],
        weights: Optional[List[float]] = None,
        trim_threshold: Optional[float] = None,
        k_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply TIES-Merging algorithm.
        
        Args:
            model_names: List of model identifiers to merge
            weights: Optional weights for each model
            trim_threshold: Optional trim threshold
            k_ratio: Optional k ratio for top-k selection
        
        Returns:
            Merged weights
        """
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)
        
        logger.info(f"Applying TIES-Merging to {len(model_names)} models")
        
        # Load models and compute delta weights
        delta_weights_list = []
        for model_name, weight in zip(model_names, weights):
            delta_weights = self.get_delta_weights(model_name)
            
            # Scale by weight
            for key in delta_weights:
                delta_weights[key] *= weight
            
            delta_weights_list.append(delta_weights)
        
        # Step 1: Trim
        logger.info("Step 1: Trim")
        trimmed_list = [
            self.trim_deltas(deltas, trim_threshold)
            for deltas in delta_weights_list
        ]
        
        # Step 2: Elect
        logger.info("Step 2: Elect")
        elected_list = [
            self.elect_top_k(deltas, k_ratio)
            for deltas in trimmed_list
        ]
        
        # Step 3: Sign
        logger.info("Step 3: Sign")
        resolved_deltas = self.resolve_signs(elected_list)
        
        # Step 4: Merge
        logger.info("Step 4: Merge")
        merged_weights = self.get_model_weights(self.base_model_name)
        
        for key, delta in resolved_deltas.items():
            if key in merged_weights:
                merged_weights[key] += delta
        
        return merged_weights
    
    def merge(
        self,
        model_names: List[str],
        weights: Optional[List[float]] = None,
        trim_threshold: Optional[float] = None,
        k_ratio: Optional[float] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Merge models using TIES-Merging algorithm.
        
        Args:
            model_names: List of model identifiers
            weights: Optional weights for each model
            trim_threshold: Optional trim threshold
            k_ratio: Optional k ratio
            **kwargs: Additional parameters
        
        Returns:
            Merged model
        """
        logger.info(f"TIES-Merging {len(model_names)} models")
        
        # Load all models
        for model_name in model_names:
            if model_name not in self.models:
                self.load_model(model_name, model_name)
        
        # Load base model if not loaded
        if self.base_model_name not in self.models:
            self.load_model(self.base_model_name, self.base_model_name)
        
        # Apply TIES-Merging
        merged_weights = self.ties_merge(
            model_names,
            weights,
            trim_threshold,
            k_ratio,
        )
        
        # Create new model and apply weights
        base_model = self.models[self.base_model_name].__class__.from_pretrained(
            self.base_model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        
        merged_model = self.apply_weights(base_model, merged_weights)
        
        return merged_model