"""
merging/dare_merging.py
~~~~~~~~~~~~~~~~~~~~~~
DARE (Drop And REscale) implementation based on research paper:

DARE: Drop And REscale for Model Merging
https://arxiv.org/abs/2311.03099

Algorithm: Random drop + rescaling of delta parameters
- Drop: Randomly drop delta parameters
- Rescale: Rescale remaining parameters to preserve expectation
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from merging.model_merger import ModelMerger

logger = logging.getLogger(__name__)


class DAREMerger(ModelMerger):
    """
    DARE (Drop And REscale) implementation for model merging.
    
    Based on: DARE: Drop And REscale for Model Merging
    https://arxiv.org/abs/2311.03099
    
    Key innovations:
    - Random drop: Randomly drop delta parameters
    - Rescaling: Rescale remaining parameters to preserve expectation
    - Theoretical guarantee: Preserves expected model performance
    """
    
    def __init__(
        self,
        base_model: Optional[str] = None,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        drop_rate: float = 0.2,
        rescale: bool = True,
    ):
        super().__init__(base_model, device, dtype)
        
        self.drop_rate = drop_rate
        self.rescale = rescale
        
        logger.info(
            f"DAREMerger initialized with drop_rate={drop_rate}, "
            f"rescale={rescale}"
        )
    
    def drop_deltas(
        self,
        delta_weights: Dict[str, torch.Tensor],
        drop_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Randomly drop delta parameters.
        
        Args:
            delta_weights: Dictionary of delta weights
            drop_rate: Fraction of parameters to drop
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (dropped deltas, drop masks)
        """
        drop_rate = drop_rate or self.drop_rate
        
        if seed is not None:
            torch.manual_seed(seed)
        
        logger.info(f"Dropping deltas with rate {drop_rate}")
        
        dropped_deltas = {}
        drop_masks = {}
        
        for key, delta in delta_weights.items():
            # Create random mask
            num_params = delta.numel()
            num_drop = int(num_params * drop_rate)
            
            # Randomly select indices to drop
            indices = torch.randperm(num_params)[:num_drop]
            
            # Create drop mask (1 = keep, 0 = drop)
            mask = torch.ones_like(delta).flatten()
            mask[indices] = 0.0
            mask = mask.reshape(delta.shape)
            
            # Apply mask
            dropped_deltas[key] = delta * mask
            drop_masks[key] = mask
        
        return dropped_deltas, drop_masks
    
    def rescale_deltas(
        self,
        delta_weights: Dict[str, torch.Tensor],
        drop_masks: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Rescale delta parameters to preserve expectation.
        
        Args:
            delta_weights: Dictionary of dropped delta weights
            drop_masks: Dictionary of drop masks
        
        Returns:
            Rescaled delta weights
        """
        if not self.rescale:
            logger.info("Skipping rescaling")
            return delta_weights
        
        logger.info("Rescaling deltas to preserve expectation")
        
        rescaled_deltas = {}
        
        for key in delta_weights:
            delta = delta_weights[key]
            mask = drop_masks[key]
            
            # Calculate keep ratio
            keep_ratio = mask.mean()
            
            if keep_ratio > 0:
                # Rescale to preserve expectation
                rescaled_deltas[key] = delta / keep_ratio
            else:
                # All parameters dropped, keep as is
                rescaled_deltas[key] = delta
        
        return rescaled_deltas
    
    def dare_merge(
        self,
        model_names: List[str],
        weights: Optional[List[float]] = None,
        drop_rate: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply DARE algorithm for model merging.
        
        Args:
            model_names: List of model identifiers to merge
            weights: Optional weights for each model
            drop_rate: Optional drop rate
            seed: Random seed for reproducibility
        
        Returns:
            Merged weights
        """
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)
        
        logger.info(f"Applying DARE to {len(model_names)} models")
        
        # Load models and compute delta weights
        delta_weights_list = []
        for model_name, weight in zip(model_names, weights):
            delta_weights = self.get_delta_weights(model_name)
            
            # Scale by weight
            for key in delta_weights:
                delta_weights[key] *= weight
            
            delta_weights_list.append(delta_weights)
        
        # Combine delta weights
        combined_deltas = {}
        for deltas in delta_weights_list:
            for key, delta in deltas.items():
                if key not in combined_deltas:
                    combined_deltas[key] = delta
                else:
                    combined_deltas[key] += delta
        
        # Apply DARE: Drop
        logger.info("DARE Step 1: Drop")
        dropped_deltas, drop_masks = self.drop_deltas(combined_deltas, drop_rate, seed)
        
        # Apply DARE: Rescale
        logger.info("DARE Step 2: Rescale")
        rescaled_deltas = self.rescale_deltas(dropped_deltas, drop_masks)
        
        # Merge with base model
        logger.info("DARE Step 3: Merge")
        merged_weights = self.get_model_weights(self.base_model_name)
        
        for key, delta in rescaled_deltas.items():
            if key in merged_weights:
                merged_weights[key] += delta
        
        return merged_weights
    
    def merge(
        self,
        model_names: List[str],
        weights: Optional[List[float]] = None,
        drop_rate: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Merge models using DARE algorithm.
        
        Args:
            model_names: List of model identifiers
            weights: Optional weights for each model
            drop_rate: Optional drop rate
            seed: Random seed for reproducibility
            **kwargs: Additional parameters
        
        Returns:
            Merged model
        """
        logger.info(f"DARE merging {len(model_names)} models")
        
        # Load all models
        for model_name in model_names:
            if model_name not in self.models:
                self.load_model(model_name, model_name)
        
        # Load base model if not loaded
        if self.base_model_name not in self.models:
            self.load_model(self.base_model_name, self.base_model_name)
        
        # Apply DARE
        merged_weights = self.dare_merge(
            model_names,
            weights,
            drop_rate,
            seed,
        )
        
        # Create new model and apply weights
        base_model = self.models[self.base_model_name].__class__.from_pretrained(
            self.base_model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        
        merged_model = self.apply_weights(base_model, merged_weights)
        
        return merged_model