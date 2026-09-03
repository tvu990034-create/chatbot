"""
merging/slerp_merging.py
~~~~~~~~~~~~~~~~~~~~~~~~
SLERP (Spherical Linear Interpolation) implementation based on research:

SLERP: Spherical Linear Interpolation for Model Merging
https://arxiv.org/abs/2401.02905

Algorithm: Spherical interpolation in weight space
- Interpolates on the sphere rather than straight line
- Preserves norm during interpolation
- Better for merging models with different magnitudes
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from merging.model_merger import ModelMerger

logger = logging.getLogger(__name__)


class SLERPMerger(ModelMerger):
    """
    SLERP (Spherical Linear Interpolation) implementation.
    
    Based on: SLERP: Spherical Linear Interpolation for Model Merging
    https://arxiv.org/abs/2401.02905
    
    Key innovations:
    - Spherical interpolation in weight space
    - Preserves norm during interpolation
    - Better geometric properties than linear interpolation
    - Widely used in MergeKit for model merging
    """
    
    def __init__(
        self,
        base_model: Optional[str] = None,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(base_model, device, dtype)
        
        logger.info("SLERPMerger initialized")
    
    def slerp(
        self,
        v1: torch.Tensor,
        v2: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """
        Spherical linear interpolation between two vectors.
        
        SLERP formula:
        slerp(v1, v2, t) = (sin((1-t)Ω) / sin(Ω)) * v1 + (sin(tΩ) / sin(Ω)) * v2
        where Ω = arccos(v1 · v2 / (|v1| * |v2|))
        
        Args:
            v1: First vector
            v2: Second vector
            t: Interpolation parameter (0 to 1)
        
        Returns:
            Interpolated vector
        """
        # Normalize vectors
        v1_norm = v1 / (v1.norm() + 1e-8)
        v2_norm = v2 / (v2.norm() + 1e-8)
        
        # Compute dot product and angle
        dot = torch.clamp(torch.dot(v1_norm.flatten(), v2_norm.flatten()), -1.0, 1.0)
        omega = torch.acos(dot)
        
        # Handle degenerate case
        if omega < 1e-6:
            return (1 - t) * v1 + t * v2
        
        # Compute SLERP
        sin_omega = torch.sin(omega)
        weight1 = torch.sin((1 - t) * omega) / sin_omega
        weight2 = torch.sin(t * omega) / sin_omega
        
        return weight1 * v1 + weight2 * v2
    
    def slerp_merge(
        self,
        model_names: List[str],
        interpolation: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply SLERP merging between two models.
        
        Args:
            model_names: List of exactly 2 model identifiers
            interpolation: Interpolation parameter (0 to 1)
        
        Returns:
            SLERP-merged weights
        """
        if len(model_names) != 2:
            raise ValueError("SLERP requires exactly 2 models")
        
        if not 0 <= interpolation <= 1:
            raise ValueError("Interpolation must be between 0 and 1")
        
        logger.info(
            f"Applying SLERP merging between {model_names[0]} and {model_names[1]} "
            f"with t={interpolation}"
        )
        
        # Get weights for both models
        weights1 = self.get_model_weights(model_names[0])
        weights2 = self.get_model_weights(model_names[1])
        
        # Apply SLERP parameter-wise
        merged_weights = {}
        
        for key in weights1:
            if key in weights2:
                # Flatten for SLERP computation
                v1 = weights1[key].flatten()
                v2 = weights2[key].flatten()
                
                # Apply SLERP
                merged_flat = self.slerp(v1, v2, interpolation)
                
                # Reshape back
                merged_weights[key] = merged_flat.reshape(weights1[key].shape)
            else:
                # Parameter only in first model
                merged_weights[key] = weights1[key] * (1 - interpolation)
        
        # Handle parameters only in second model
        for key in weights2:
            if key not in weights1:
                merged_weights[key] = weights2[key] * interpolation
        
        return merged_weights
    
    def multi_slerp_merge(
        self,
        model_names: List[str],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply SLERP merging for multiple models using pairwise SLERP.
        
        Args:
            model_names: List of model identifiers
            weights: Optional weights for each model
        
        Returns:
            SLERP-merged weights
        """
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)
        
        if len(model_names) != len(weights):
            raise ValueError("Number of models must match number of weights")
        
        logger.info(f"Applying multi-model SLERP merging to {len(model_names)} models")
        
        # Start with first model
        current_weights = self.get_model_weights(model_names[0])
        current_weight = weights[0]
        
        # Iteratively merge with remaining models
        for i, (model_name, weight) in enumerate(zip(model_names[1:], weights[1:]), 1):
            next_weights = self.get_model_weights(model_name)
            
            # Compute interpolation parameter
            total_weight = current_weight + weight
            t = weight / total_weight if total_weight > 0 else 0.5
            
            # Apply SLERP
            merged_weights = {}
            for key in current_weights:
                if key in next_weights:
                    v1 = current_weights[key].flatten()
                    v2 = next_weights[key].flatten()
                    merged_flat = self.slerp(v1, v2, t)
                    merged_weights[key] = merged_flat.reshape(current_weights[key].shape)
                else:
                    merged_weights[key] = current_weights[key] * (1 - t)
            
            # Handle parameters only in next model
            for key in next_weights:
                if key not in current_weights:
                    merged_weights[key] = next_weights[key] * t
            
            current_weights = merged_weights
            current_weight = total_weight
        
        return current_weights
    
    def merge(
        self,
        model_names: List[str],
        interpolation: Optional[float] = None,
        weights: Optional[List[float]] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Merge models using SLERP.
        
        Args:
            model_names: List of model identifiers
            interpolation: Interpolation parameter (for 2 models)
            weights: Weights for multi-model SLERP
            **kwargs: Additional parameters
        
        Returns:
            Merged model
        """
        logger.info(f"SLERP merging {len(model_names)} models")
        
        # Load all models
        for model_name in model_names:
            if model_name not in self.models:
                self.load_model(model_name, model_name)
        
        # Apply appropriate SLERP method
        if len(model_names) == 2 and interpolation is not None:
            merged_weights = self.slerp_merge(model_names, interpolation)
        else:
            merged_weights = self.multi_slerp_merge(model_names, weights)
        
        # Create new model and apply weights
        # Use first model as base for architecture
        base_model = self.models[model_names[0]].__class__.from_pretrained(
            self.base_model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        
        merged_model = self.apply_weights(base_model, merged_weights)
        
        return merged_model