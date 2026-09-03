"""
merging/git_rebasin.py
~~~~~~~~~~~~~~~~~~~~~~
Git Re-Basin implementation based on research paper:

Git Re-Basin: Merging Models Modulo Permutation Symmetries
https://arxiv.org/abs/2209.04836

Algorithm: Align neuron permutations before averaging
- Finds optimal permutation of neurons
- Aligns models in weight space
- Enables merging of independently fine-tuned models
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

from merging.model_merger import ModelMerger

logger = logging.getLogger(__name__)


class GitReBasin(ModelMerger):
    """
    Git Re-Basin implementation for permutation alignment.
    
    Based on: Git Re-Basin: Merging Models Modulo Permutation Symmetries
    https://arxiv.org/abs/2209.04836
    
    Key innovations:
    - Permutation alignment of neurons
    - Weight matching algorithm
    - Enables merging of independently fine-tuned models
    - Based on permutation symmetry in neural networks
    """
    
    def __init__(
        self,
        base_model: Optional[str] = None,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        max_iterations: int = 100,
    ):
        super().__init__(base_model, device, dtype)
        
        self.max_iterations = max_iterations
        
        logger.info(f"GitReBasin initialized with max_iterations={max_iterations}")
    
    def weight_matching(
        self,
        weights1: torch.Tensor,
        weights2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Find optimal permutation to align two weight matrices.
        
        Uses Hungarian algorithm for optimal assignment.
        
        Args:
            weights1: First weight matrix (out_features, in_features)
            weights2: Second weight matrix (out_features, in_features)
        
        Returns:
            Permutation indices
        """
        # Compute cost matrix (negative correlation)
        # We want to maximize correlation, so minimize negative correlation
        weights1_norm = weights1 / (weights1.norm(dim=1, keepdim=True) + 1e-8)
        weights2_norm = weights2 / (weights2.norm(dim=1, keepdim=True) + 1e-8)
        
        # Compute correlation matrix
        correlation = torch.mm(weights1_norm, weights2_norm.T)
        
        # Convert to cost matrix (maximize correlation = minimize negative correlation)
        cost_matrix = -correlation.cpu().numpy()
        
        # Hungarian algorithm for optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        return torch.tensor(col_ind, dtype=torch.long)
    
    def align_layer_weights(
        self,
        weights1: torch.Tensor,
        weights2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align two weight matrices by permutation.
        
        Args:
            weights1: First weight matrix
            weights2: Second weight matrix
        
        Returns:
            Tuple of (aligned_weights1, aligned_weights2)
        """
        # Handle different shapes
        if weights1.shape != weights2.shape:
            logger.warning(
                f"Weight shapes differ: {weights1.shape} vs {weights2.shape}, "
                "skipping alignment"
            )
            return weights1, weights2
        
        # Find optimal permutation
        permutation = self.weight_matching(weights1, weights2)
        
        # Apply permutation to second weights
        aligned_weights2 = weights2[permutation]
        
        return weights1, aligned_weights2
    
    def align_model_weights(
        self,
        model1_name: str,
        model2_name: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Align weights of two models layer by layer.
        
        Args:
            model1_name: First model identifier
            model2_name: Second model identifier
        
        Returns:
            Dictionary of aligned weights for model2
        """
        logger.info(f"Aligning weights between {model1_name} and {model2_name}")
        
        weights1 = self.get_model_weights(model1_name)
        weights2 = self.get_model_weights(model2_name)
        
        aligned_weights2 = {}
        
        # Align each layer
        for key in weights1:
            if key in weights2:
                w1 = weights1[key]
                w2 = weights2[key]
                
                # Only align linear layers (weight matrices)
                if len(w1.shape) == 2 and len(w2.shape) == 2:
                    aligned_w1, aligned_w2 = self.align_layer_weights(w1, w2)
                    aligned_weights2[key] = aligned_w2
                else:
                    # Skip other parameters (biases, etc.)
                    aligned_weights2[key] = w2
            else:
                logger.warning(f"Parameter {key} not in model2")
        
        return aligned_weights2
    
    def iterative_alignment(
        self,
        model_names: List[str],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Iteratively align multiple models.
        
        Args:
            model_names: List of model identifiers
        
        Returns:
            List of aligned weight dictionaries
        """
        logger.info(f"Iteratively aligning {len(model_names)} models")
        
        # Start with first model as reference
        aligned_weights_list = [self.get_model_weights(model_names[0])]
        
        # Align each subsequent model to the reference
        for i in range(1, len(model_names)):
            aligned_weights = self.align_model_weights(
                model_names[0],
                model_names[i],
            )
            aligned_weights_list.append(aligned_weights)
        
        return aligned_weights_list
    
    def merge(
        self,
        model_names: List[str],
        weights: Optional[List[float]] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Merge models using Git Re-Basin permutation alignment.
        
        Args:
            model_names: List of model identifiers
            weights: Optional weights for each model
            **kwargs: Additional parameters
        
        Returns:
            Merged model
        """
        logger.info(f"Git Re-Basin merging {len(model_names)} models")
        
        # Load all models
        for model_name in model_names:
            if model_name not in self.models:
                self.load_model(model_name, model_name)
        
        # Align models
        aligned_weights_list = self.iterative_alignment(model_names)
        
        # Simple average of aligned weights
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)
        
        merged_weights = {}
        for key in aligned_weights_list[0]:
            merged_weights[key] = aligned_weights_list[0][key] * weights[0]
            
            for aligned_weights, weight in zip(aligned_weights_list[1:], weights[1:]):
                if key in aligned_weights:
                    merged_weights[key] += aligned_weights[key] * weight
        
        # Create new model and apply weights
        base_model = self.models[model_names[0]].__class__.from_pretrained(
            self.base_model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        
        merged_model = self.apply_weights(base_model, merged_weights)
        
        return merged_model