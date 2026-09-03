"""
merging/model_stock.py
~~~~~~~~~~~~~~~~~~~~~~
Model Stock implementation based on research paper:

Model Stock: All We Need is Just a Few Fine-Tuned Models
https://arxiv.org/abs/2403.19522

Algorithm: Geometric median in weight space
- Robust merging from as few as 2 models
- Uses geometric median instead of mean
- More robust to outliers than simple averaging
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from merging.model_merger import ModelMerger

logger = logging.getLogger(__name__)


class ModelStock(ModelMerger):
    """
    Model Stock implementation using geometric median.
    
    Based on: Model Stock: All We Need is Just a Few Fine-Tuned Models
    https://arxiv.org/abs/2403.19522
    
    Key innovations:
    - Geometric median in weight space
    - Robust to outliers
    - Works with as few as 2 models
    - Better generalization than simple averaging
    """
    
    def __init__(
        self,
        base_model: Optional[str] = None,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ):
        super().__init__(base_model, device, dtype)
        
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        logger.info(
            f"ModelStock initialized with max_iterations={max_iterations}, "
            f"tolerance={tolerance}"
        )
    
    def geometric_median(
        self,
        points: List[torch.Tensor],
        max_iterations: Optional[int] = None,
        tolerance: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute geometric median of points using Weiszfeld algorithm.
        
        Geometric median minimizes sum of distances: argmin_x Σ||x - xi||
        
        Args:
            points: List of tensors (points in weight space)
            max_iterations: Maximum iterations for algorithm
            tolerance: Convergence tolerance
        
        Returns:
            Geometric median tensor
        """
        max_iter = max_iterations or self.max_iterations
        tol = tolerance or self.tolerance
        
        # Initialize with mean
        median = torch.stack(points).mean(dim=0)
        
        for iteration in range(max_iter):
            # Compute distances
            distances = torch.stack([
                torch.norm(point - median) for point in points
            ])
            
            # Handle zero distance case
            distances = torch.clamp(distances, min=1e-8)
            
            # Weiszfeld update
            weights = 1.0 / distances
            weights = weights / weights.sum()
            
            new_median = torch.stack([
                weight * point for weight, point in zip(weights, points)
            ]).sum(dim=0)
            
            # Check convergence
            if torch.norm(new_median - median) < tol:
                logger.info(f"Geometric median converged at iteration {iteration}")
                break
            
            median = new_median
        
        return median
    
    def compute_model_stock(
        self,
        model_names: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Model Stock (geometric median) of models.
        
        Args:
            model_names: List of model identifiers
        
        Returns:
            Geometric median weights
        """
        logger.info(f"Computing Model Stock for {len(model_names)} models")
        
        # Load all model weights
        all_weights = [self.get_model_weights(name) for name in model_names]
        
        # Compute geometric median for each parameter
        stock_weights = {}
        
        for key in all_weights[0]:
            # Collect weights for this parameter
            param_weights = [w[key] for w in all_weights if key in w]
            
            if not param_weights:
                continue
            
            # Flatten for geometric median computation
            flattened_weights = [w.flatten() for w in param_weights]
            
            # Compute geometric median
            median_flat = self.geometric_median(flattened_weights)
            
            # Reshape back
            stock_weights[key] = median_flat.reshape(param_weights[0].shape)
        
        return stock_weights
    
    def merge(
        self,
        model_names: List[str],
        **kwargs,
    ) -> nn.Module:
        """
        Merge models using Model Stock (geometric median).
        
        Args:
            model_names: List of model identifiers
            **kwargs: Additional parameters
        
        Returns:
            Merged model
        """
        logger.info(f"Model Stock merging {len(model_names)} models")
        
        # Load all models
        for model_name in model_names:
            if model_name not in self.models:
                self.load_model(model_name, model_name)
        
        # Apply Model Stock
        stock_weights = self.compute_model_stock(model_names)
        
        # Create new model and apply weights
        base_model = self.models[model_names[0]].__class__.from_pretrained(
            self.base_model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        
        merged_model = self.apply_weights(base_model, stock_weights)
        
        return merged_model