"""
merging/model_merger.py
~~~~~~~~~~~~~~~~~~~~~~~
Base model merger with common functionality for all merging algorithms.

Based on foundational research on linear mode connectivity and model merging:
- Model Soups: https://arxiv.org/abs/2203.05482
- Linear Mode Connectivity: https://arxiv.org/abs/1912.05671
- Loss Surfaces: https://arxiv.org/abs/1802.10026
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import settings

logger = logging.getLogger(__name__)


class ModelMerger:
    """
    Base class for model merging algorithms.
    
    Implements common functionality:
    - Model loading and saving
    - Weight extraction and manipulation
    - Layer-wise merging operations
    - Evaluation and validation
    """
    
    def __init__(
        self,
        base_model: Optional[str] = None,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
    ):
        self.base_model_name = base_model or settings.local_model_name
        self.device = self._get_device(device)
        self.dtype = dtype
        
        self.models: Dict[str, nn.Module] = {}
        self.tokenizers: Dict[str, Any] = {}
        
        logger.info(f"ModelMerger initialized with base model: {self.base_model_name}")
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the appropriate device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
    
    def load_model(
        self,
        model_path: str,
        model_name: str,
        is_adapter: bool = False,
    ) -> nn.Module:
        """
        Load a model from path.
        
        Args:
            model_path: Path to model or adapter
            model_name: Identifier for the model
            is_adapter: Whether this is a LoRA adapter
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading model: {model_name} from {model_path}")
        
        if is_adapter:
            try:
                from peft import PeftModel
                base_model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    torch_dtype=self.dtype,
                    device_map="auto",
                )
                model = PeftModel.from_pretrained(base_model, model_path)
            except ImportError:
                logger.warning("PEFT not installed, loading as regular model")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=self.dtype,
                    device_map="auto",
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                device_map="auto",
            )
        
        self.models[model_name] = model
        logger.info(f"Model {model_name} loaded successfully")
        return model
    
    def load_tokenizer(self, model_path: str, model_name: str):
        """Load tokenizer for a model."""
        logger.info(f"Loading tokenizer for: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizers[model_name] = tokenizer
        return tokenizer
    
    def get_model_weights(self, model_name: str) -> Dict[str, torch.Tensor]:
        """
        Extract model weights as a dictionary.
        
        Args:
            model_name: Identifier for the model
        
        Returns:
            Dictionary of parameter names to tensors
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        weights = {}
        
        for name, param in model.named_parameters():
            weights[name] = param.data.clone()
        
        return weights
    
    def get_delta_weights(
        self,
        model_name: str,
        base_model_name: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute delta weights (fine-tuned - base).
        
        Based on Task Arithmetic: https://arxiv.org/abs/2212.04089
        
        Args:
            model_name: Fine-tuned model identifier
            base_model_name: Base model identifier (uses default if None)
        
        Returns:
            Dictionary of delta weights
        """
        base_name = base_model_name or self.base_model_name
        
        if model_name not in self.models or base_name not in self.models:
            raise ValueError(f"Models {model_name} and {base_name} must be loaded")
        
        fine_tuned_weights = self.get_model_weights(model_name)
        base_weights = self.get_model_weights(base_name)
        
        delta_weights = {}
        for key in fine_tuned_weights:
            if key in base_weights:
                delta_weights[key] = fine_tuned_weights[key] - base_weights[key]
        
        return delta_weights
    
    def simple_average(
        self,
        model_names: List[str],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Simple weight averaging (Model Soups).
        
        Based on Model Soups: https://arxiv.org/abs/2203.05482
        
        Args:
            model_names: List of model identifiers to merge
            weights: Optional weights for each model (default: uniform)
        
        Returns:
            Averaged weights dictionary
        """
        if weights is None:
            weights = [1.0 / len(model_names)] * len(model_names)
        
        if len(model_names) != len(weights):
            raise ValueError("Number of models must match number of weights")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        logger.info(f"Computing simple average of {len(model_names)} models")
        
        # Load first model to get structure
        first_weights = self.get_model_weights(model_names[0])
        averaged_weights = {k: v * weights[0] for k, v in first_weights.items()}
        
        # Add weighted contributions from other models
        for model_name, weight in zip(model_names[1:], weights[1:]):
            model_weights = self.get_model_weights(model_name)
            for key in averaged_weights:
                if key in model_weights:
                    averaged_weights[key] += model_weights[key] * weight
        
        return averaged_weights
    
    def layer_wise_merge(
        self,
        model_names: List[str],
        layer_weights: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Layer-wise merging with different weights per layer.
        
        Based on Layer-Wise Model Merging: https://arxiv.org/abs/2405.14504
        
        Args:
            model_names: List of model identifiers
            layer_weights: Optional dict mapping layer names to weights
        
        Returns:
            Merged weights with layer-specific weighting
        """
        if layer_weights is None:
            # Default: uniform weights for all layers
            layer_weights = {}
        
        logger.info("Computing layer-wise merge")
        
        first_weights = self.get_model_weights(model_names[0])
        merged_weights = {}
        
        for param_name in first_weights:
            # Determine layer name from parameter name
            layer_name = self._get_layer_name(param_name)
            
            # Get weights for this layer
            if layer_name in layer_weights:
                weights = layer_weights[layer_name]
            else:
                weights = [1.0 / len(model_names)] * len(model_names)
            
            # Compute weighted average
            merged_weights[param_name] = first_weights[param_name] * weights[0]
            
            for model_name, weight in zip(model_names[1:], weights[1:]):
                model_weights = self.get_model_weights(model_name)
                if param_name in model_weights:
                    merged_weights[param_name] += model_weights[param_name] * weight
        
        return merged_weights
    
    def _get_layer_name(self, param_name: str) -> str:
        """Extract layer name from parameter name."""
        # Handle various parameter naming conventions
        parts = param_name.split('.')
        
        # Common patterns: "layers.0", "layer.0", "encoder.layer.0"
        for i, part in enumerate(parts):
            if part.isdigit() or (part.startswith('layer') and part[5:].isdigit()):
                # Return the prefix up to this layer
                return '.'.join(parts[:i+1])
        
        # Fallback: return first component
        return parts[0] if parts else "default"
    
    def apply_weights(
        self,
        target_model: nn.Module,
        weights: Dict[str, torch.Tensor],
    ) -> nn.Module:
        """
        Apply merged weights to a model.
        
        Args:
            target_model: Model to apply weights to
            weights: Dictionary of parameter names to tensors
        
        Returns:
            Model with applied weights
        """
        logger.info("Applying merged weights to model")
        
        with torch.no_grad():
            for name, param in target_model.named_parameters():
                if name in weights:
                    param.data = weights[name].to(param.device)
                else:
                    logger.warning(f"Parameter {name} not in merged weights")
        
        return target_model
    
    def save_merged_model(
        self,
        model: nn.Module,
        output_path: Path,
        tokenizer: Optional[Any] = None,
    ):
        """
        Save merged model and tokenizer.
        
        Args:
            model: Merged model
            output_path: Path to save model
            tokenizer: Optional tokenizer to save
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving merged model to {output_path}")
        
        model.save_pretrained(output_path)
        
        if tokenizer is not None:
            tokenizer.save_pretrained(output_path)
        
        logger.info(f"Merged model saved successfully")
    
    def compute_similarity(
        self,
        model1_name: str,
        model2_name: str,
    ) -> float:
        """
        Compute cosine similarity between two models in weight space.
        
        Based on linear mode connectivity analysis.
        
        Args:
            model1_name: First model identifier
            model2_name: Second model identifier
        
        Returns:
            Cosine similarity score
        """
        weights1 = self.get_model_weights(model1_name)
        weights2 = self.get_model_weights(model2_name)
        
        # Flatten weights
        flat1 = torch.cat([w.flatten() for w in weights1.values()])
        flat2 = torch.cat([w.flatten() for w in weights2.values()])
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            flat1.unsqueeze(0),
            flat2.unsqueeze(0),
        ).item()
        
        return similarity
    
    def merge(
        self,
        model_names: List[str],
        method: str = "average",
        **kwargs,
    ) -> nn.Module:
        """
        Merge models using specified method.
        
        Args:
            model_names: List of model identifiers to merge
            method: Merging method (average, ties, dare, task_arithmetic, etc.)
            **kwargs: Additional method-specific parameters
        
        Returns:
            Merged model
        """
        logger.info(f"Merging {len(model_names)} models using method: {method}")
        
        # Load all models if not already loaded
        for model_name in model_names:
            if model_name not in self.models:
                self.load_model(model_name, model_name)
        
        # Apply merging method
        if method == "average":
            merged_weights = self.simple_average(model_names, kwargs.get("weights"))
        elif method == "layer_wise":
            merged_weights = self.layer_wise_merge(model_names, kwargs.get("layer_weights"))
        else:
            raise ValueError(f"Unknown merging method: {method}")
        
        # Create base model and apply weights
        base_model = self.models[model_names[0]].__class__.from_pretrained(
            self.base_model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        
        merged_model = self.apply_weights(base_model, merged_weights)
        
        return merged_model
    
    def cleanup(self):
        """Clean up loaded models to free memory."""
        for name in list(self.models.keys()):
            del self.models[name]
        
        for name in list(self.tokenizers.keys()):
            del self.tokenizers[name]
        
        torch.cuda.empty_cache()
        logger.info("ModelMerger cleaned up")