"""
merging/task_arithmetic.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Task Arithmetic implementation based on research paper:

Editing Models with Task Arithmetic
https://arxiv.org/abs/2212.04089

Algorithm: Task vectors = fine-tuned weights - base weights
- Add task vectors to add skills
- Subtract task vectors to remove skills
- Scale task vectors to control skill strength
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from merging.model_merger import ModelMerger

logger = logging.getLogger(__name__)


class TaskArithmetic(ModelMerger):
    """
    Task Arithmetic implementation for model editing.
    
    Based on: Editing Models with Task Arithmetic
    https://arxiv.org/abs/2212.04089
    
    Key innovations:
    - Task vectors: ΔW = W_finetuned - W_base
    - Additive editing: W_new = W_base + Σ α_i * ΔW_i
    - Subtractive editing: W_new = W_base - Σ α_i * ΔW_i
    - Skill composition through vector arithmetic
    """
    
    def __init__(
        self,
        base_model: Optional[str] = None,
        device: str = "auto",
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__(base_model, device, dtype)
        
        logger.info("TaskArithmetic initialized")
    
    def compute_task_vector(
        self,
        finetuned_model_name: str,
        base_model_name: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute task vector for a fine-tuned model.
        
        Task vector = fine-tuned weights - base weights
        
        Args:
            finetuned_model_name: Fine-tuned model identifier
            base_model_name: Base model identifier
        
        Returns:
            Task vector dictionary
        """
        logger.info(f"Computing task vector for {finetuned_model_name}")
        
        task_vector = self.get_delta_weights(finetuned_model_name, base_model_name)
        
        # Log statistics
        total_norm = 0.0
        for delta in task_vector.values():
            total_norm += delta.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        logger.info(f"Task vector norm: {total_norm:.4f}")
        
        return task_vector
    
    def add_task_vectors(
        self,
        task_vectors: List[Dict[str, torch.Tensor]],
        coefficients: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Add task vectors with optional coefficients.
        
        Args:
            task_vectors: List of task vector dictionaries
            coefficients: Optional coefficients for each task vector
        
        Returns:
            Combined task vector
        """
        if coefficients is None:
            coefficients = [1.0] * len(task_vectors)
        
        if len(task_vectors) != len(coefficients):
            raise ValueError("Number of task vectors must match number of coefficients")
        
        logger.info(f"Adding {len(task_vectors)} task vectors")
        
        combined_vector = {}
        
        # Initialize with first task vector
        for key in task_vectors[0]:
            combined_vector[key] = task_vectors[0][key] * coefficients[0]
        
        # Add remaining task vectors
        for task_vector, coeff in zip(task_vectors[1:], coefficients[1:]):
            for key in combined_vector:
                if key in task_vector:
                    combined_vector[key] += task_vector[key] * coeff
        
        return combined_vector
    
    def subtract_task_vectors(
        self,
        task_vectors: List[Dict[str, torch.Tensor]],
        coefficients: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Subtract task vectors with optional coefficients.
        
        Args:
            task_vectors: List of task vector dictionaries
            coefficients: Optional coefficients for each task vector
        
        Returns:
            Combined task vector (subtracted)
        """
        if coefficients is None:
            coefficients = [1.0] * len(task_vectors)
        
        logger.info(f"Subtracting {len(task_vectors)} task vectors")
        
        combined_vector = {}
        
        # Initialize with first task vector
        for key in task_vectors[0]:
            combined_vector[key] = task_vectors[0][key] * coefficients[0]
        
        # Subtract remaining task vectors
        for task_vector, coeff in zip(task_vectors[1:], coefficients[1:]):
            for key in combined_vector:
                if key in task_vector:
                    combined_vector[key] -= task_vector[key] * coeff
        
        return combined_vector
    
    def apply_task_vector(
        self,
        task_vector: Dict[str, torch.Tensor],
        base_model_name: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply task vector to base model.
        
        Args:
            task_vector: Task vector to apply
            base_model_name: Base model identifier
        
        Returns:
            Modified weights
        """
        base_name = base_model_name or self.base_model_name
        
        logger.info(f"Applying task vector to base model {base_name}")
        
        base_weights = self.get_model_weights(base_name)
        modified_weights = {}
        
        for key in base_weights:
            if key in task_vector:
                modified_weights[key] = base_weights[key] + task_vector[key]
            else:
                modified_weights[key] = base_weights[key]
        
        return modified_weights
    
    def task_arithmetic_merge(
        self,
        model_names: List[str],
        operations: List[str],
        coefficients: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply task arithmetic with multiple operations.
        
        Args:
            model_names: List of model identifiers
            operations: List of operations ("add" or "subtract")
            coefficients: Optional coefficients for each operation
        
        Returns:
            Modified weights
        """
        if len(model_names) != len(operations):
            raise ValueError("Number of models must match number of operations")
        
        if coefficients is None:
            coefficients = [1.0] * len(model_names)
        
        logger.info(f"Applying task arithmetic to {len(model_names)} models")
        
        # Compute task vectors
        task_vectors = []
        for model_name in model_names:
            task_vector = self.compute_task_vector(model_name)
            task_vectors.append(task_vector)
        
        # Apply operations
        current_vector = None
        
        for task_vector, operation, coeff in zip(task_vectors, operations, coefficients):
            if operation == "add":
                if current_vector is None:
                    current_vector = {k: v * coeff for k, v in task_vector.items()}
                else:
                    for key in current_vector:
                        if key in task_vector:
                            current_vector[key] += task_vector[key] * coeff
            elif operation == "subtract":
                if current_vector is None:
                    current_vector = {k: v * coeff for k, v in task_vector.items()}
                else:
                    for key in current_vector:
                        if key in task_vector:
                            current_vector[key] -= task_vector[key] * coeff
            else:
                raise ValueError(f"Unknown operation: {operation}")
        
        # Apply to base model
        if current_vector is not None:
            modified_weights = self.apply_task_vector(current_vector)
        else:
            modified_weights = self.get_model_weights(self.base_model_name)
        
        return modified_weights
    
    def merge(
        self,
        model_names: List[str],
        operations: Optional[List[str]] = None,
        coefficients: Optional[List[float]] = None,
        **kwargs,
    ) -> nn.Module:
        """
        Merge models using task arithmetic.
        
        Args:
            model_names: List of model identifiers
            operations: List of operations ("add" or "subtract")
            coefficients: Optional coefficients
            **kwargs: Additional parameters
        
        Returns:
            Merged model
        """
        if operations is None:
            operations = ["add"] * len(model_names)
        
        logger.info(f"Task arithmetic merging {len(model_names)} models")
        
        # Load all models
        for model_name in model_names:
            if model_name not in self.models:
                self.load_model(model_name, model_name)
        
        # Load base model if not loaded
        if self.base_model_name not in self.models:
            self.load_model(self.base_model_name, self.base_model_name)
        
        # Apply task arithmetic
        merged_weights = self.task_arithmetic_merge(
            model_names,
            operations,
            coefficients,
        )
        
        # Create new model and apply weights
        base_model = self.models[self.base_model_name].__class__.from_pretrained(
            self.base_model_name,
            torch_dtype=self.dtype,
            device_map="auto",
        )
        
        merged_model = self.apply_weights(base_model, merged_weights)
        
        return merged_model