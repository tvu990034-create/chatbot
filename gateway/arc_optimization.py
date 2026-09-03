"""
ARC Optimization Module
Implements 2D positional encoding and pixel-level representation for ARC tasks
Based on research: "The role of positional encodings in the ARC benchmark" (2025)
And "Tackling the Abstraction and Reasoning Corpus with Vision Transformers" (2024)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ARCOptimizer:
    """Optimizer for ARC (Abstraction and Reasoning Corpus) tasks."""
    
    def __init__(self, use_2d_encoding: bool = True, pixel_level_repr: bool = True):
        self.use_2d_encoding = use_2d_encoding
        self.pixel_level_repr = pixel_level_repr
        logger.info("ARC Optimizer initialized with 2D encoding and pixel-level representation")
    
    def get_2d_positional_encoding(self, grid_size: int, d_model: int = 512) -> torch.Tensor:
        """
        Generate 2D positional encoding for grid-based tasks.
        
        Args:
            grid_size: Size of the grid (assuming square grid)
            d_model: Dimension of the positional encoding
            
        Returns:
            2D positional encoding tensor
        """
        # Create separate encodings for x and y positions
        x_pos = torch.arange(grid_size).unsqueeze(1).expand(grid_size, grid_size)
        y_pos = torch.arange(grid_size).unsqueeze(0).expand(grid_size, grid_size)
        
        # Apply sinusoidal encoding to both dimensions
        def positional_encoding(pos, d_model):
            position = pos.unsqueeze(-1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
            pe = torch.zeros(pos.shape + (d_model,))
            pe[..., 0::2] = torch.sin(position * div_term)
            pe[..., 1::2] = torch.cos(position * div_term)
            return pe
        
        x_encoding = positional_encoding(x_pos, d_model)
        y_encoding = positional_encoding(y_pos, d_model)
        
        # Combine x and y encodings
        combined_encoding = x_encoding + y_encoding
        
        return combined_encoding
    
    def process_arc_grid(self, grid: List[List[int]]) -> np.ndarray:
        """
        Process ARC grid with pixel-level representation.
        
        Args:
            grid: 2D list representing the ARC grid
            
        Returns:
            Processed grid as numpy array preserving spatial structure
        """
        grid_array = np.array(grid)
        
        if self.pixel_level_repr:
            # Flatten while preserving spatial order for pixel-level processing
            # This maintains the 2D structure information
            pixels = grid_array.flatten()
            return pixels.reshape(-1, 1)  # Maintain as column vector
        
        return grid_array
    
    def enhance_arc_prompt(self, prompt: str, grid_size: int = 10) -> str:
        """
        Enhance ARC prompt with spatial reasoning instructions.
        
        Args:
            prompt: Original prompt
            grid_size: Size of the grid for positional encoding
            
        Returns:
            Enhanced prompt with spatial reasoning guidance
        """
        if self.use_2d_encoding:
            enhanced_prompt = f"""Analyze the following pattern transformation problem using spatial reasoning.

{prompt}

Instructions:
1. Pay attention to the 2D spatial structure of the grid
2. Consider the position of each element in both x and y dimensions
3. Look for patterns that involve horizontal, vertical, and diagonal relationships
4. Identify transformations that preserve or modify spatial relationships
5. Consider both local patterns (neighborhoods) and global patterns (entire grid)

Provide your step-by-step reasoning about the spatial pattern."""
            return enhanced_prompt
        
        return prompt
    
    def extract_spatial_features(self, grid: List[List[int]]) -> Dict[str, Any]:
        """
        Extract spatial features from ARC grid.
        
        Args:
            grid: 2D list representing the ARC grid
            
        Returns:
            Dictionary of spatial features
        """
        grid_array = np.array(grid)
        features = {}
        
        # Basic spatial features
        features['grid_size'] = grid_array.shape
        features['unique_colors'] = len(np.unique(grid_array))
        features['num_pixels'] = grid_array.size
        
        # Position-based features
        features['color_positions'] = {}
        for color in np.unique(grid_array):
            positions = np.where(grid_array == color)
            features['color_positions'][int(color)] = {
                'x_coords': positions[1].tolist(),
                'y_coords': positions[0].tolist()
            }
        
        # Adjacency features
        features['adjacency'] = self._compute_adjacency(grid_array)
        
        return features
    
    def _compute_adjacency(self, grid: np.ndarray) -> Dict[str, List]:
        """Compute adjacency relationships in the grid."""
        rows, cols = grid.shape
        adjacency = {'horizontal': [], 'vertical': [], 'diagonal': []}
        
        for i in range(rows):
            for j in range(cols):
                current = grid[i, j]
                
                # Horizontal adjacency
                if j < cols - 1:
                    right = grid[i, j + 1]
                    adjacency['horizontal'].append((current, right))
                
                # Vertical adjacency
                if i < rows - 1:
                    down = grid[i + 1, j]
                    adjacency['vertical'].append((current, down))
                
                # Diagonal adjacency
                if i < rows - 1 and j < cols - 1:
                    diagonal = grid[i + 1, j + 1]
                    adjacency['diagonal'].append((current, diagonal))
        
        return adjacency
    
    def is_arc_task(self, prompt: str) -> bool:
        """
        Detect if the task is an ARC pattern completion task.
        
        Args:
            prompt: The input prompt
            
        Returns:
            True if this appears to be an ARC task
        """
        arc_keywords = [
            'pattern', 'sequence', 'next in the pattern',
            'complete the pattern', 'what comes next',
            'grid', 'transformation', 'input-output'
        ]
        
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in arc_keywords)


class ARCReasoningEnhancer:
    """Enhanced reasoning for ARC tasks using spatial logic."""
    
    def __init__(self):
        self.spatial_rules = [
            "Horizontal patterns (left-to-right transformations)",
            "Vertical patterns (top-to-bottom transformations)",
            "Diagonal patterns (corner-to-corner transformations)",
            "Rotation patterns (clockwise/counterclockwise)",
            "Reflection patterns (horizontal/vertical mirrors)",
            "Color substitution patterns",
            "Size transformation patterns",
            "Position shift patterns"
        ]
    
    def generate_spatial_reasoning_prompt(self, prompt: str) -> str:
        """Generate enhanced prompt with spatial reasoning guidance."""
        reasoning_prompt = f"""{prompt}

Systematic Spatial Analysis:
Consider the following transformation types:
{chr(10).join(f"- {rule}" for rule in self.spatial_rules)}

For each transformation type, check:
1. Does it apply to this problem?
2. If yes, what are the specific rules?
3. How do the rules apply to each element?

Provide your final answer after this systematic analysis."""
        
        return reasoning_prompt