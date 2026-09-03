"""
factuality/logit_lens.py
~~~~~~~~~~~~~~~~~~~~~~~~
Logit Lens implementation for uncertainty probing.

Based on research paper:
Logit Lens: Extracting Latent Predictions from Transformer Layers
https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/logit-lens

Algorithm: Early-layer logits can indicate uncertainty in predictions
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Dict
import math

import torch
import torch.nn as nn

from factuality.logit_processor import LogitProcessor

logger = logging.getLogger(__name__)


class LogitLens(LogitProcessor):
    """
    Logit Lens implementation for uncertainty probing.
    
    Based on: Logit Lens: Extracting Latent Predictions from Transformer Layers
    https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/logit-lens
    
    Key innovations:
    - Examines early layer logits to detect uncertainty
    - Early layers show more uncertainty than final layers
    - Compares predictions across layers
    - Can detect when model is uncertain early in processing
    - Useful for early stopping and uncertainty estimation
    
    Mathematical formulation:
    Uncertainty = disagreement between early and final layer predictions
    Layer-wise probability tracking
    Prediction stability across layers
    """
    
    def __init__(
        self,
        base_temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        examine_layers: List[int] = None,
        stability_threshold: float = 0.8,
        use_embedding_projection: bool = True,
    ):
        super().__init__(base_temperature, top_k, top_p)
        
        self.examine_layers = examine_layers or [1, 3, 6, 12]  # Layers to examine
        self.stability_threshold = stability_threshold
        self.use_embedding_projection = use_embedding_projection
        
        logger.info(
            f"LogitLens initialized with examine_layers={examine_layers}, "
            f"stability_threshold={stability_threshold}"
        )
    
    def extract_layer_logits(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract logits from specified layers.
        
        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Optional attention mask
        
        Returns:
            Dictionary mapping layer indices to logits
        """
        model.eval()
        layer_logits = {}
        
        # Hook to capture intermediate layer outputs
        hooks = []
        
        def make_hook(layer_idx):
            def hook(module, input, output):
                # Project embeddings to logits if needed
                if self.use_embedding_projection and hasattr(model, 'lm_head'):
                    # Use language model head to project
                    logits = model.lm_head(output[0] if isinstance(output, tuple) else output)
                else:
                    # Simple projection (placeholder)
                    logits = output[0] if isinstance(output, tuple) else output
                
                layer_logits[layer_idx] = logits.detach()
            return hook
        
        # Register hooks for specified layers
        for i, layer in enumerate(model.transformer.h if hasattr(model, 'transformer') else model.model.layers):
            if i in self.examine_layers:
                hook = make_hook(i)
                hooks.append(layer.register_forward_hook(hook))
        
        # Forward pass
        with torch.no_grad():
            _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return layer_logits
    
    def compute_layer_predictions(
        self,
        layer_logits: Dict[int, torch.Tensor],
    ) -> Dict[int, Tuple[int, float]]:
        """
        Compute predictions for each layer.
        
        Args:
            layer_logits: Dictionary of layer logits
        
        Returns:
            Dictionary mapping layer indices to (predicted_token, confidence)
        """
        layer_predictions = {}
        
        for layer_idx, logits in layer_logits.items():
            # Get top prediction
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
            top_log_prob, top_token = torch.max(log_probs, dim=-1)
            
            confidence = torch.exp(top_log_prob).item()
            predicted_token = top_token.item()
            
            layer_predictions[layer_idx] = (predicted_token, confidence)
        
        return layer_predictions
    
    def compute_prediction_stability(
        self,
        layer_predictions: Dict[int, Tuple[int, float]],
    ) -> float:
        """
        Compute stability of predictions across layers.
        
        Args:
            layer_predictions: Dictionary of layer predictions
        
        Returns:
            Stability score (0 to 1)
        """
        if len(layer_predictions) < 2:
            return 1.0
        
        # Get predicted tokens from each layer
        predicted_tokens = [pred[0] for pred in layer_predictions.values()]
        
        # Count frequency of each prediction
        from collections import Counter
        token_counts = Counter(predicted_tokens)
        
        # Stability = frequency of most common prediction
        most_common_count = token_counts.most_common(1)[0][1]
        stability = most_common_count / len(predicted_tokens)
        
        return stability
    
    def compute_confidence_trajectory(
        self,
        layer_predictions: Dict[int, Tuple[int, float]],
    ) -> Tuple[List[float], List[int]]:
        """
        Compute confidence trajectory across layers.
        
        Args:
            layer_predictions: Dictionary of layer predictions
        
        Returns:
            Tuple of (confidence_values, layer_indices)
        """
        sorted_layers = sorted(layer_predictions.keys())
        confidence_values = []
        layer_indices = []
        
        for layer_idx in sorted_layers:
            _, confidence = layer_predictions[layer_idx]
            confidence_values.append(confidence)
            layer_indices.append(layer_idx)
        
        return confidence_values, layer_indices
    
    def detect_uncertainty(
        self,
        layer_predictions: Dict[int, Tuple[int, float]],
    ) -> Tuple[bool, float, Dict[str, any]]:
        """
        Detect uncertainty using logit lens analysis.
        
        Args:
            layer_predictions: Dictionary of layer predictions
        
        Returns:
            Tuple of (is_uncertain, uncertainty_score, metadata)
        """
        # Compute prediction stability
        stability = self.compute_prediction_stability(layer_predictions)
        
        # Compute confidence trajectory
        confidence_values, layer_indices = self.compute_confidence_trajectory(layer_predictions)
        
        # Early layer confidence (first examined layer)
        early_confidence = confidence_values[0] if confidence_values else 0.0
        
        # Final layer confidence (last examined layer)
        final_confidence = confidence_values[-1] if confidence_values else 0.0
        
        # Confidence growth (improvement from early to final)
        confidence_growth = final_confidence - early_confidence
        
        # Overall uncertainty score
        uncertainty_score = 1.0 - stability
        
        # Determine if uncertain
        is_uncertain = stability < self.stability_threshold
        
        metadata = {
            "stability": stability,
            "early_confidence": early_confidence,
            "final_confidence": final_confidence,
            "confidence_growth": confidence_growth,
            "confidence_trajectory": confidence_values,
            "layer_indices": layer_indices,
        }
        
        return is_uncertain, uncertainty_score, metadata
    
    def analyze_layer_disagreement(
        self,
        layer_logits: Dict[int, torch.Tensor],
    ) -> Dict[str, any]:
        """
        Analyze disagreement between layers in detail.
        
        Args:
            layer_logits: Dictionary of layer logits
        
        Returns:
            Dictionary with disagreement analysis
        """
        # Compute pairwise disagreements
        sorted_layers = sorted(layer_logits.keys())
        disagreements = []
        
        for i in range(len(sorted_layers)):
            for j in range(i + 1, len(sorted_layers)):
                layer_i = sorted_layers[i]
                layer_j = sorted_layers[j]
                
                logits_i = layer_logits[layer_i][:, -1, :]
                logits_j = layer_logits[layer_j][:, -1, :]
                
                # Compute disagreement metrics
                prob_i = torch.softmax(logits_i, dim=-1)
                prob_j = torch.softmax(logits_j, dim=-1)
                
                # KL divergence
                kl_div = torch.nn.functional.kl_div(
                    torch.log(prob_i + 1e-10),
                    prob_j,
                    reduction='batchmean'
                ).item()
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    logits_i, logits_j, dim=-1
                ).item()
                
                disagreements.append({
                    "layer_i": layer_i,
                    "layer_j": layer_j,
                    "kl_divergence": kl_div,
                    "cosine_similarity": cos_sim,
                })
        
        return {
            "pairwise_disagreements": disagreements,
            "avg_kl_divergence": sum(d["kl_divergence"] for d in disagreements) / len(disagreements) if disagreements else 0.0,
            "avg_cosine_similarity": sum(d["cosine_similarity"] for d in disagreements) / len(disagreements) if disagreements else 0.0,
        }
    
    def get_early_uncertainty_signal(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        early_layer: int = 1,
    ) -> Tuple[bool, float]:
        """
        Get early uncertainty signal from a single early layer.
        
        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            early_layer: Layer to examine for early signal
        
        Returns:
            Tuple of (is_uncertain, confidence)
        """
        # Extract logits from early layer only
        layer_logits = self.extract_layer_logits(
            model, input_ids, attention_mask
        )
        
        if early_layer not in layer_logits:
            logger.warning(f"Layer {early_layer} not in extracted logits")
            return False, 0.5
        
        logits = layer_logits[early_layer][:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get top prediction confidence
        top_log_prob, _ = torch.max(log_probs, dim=-1)
        confidence = torch.exp(top_log_prob).item()
        
        # Early uncertainty if confidence is low
        is_uncertain = confidence < self.stability_threshold
        
        return is_uncertain, confidence
    
    def detect_hallucination(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[bool, float, Dict[str, any]]:
        """
        Detect hallucination using logit lens uncertainty analysis.
        
        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            threshold: Hallucination threshold
        
        Returns:
            Tuple of (is_hallucination, uncertainty_score, metadata)
        """
        # Extract layer logits
        layer_logits = self.extract_layer_logits(model, input_ids, attention_mask)
        
        # Compute layer predictions
        layer_predictions = self.compute_layer_predictions(layer_logits)
        
        # Detect uncertainty
        is_uncertain, uncertainty_score, metadata = self.detect_uncertainty(layer_predictions)
        
        # Analyze layer disagreement
        disagreement_analysis = self.analyze_layer_disagreement(layer_logits)
        metadata.update(disagreement_analysis)
        
        # Determine if hallucination
        is_hallucination = uncertainty_score > threshold
        
        return is_hallucination, uncertainty_score, metadata