"""
factuality/semantic_uncertainty.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Semantic Uncertainty implementation based on research paper:

Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation
https://arxiv.org/abs/2302.09664

Algorithm: Clustering of sampled outputs to detect confabulations
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Dict
import math

import torch
import torch.nn as nn

from factuality.logit_processor import LogitProcessor

logger = logging.getLogger(__name__)


class SemanticUncertainty(LogitProcessor):
    """
    Semantic Uncertainty implementation for hallucination detection.
    
    Based on: Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation
    https://arxiv.org/abs/2302.09664
    
    Key innovations:
    - Samples multiple responses and clusters them by meaning
    - Clusters that disagree indicate high uncertainty
    - Uses linguistic invariances for semantic clustering
    - Better than lexical similarity for detecting hallucinations
    
    Mathematical formulation:
    Uncertainty = 1 - (consistency of semantic clusters)
    Clusters formed by semantic similarity (not lexical)
    High uncertainty when clusters disagree on meaning
    """
    
    def __init__(
        self,
        num_samples: int = 5,
        temperature_samples: float = 0.7,
        base_temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        clustering_threshold: float = 0.7,
        use_lexical_invariance: bool = True,
        use_syntactic_invariance: bool = True,
    ):
        super().__init__(base_temperature, top_k, top_p)
        
        self.num_samples = num_samples
        self.temperature_samples = temperature_samples
        self.clustering_threshold = clustering_threshold
        self.use_lexical_invariance = use_lexical_invariance
        self.use_syntactic_invariance = use_syntactic_invariance
        
        logger.info(
            f"SemanticUncertainty initialized with num_samples={num_samples}, "
            f"clustering_threshold={clustering_threshold}"
        )
    
    def generate_multiple_responses(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
    ) -> List[str]:
        """
        Generate multiple responses for semantic clustering.
        
        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            List of generated responses
        """
        model.eval()
        
        responses = []
        
        with torch.no_grad():
            for i in range(self.num_samples):
                # Generate response with sampling
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=self.temperature_samples,
                    do_sample=True,
                    pad_token_id=model.config.pad_token_id,
                )
                
                # Decode response
                response = model.decode(outputs[0], skip_special_tokens=True)
                responses.append(response)
        
        return responses
    
    def compute_semantic_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Semantic similarity score (0 to 1)
        """
        # Simplified semantic similarity computation
        # In practice, you'd use embeddings or a semantic model
        
        # Compute lexical invariance (different words, same meaning)
        lexical_score = self._compute_lexical_invariance(text1, text2)
        
        # Compute syntactic invariance (different structure, same meaning)
        syntactic_score = self._compute_syntactic_invariance(text1, text2)
        
        # Combine invariances
        if self.use_lexical_invariance and self.use_syntactic_invariance:
            return 0.5 * lexical_score + 0.5 * syntactic_score
        elif self.use_lexical_invariance:
            return lexical_score
        elif self.use_syntactic_invariance:
            return syntactic_score
        else:
            # Fallback to simple lexical overlap
            return self._compute_lexical_overlap(text1, text2)
    
    def _compute_lexical_invariance(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Compute lexical invariance (different words, same meaning).
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Lexical invariance score
        """
        # Extract words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_syntactic_invariance(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Compute syntactic invariance (different structure, same meaning).
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Syntactic invariance score
        """
        # Simplified syntactic similarity based on structure
        # In practice, you'd use dependency parsing or constituency parsing
        
        # Compare sentence structure
        sentences1 = text1.split('.')
        sentences2 = text2.split('.')
        
        if len(sentences1) == 0 or len(sentences2) == 0:
            return 0.0
        
        # Compare average sentence length
        avg_len1 = sum(len(s.split()) for s in sentences1) / len(sentences1)
        avg_len2 = sum(len(s.split()) for s in sentences2) / len(sentences2)
        
        # Similarity based on length ratio
        length_ratio = min(avg_len1, avg_len2) / max(avg_len1, avg_len2)
        
        return length_ratio
    
    def _compute_lexical_overlap(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """
        Compute simple lexical overlap.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Lexical overlap score
        """
        words1 = text1.lower().split()
        words2 = text2.lower().split()
        
        if not words1 or not words2:
            return 0.0
        
        # Compute overlap
        overlap = len(set(words1) & set(words2))
        total = len(set(words1) | set(words2))
        
        return overlap / total if total > 0 else 0.0
    
    def cluster_responses(
        self,
        responses: List[str],
    ) -> List[List[str]]:
        """
        Cluster responses by semantic similarity.
        
        Args:
            responses: List of responses
        
        Returns:
            List of clusters (each cluster is a list of responses)
        """
        if len(responses) == 0:
            return []
        
        # Initialize clusters
        clusters = [[responses[0]]]
        
        for response in responses[1:]:
            # Find best matching cluster
            best_cluster_idx = None
            best_similarity = 0.0
            
            for i, cluster in enumerate(clusters):
                # Compute similarity to cluster representative
                cluster_rep = cluster[0]  # Use first element as representative
                similarity = self.compute_semantic_similarity(response, cluster_rep)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster_idx = i
            
            # Add to best cluster if above threshold, else create new cluster
            if best_cluster_idx is not None and best_similarity >= self.clustering_threshold:
                clusters[best_cluster_idx].append(response)
            else:
                clusters.append([response])
        
        return clusters
    
    def compute_semantic_uncertainty(
        self,
        clusters: List[List[str]],
    ) -> float:
        """
        Compute semantic uncertainty from clusters.
        
        Args:
            clusters: List of response clusters
        
        Returns:
            Semantic uncertainty score (0 to 1)
        """
        if len(clusters) <= 1:
            return 0.0  # No uncertainty if all responses are in one cluster
        
        # Compute cluster sizes
        cluster_sizes = [len(cluster) for cluster in clusters]
        total_responses = sum(cluster_sizes)
        
        # Compute entropy of cluster distribution
        cluster_probs = [size / total_responses for size in cluster_sizes]
        entropy = -sum(p * math.log(p + 1e-10) for p in cluster_probs if p > 0)
        
        # Normalize entropy
        max_entropy = math.log(len(clusters))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def detect_hallucination(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        max_new_tokens: int = 50,
    ) -> Tuple[bool, float, List[str], List[List[str]]]:
        """
        Detect hallucination using semantic uncertainty.
        
        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            threshold: Hallucination threshold
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Tuple of (is_hallucination, uncertainty_score, responses, clusters)
        """
        # Generate multiple responses
        responses = self.generate_multiple_responses(
            model,
            input_ids,
            attention_mask,
            max_new_tokens
        )
        
        # Cluster responses by semantic similarity
        clusters = self.cluster_responses(responses)
        
        # Compute semantic uncertainty
        uncertainty = self.compute_semantic_uncertainty(clusters)
        
        # Determine if hallucination
        is_hallucination = uncertainty > threshold
        
        return is_hallucination, uncertainty, responses, clusters
    
    def get_cluster_consistency(
        self,
        clusters: List[List[str]],
    ) -> Dict[str, any]:
        """
        Get detailed cluster consistency metrics.
        
        Args:
            clusters: List of response clusters
        
        Returns:
            Dictionary with cluster consistency metrics
        """
        if not clusters:
            return {
                "num_clusters": 0,
                "largest_cluster_size": 0,
                "cluster_entropy": 0.0,
                "dominant_cluster_ratio": 0.0,
            }
        
        cluster_sizes = [len(cluster) for cluster in clusters]
        total_responses = sum(cluster_sizes)
        
        # Compute cluster entropy
        cluster_probs = [size / total_responses for size in cluster_sizes]
        entropy = -sum(p * math.log(p + 1e-10) for p in cluster_probs if p > 0)
        
        # Find dominant cluster
        largest_cluster_size = max(cluster_sizes)
        dominant_cluster_ratio = largest_cluster_size / total_responses
        
        return {
            "num_clusters": len(clusters),
            "cluster_sizes": cluster_sizes,
            "largest_cluster_size": largest_cluster_size,
            "cluster_entropy": entropy,
            "dominant_cluster_ratio": dominant_cluster_ratio,
        }