"""
factuality/selfcheck_gpt.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
SelfCheckGPT implementation based on research paper:

SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for LLMs
https://arxiv.org/abs/2303.08896

Algorithm: Detects hallucination by comparing multiple sampled responses
- Generates multiple responses for the same query
- Compares responses to detect inconsistencies
- Probability-based confidence scoring
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from factuality.logit_processor import LogitProcessor

logger = logging.getLogger(__name__)


class SelfCheckGPT(LogitProcessor):
    """
    SelfCheckGPT implementation for hallucination detection.
    
    Based on: SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for LLMs
    https://arxiv.org/abs/2303.08896
    
    Key innovations:
    - Generates multiple responses for the same query
    - Compares responses to detect inconsistencies
    - Probability-based confidence scoring
    - Zero-resource (no additional models needed)
    
    Mathematical formulation:
    Consistency = P(agreement among samples)
    Confidence = 1 - (entropy of responses)
    """
    
    def __init__(
        self,
        num_samples: int = 5,
        temperature_samples: float = 0.7,
        base_temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        super().__init__(base_temperature, top_k, top_p)
        
        self.num_samples = num_samples
        self.temperature_samples = temperature_samples
        
        logger.info(
            f"SelfCheckGPT initialized with num_samples={num_samples}, "
            f"temperature_samples={temperature_samples}"
        )
    
    def generate_multiple_responses(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
    ) -> List[str]:
        """
        Generate multiple responses for consistency checking.
        
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
    
    def compute_consistency_score(
        self,
        responses: List[str],
    ) -> float:
        """
        Compute consistency score across multiple responses using advanced metrics.
        
        Args:
            responses: List of generated responses
        
        Returns:
            Consistency score (0 to 1)
        """
        if len(responses) < 2:
            return 1.0
        
        # Check if all responses are identical
        if len(set(responses)) == 1:
            return 1.0
        
        # Compute multiple consistency metrics
        ngram_score = self._compute_ngram_overlap(responses)
        semantic_score = self._compute_semantic_similarity(responses)
        probability_score = self._compute_probability_consistency(responses)
        
        # Combine metrics
        consistency = 0.4 * ngram_score + 0.4 * semantic_score + 0.2 * probability_score
        
        return consistency
    
    def _compute_ngram_overlap(
        self,
        responses: List[str],
        n: int = 3,
    ) -> float:
        """
        Compute n-gram overlap consistency.
        
        Args:
            responses: List of responses
            n: N-gram size
        
        Returns:
            N-gram overlap score
        """
        def get_ngrams(text: str, n: int) -> set:
            words = text.lower().split()
            return set(zip(*[words[i:] for i in range(n)]))
        
        if len(responses) < 2:
            return 1.0
        
        # Get n-grams for each response
        ngrams_list = [get_ngrams(response, n) for response in responses]
        
        # Compute pairwise overlap
        total_overlap = 0.0
        num_pairs = 0
        
        for i in range(len(ngrams_list)):
            for j in range(i + 1, len(ngrams_list)):
                if not ngrams_list[i] or not ngrams_list[j]:
                    continue
                
                intersection = len(ngrams_list[i] & ngrams_list[j])
                union = len(ngrams_list[i] | ngrams_list[j])
                overlap = intersection / union if union > 0 else 0.0
                
                total_overlap += overlap
                num_pairs += 1
        
        return total_overlap / num_pairs if num_pairs > 0 else 0.0
    
    def _compute_semantic_similarity(
        self,
        responses: List[str],
    ) -> float:
        """
        Compute semantic similarity across responses.
        
        Args:
            responses: List of responses
        
        Returns:
            Semantic similarity score
        """
        # Simplified semantic similarity using word overlap
        def word_overlap(text1: str, text2: str) -> float:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
        
        if len(responses) < 2:
            return 1.0
        
        # Compute pairwise semantic similarity
        total_similarity = 0.0
        num_pairs = 0
        
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = word_overlap(responses[i], responses[j])
                total_similarity += similarity
                num_pairs += 1
        
        return total_similarity / num_pairs if num_pairs > 0 else 0.0
    
    def _compute_probability_consistency(
        self,
        responses: List[str],
    ) -> float:
        """
        Compute probability-based consistency (response length distribution).
        
        Args:
            responses: List of responses
        
        Returns:
            Probability consistency score
        """
        if len(responses) < 2:
            return 1.0
        
        # Compute response lengths
        lengths = [len(response.split()) for response in responses]
        
        # Compute coefficient of variation
        if len(lengths) == 0:
            return 1.0
        
        mean_length = sum(lengths) / len(lengths)
        if mean_length == 0:
            return 1.0
        
        std_length = (sum((l - mean_length) ** 2 for l in lengths) / len(lengths)) ** 0.5
        cv = std_length / mean_length if mean_length > 0 else 0.0
        
        # Lower CV = higher consistency
        consistency = 1.0 / (1.0 + cv)
        
        return consistency
    
    def compute_hallucination_probability(
        self,
        responses: List[str],
    ) -> float:
        """
        Compute probability of hallucination.
        
        Args:
            responses: List of generated responses
        
        Returns:
            Hallucination probability (0 to 1)
        """
        consistency = self.compute_consistency_score(responses)
        
        # Higher consistency = lower hallucination probability
        hallucination_prob = 1.0 - consistency
        
        return hallucination_prob
    
    def detect_hallucination(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        max_new_tokens: int = 50,
    ) -> Tuple[bool, float, List[str]]:
        """
        Detect hallucination in model output.
        
        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            threshold: Hallucination threshold
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Tuple of (is_hallucination, confidence_score, responses)
        """
        # Generate multiple responses
        responses = self.generate_multiple_responses(
            model,
            input_ids,
            attention_mask,
            max_new_tokens
        )
        
        # Compute hallucination probability
        hallucination_prob = self.compute_hallucination_probability(responses)
        
        # Determine if hallucination
        is_hallucination = hallucination_prob > threshold
        
        return is_hallucination, hallucination_prob, responses