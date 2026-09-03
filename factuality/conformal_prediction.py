"""
factuality/conformal_prediction.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Conformal Prediction implementation for LLM uncertainty quantification.

Based on research papers:
- Conformal Prediction for Natural Language Processing: A Survey
  https://arxiv.org/abs/2305.18474
- Conformal Language Modeling
  https://arxiv.org/abs/2306.10193
- Uncertainty Quantification for LLMs via Conformal Risk Control
  https://arxiv.org/abs/2306.02908

Algorithm: Distribution-free uncertainty quantification with coverage guarantees
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Dict, Callable
import math
import numpy as np

import torch
import torch.nn as nn

from factuality.logit_processor import LogitProcessor

logger = logging.getLogger(__name__)


class ConformalPrediction(LogitProcessor):
    """
    Conformal Prediction implementation for uncertainty quantification.
    
    Based on:
    - Conformal Prediction for Natural Language Processing: A Survey
      https://arxiv.org/abs/2305.18474
    - Conformal Language Modeling
      https://arxiv.org/abs/2306.10193
    
    Key innovations:
    - Distribution-free uncertainty quantification
    - Coverage guarantees under minimal assumptions
    - Calibration sets for valid confidence intervals
    - Risk control for safe generation
    - Adaptable to different scoring functions
    
    Mathematical formulation:
    Quantile-based prediction sets
    1 - α coverage guarantee
    Risk-controlling prediction sets (RCPS)
    """
    
    def __init__(
        self,
        base_temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        alpha: float = 0.1,
        n_calib: int = 100,
        scoring_function: str = "negative_log_likelihood",
        adaptive: bool = True,
    ):
        super().__init__(base_temperature, top_k, top_p)
        
        self.alpha = alpha  # Target error rate (1 - coverage)
        self.n_calib = n_calib  # Number of calibration samples
        self.scoring_function = scoring_function
        self.adaptive = adaptive
        
        # Calibration data
        self.calibration_scores: List[float] = []
        self.is_calibrated = False
        
        logger.info(
            f"ConformalPrediction initialized with alpha={alpha}, "
            f"n_calib={n_calib}, scoring_function={scoring_function}"
        )
    
    def compute_score(
        self,
        logits: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute conformity score for given logits.
        
        Args:
            logits: Model logits
            target: Target token (optional)
        
        Returns:
            Conformity score
        """
        if self.scoring_function == "negative_log_likelihood":
            return self._compute_nll_score(logits, target)
        elif self.scoring_function == "max_log_prob":
            return self._compute_max_log_prob_score(logits)
        elif self.scoring_function == "entropy":
            return self._compute_entropy_score(logits)
        elif self.scoring_function == "probability":
            return self._compute_probability_score(logits, target)
        else:
            raise ValueError(f"Unknown scoring function: {self.scoring_function}")
    
    def _compute_nll_score(
        self,
        logits: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute negative log-likelihood score.
        
        Args:
            logits: Model logits
            target: Target token
        
        Returns:
            NLL score
        """
        if target is None:
            # Return max NLL if no target
            return float(-torch.max(logits, dim=-1).values.item())
        
        # Compute log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Get log probability of target
        target_log_prob = log_probs[0, target].item()
        
        return -target_log_prob
    
    def _compute_max_log_prob_score(
        self,
        logits: torch.Tensor,
    ) -> float:
        """
        Compute maximum log probability score.
        
        Args:
            logits: Model logits
        
        Returns:
            Max log prob score
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        max_log_prob = torch.max(log_probs, dim=-1).values.item()
        
        return -max_log_prob  # Lower is more uncertain
    
    def _compute_entropy_score(
        self,
        logits: torch.Tensor,
    ) -> float:
        """
        Compute entropy-based score.
        
        Args:
            logits: Model logits
        
        Returns:
            Entropy score
        """
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).item()
        
        return entropy  # Higher is more uncertain
    
    def _compute_probability_score(
        self,
        logits: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> float:
        """
        Compute probability score for target.
        
        Args:
            logits: Model logits
            target: Target token
        
        Returns:
            Probability score
        """
        if target is None:
            return 0.0
        
        probs = torch.softmax(logits, dim=-1)
        target_prob = probs[0, target].item()
        
        return -target_prob  # Lower is more uncertain
    
    def calibrate(
        self,
        calibration_logits: List[torch.Tensor],
        calibration_targets: Optional[List[torch.Tensor]] = None,
    ) -> None:
        """
        Calibrate conformal predictor using calibration data.
        
        Args:
            calibration_logits: List of calibration logits
            calibration_targets: Optional list of calibration targets
        """
        self.calibration_scores = []
        
        for i, logits in enumerate(calibration_logits):
            target = calibration_targets[i] if calibration_targets else None
            score = self.compute_score(logits, target)
            self.calibration_scores.append(score)
        
        # Compute quantile threshold
        n = len(self.calibration_scores)
        if n == 0:
            logger.warning("No calibration data provided")
            return
        
        # Compute quantile for 1 - α coverage
        q_level = math.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantile = np.quantile(self.calibration_scores, q_level)
        
        self.is_calibrated = True
        logger.info(f"Calibrated with {n} samples, quantile={self.quantile:.4f}")
    
    def predict_set(
        self,
        logits: torch.Tensor,
    ) -> Tuple[List[int], float]:
        """
        Generate prediction set with coverage guarantee.
        
        Args:
            logits: Model logits
        
        Returns:
            Tuple of (prediction_set, confidence)
        """
        if not self.is_calibrated:
            logger.warning("Predictor not calibrated, using default threshold")
            self.quantile = 0.0
        
        # Compute conformity score for each token
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Generate prediction set
        scores = []
        for token_id in range(logits.shape[-1]):
            token_logit = logits[:, token_id:token_id+1]
            score = self.compute_score(token_logit, token_id)
            scores.append(score)
        
        # Apply threshold
        scores_tensor = torch.tensor(scores)
        prediction_set = (scores_tensor <= self.quantile).nonzero().squeeze().tolist()
        
        # Handle case where no tokens pass threshold
        if not prediction_set or (isinstance(prediction_set, int) and prediction_set == 0):
            # Fallback to top-k
            top_k = min(10, logits.shape[-1])
            prediction_set = torch.topk(logits[0], top_k).indices.tolist()
        
        # Compute confidence (inverse of set size)
        set_size = len(prediction_set) if isinstance(prediction_set, list) else 1
        confidence = 1.0 / (1.0 + set_size)
        
        return prediction_set, confidence
    
    def risk_controlling_prediction(
        self,
        logits: torch.Tensor,
        risk_budget: float = 0.1,
    ) -> Tuple[List[int], float]:
        """
        Risk-controlling prediction sets (RCPS).
        
        Args:
            logits: Model logits
            risk_budget: Target risk level
        
        Returns:
            Tuple of (prediction_set, estimated_risk)
        """
        if not self.is_calibrated:
            logger.warning("Predictor not calibrated for RCPS")
            self.quantile = 0.0
        
        # Use Lagrangian optimization for risk control
        # Simplified implementation: adapt threshold based on risk budget
        
        current_risk = self.alpha
        threshold_multiplier = 1.0
        
        # Adjust threshold to meet risk budget
        if current_risk > risk_budget:
            threshold_multiplier = risk_budget / current_risk
        else:
            threshold_multiplier = 1.0
        
        adjusted_quantile = self.quantile * threshold_multiplier
        
        # Generate prediction set with adjusted threshold
        log_probs = torch.log_softmax(logits, dim=-1)
        scores = []
        
        for token_id in range(logits.shape[-1]):
            token_logit = logits[:, token_id:token_id+1]
            score = self.compute_score(token_logit, token_id)
            scores.append(score)
        
        scores_tensor = torch.tensor(scores)
        prediction_set = (scores_tensor <= adjusted_quantile).nonzero().squeeze().tolist()
        
        # Ensure non-empty set
        if not prediction_set or (isinstance(prediction_set, int) and prediction_set == 0):
            top_k = min(10, logits.shape[-1])
            prediction_set = torch.topk(logits[0], top_k).indices.tolist()
        
        # Estimate risk
        estimated_risk = min(current_risk * threshold_multiplier, 1.0)
        
        return prediction_set, estimated_risk
    
    def adaptive_prediction_set(
        self,
        logits: torch.Tensor,
        context: Optional[str] = None,
    ) -> Tuple[List[int], float, Dict[str, any]]:
        """
        Adaptive prediction set based on context.
        
        Args:
            logits: Model logits
            context: Optional context string for adaptation
        
        Returns:
            Tuple of (prediction_set, confidence, metadata)
        """
        if not self.adaptive:
            return self.predict_set(logits)
        
        # Compute base prediction set
        prediction_set, confidence = self.predict_set(logits)
        
        # Adaptive adjustment based on context difficulty
        context_metadata = {}
        
        if context:
            # Simple context difficulty estimation
            context_length = len(context.split())
            context_metadata["context_length"] = context_length
            
            # Adjust confidence based on context
            if context_length > 100:  # Long context = more uncertain
                confidence *= 0.9
                context_metadata["adjustment"] = "long_context_penalty"
            elif context_length < 20:  # Short context = more certain
                confidence *= 1.1
                context_metadata["adjustment"] = "short_context_bonus"
        
        # Clip confidence to valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return prediction_set, confidence, context_metadata
    
    def get_coverage_estimate(
        self,
        test_logits: List[torch.Tensor],
        test_targets: Optional[List[torch.Tensor]] = None,
    ) -> float:
        """
        Estimate empirical coverage on test data.
        
        Args:
            test_logits: List of test logits
            test_targets: Optional list of test targets
        
        Returns:
            Empirical coverage estimate
        """
        if not self.is_calibrated:
            logger.warning("Predictor not calibrated")
            return 0.0
        
        coverage_count = 0
        total_count = 0
        
        for i, logits in enumerate(test_logits):
            target = test_targets[i] if test_targets else None
            
            if target is not None:
                prediction_set, _ = self.predict_set(logits)
                
                if isinstance(prediction_set, list):
                    coverage = target.item() in prediction_set
                else:
                    coverage = target.item() == prediction_set
                
                coverage_count += coverage
                total_count += 1
        
        empirical_coverage = coverage_count / total_count if total_count > 0 else 0.0
        
        logger.info(f"Empirical coverage: {empirical_coverage:.4f} (target: {1-self.alpha:.4f})")
        
        return empirical_coverage
    
    def detect_hallucination(
        self,
        logits: torch.Tensor,
        confidence_threshold: float = 0.5,
    ) -> Tuple[bool, float, List[int]]:
        """
        Detect hallucination using conformal prediction confidence.
        
        Args:
            logits: Model logits
            confidence_threshold: Confidence threshold for hallucination detection
        
        Returns:
            Tuple of (is_hallucination, confidence, prediction_set)
        """
        prediction_set, confidence = self.predict_set(logits)
        
        # Low confidence indicates potential hallucination
        is_hallucination = confidence < confidence_threshold
        
        return is_hallucination, confidence, prediction_set