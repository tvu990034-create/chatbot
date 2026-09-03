"""
factuality/hallucination_detector.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Comprehensive hallucination detector combining multiple techniques.

Based on research papers:
- SelfCheckGPT: https://arxiv.org/abs/2303.08896
- Language Models Know What They Know: https://arxiv.org/abs/2207.05221
- The Internal State of an LLM Knows When It's Lying: https://arxiv.org/abs/2304.13734
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from factuality.logit_processor import LogitProcessor
from factuality.selfcheck_gpt import SelfCheckGPT

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Comprehensive hallucination detector combining multiple techniques.
    
    Combines:
    - SelfCheckGPT: Response consistency checking
    - Logit confidence: Uncertainty quantification
    - Token uncertainty: Per-token uncertainty scores
    - Confidence thresholding: Low-confidence detection
    """
    
    def __init__(
        self,
        consistency_threshold: float = 0.5,
        confidence_threshold: float = 0.5,
        uncertainty_threshold: float = 0.5,
    ):
        self.consistency_threshold = consistency_threshold
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        
        self.logit_processor = LogitProcessor()
        self.selfcheck_gpt = SelfCheckGPT()
        
        logger.info(
            f"HallucinationDetector initialized with "
            f"consistency_threshold={consistency_threshold}, "
            f"confidence_threshold={confidence_threshold}, "
            f"uncertainty_threshold={uncertainty_threshold}"
        )
    
    def detect(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, any]:
        """
        Detect hallucination using multiple techniques.
        
        Args:
            model: Language model
            input_ids: Input token IDs
            attention_mask: Optional attention mask
            logits: Optional pre-computed logits
        
        Returns:
            Dictionary with detection results
        """
        results = {
            "is_hallucination": False,
            "confidence": 0.0,
            "techniques_used": [],
        }
        
        # Get logits if not provided
        if logits is None:
            model.eval()
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = outputs.logits[:, -1, :]
        
        # Technique 1: Logit confidence
        confidence = self.logit_processor.compute_confidence(logits)
        results["logit_confidence"] = confidence.item()
        results["techniques_used"].append("logit_confidence")
        
        if confidence.item() < self.confidence_threshold:
            results["low_confidence"] = True
        else:
            results["low_confidence"] = False
        
        # Technique 2: Token uncertainty
        uncertainty = self.logit_processor.compute_token_uncertainty(logits)
        results["token_uncertainty"] = uncertainty.item()
        results["techniques_used"].append("token_uncertainty")
        
        if uncertainty.item() > self.uncertainty_threshold:
            results["high_uncertainty"] = True
        else:
            results["high_uncertainty"] = False
        
        # Technique 3: SelfCheckGPT consistency
        try:
            is_hallucination, consistency_prob, responses = self.selfcheck_gpt.detect_hallucination(
                model,
                input_ids,
                attention_mask,
                threshold=self.consistency_threshold
            )
            results["selfcheck_hallucination"] = is_hallucination
            results["selfcheck_consistency"] = 1.0 - consistency_prob
            results["selfcheck_responses"] = responses
            results["techniques_used"].append("selfcheck_gpt")
        except Exception as e:
            logger.warning(f"SelfCheckGPT failed: {e}")
            results["selfcheck_hallucination"] = False
            results["selfcheck_consistency"] = 1.0
        
        # Combine results
        results["confidence"] = (confidence.item() + 
                              (1.0 - uncertainty.item()) + 
                              results.get("selfcheck_consistency", 1.0)) / 3.0
        
        # Determine final hallucination status
        results["is_hallucination"] = (
            results.get("low_confidence", False) or
            results.get("high_uncertainty", False) or
            results.get("selfcheck_hallucination", False)
        )
        
        return results
    
    def get_hallucination_report(
        self,
        detection_results: Dict[str, any],
    ) -> str:
        """
        Generate human-readable hallucination report.
        
        Args:
            detection_results: Results from detect() method
        
        Returns:
            Human-readable report
        """
        report = []
        report.append("=== Hallucination Detection Report ===")
        report.append(f"Overall Hallucination: {detection_results['is_hallucination']}")
        report.append(f"Overall Confidence: {detection_results['confidence']:.2f}")
        report.append("")
        report.append("Technique Results:")
        
        for technique in detection_results["techniques_used"]:
            if technique == "logit_confidence":
                report.append(f"  Logit Confidence: {detection_results['logit_confidence']:.2f}")
            elif technique == "token_uncertainty":
                report.append(f"  Token Uncertainty: {detection_results['token_uncertainty']:.2f}")
            elif technique == "selfcheck_gpt":
                report.append(f"  SelfCheckGPT Consistency: {detection_results['selfcheck_consistency']:.2f}")
        
        if "selfcheck_responses" in detection_results:
            report.append("")
            report.append("SelfCheckGPT Responses:")
            for i, response in enumerate(detection_results["selfcheck_responses"]):
                report.append(f"  Response {i+1}: {response[:100]}...")
        
        return "\n".join(report)