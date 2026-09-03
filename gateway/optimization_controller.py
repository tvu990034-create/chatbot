"""
Optimization Controller
Central coordinator for all optimization modules
"""

import logging
from typing import Dict, List, Tuple
from config import settings

logger = logging.getLogger(__name__)


class OptimizationController:
    """
    Central controller that coordinates all optimization modules.
    
    Analyzes query characteristics and determines which optimizations
    to apply in what order, with dynamic adaptation based on performance.
    """
    
    def __init__(self):
        self.query_classifier = None
        self.math_enhancer = None
        self.adversarial_detector = None
        self.cot = None
        
        # Performance tracking
        self.optimization_stats = {
            "total_queries": 0,
            "optimization_counts": {},
            "performance_by_domain": {}
        }
        
        # Initialize modules
        self._initialize_modules()
    
    def _initialize_modules(self):
        """Initialize all optimization modules."""
        try:
            from .query_classifier import get_query_classifier
            self.query_classifier = get_query_classifier()
            logger.info("Query classifier initialized")
        except Exception as e:
            logger.warning(f"Query classifier initialization failed: {e}")
        
        try:
            from .mathematical_enhancer import get_mathematical_enhancer
            self.math_enhancer = get_mathematical_enhancer()
            logger.info("Mathematical enhancer initialized")
        except Exception as e:
            logger.warning(f"Mathematical enhancer initialization failed: {e}")
        
        try:
            from .adversarial_detector import get_adversarial_detector
            self.adversarial_detector = get_adversarial_detector()
            logger.info("Adversarial detector initialized")
        except Exception as e:
            logger.warning(f"Adversarial detector initialization failed: {e}")
        
        try:
            from reasoning.chain_of_thought import ChainOfThought
            self.cot = ChainOfThought()
            logger.info("Chain-of-Thought initialized")
        except Exception as e:
            logger.warning(f"Chain-of-Thought initialization failed: {e}")
    
    def analyze_query(self, query: str) -> Dict:
        """
        Analyze query to determine characteristics and optimization strategy.
        
        Args:
            query: The query string
            
        Returns:
            Dictionary with analysis results
        """
        if self.query_classifier is None:
            return {
                "domain": "general",
                "complexity": "intermediate",
                "optimizations": ["simple_cache", "adversarial_detection"],
                "suggested_temperature": settings.litellm_temperature
            }
        
        classification = self.query_classifier.classify(query)
        
        return classification
    
    def apply_optimizations(self, query: str, analysis: Dict) -> Tuple[str, Dict]:
        """
        Apply appropriate optimizations based on query analysis.
        
        Args:
            query: The original query
            analysis: Query analysis from analyze_query()
            
        Returns:
            Tuple of (optimized_query, optimization_metadata)
        """
        optimized_query = query
        applied_optimizations = []
        optimization_metadata = {
            "original_query": query,
            "applied_optimizations": [],
            "optimization_details": {}
        }
        
        # AGGRESSIVELY APPLY ALL AVAILABLE OPTIMIZATIONS
        
        # Apply adversarial detection (always)
        if self.adversarial_detector is not None:
            detection = self.adversarial_detector.detect_adversarial(query)
            if detection['is_adversarial']:
                enhanced_query = self.adversarial_detector.get_adversarial_prompt_enhancement(query, detection)
                optimized_query = enhanced_query
                applied_optimizations.append("adversarial_detection")
                optimization_metadata["optimization_details"]["adversarial"] = detection
        
        # ALWAYS apply mathematical enhancement for better responses
        if self.math_enhancer is not None:
            enhanced_query, math_info = self.math_enhancer.enhance_mathematical_query(optimized_query)
            if math_info.get('enhanced'):
                optimized_query = enhanced_query
                applied_optimizations.append("mathematical_enhancement")
                optimization_metadata["optimization_details"]["mathematical"] = math_info
        
        # Apply Chain-of-Thought for ALL queries (not just complex ones)
        if self.cot is not None:
            domain = analysis.get("domain", "general")
            cot_prompt = self.cot.build_cot_prompt(optimized_query, task_type=domain)
            optimized_query = cot_prompt
            applied_optimizations.append("chain_of_thought")
            optimization_metadata["optimization_details"]["cot"] = {
                "domain": domain,
                "complexity": analysis.get("complexity")
            }
        
        # Update metadata
        optimization_metadata["optimized_query"] = optimized_query
        optimization_metadata["applied_optimizations"] = applied_optimizations
        
        # Track statistics
        self._track_optimizations(applied_optimizations, analysis)
        
        return optimized_query, optimization_metadata
    
    def optimize_query(self, messages: list, model: str) -> list:
        """
        Main optimization entry point - applies all available optimizations.
        
        Args:
            messages: Original message list
            model: Model being used
            
        Returns:
            Optimized message list
        """
        if not messages:
            return messages
            
        query = messages[-1].get('content', '')
        if not query:
            return messages
            
        # Analyze query
        analysis = self.analyze_query(query)
        
        # Apply optimizations
        optimized_query, metadata = self.apply_optimizations(query, analysis)
        
        # Create optimized messages
        optimized_messages = messages.copy()
        optimized_messages[-1]['content'] = optimized_query
        
        logger.info(f"Applied optimizations: {metadata['applied_optimizations']}")
        
        return optimized_messages
    
    def get_optimal_temperature(self, analysis: Dict) -> float:
        """
        Get optimal temperature based on query analysis.
        
        Args:
            analysis: Query analysis from analyze_query()
            
        Returns:
            Optimal temperature value
        """
        suggested_temp = analysis.get("suggested_temperature")
        if suggested_temp is not None:
            return suggested_temp
        
        return settings.litellm_temperature
    
    def _track_optimizations(self, applied_optimizations: List[str], analysis: Dict):
        """Track optimization usage statistics."""
        self.optimization_stats["total_queries"] += 1
        
        for opt in applied_optimizations:
            if opt not in self.optimization_stats["optimization_counts"]:
                self.optimization_stats["optimization_counts"][opt] = 0
            self.optimization_stats["optimization_counts"][opt] += 1
        
        # Track by domain
        domain = analysis.get("domain", "general")
        if domain not in self.optimization_stats["performance_by_domain"]:
            self.optimization_stats["performance_by_domain"][domain] = {
                "count": 0,
                "optimizations": {}
            }
        
        self.optimization_stats["performance_by_domain"][domain]["count"] += 1
        for opt in applied_optimizations:
            if opt not in self.optimization_stats["performance_by_domain"][domain]["optimizations"]:
                self.optimization_stats["performance_by_domain"][domain]["optimizations"][opt] = 0
            self.optimization_stats["performance_by_domain"][domain]["optimizations"][opt] += 1
    
    def get_statistics(self) -> Dict:
        """Get optimization statistics."""
        return self.optimization_stats
    
    def reset_statistics(self):
        """Reset optimization statistics."""
        self.optimization_stats = {
            "total_queries": 0,
            "optimization_counts": {},
            "performance_by_domain": {}
        }


# Singleton instance
_controller = None


def get_optimization_controller() -> OptimizationController:
    """Get the singleton optimization controller instance."""
    global _controller
    if _controller is None:
        _controller = OptimizationController()
    return _controller
