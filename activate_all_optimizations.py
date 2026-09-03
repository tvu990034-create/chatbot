"""
Systematic Activation of All Available Optimizations
This script enables every optimization module in the gateway folder
"""

import logging
import sys
from pathlib import Path

# Add gateway to path
gateway_path = Path(__file__).parent / "gateway"
sys.path.insert(0, str(gateway_path))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def activate_all_optimizations():
    """Systematically activate all available optimization modules"""
    
    logger.info("=== ACTIVATING ALL OPTIMIZATION MODULES ===")
    
    # Level 1: Basic Optimizations
    logger.info("Level 1: Basic Optimizations")
    try:
        from simple_cache import get_cache
        cache = get_cache()
        logger.info(f"✅ Simple Cache: {len(cache.cache)} entries")
    except Exception as e:
        logger.warning(f"❌ Simple Cache: {e}")
    
    try:
        from semantic_cache import SemanticCache
        semantic_cache = SemanticCache()
        logger.info("✅ Semantic Cache: Activated")
    except Exception as e:
        logger.warning(f"❌ Semantic Cache: {e}")
    
    # Level 2: Advanced Optimizations
    logger.info("Level 2: Advanced Optimizations")
    try:
        from advanced_optimization import TokenCostOptimizer
        cost_optimizer = TokenCostOptimizer(lambda_cost=0.3)
        logger.info("✅ Token Cost Optimizer: Activated")
    except Exception as e:
        logger.warning(f"❌ Token Cost Optimizer: {e}")
    
    try:
        from inference_optimization import InferenceActionMinimizer
        action_minimizer = InferenceActionMinimizer(lambda_time=0.1, mu_uncertainty=0.2)
        logger.info("✅ Inference Action Minimizer: Activated")
    except Exception as e:
        logger.warning(f"❌ Inference Action Minimizer: {e}")
    
    # Level 3: Performance Optimizations
    logger.info("Level 3: Performance Optimizations")
    try:
        from performance_optimizer import CacheOptimizer
        cache_optimizer = CacheOptimizer()
        logger.info("✅ Cache Optimizer: Activated")
    except Exception as e:
        logger.warning(f"❌ Cache Optimizer: {e}")
    
    # Level 4: Intelligence Optimizations
    logger.info("Level 4: Intelligence Optimizations")
    try:
        from query_classifier import get_query_classifier
        classifier = get_query_classifier()
        logger.info("✅ Query Classifier: Activated")
    except Exception as e:
        logger.warning(f"❌ Query Classifier: {e}")
    
    try:
        from mathematical_enhancer import get_mathematical_enhancer
        math_enhancer = get_mathematical_enhancer()
        logger.info("✅ Mathematical Enhancer: Activated")
    except Exception as e:
        logger.warning(f"❌ Mathematical Enhancer: {e}")
    
    try:
        from adversarial_detector import get_adversarial_detector
        detector = get_adversarial_detector()
        logger.info("✅ Adversarial Detector: Activated")
    except Exception as e:
        logger.warning(f"❌ Adversarial Detector: {e}")
    
    # Level 5: Advanced Intelligence
    logger.info("Level 5: Advanced Intelligence")
    try:
        from human_like_intelligence import HumanLikeIntelligenceSuite
        hli = HumanLikeIntelligenceSuite()
        logger.info("✅ Human-Like Intelligence Suite: Activated")
    except Exception as e:
        logger.warning(f"❌ Human-Like Intelligence Suite: {e}")
    
    try:
        from meta_reasoning import MetaReasoningEngine
        meta_engine = MetaReasoningEngine()
        logger.info("✅ Meta Reasoning Engine: Activated")
    except Exception as e:
        logger.warning(f"❌ Meta Reasoning Engine: {e}")
    
    # Level 6: Specialized Optimizations
    logger.info("Level 6: Specialized Optimizations")
    try:
        from arc_optimization import ARCOptimizer
        arc_optimizer = ARCOptimizer()
        logger.info("✅ ARC Optimizer: Activated")
    except Exception as e:
        logger.warning(f"❌ ARC Optimizer: {e}")
    
    try:
        from reasoning_optimization import ReasoningOptimizer
        reasoning_optimizer = ReasoningOptimizer()
        logger.info("✅ Reasoning Optimizer: Activated")
    except Exception as e:
        logger.warning(f"❌ Reasoning Optimizer: {e}")
    
    # Level 7: Universal Gateway
    logger.info("Level 7: Universal Enhanced Gateway")
    try:
        from universal_enhanced_gateway import UniversalEnhancedGateway
        universal_gateway = UniversalEnhancedGateway(
            model_name="gemma2:2b",
            enable_all_optimizations=True,
            performance_mode="speed"
        )
        logger.info("✅ Universal Enhanced Gateway: Fully Activated")
    except Exception as e:
        logger.warning(f"❌ Universal Enhanced Gateway: {e}")
    
    # Level 8: Chain of Thought
    logger.info("Level 8: Chain of Thought")
    try:
        from reasoning.chain_of_thought import ChainOfThought
        cot = ChainOfThought()
        logger.info("✅ Chain of Thought: Activated")
    except Exception as e:
        logger.warning(f"❌ Chain of Thought: {e}")
    
    logger.info("=== OPTIMIZATION ACTIVATION COMPLETE ===")

if __name__ == "__main__":
    activate_all_optimizations()