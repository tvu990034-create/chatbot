"""
Fixed Enhanced Gateway with Effective Speed and Smart Optimizations
Fixed to deliver actual speed improvements and better intelligence
"""

import logging
import time
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np

import litellm
from litellm import completion
from litellm.exceptions import (
    APIConnectionError,
    RateLimitError,
    ServiceUnavailableError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import get_litellm_model, settings
from .simple_cache import get_cache
from .calibration_metrics import CalibrationMetrics, estimate_confidence_from_response
from .inference_optimization import InferenceOptimizer
from .meta_reasoning import MetaReasoningSuite
from .performance_optimizer import PerformanceOptimizerSuite
from .advanced_optimization import AdvancedOptimizationSuite
from .training_diagnostics import TrainingDiagnosticsSuite

# Try to import advanced monitoring
try:
    from .advanced_monitoring import update_monitoring, get_monitoring_status
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

# Try to import extended monitoring
try:
    from .extended_monitoring import update_extended_monitoring, get_extended_monitoring_status
    EXTENDED_MONITORING_AVAILABLE = True
except ImportError:
    EXTENDED_MONITORING_AVAILABLE = False

# Try to import benchmark-specific optimizations
try:
    from .arc_optimization import ARCOptimizer, ARCReasoningEnhancer
    ARC_OPTIMIZATION_AVAILABLE = True
except ImportError:
    ARC_OPTIMIZATION_AVAILABLE = False

try:
    from .truthfulqa_optimization import TruthfulQAOptimizer, TruthfulQAEnhancedEvaluator
    TRUTHFULQA_OPTIMIZATION_AVAILABLE = True
except ImportError:
    TRUTHFULQA_OPTIMIZATION_AVAILABLE = False

try:
    from .agentharm_optimization import AgentHarmOptimizer, AgentHarmEnhancedEvaluator
    AGENTHARM_OPTIMIZATION_AVAILABLE = True
except ImportError:
    AGENTHARM_OPTIMIZATION_AVAILABLE = False

# Try to import advanced cognitive modules
try:
    from .human_like_intelligence import HumanLikeIntelligenceSuite
    HUMAN_INTELLIGENCE_AVAILABLE = True
except ImportError:
    HUMAN_INTELLIGENCE_AVAILABLE = False

try:
    from .long_term_memory import LongTermMemorySuite
    LONG_TERM_MEMORY_AVAILABLE = True
except ImportError:
    LONG_TERM_MEMORY_AVAILABLE = False

# Try to import safety and academic modules
try:
    from .safety_module import SafetySuite
    SAFETY_MODULE_AVAILABLE = True
except ImportError:
    SAFETY_MODULE_AVAILABLE = False

try:
    from .academic_module import AcademicSuite
    ACADEMIC_MODULE_AVAILABLE = True
except ImportError:
    ACADEMIC_MODULE_AVAILABLE = False

# Try to import Bayesian and causal reasoning
try:
    from .bayesian_causal_reasoning import BayesianReasoningEngine
    BAYESIAN_CAUSAL_AVAILABLE = True
except ImportError:
    BAYESIAN_CAUSAL_AVAILABLE = False

# Try to import advanced reasoning and calibration
try:
    from .advanced_reasoning_calibration import AdvancedReasoningSuite
    ADVANCED_REASONING_AVAILABLE = True
except ImportError:
    ADVANCED_REASONING_AVAILABLE = False

# Try to import reasoning optimization
try:
    from .reasoning_optimization import ReasoningOptimizationSuite
    REASONING_OPTIMIZATION_AVAILABLE = True
except ImportError:
    REASONING_OPTIMIZATION_AVAILABLE = False

# Try to import decision theory and argumentation
try:
    from .decision_argumentation import DecisionArgumentationSuite
    DECISION_ARGUMENTATION_AVAILABLE = True
except ImportError:
    DECISION_ARGUMENTATION_AVAILABLE = False

# Try to import reasoning modules
try:
    from reasoning.chain_of_thought import ChainOfThought
    COT_AVAILABLE = True
except ImportError:
    COT_AVAILABLE = False

try:
    from gateway.adversarial_detector import AdversarialDetector
    ADVERSARIAL_AVAILABLE = True
except ImportError:
    ADVERSARIAL_AVAILABLE = False

try:
    from gateway.knowledge_base import retrieve_context
    KB_AVAILABLE = True
except ImportError:
    KB_AVAILABLE = False

logger = logging.getLogger(__name__)

# Configure litellm
litellm.set_verbose = False

# Initialize cache
_cache = get_cache()

@dataclass
class QueryAnalysis:
    """Analysis of query characteristics for optimization routing."""
    is_complex: bool = False
    needs_reasoning: bool = False
    is_adversarial: bool = False
    needs_recent_info: bool = False
    is_creative: bool = False
    difficulty_score: float = 0.5
    estimated_tokens: int = 0
    confidence_estimate: float = 0.5
    # BUG 5 FIX: Add query_text field for adaptive temperature
    query_text: str = ""

class FixedEnhancedGateway:
    """Fixed enhanced gateway with effective speed and smart optimizations."""
    
    def __init__(self):
        self.cot = ChainOfThought() if COT_AVAILABLE else None
        self.adversarial_detector = AdversarialDetector() if ADVERSARIAL_AVAILABLE else None
        
        # Speed optimization settings
        self.use_cache = True  # Always enable for speed
        self.enable_prompt_compression = True
        self.compression_ratio = 0.7
        self.enable_difficulty_routing = True
        self.difficulty_threshold = 0.7
        
        # Smart optimization settings
        self.enable_adaptive_temp = True
        self.base_temp = 0.7
        
        # Calibration diagnostics
        self.calibration_metrics = CalibrationMetrics()
        
        # Inference optimization
        self.inference_optimizer = InferenceOptimizer()
        
        # Meta-reasoning
        self.meta_reasoning = MetaReasoningSuite()
        
        # Performance optimization
        self.performance_optimizer = PerformanceOptimizerSuite()
        
        # Advanced optimization
        self.advanced_optimizer = AdvancedOptimizationSuite()
        
        # Training diagnostics
        self.training_diagnostics = TrainingDiagnosticsSuite()
        
        # NEW: Weakness-specific enhancements
        self.formal_math_enhancement = True  # For MiniF2F formal proofs
        self.agent_task_enhancement = True    # For OSWorld, GAIA, AgentBench
        self.academic_knowledge_enhancement = True  # For MMLU, GPQA
        self.proof_reasoning_templates = self._load_proof_templates()
        self.agent_coordination_patterns = self._load_agent_patterns()
        self.academic_context_bank = self._load_academic_context()
        
        # Advanced monitoring
        self.monitoring_enabled = MONITORING_AVAILABLE
        self.extended_monitoring_enabled = EXTENDED_MONITORING_AVAILABLE
        
        # Benchmark-specific optimizers
        self.arc_optimizer = ARCOptimizer() if ARC_OPTIMIZATION_AVAILABLE else None
        self.truthfulqa_optimizer = TruthfulQAOptimizer() if TRUTHFULQA_OPTIMIZATION_AVAILABLE else None
        self.agentharm_optimizer = AgentHarmOptimizer() if AGENTHARM_OPTIMIZATION_AVAILABLE else None
        
        # Advanced cognitive modules
        self.human_intelligence = HumanLikeIntelligenceSuite() if HUMAN_INTELLIGENCE_AVAILABLE else None
        self.long_term_memory = LongTermMemorySuite() if LONG_TERM_MEMORY_AVAILABLE else None
        self.safety_suite = SafetySuite() if SAFETY_MODULE_AVAILABLE else None
        self.academic_suite = AcademicSuite() if ACADEMIC_MODULE_AVAILABLE else None
        
        # Bayesian and causal reasoning
        self.bayesian_causal_enabled = BAYESIAN_CAUSAL_AVAILABLE
        self.reasoning_engine = BayesianReasoningEngine() if BAYESIAN_CAUSAL_AVAILABLE else None
        
        # Advanced reasoning and calibration
        self.advanced_reasoning_enabled = ADVANCED_REASONING_AVAILABLE
        self.advanced_reasoning = AdvancedReasoningSuite() if ADVANCED_REASONING_AVAILABLE else None
        
        # Reasoning optimization
        self.reasoning_optimization_enabled = REASONING_OPTIMIZATION_AVAILABLE
        self.reasoning_optimizer = ReasoningOptimizationSuite() if REASONING_OPTIMIZATION_AVAILABLE else None
        
        # Decision argumentation
        self.decision_argumentation_enabled = DECISION_ARGUMENTATION_AVAILABLE
        self.decision_argumentation = DecisionArgumentationSuite() if DECISION_ARGUMENTATION_AVAILABLE else None
        
        # Statistics tracking
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        self.total_time = 0.0
    
    def get_cache_key(self, query: str) -> str:
        """Generate a consistent cache key (temperature-independent for better cache hits)."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Fast query analysis for optimization routing."""
        # BUG 5 FIX: Include query_text in QueryAnalysis
        analysis = QueryAnalysis(query_text=query)
        
        # Fast complexity check
        query_lower = query.lower()
        analysis.is_complex = len(query) > 100 or any(word in query_lower for word in ['explain', 'analyze', 'compare', 'evaluate', 'derive'])
        analysis.needs_reasoning = any(word in query_lower for word in ['why', 'how', 'because', 'reason', 'prove', 'show'])
        analysis.is_adversarial = any(word in query_lower for word in ['hack', 'exploit', 'bypass', 'override', 'jailbreak'])
        analysis.needs_recent_info = any(word in query_lower for word in ['latest', 'recent', 'current', 'today', 'now'])
        analysis.is_creative = any(word in query_lower for word in ['write', 'create', 'generate', 'imagine', 'story'])
        
        # Fast difficulty scoring
        analysis.difficulty_score = 0.3
        if analysis.is_complex:
            analysis.difficulty_score += 0.3
        if analysis.needs_reasoning:
            analysis.difficulty_score += 0.2
        if analysis.needs_recent_info:
            analysis.difficulty_score += 0.1
        if '?' in query and query.count('?') > 1:
            analysis.difficulty_score += 0.1
        
        # Estimate tokens
        analysis.estimated_tokens = len(query.split()) * 1.3
        
        # Confidence estimate
        analysis.confidence_estimate = 0.8 if not analysis.is_complex else 0.5
        
        return analysis
    
    def compress_prompt(self, prompt: str) -> str:
        """Fast prompt compression."""
        if not self.enable_prompt_compression:
            return prompt
        
        # Simple compression: remove extra whitespace
        compressed = ' '.join(prompt.split())
        
        # If still too long, truncate intelligently
        if len(compressed) > 2000:
            compressed = compressed[:2000] + "..."
        
        return compressed
    
    def apply_smart_enhancements(self, query: str, analysis: QueryAnalysis) -> str:
        """Apply smart enhancements for better responses."""
        enhanced = query
        
        # Add comprehensive math reasoning guidance for math problems
        if any(word in query.lower() for word in ['calculate', 'solve', 'how many', 'math', 'add', 'subtract', 'multiply', 'divide', 'sum', 'total', 'if', 'have', 'then', 'buy', 'apples', 'oranges', 'train', 'travel', 'speed', 'time', 'distance', 'perimeter', 'area', 'x=', 'equation']):
            enhanced = f"Solve this step by step: 1) Identify the numbers and operations, 2) Show your work, 3) Give the final answer clearly. Question: {query}"
        
        # Add reasoning guidance for complex queries
        elif analysis.is_complex or analysis.needs_reasoning:
            if self.cot:
                enhanced = f"Think step by step and explain your reasoning: {query}"
        
        return enhanced
    
    def get_adaptive_temperature(self, analysis: QueryAnalysis) -> float:
        """Get adaptive temperature based on query analysis."""
        if not self.enable_adaptive_temp:
            return self.base_temp
        
        # Lower temperature for factual queries, higher for creative
        if analysis.is_creative:
            return min(1.0, self.base_temp + 0.2)
        elif analysis.needs_reasoning:
            return self.base_temp
        else:
            return max(0.3, self.base_temp - 0.2)
    
    def apply_speed_optimizations(self, messages: List[Dict[str, str]], analysis: QueryAnalysis) -> List[Dict[str, str]]:
        """Apply speed optimizations to messages."""
        optimized_messages = messages.copy()
        
        # Only compress if query is long enough to benefit
        if self.enable_prompt_compression and analysis.estimated_tokens > 100:
            for msg in optimized_messages:
                if msg['role'] == 'user':
                    original_length = len(msg['content'])
                    msg['content'] = self.compress_prompt(msg['content'])
                    compression_ratio = len(msg['content']) / original_length if original_length > 0 else 1.0
                    logger.debug(f"Prompt compression: {compression_ratio:.2f} ratio")
        
        return optimized_messages
    
    @retry(
        retry=retry_if_exception_type((APIConnectionError, RateLimitError, ServiceUnavailableError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        api_base: str = None,
        use_cache: bool = True,
    ) -> str:
        """
        Fixed enhanced chat with effective speed and smart optimizations.
        """
        self.total_requests += 1
        start_time = time.time()
        
        if model is None:
            model = get_litellm_model()
        
        if max_tokens is None:
            max_tokens = min(512, settings.litellm_max_tokens)  # Reduce for speed
        
        # Get query for analysis
        query = messages[-1]['content'] if messages else ""
        
        # Check cache first (most important speed optimization)
        if use_cache and query and self.use_cache:
            cache_key = self.get_cache_key(query)
            cached = _cache.get(cache_key)
            if cached:
                self.cache_hits += 1
                logger.debug(f"Cache hit for query: {query[:40]}...")
                self.total_time += time.time() - start_time
                return cached
            else:
                self.cache_misses += 1
        
        # Fast query analysis
        analysis = self.analyze_query(query)
        logger.debug(f"Query analysis: complexity={analysis.is_complex}, difficulty={analysis.difficulty_score:.2f}")
        
        # Apply benchmark-specific optimizations (skip for speed unless needed)
        enhanced_query = query
        if self.arc_optimizer and self.arc_optimizer.is_arc_task(query):
            logger.debug("ARC task detected, applying ARC optimizations")
            enhanced_query = self.arc_optimizer.enhance_arc_prompt(query)
            analysis.is_complex = True
        
        # NEW: Apply weakness-specific enhancements
        if self.formal_math_enhancement and self._is_formal_math_task(query):
            logger.debug("Formal math task detected, applying proof enhancement")
            enhanced_query = self._enhance_formal_math_prompt(query)
            analysis.is_complex = True
            analysis.needs_reasoning = True
            
        if self.agent_task_enhancement and self._is_agent_task(query):
            logger.debug("Agent task detected, applying coordination enhancement")
            enhanced_query = self._enhance_agent_coordination_prompt(query)
            analysis.is_complex = True
            analysis.needs_reasoning = True
            
        if self.academic_knowledge_enhancement and self._is_academic_task(query):
            logger.debug("Academic task detected, applying knowledge enhancement")
            enhanced_query = self._enhance_academic_knowledge_prompt(query)
            analysis.is_complex = True
            
        if self.academic_knowledge_enhancement and self._is_gpqa_task(query):
            logger.debug("GPQA task detected, applying graduate-level enhancement")
            enhanced_query = self._enhance_gpqa_prompt(query)
            analysis.is_complex = True
            analysis.needs_reasoning = True
        
        # Apply smart enhancements only for complex queries
        if analysis.is_complex or analysis.needs_reasoning:
            enhanced_query = self.apply_smart_enhancements(query, analysis)
        
        # Update messages with enhanced query
        optimized_messages = messages.copy()
        optimized_messages[-1]['content'] = enhanced_query
        
        # Apply speed optimizations
        optimized_messages = self.apply_speed_optimizations(optimized_messages, analysis)
        
        # Get adaptive temperature
        if temperature is None:
            temperature = self.get_adaptive_temperature(analysis)
        
        # Skip adversarial detection for speed unless query looks suspicious
        if analysis.is_adversarial and self.adversarial_detector:
            detection = self.adversarial_detector.detect_adversarial(query)
            if detection['is_adversarial']:
                logger.warning(f"Adversarial query detected: {detection['warnings']}")
                warning = "\n\nNote: This query may contain adversarial patterns. Please think carefully before responding."
                optimized_messages[-1]['content'] = enhanced_query + warning
        
        # API call with optimizations
        kwargs = {
            "model": model,
            "messages": optimized_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        
        if api_base:
            kwargs["api_base"] = api_base
        
        api_start = time.time()
        response = completion(**kwargs)
        api_time = time.time() - api_start
        
        reply = response.choices[0].message.content if response.choices else ""
        
        total_elapsed = time.time() - start_time
        self.total_time += total_elapsed
        
        logger.debug(f"Fixed enhanced_chat() {total_elapsed:.2f}s (API: {api_time:.2f}s) | model={model} | temp={temperature:.2f}")
        
        # Cache the response
        if use_cache and query and self.use_cache:
            cache_key = self.get_cache_key(query)
            _cache.set(cache_key, reply)
        
        # Minimal monitoring (skip for speed unless needed)
        if analysis.is_complex and self.monitoring_enabled:
            try:
                quality_estimate = 1.0 if len(reply) > 10 else 0.5
                update_monitoring(quality_estimate, analysis.confidence_estimate)
            except Exception as e:
                logger.debug(f"Monitoring update failed: {e}")
        
        return reply
    
    def _load_proof_templates(self) -> Dict[str, str]:
        """Load formal proof reasoning templates for MiniF2F enhancement."""
        return {
            "mathematical_induction": "Use mathematical induction: Base case: show P(1) holds. Inductive step: assume P(k) holds, prove P(k+1) holds.",
            "direct_proof": "Construct a direct proof: Assume hypothesis, use logical steps to reach conclusion.",
            "contradiction": "Proof by contradiction: Assume the negation of what you want to prove, derive a contradiction.",
            "formal_structure": "Structure your proof clearly: Given: [axioms]. To prove: [statement]. Proof: [steps]. QED."
        }
    
    def _load_agent_patterns(self) -> Dict[str, str]:
        """Load agent coordination patterns for OSWorld/GAIA/AgentBench enhancement."""
        return {
            "task_decomposition": "Break down complex tasks into clear sub-steps with dependencies.",
            "coordination": "When coordinating multiple agents, clearly define roles and communication protocols.",
            "tool_usage": "For system interactions, specify exact tool calls and expected outputs.",
            "error_recovery": "Include fallback strategies and error handling in your plan."
        }
    
    def _load_academic_context(self) -> Dict[str, str]:
        """Load academic knowledge context for MMLU/GPQA enhancement."""
        return {
            "step_by_step": "Work through academic problems systematically: identify key concepts, apply relevant theories, verify assumptions.",
            "domain_knowledge": "Consider domain-specific knowledge and standard methodologies for this field.",
            "elimination": "For multiple choice, use elimination: rule out clearly wrong answers, analyze remaining options carefully.",
            "verification": "Always verify your answer by cross-checking with domain principles."
        }
    
    def _is_formal_math_task(self, query: str) -> bool:
        """Detect formal mathematical proof tasks (MiniF2F style)."""
        formal_indicators = [
            "prove", "theorem", "lemma", "axiom", "formal", "proof", 
            "mathematical", "induction", "contradiction", "q.e.d", "qed",
            "theorem proof", "formal proof", "mathematical proof"
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in formal_indicators)
    
    def _is_agent_task(self, query: str) -> bool:
        """Detect agent-based tasks (OSWorld/GAIA/AgentBench style)."""
        agent_indicators = [
            "agent", "coordinate", "system", "operating system", "os", 
            "environment", "task", "action", "step", "execute", "tool",
            "gaia", "osworld", "agentbench", "multi-agent"
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in agent_indicators)
    
    def _is_academic_task(self, query: str) -> bool:
        """Detect academic knowledge tasks (MMLU/GPQA style)."""
        academic_indicators = [
            "multiple choice", "choose", "select", "answer", "academic",
            "knowledge", "test", "question", "exam", "professor", "university",
            "mmlu", "graduate", "subject", "discipline",
            "which of the following", "correct answer", "a) b) c) d)", "abstract algebra",
            "group theory", "physics", "chemistry", "biology", "history"
        ]
        # Exclude GPQA from academic enhancement (needs specialized handling)
        gpqa_indicators = ["gpqa", "graduate level", "phd", "expert"]
        query_lower = query.lower()
        
        has_academic = any(indicator in query_lower for indicator in academic_indicators)
        is_gpqa = any(indicator in query_lower for indicator in gpqa_indicators)
        
        return has_academic and not is_gpqa
    
    def _enhance_formal_math_prompt(self, query: str) -> str:
        """Enhance formal math tasks with proof templates and structure."""
        enhanced = query
        # Add proof structure guidance
        enhanced += "\n\nProof Structure Guidance:\n"
        enhanced += "1. Clearly state what you need to prove\n"
        enhanced += "2. List given information and axioms\n"
        enhanced += "3. Choose appropriate proof method (induction, contradiction, direct)\n"
        enhanced += "4. Present each step clearly with logical justification\n"
        enhanced += "5. Conclude with QED\n\n"
        # Add mathematical rigor emphasis
        enhanced += "Use formal mathematical notation and ensure each step follows logically from the previous one."
        return enhanced
    
    def _enhance_agent_coordination_prompt(self, query: str) -> str:
        """Enhance agent tasks with coordination patterns and system awareness."""
        enhanced = query
        # Add task decomposition guidance
        enhanced += "\n\nTask Coordination Framework:\n"
        enhanced += "1. Analyze the environment and available tools\n"
        enhanced += "2. Decompose the task into clear, sequential steps\n"
        enhanced += "3. Specify exact actions and expected outcomes for each step\n"
        enhanced += "4. Include error handling and fallback strategies\n"
        enhanced += "5. Define how to verify successful completion\n\n"
        # Add system interaction guidance
        enhanced += "For system interactions: specify exact commands, expected outputs, and verification methods."
        return enhanced
    
    def _is_gpqa_task(self, query: str) -> bool:
        """Detect GPQA graduate-level academic tasks."""
        gpqa_indicators = [
            "gpqa", "graduate-level", "graduate level", "phd", "expert", "professor",
            "advanced", "specialized", "domain expert", "academic expert", "expert question"
        ]
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in gpqa_indicators)
    
    def _enhance_gpqa_prompt(self, query: str) -> str:
        """Enhance GPQA tasks with graduate-level reasoning guidance."""
        enhanced = query
        # Add graduate-level problem-solving approach
        enhanced += "\n\nGraduate-Level Problem-Solving Approach:\n"
        enhanced += "1. Identify the domain and level of expertise required\n"
        enhanced += "2. Apply specialized knowledge and advanced methodologies\n"
        enhanced += "3. Consider edge cases and subtle distinctions\n"
        enhanced += "4. Use precise terminology and technical accuracy\n"
        enhanced += "5. Provide clear reasoning for your answer\n\n"
        # Add careful analysis emphasis
        enhanced += "Apply graduate-level analytical thinking and domain expertise."
        return enhanced

    def _enhance_academic_knowledge_prompt(self, query: str) -> str:
        """Enhance academic tasks with knowledge retrieval and reasoning guidance."""
        enhanced = query
        # Add systematic problem-solving approach
        enhanced += "\n\nAcademic Problem-Solving Approach:\n"
        enhanced += "1. Identify the core concepts and domain of the question\n"
        enhanced += "2. Recall relevant theories, formulas, and methodologies\n"
        enhanced += "3. Work through the problem systematically step by step\n"
        enhanced += "4. For multiple choice: eliminate incorrect options, analyze remaining ones\n"
        enhanced += "5. Verify your answer using domain knowledge principles\n\n"
        # Add careful analysis emphasis
        enhanced += "Take time to consider all aspects of the question and apply domain-specific knowledge."
        return enhanced

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        cache_hit_rate = self.cache_hits / self.total_requests if self.total_requests > 0 else 0
        avg_time = self.total_time / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "performance_equations_enabled": True,
            "cache_enabled": self.use_cache,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "total_requests": self.total_requests,
            "average_response_time": avg_time,
            "total_time": self.total_time,
            "optimizations": {
                "human_intelligence": {"enabled": HUMAN_INTELLIGENCE_AVAILABLE},
                "long_term_memory": {"enabled": LONG_TERM_MEMORY_AVAILABLE},
                "safety_suite": {"enabled": SAFETY_MODULE_AVAILABLE},
                "academic_suite": {"enabled": ACADEMIC_MODULE_AVAILABLE}
            }
        }

# Create global instance
_fixed_enhanced_gateway = FixedEnhancedGateway()

def get_fixed_enhanced_gateway() -> FixedEnhancedGateway:
    """Get the fixed enhanced gateway instance."""
    return _fixed_enhanced_gateway