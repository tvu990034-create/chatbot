"""
Query Classification System
Analyzes queries to determine domain, complexity, and required optimizations
"""

import re
from typing import Dict, List
from enum import Enum


class QueryDomain(Enum):
    """Query domain types."""
    MATHEMATICAL = "mathematical"
    CREATIVE = "creative"
    FACTUAL = "factual"
    REASONING = "reasoning"
    CODING = "coding"
    LANGUAGE = "language"
    GENERAL = "general"


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    INTERMEDIATE = "intermediate"
    HARD = "hard"
    VERY_HARD = "very_hard"
    EXPERT = "expert"


class QueryClassifier:
    """
    Classifies queries to enable intelligent routing and optimization selection.
    
    Uses pattern matching and heuristics to determine:
    - Query domain (mathematical, creative, factual, etc.)
    - Complexity level (simple to expert)
    - Required optimizations (CoT, factuality, etc.)
    """
    
    def __init__(self):
        # Domain-specific patterns
        self.mathematical_patterns = [
            r'\b\d+\s*[\+\-\*\/]\s*\d+',  # Simple arithmetic
            r'\b(calculat|solve|compute|find)\b.*\b\d+',  # Calculation verbs
            r'\b(percentage|fraction|ratio|proportion)\b',  # Math terms
            r'\b(equation|formula|theorem|proof)\b',  # Advanced math
            r'\b(sums?|product|average|mean|median)\b',  # Statistics
            r'[a-zA-Z]\d+.*\d+',  # Variables and numbers
            r'\b(prime|factor|divisible|multiple)\b',  # Number theory
        ]
        
        self.creative_patterns = [
            r'\b(invent|create|design|imagine|innovate)\b',
            r'\b(story|poem|narrative|creative|fiction)\b',
            r'\b(what if|imagine.*scenario|hypothetical)\b',
            r'\b(new.*idea|novel.*approach|creative.*solution)\b',
        ]
        
        self.factual_patterns = [
            r'\b(what is|who is|when did|where is|why did)\b',
            r'\b(capital|president|inventor|discovered)\b',
            r'\b(explain|describe|define)\b',
            r'\b(history|geography|science|biography)\b',
        ]
        
        self.reasoning_patterns = [
            r'\b(why|how|because|therefore|consequently)\b',
            r'\b(logic|reason|inference|deduction)\b',
            r'\b(cause|effect|relationship|connection)\b',
            r'\b(prove|argue|convince)\b',
        ]
        
        self.coding_patterns = [
            r'\b(code|function|class|algorithm|program)\b',
            r'\b(python|javascript|java|c\+\+|rust)\b',
            r'\b(debug|fix|implement|optimize)\b',
            r'\b(syntax|error|exception|bug)\b',
        ]
        
        # Complexity indicators
        self.complexity_keywords = {
            QueryComplexity.SIMPLE: [
                'what is', 'who is', 'basic', 'simple', 'easy'
            ],
            QueryComplexity.INTERMEDIATE: [
                'explain', 'describe', 'how does', 'why does'
            ],
            QueryComplexity.HARD: [
                'analyze', 'compare', 'evaluate', 'complex'
            ],
            QueryComplexity.VERY_HARD: [
                'optimize', 'design', 'implement', 'synthesize'
            ],
            QueryComplexity.EXPERT: [
                'prove', 'derive', 'research', 'advanced', 'theoretical'
            ]
        }
        
        # Length-based complexity thresholds
        self.length_thresholds = {
            QueryComplexity.SIMPLE: 50,
            QueryComplexity.INTERMEDIATE: 100,
            QueryComplexity.HARD: 200,
            QueryComplexity.VERY_HARD: 300,
        }
    
    def classify_domain(self, query: str) -> QueryDomain:
        """
        Classify the domain of the query.
        
        Args:
            query: The query string
            
        Returns:
            QueryDomain enum value
        """
        query_lower = query.lower()
        
        # Check each domain's patterns
        domain_scores = {
            QueryDomain.MATHEMATICAL: self._score_patterns(query_lower, self.mathematical_patterns),
            QueryDomain.CREATIVE: self._score_patterns(query_lower, self.creative_patterns),
            QueryDomain.FACTUAL: self._score_patterns(query_lower, self.factual_patterns),
            QueryDomain.REASONING: self._score_patterns(query_lower, self.reasoning_patterns),
            QueryDomain.CODING: self._score_patterns(query_lower, self.coding_patterns),
        }
        
        # Find highest scoring domain
        max_score = max(domain_scores.values())
        if max_score > 0:
            return max(domain_scores, key=domain_scores.get)
        
        return QueryDomain.GENERAL
    
    def classify_complexity(self, query: str, domain: QueryDomain) -> QueryComplexity:
        """
        Classify the complexity of the query.
        
        Args:
            query: The query string
            domain: The classified domain
            
        Returns:
            QueryComplexity enum value
        """
        query_lower = query.lower()
        
        # Check for complexity keywords
        complexity_scores = {}
        for complexity, keywords in self.complexity_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            complexity_scores[complexity] = score
        
        # Check length-based complexity
        query_length = len(query)
        length_complexity = QueryComplexity.SIMPLE
        for complexity, threshold in self.length_thresholds.items():
            if query_length > threshold:
                length_complexity = complexity
        
        # Combine keyword and length scores
        if max(complexity_scores.values()) > 0:
            return max(complexity_scores, key=complexity_scores.get)
        
        return length_complexity
    
    def determine_optimizations(self, domain: QueryDomain, complexity: QueryComplexity) -> List[str]:
        """
        Determine which optimizations should be applied.
        
        Args:
            domain: The classified domain
            complexity: The classified complexity
            
        Returns:
            List of optimization names to apply
        """
        optimizations = []
        
        # Domain-specific optimizations
        if domain == QueryDomain.MATHEMATICAL:
            optimizations.append("mathematical_enhancement")
            if complexity.value in [QueryComplexity.HARD.value, QueryComplexity.VERY_HARD.value, QueryComplexity.EXPERT.value]:
                optimizations.append("chain_of_thought")
        
        elif domain == QueryDomain.CREATIVE:
            optimizations.append("creative_thinking")
            # Higher temperature for creative tasks
        
        elif domain == QueryDomain.FACTUAL:
            if complexity.value in [QueryComplexity.VERY_HARD.value, QueryComplexity.EXPERT.value]:
                optimizations.append("factuality_check")
        
        elif domain == QueryDomain.REASONING:
            if complexity.value in [QueryComplexity.INTERMEDIATE.value, QueryComplexity.HARD.value]:
                optimizations.append("chain_of_thought")
            if complexity.value in [QueryComplexity.VERY_HARD.value, QueryComplexity.EXPERT.value]:
                optimizations.append("self_consistency")
        
        elif domain == QueryDomain.CODING:
            optimizations.append("code_aware_processing")
        
        # Complexity-based optimizations
        if complexity == QueryComplexity.EXPERT:
            optimizations.append("enhanced_reasoning")
        
        # Always apply basic optimizations
        optimizations.append("simple_cache")
        optimizations.append("adversarial_detection")
        
        return list(set(optimizations))  # Remove duplicates
    
    def get_suggested_temperature(self, domain: QueryDomain, complexity: QueryComplexity) -> float:
        """
        Get the suggested temperature for the query.
        
        Args:
            domain: The classified domain
            complexity: The classified complexity
            
        Returns:
            Suggested temperature value
        """
        # Domain-based temperature
        if domain == QueryDomain.MATHEMATICAL:
            base_temp = 0.0  # Deterministic for math
        elif domain == QueryDomain.CREATIVE:
            base_temp = 0.8  # High for creative
        elif domain == QueryDomain.FACTUAL:
            base_temp = 0.1  # Low for factual
        elif domain == QueryDomain.REASONING:
            base_temp = 0.3  # Medium for reasoning
        elif domain == QueryDomain.CODING:
            base_temp = 0.2  # Low for coding
        else:
            base_temp = 0.1  # Default
        
        # Complexity adjustment
        if complexity == QueryComplexity.EXPERT:
            return base_temp + 0.1
        elif complexity == QueryComplexity.VERY_HARD:
            return base_temp + 0.05
        elif complexity == QueryComplexity.SIMPLE:
            return max(0.0, base_temp - 0.05)
        
        return base_temp
    
    def classify(self, query: str) -> Dict:
        """
        Perform full classification of the query.
        
        Args:
            query: The query string
            
        Returns:
            Dictionary with classification results
        """
        domain = self.classify_domain(query)
        complexity = self.classify_complexity(query, domain)
        optimizations = self.determine_optimizations(domain, complexity)
        temperature = self.get_suggested_temperature(domain, complexity)
        
        return {
            "query": query,
            "domain": domain.value,
            "complexity": complexity.value,
            "optimizations": optimizations,
            "suggested_temperature": temperature,
            "requires_cot": "chain_of_thought" in optimizations,
            "requires_factuality": "factuality_check" in optimizations,
            "requires_creative": "creative_thinking" in optimizations,
        }
    
    def _score_patterns(self, text: str, patterns: List[str]) -> int:
        """
        Score how many patterns match the text.
        
        Args:
            text: Text to search
            patterns: List of regex patterns
            
        Returns:
            Number of pattern matches
        """
        score = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1
        return score


# Singleton instance
_classifier = None


def get_query_classifier() -> QueryClassifier:
    """Get the singleton query classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier()
    return _classifier
