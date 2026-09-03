"""
Mathematical Reasoning Enhancement
Simple enhancement for mathematical reasoning questions
"""

import re
from typing import Dict, Tuple

class MathematicalEnhancer:
    """Simple mathematical reasoning enhancement."""
    
    def __init__(self):
        self.math_patterns = [
            r'(?i)calculate|compute|solve|find|determine',
            r'(?i)\d+.*\+.*\d+',
            r'(?i)\d+.*\*.*\d+',
            r'(?i)\d+.*\/.*\d+',
            r'(?i)equation|formula|function',
            r'(?i)sum|average|mean|median',
            r'(?i)prime|factor|multiple',
            r'(?i)rate|ratio|proportion',
            r'(?i)area|volume|perimeter',
        ]
    
    def detect_mathematical(self, query: str) -> bool:
        """Detect if query is mathematical."""
        query_lower = query.lower()
        return any(re.search(pattern, query_lower) for pattern in self.math_patterns)
    
    def enhance_mathematical_query(self, query: str) -> Tuple[str, Dict]:
        """Enhance mathematical query with step-by-step guidance."""
        enhancement_info = {
            'enhanced': False,
            'method': None
        }
        
        if not self.detect_mathematical(query):
            return query, enhancement_info
        
        enhancement = f"""{query}

Please solve this step by step:
1. Identify what type of mathematical problem this is
2. Identify the given information
3. Plan your approach
4. Show your calculations
5. Verify your answer
"""
        
        enhancement_info['enhanced'] = True
        enhancement_info['method'] = 'step_by_step'
        
        return enhancement, enhancement_info


def get_mathematical_enhancer() -> MathematicalEnhancer:
    """Get mathematical enhancer instance."""
    return MathematicalEnhancer()
