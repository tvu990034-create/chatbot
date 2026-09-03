"""
Efficient Reasoning Module for MMLU-Pro Benchmark
Optimized implementations of key reasoning techniques for benchmark performance.
"""

import re
from typing import Dict, List, Tuple
from config import settings


class EfficientReasoner:
    """Optimized reasoning that combines best techniques for speed and accuracy."""
    
    def __init__(self, model: str):
        self.model = model
        
    def answer_with_reasoning(self, question: str, choices: List[str]) -> Tuple[str, float]:
        """Answer using optimized chain-of-thought with verification."""
        from litellm import completion
        
        # Step 1: Initial reasoning
        cot_prompt = f"""Solve this step by step:

Question: {question}
A. {choices[0]}
B. {choices[1]}  
C. {choices[2]}
D. {choices[3]}

Think through this methodically:
1. What is the question asking?
2. What information do I need?
3. Which option best answers the question?

Answer with just the letter (A, B, C, or D)."""
        
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": cot_prompt}],
                max_tokens=256,
                temperature=0.0
            )
            
            initial_answer = self._extract_letter(response.choices[0].message.content)
            
            # Step 2: Quick verification for high-stakes questions
            verify_prompt = f"""Double-check this answer:

Question: {question}
A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

My answer: {initial_answer}

Is this correct? Answer YES or NO, then the correct letter if NO."""
            
            verify_response = completion(
                model=self.model,
                messages=[{"role": "user", "content": verify_prompt}],
                max_tokens=20,
                temperature=0.0
            )
            
            verify_text = verify_response.choices[0].message.content.upper()
            
            if "NO" in verify_text:
                # Extract corrected answer
                corrected = self._extract_letter(verify_text)
                if corrected and corrected != initial_answer:
                    return corrected, 0.9  # High confidence after correction
            
            return initial_answer or "A", 0.8
            
        except Exception as e:
            print(f"Reasoning error: {e}")
            return "A", 0.0
    
    def _extract_letter(self, text: str) -> str:
        """Extract letter from response with improved accuracy."""
        text_upper = text.upper()
        
        # Try to find standalone letter patterns first (most accurate)
        import re
        # Pattern: letter followed by period or parenthesis, or at end of line
        standalone_patterns = [
            r'\b([A-D])\.\s*$',  # "A." at end
            r'\b([A-D])\)\s*$',  # "A)" at end
            r'^\s*([A-D])\s*$',  # Just "A"
            r'\b([A-D])\s*$',    # "A" at end
            r'^\s*([A-D])[\.\)]', # "A." or "A)" at start
        ]
        
        for pattern in standalone_patterns:
            match = re.search(pattern, text_upper, re.MULTILINE)
            if match:
                return match.group(1)
        
        # Try to find letter followed by "is" or "answer is"
        answer_patterns = [
            r'answer\s+is\s+([A-D])',
            r'([A-D])\s+is\s+the\s+answer',
            r'correct\s+answer\s+is\s+([A-D])',
            r'solution\s+is\s+([A-D])',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text_upper)
            if match:
                return match.group(1)
        
        # Fallback: find last occurrence of any letter (least accurate)
        # This is the original behavior but as a last resort
        for letter in ["D", "C", "B", "A"]:  # Search backwards
            if letter in text_upper:
                # Check if it's likely a standalone letter
                if re.search(rf'\b{letter}\b', text_upper):
                    return letter
        
        return ""


class OptimizedBenchmark:
    """Benchmark with efficient multi-strategy voting."""
    
    def __init__(self, model: str):
        self.model = model
        self.reasoner = EfficientReasoner(model)
        
    def answer_question(self, question: str, choices: List[str]) -> Tuple[str, float]:
        """Answer using strategy ensemble for maximum accuracy."""
        from litellm import completion
        
        strategies = []
        
        # Strategy 1: Direct with strong instructions
        direct_prompt = f"""Answer this question accurately:

{question}

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Choose the single best answer. Respond with just the letter."""
        
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": direct_prompt}],
                max_tokens=5,
                temperature=0.0
            )
            direct_answer = self._extract_letter(response.choices[0].message.content)
            strategies.append(("direct", direct_answer, 0.6))
        except:
            strategies.append(("direct", "A", 0.0))
        
        # Strategy 2: Elimination method
        elimination_prompt = f"""Use elimination to find the correct answer:

{question}

A. {choices[0]}
B. {choices[1]}
C. {choices[2]}
D. {choices[3]}

Eliminate wrong options step by step, then give the final answer letter."""
        
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": elimination_prompt}],
                max_tokens=80,
                temperature=0.0
            )
            elim_answer = self._extract_letter(response.choices[0].message.content)
            strategies.append(("elimination", elim_answer, 0.7))
        except:
            strategies.append(("elimination", "A", 0.0))
        
        # Strategy 3: Analytical reasoning
        analytical_prompt = f"""Analyze this carefully:

{question}

Options:
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Consider each option's validity. What is the most reasonable answer? Give the letter."""
        
        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": analytical_prompt}],
                max_tokens=60,
                temperature=0.0
            )
            analytical_answer = self._extract_letter(response.choices[0].message.content)
            strategies.append(("analytical", analytical_answer, 0.8))
        except:
            strategies.append(("analytical", "A", 0.0))
        
        # Weighted voting
        votes = {}
        total_weight = 0.0
        
        for strategy, answer, weight in strategies:
            if answer:  # Only count non-empty answers
                votes[answer] = votes.get(answer, 0) + weight
                total_weight += weight
        
        if not votes:
            return "A", 0.0
        
        # Get winner
        winner = max(votes, key=votes.get)
        confidence = votes[winner] / total_weight if total_weight > 0 else 0.0
        
        return winner, confidence
    
    def _extract_letter(self, text: str) -> str:
        """Extract letter from response with improved accuracy."""
        text_upper = text.upper()
        
        # Try to find standalone letter patterns first (most accurate)
        import re
        # Pattern: letter followed by period or parenthesis, or at end of line
        standalone_patterns = [
            r'\b([A-D])\.\s*$',  # "A." at end
            r'\b([A-D])\)\s*$',  # "A)" at end
            r'^\s*([A-D])\s*$',  # Just "A"
            r'\b([A-D])\s*$',    # "A" at end
            r'^\s*([A-D])[\.\)]', # "A." or "A)" at start
        ]
        
        for pattern in standalone_patterns:
            match = re.search(pattern, text_upper, re.MULTILINE)
            if match:
                return match.group(1)
        
        # Try to find letter followed by "is" or "answer is"
        answer_patterns = [
            r'answer\s+is\s+([A-D])',
            r'([A-D])\s+is\s+the\s+answer',
            r'correct\s+answer\s+is\s+([A-D])',
            r'solution\s+is\s+([A-D])',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, text_upper)
            if match:
                return match.group(1)
        
        # Fallback: find last occurrence of any letter (least accurate)
        # This is the original behavior but as a last resort
        for letter in ["D", "C", "B", "A"]:  # Search backwards
            if letter in text_upper:
                # Check if it's likely a standalone letter
                if re.search(rf'\b{letter}\b', text_upper):
                    return letter
        
        return ""