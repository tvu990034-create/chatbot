"""
reasoning/chain_of_thought.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Chain-of-Thought (CoT) implementation based on research paper:

Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
https://arxiv.org/abs/2201.11903

Algorithm: Intermediate reasoning steps improve multi-step reasoning
"""

from __future__ import annotations

import logging
from typing import List, Optional, Dict, Tuple
import re

logger = logging.getLogger(__name__)


class ChainOfThought:
    """
    Chain-of-Thought reasoning implementation.
    
    Based on: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
    https://arxiv.org/abs/2201.11903
    
    Key innovations:
    - Uses intermediate reasoning steps
    - Improves multi-step reasoning tasks
    - Better performance on math, commonsense, and symbolic reasoning
    - Can be applied in few-shot and zero-shot settings
    
    Mathematical formulation:
    Question → [Reasoning steps] → Answer
    Each step builds on previous reasoning
    Final answer is derived from accumulated reasoning
    """
    
    def __init__(
        self,
        use_few_shot: bool = True,
        max_reasoning_steps: int = 10,
        enforce_step_format: bool = True,
        step_delimiter: str = "->",
        enable_prompt_optimization: bool = True,
        max_examples_in_prompt: int = 3,
    ):
        self.use_few_shot = use_few_shot
        self.max_reasoning_steps = max_reasoning_steps
        self.enforce_step_format = enforce_step_format
        self.step_delimiter = step_delimiter
        self.enable_prompt_optimization = enable_prompt_optimization
        self.max_examples_in_prompt = max_examples_in_prompt
        
        # Few-shot examples for common reasoning tasks
        self.few_shot_examples = self._load_few_shot_examples()
        
        logger.info(
            f"ChainOfThought initialized with use_few_shot={use_few_shot}, "
            f"max_reasoning_steps={max_reasoning_steps}, "
            f"enable_prompt_optimization={enable_prompt_optimization}"
        )
    
    def _load_few_shot_examples(self) -> Dict[str, List[Dict]]:
        """Load few-shot examples for different reasoning tasks."""
        return {
            "math": [
                {
                    "question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?",
                    "reasoning": "Roger starts with 5 balls. He buys 2 cans, each with 3 balls, so he gets 2 * 3 = 6 more balls. Total balls = 5 + 6 = 11.",
                    "answer": "11"
                },
                {
                    "question": "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?",
                    "reasoning": "They started with 23 apples. They used 20, so 23 - 20 = 3 apples left. They bought 6 more, so 3 + 6 = 9 apples.",
                    "answer": "9"
                }
            ],
            "commonsense": [
                {
                    "question": "If I put a heavy book on a weak table, what will happen?",
                    "reasoning": "A weak table cannot support heavy objects. The weight of the book exceeds the table's capacity. Therefore, the table will likely break or collapse.",
                    "answer": "The table will break or collapse"
                },
                {
                    "question": "If I leave ice cream in the sun, what will happen?",
                    "reasoning": "Ice cream melts when exposed to heat. The sun provides heat energy. Therefore, the ice cream will melt into liquid form.",
                    "answer": "The ice cream will melt"
                }
            ]
        }
    
    def build_cot_prompt(
        self,
        question: str,
        task_type: str = "math",
    ) -> str:
        """
        Build Chain-of-Thought prompt with examples.
        
        Args:
            question: The question to answer
            task_type: Type of reasoning task (math, commonsense, etc.)
        
        Returns:
            Formatted CoT prompt
        """
        if self.use_few_shot and task_type in self.few_shot_examples:
            # Build optimized few-shot prompt
            if self.enable_prompt_optimization:
                # Use limited number of diverse examples
                examples = self._select_diverse_examples(task_type, question)
                prompt = "Let's think step by step to solve this problem.\n\n"
                
                for example in examples:
                    prompt += f"Q: {example['question']}\n"
                    prompt += f"A: {example['reasoning']} The answer is {example['answer']}.\n\n"
                
                prompt += f"Q: {question}\n"
                prompt += "A: "
            else:
                # Original approach with all examples
                prompt = "Let's think step by step to solve this problem.\n\n"
                
                for example in self.few_shot_examples[task_type]:
                    prompt += f"Q: {example['question']}\n"
                    prompt += f"A: {example['reasoning']} The answer is {example['answer']}.\n\n"
                
                prompt += f"Q: {question}\n"
                prompt += "A: "
        else:
            # Zero-shot CoT
            prompt = f"Q: {question}\n"
            prompt += "A: Let's think step by step.\n"
        
        return prompt
    
    def _select_diverse_examples(
        self,
        task_type: str,
        question: str,
    ) -> List[Dict]:
        """
        Select diverse examples for few-shot prompting.
        
        Args:
            task_type: Type of reasoning task
            question: The target question
        
        Returns:
            Selected examples
        """
        if task_type not in self.few_shot_examples:
            return []
        
        all_examples = self.few_shot_examples[task_type]
        
        # If we have few examples, return all of them
        if len(all_examples) <= self.max_examples_in_prompt:
            return all_examples
        
        # Select examples based on similarity to target question
        import re
        question_numbers = set(re.findall(r'\d+', question))
        
        # Score examples by diversity and relevance
        scored_examples = []
        for example in all_examples:
            example_numbers = set(re.findall(r'\d+', example['question']))
            
            # Calculate overlap
            overlap = len(question_numbers & example_numbers)
            
            # Prefer examples with some overlap but not identical
            score = overlap if 0 < overlap < len(question_numbers) else 0
            
            scored_examples.append((score, example))
        
        # Sort by score and take top examples
        scored_examples.sort(key=lambda x: x[0], reverse=True)
        
        # If no good matches, take first N examples
        if scored_examples[0][0] == 0:
            return all_examples[:self.max_examples_in_prompt]
        
        return [example for score, example in scored_examples[:self.max_examples_in_prompt]]
    
    def parse_reasoning_steps(
        self,
        response: str,
    ) -> List[str]:
        """
        Parse reasoning steps from model response.
        
        Args:
            response: Model response containing reasoning
        
        Returns:
            List of reasoning steps
        """
        # Try to extract numbered steps
        numbered_steps = re.findall(r'\d+\.?\s*([^.!?]+[.!?])', response)
        if numbered_steps:
            return numbered_steps
        
        # Try to extract steps delimited by specific markers
        if self.step_delimiter in response:
            steps = [step.strip() for step in response.split(self.step_delimiter)]
            return steps
        
        # Try to extract sentences as steps
        sentences = re.split(r'[.!?]+', response)
        steps = [s.strip() for s in sentences if s.strip()]
        
        return steps[:self.max_reasoning_steps]
    
    def extract_answer(
        self,
        response: str,
    ) -> str:
        """
        Extract final answer from reasoning response.
        
        Args:
            response: Model response
        
        Returns:
            Extracted answer
        """
        # Look for answer patterns
        answer_patterns = [
            r'The answer is\s*([^.!?]+)',
            r'Answer:\s*([^.!?]+)',
            r'Therefore,?\s*([^.!?]+)',
            r'So,?\s*([^.!?]+)',
            r'Final answer:\s*([^.!?]+)',
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Try to extract multiple choice letter if present
        letter_patterns = [
            r'\b([A-D])\.\s*$',  # "A." at end
            r'\b([A-D])\)\s*$',  # "A)" at end
            r'^\s*([A-D])\s*$',  # Just "A"
            r'answer\s+is\s+([A-D])',
            r'correct\s+answer\s+is\s+([A-D])',
        ]
        
        for pattern in letter_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # If no pattern found, return last sentence
        sentences = re.split(r'[.!?]+', response)
        if sentences:
            return sentences[-1].strip()
        
        return response.strip()
    
    def reason(
        self,
        question: str,
        model_fn: callable,
        task_type: str = "math",
    ) -> Dict[str, any]:
        """
        Perform Chain-of-Thought reasoning.
        
        Args:
            question: Question to reason about
            model_fn: Function to call the model
            task_type: Type of reasoning task
        
        Returns:
            Dictionary with reasoning results
        """
        # Build CoT prompt
        prompt = self.build_cot_prompt(question, task_type)
        
        # Get model response
        response = model_fn(prompt)
        
        # Parse reasoning steps
        reasoning_steps = self.parse_reasoning_steps(response)
        
        # Extract final answer
        answer = self.extract_answer(response)
        
        return {
            "question": question,
            "prompt": prompt,
            "response": response,
            "reasoning_steps": reasoning_steps,
            "answer": answer,
            "num_steps": len(reasoning_steps),
            "task_type": task_type,
        }
    
    def batch_reason(
        self,
        questions: List[str],
        model_fn: callable,
        task_type: str = "math",
    ) -> List[Dict[str, any]]:
        """
        Perform Chain-of-Thought reasoning on multiple questions.
        
        Args:
            questions: List of questions
            model_fn: Function to call the model
            task_type: Type of reasoning task
        
        Returns:
            List of reasoning results
        """
        results = []
        
        for question in questions:
            result = self.reason(question, model_fn, task_type)
            results.append(result)
        
        return results
    
    def evaluate_reasoning_quality(
        self,
        reasoning_result: Dict[str, any],
    ) -> Dict[str, any]:
        """
        Evaluate the quality of reasoning.
        
        Args:
            reasoning_result: Result from reasoning
        
        Returns:
            Dictionary with quality metrics
        """
        reasoning_steps = reasoning_result["reasoning_steps"]
        
        # Basic quality metrics
        metrics = {
            "num_steps": len(reasoning_steps),
            "avg_step_length": sum(len(step) for step in reasoning_steps) / len(reasoning_steps) if reasoning_steps else 0,
            "has_intermediate_steps": len(reasoning_steps) > 1,
            "step_diversity": len(set(reasoning_steps)) / len(reasoning_steps) if reasoning_steps else 0,
        }
        
        # Check for step progression
        if len(reasoning_steps) > 1:
            # Look for progression indicators
            progression_indicators = ["then", "next", "after", "following", "so", "therefore"]
            has_progression = any(
                any(indicator in step.lower() for indicator in progression_indicators)
                for step in reasoning_steps
            )
            metrics["has_progression"] = has_progression
        
        return metrics
    
    def format_reasoning_output(
        self,
        reasoning_result: Dict[str, any],
    ) -> str:
        """
        Format reasoning result for display.
        
        Args:
            reasoning_result: Result from reasoning
        
        Returns:
            Formatted string
        """
        output = f"Question: {reasoning_result['question']}\n\n"
        output += "Reasoning:\n"
        
        for i, step in enumerate(reasoning_result["reasoning_steps"], 1):
            output += f"{i}. {step}\n"
        
        output += f"\nAnswer: {reasoning_result['answer']}"
        
        return output
    
    def enhance_query(self, query: str) -> str:
        """
        Enhanced query processing with task-specific optimizations.
        
        Args:
            query: Original query
        
        Returns:
            Enhanced query with appropriate prompts
        """
        query_lower = query.lower()
        
        # Creative task enhancement
        creative_keywords = ['metaphor', 'analogy', 'creative', 'story', 'imagine', 'design', 'create']
        if any(kw in query_lower for kw in creative_keywords):
            return f"Think creatively and use vivid imagery. Provide a thoughtful and imaginative response.\n\n{query}"
        
        # Mathematical enhancement
        math_keywords = ['calculate', 'solve', 'compute', 'average', 'sum', 'multiply', 'divide']
        if any(kw in query_lower for kw in math_keywords):
            return f"Show your work step by step. Be precise with calculations.\n\n{query}"
        
        # Logical reasoning enhancement
        logic_keywords = ['if', 'implies', 'therefore', 'conclude', 'deduce', 'logic', 'reasoning']
        if any(kw in query_lower for kw in logic_keywords):
            return f"Think step by step through the logical implications. Be careful with assumptions.\n\n{query}"
        
        # Default Chain-of-Thought
        return f"Let's think step by step to solve this.\n\n{query}"