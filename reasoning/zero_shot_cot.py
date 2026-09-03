"""
reasoning/zero_shot_cot.py
~~~~~~~~~~~~~~~~~~~~~~~~
Zero-shot Chain-of-Thought implementation based on research paper:

Large Language Models are Zero-Shot Reasoners
https://arxiv.org/abs/2205.11916

Algorithm: Simple prefix "Let's think step by step" enables CoT without examples
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from .chain_of_thought import ChainOfThought

logger = logging.getLogger(__name__)


class ZeroShotCoT(ChainOfThought):
    """
    Zero-shot Chain-of-Thought implementation.
    
    Based on: Large Language Models are Zero-Shot Reasoners
    https://arxiv.org/abs/2205.11916
    
    Key innovations:
    - No few-shot examples needed
    - Simple prefix "Let's think step by step" triggers reasoning
    - Works across different models and tasks
    - Simpler and more general than few-shot CoT
    
    Mathematical formulation:
    Prompt = "Let's think step by step.\n" + Question
    Model generates reasoning automatically
    No training or examples required
    """
    
    def __init__(
        self,
        reasoning_prefix: str = "Let's think step by step.",
        max_reasoning_steps: int = 10,
        enforce_step_format: bool = True,
    ):
        # Zero-shot doesn't use few-shot examples
        super().__init__(use_few_shot=False, max_reasoning_steps=max_reasoning_steps, 
                       enforce_step_format=enforce_step_format)
        
        self.reasoning_prefix = reasoning_prefix
        
        logger.info(
            f"ZeroShotCoT initialized with reasoning_prefix='{reasoning_prefix}'"
        )
    
    def build_cot_prompt(
        self,
        question: str,
        task_type: str = "math",
    ) -> str:
        """
        Build zero-shot Chain-of-Thought prompt.
        
        Args:
            question: The question to answer
            task_type: Type of reasoning task (ignored in zero-shot)
        
        Returns:
            Formatted zero-shot CoT prompt
        """
        # Zero-shot uses only the reasoning prefix
        prompt = f"{self.reasoning_prefix}\n\n"
        prompt += f"Q: {question}\n"
        prompt += "A: "
        
        return prompt
    
    def reason(
        self,
        question: str,
        model_fn: callable,
        task_type: str = "math",
    ) -> Dict[str, any]:
        """
        Perform zero-shot Chain-of-Thought reasoning.
        
        Args:
            question: Question to reason about
            model_fn: Function to call the model
            task_type: Type of reasoning task (ignored in zero-shot)
        
        Returns:
            Dictionary with reasoning results
        """
        # Build zero-shot CoT prompt
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
            "method": "zero_shot_cot",
        }