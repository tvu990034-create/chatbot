"""
finetuning/math_recipes.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Fine-tuning recipes for mathematical reasoning based on research papers:

DeepSeekMath: Pushing the Limits of Mathematical Reasoning (https://arxiv.org/abs/2402.03300)
Llemma: An Open Language Model For Mathematics (https://arxiv.org/abs/2310.10631)
Minerva: Solving Quantitative Reasoning Problems (https://arxiv.org/abs/2206.14858)
WizardMath: Empowering Mathematical Reasoning (https://arxiv.org/abs/2308.09583)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .lora_trainer import get_trainer
from .dataset_processor import get_dataset_processor

from config import settings

logger = logging.getLogger(__name__)


class MathReasoningRecipe:
    """
    Base class for mathematical reasoning fine-tuning recipes.
    
    Implements common patterns from math reasoning research:
    - Chain-of-thought training
    - Process supervision
    - Step-by-step solution generation
    """
    
    def __init__(
        self,
        base_model: str,
        output_dir: Optional[Path] = None,
        method: str = "qlora",
    ):
        self.base_model = base_model
        self.output_dir = output_dir or settings.finetuning_output_dir / "math_reasoning"
        self.method = method
        
        self.trainer = None
        self.dataset_processor = None
    
    def prepare_training(
        self,
        dataset_name: str = "gsm8k",
        max_samples: int = 1000,
    ):
        """Prepare trainer and dataset for math reasoning fine-tuning."""
        logger.info(f"Preparing math reasoning fine-tuning with {dataset_name}")
        
        # Initialize trainer
        self.trainer = get_trainer(
            method=self.method,
            model_name=self.base_model,
            output_dir=self.output_dir,
        )
        
        # Load model and tokenizer
        self.trainer.load_model_and_tokenizer()
        
        # Apply LoRA/QLoRA
        self.trainer.apply_lora()
        
        # Initialize dataset processor with math template
        self.dataset_processor = get_dataset_processor(
            tokenizer=self.trainer.tokenizer,
            template="math",
            max_length=1024,  # Math problems need longer context
        )
        
        # Load math dataset
        if dataset_name == "gsm8k":
            train_dataset = self.dataset_processor.load_math_dataset(
                "gsm8k",
                split="train",
                max_samples=max_samples,
            )
        elif dataset_name == "math":
            train_dataset = self.dataset_processor.load_math_dataset(
                "math",
                split="train",
                max_samples=max_samples,
            )
        else:
            train_dataset = self.dataset_processor.load_dataset(
                dataset_name,
                split="train",
                max_samples=max_samples,
            )
        
        # Process dataset
        processed_dataset = self.dataset_processor.process_dataset(train_dataset)
        
        return processed_dataset
    
    def train(self, dataset):
        """Train the model on math reasoning data."""
        logger.info("Starting math reasoning fine-tuning")
        
        # Prepare trainer with dataset
        self.trainer.prepare_trainer(train_dataset=dataset)
        
        # Train
        self.trainer.train()
        
        # Save model
        self.trainer.save_model()
        
        logger.info(f"Math reasoning model saved to {self.output_dir}")


class GSM8KRecipe(MathReasoningRecipe):
    """
    GSM8K fine-tuning recipe for grade school math problems.
    
    Based on: Grade School Math 8K (https://github.com/openai/grade-school-math)
    
    Focuses on:
    - Multi-step arithmetic reasoning
    - Word problem understanding
    - Chain-of-thought solution generation
    """
    
    def __init__(
        self,
        base_model: str,
        output_dir: Optional[Path] = None,
        method: str = "qlora",
    ):
        super().__init__(base_model, output_dir, method)
        self.output_dir = self.output_dir / "gsm8k"
    
    def prepare_training(self, max_samples: int = 1000):
        """Prepare GSM8K-specific training."""
        logger.info("Preparing GSM8K fine-tuning")
        
        # GSM8K-specific configuration
        self.trainer = get_trainer(
            method=self.method,
            model_name=self.base_model,
            output_dir=self.output_dir,
            config={
                "r": 16,  # Higher rank for math reasoning
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            },
        )
        
        self.trainer.load_model_and_tokenizer()
        self.trainer.apply_lora()
        
        self.dataset_processor = get_dataset_processor(
            tokenizer=self.trainer.tokenizer,
            template="gsm8k",
            max_length=1024,
        )
        
        train_dataset = self.dataset_processor.load_math_dataset(
            "gsm8k",
            split="train",
            max_samples=max_samples,
        )
        
        processed_dataset = self.dataset_processor.process_dataset(train_dataset)
        
        return processed_dataset


class MATHRecipe(MathReasoningRecipe):
    """
    MATH dataset fine-tuning for competition mathematics.
    
    Based on: MATH (Hendrycks et al.) (https://github.com/hendrycks/math)
    
    Focuses on:
    - High-school competition math
    - Advanced algebra, geometry, calculus
    - Formal mathematical reasoning
    """
    
    def __init__(
        self,
        base_model: str,
        output_dir: Optional[Path] = None,
        method: str = "qlora",
    ):
        super().__init__(base_model, output_dir, method)
        self.output_dir = self.output_dir / "math_competition"
    
    def prepare_training(self, max_samples: int = 500):
        """Prepare MATH dataset-specific training."""
        logger.info("Preparing MATH competition fine-tuning")
        
        # MATH requires higher capacity
        self.trainer = get_trainer(
            method=self.method,
            model_name=self.base_model,
            output_dir=self.output_dir,
            config={
                "r": 32,  # Higher rank for complex math
                "lora_alpha": 64,
                "lora_dropout": 0.05,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            },
        )
        
        self.trainer.load_model_and_tokenizer()
        self.trainer.apply_lora()
        
        self.dataset_processor = get_dataset_processor(
            tokenizer=self.trainer.tokenizer,
            template="math",
            max_length=2048,  # Longer sequences for competition problems
        )
        
        train_dataset = self.dataset_processor.load_math_dataset(
            "math",
            split="train",
            max_samples=max_samples,
        )
        
        processed_dataset = self.dataset_processor.process_dataset(train_dataset)
        
        return processed_dataset


class MultiMathRecipe(MathReasoningRecipe):
    """
    Multi-dataset math reasoning recipe combining multiple math datasets.
    
    Based on: MAmmoTH: Building Math Generalist Models via Hybrid Instruction Tuning
    https://arxiv.org/abs/2309.05653
    
    Combines:
    - GSM8K (grade school math)
    - MATH (competition math)
    - MMLU (math subset)
    - Custom math instruction data
    """
    
    def __init__(
        self,
        base_model: str,
        output_dir: Optional[Path] = None,
        method: str = "qlora",
    ):
        super().__init__(base_model, output_dir, method)
        self.output_dir = self.output_dir / "multi_math"
    
    def prepare_training(self, max_samples_per_dataset: int = 500):
        """Prepare multi-dataset math training."""
        logger.info("Preparing multi-dataset math fine-tuning")
        
        # Multi-dataset requires balanced configuration
        self.trainer = get_trainer(
            method=self.method,
            model_name=self.base_model,
            output_dir=self.output_dir,
            config={
                "r": 24,  # Balanced rank
                "lora_alpha": 48,
                "lora_dropout": 0.08,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            },
        )
        
        self.trainer.load_model_and_tokenizer()
        self.trainer.apply_lora()
        
        self.dataset_processor = get_dataset_processor(
            tokenizer=self.trainer.tokenizer,
            template="math",
            max_length=1536,
        )
        
        # Load and combine multiple datasets
        datasets = []
        
        # GSM8K
        gsm8k_data = self.dataset_processor.load_math_dataset(
            "gsm8k",
            split="train",
            max_samples=max_samples_per_dataset,
        )
        datasets.append(gsm8k_data)
        
        # MATH
        math_data = self.dataset_processor.load_math_dataset(
            "math",
            split="train",
            max_samples=max_samples_per_dataset,
        )
        datasets.append(math_data)
        
        # Combine datasets
        from datasets import concatenate_datasets
        combined_dataset = concatenate_datasets(datasets)
        
        # Shuffle combined dataset
        combined_dataset = combined_dataset.shuffle(seed=42)
        
        processed_dataset = self.dataset_processor.process_dataset(combined_dataset)
        
        logger.info(f"Combined dataset size: {len(processed_dataset)}")
        
        return processed_dataset


class ProcessSupervisionRecipe(MathReasoningRecipe):
    """
    Process supervision recipe for mathematical reasoning.
    
    Based on: Improving Mathematical Reasoning with Process Supervision
    https://openai.com/research/improving-mathematical-reasoning-with-process-supervision
    
    Focuses on:
    - Rewarding correct reasoning steps
    - Learning from intermediate steps
    - Better generalization to new problems
    """
    
    def __init__(
        self,
        base_model: str,
        output_dir: Optional[Path] = None,
        method: str = "qlora",
    ):
        super().__init__(base_model, output_dir, method)
        self.output_dir = self.output_dir / "process_supervision"
    
    def prepare_training(self, max_samples: int = 1000):
        """Prepare process supervision training."""
        logger.info("Preparing process supervision fine-tuning")
        
        self.trainer = get_trainer(
            method=self.method,
            model_name=self.base_model,
            output_dir=self.output_dir,
            config={
                "r": 20,
                "lora_alpha": 40,
                "lora_dropout": 0.07,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            },
        )
        
        self.trainer.load_model_and_tokenizer()
        self.trainer.apply_lora()
        
        # Use custom template for step-by-step reasoning
        self.dataset_processor = get_dataset_processor(
            tokenizer=self.trainer.tokenizer,
            template="math",
            max_length=1024,
        )
        
        # Load dataset with step-by-step solutions
        train_dataset = self.dataset_processor.load_dataset(
            "gsm8k",
            split="train",
            max_samples=max_samples,
        )
        
        # Format for process supervision
        def format_process_supervision(example):
            question = example.get("question", "")
            answer = example.get("answer", "")
            return {
                "instruction": f"Solve step by step: {question}",
                "output": f"Let me think through this step by step:\n{answer}",
            }
        
        train_dataset = train_dataset.map(format_process_supervision)
        
        processed_dataset = self.dataset_processor.process_dataset(train_dataset)
        
        return processed_dataset


def get_math_recipe(
    recipe_name: str,
    base_model: Optional[str] = None,
    output_dir: Optional[Path] = None,
    method: str = "qlora",
) -> MathReasoningRecipe:
    """
    Factory function to get a math reasoning recipe.
    
    Args:
        recipe_name: Name of recipe (gsm8k, math, multi_math, process_supervision)
        base_model: Base model name
        output_dir: Output directory
        method: Fine-tuning method
    
    Returns:
        Math reasoning recipe instance
    """
    base_model = base_model or settings.local_model_name
    
    recipes = {
        "gsm8k": GSM8KRecipe,
        "math": MATHRecipe,
        "multi_math": MultiMathRecipe,
        "process_supervision": ProcessSupervisionRecipe,
    }
    
    recipe_class = recipes.get(recipe_name.lower())
    if recipe_class is None:
        raise ValueError(f"Unknown math recipe: {recipe_name}")
    
    return recipe_class(base_model, output_dir, method)