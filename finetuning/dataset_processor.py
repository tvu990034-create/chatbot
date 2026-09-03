"""
finetuning/dataset_processor.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dataset processing for fine-tuning with support for various instruction formats
and math reasoning datasets.

Supported datasets:
- GSM8K: Grade School Math 8K (https://github.com/openai/grade-school-math)
- MATH: Competition mathematics (https://github.com/hendrycks/math)
- MMLU: Massive Multitask Language Understanding (https://github.com/hendrycks/test)
- Custom datasets with various instruction templates
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from config import settings

logger = logging.getLogger(__name__)


class DatasetProcessor:
    """
    Process datasets for fine-tuning with various instruction templates.
    
    Supports multiple instruction formats:
    - Alpaca: Simple instruction-following format
    - Vicuna: Chat-based format
    - ChatML: OpenAI-style chat format
    - Math: Chain-of-thought math reasoning format
    """
    
    # Instruction templates
    TEMPLATES = {
        "alpaca": {
            "prompt": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            ),
            "response_field": "output",
        },
        "vicuna": {
            "prompt": (
                "A chat between a curious user and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the user's questions.\n\n"
                "USER: {instruction}\nASSISTANT:"
            ),
            "response_field": "output",
        },
        "chatml": {
            "prompt": (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n{instruction}<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "response_field": "output",
        },
        "math": {
            "prompt": (
                "Solve the following math problem step by step:\n\n"
                "Problem: {instruction}\n\n"
                "Solution:"
            ),
            "response_field": "solution",
        },
        "gsm8k": {
            "prompt": "Question: {question}\nAnswer:",
            "response_field": "answer",
        },
    }
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        template: str = "alpaca",
        max_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.template_name = template
        self.max_length = max_length
        self.template = self.TEMPLATES.get(template, self.TEMPLATES["alpaca"])
    
    def load_dataset(
        self,
        dataset_name: str,
        split: str = "train",
        max_samples: int = -1,
    ) -> Dataset:
        """
        Load a dataset from HuggingFace or local path.
        
        Args:
            dataset_name: Name of the dataset or local path
            split: Dataset split to load
            max_samples: Maximum number of samples (-1 for all)
        
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading dataset: {dataset_name} (split: {split})")
        
        # Check if it's a local path
        if Path(dataset_name).exists():
            dataset = load_dataset("json", data_files=dataset_name, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        
        # Limit samples if specified
        if max_samples > 0 and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
            logger.info(f"Limited dataset to {max_samples} samples")
        
        logger.info(f"Loaded {len(dataset)} samples")
        return dataset
    
    def format_instruction(self, example: dict) -> str:
        """Format a single example according to the template."""
        # Handle different dataset structures
        if self.template_name == "gsm8k":
            return self.template["prompt"].format(
                question=example.get("question", example.get("instruction", ""))
            )
        elif self.template_name == "math":
            return self.template["prompt"].format(
                instruction=example.get("problem", example.get("instruction", ""))
            )
        else:
            return self.template["prompt"].format(
                instruction=example.get("instruction", example.get("input", ""))
            )
    
    def format_response(self, example: dict) -> str:
        """Extract the response from an example."""
        response_field = self.template["response_field"]
        
        # Try multiple common field names
        for field in [response_field, "output", "response", "answer", "text"]:
            if field in example:
                return str(example[field])
        
        # Fallback to first non-instruction field
        for key, value in example.items():
            if key not in ["instruction", "input", "question", "problem"]:
                return str(value)
        
        return ""
    
    def preprocess_function(self, examples: dict) -> dict:
        """
        Preprocess dataset examples for training.
        
        Args:
            examples: Batch of examples from dataset
        
        Returns:
            Tokenized inputs
        """
        # Format instructions and responses
        instructions = [self.format_instruction(ex) for ex in examples]
        responses = [self.format_response(ex) for ex in examples]
        
        # Combine for training
        texts = [
            inst + resp if not inst.endswith((":", "\n")) else inst + " " + resp
            for inst, resp in zip(instructions, responses)
        ]
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        return tokenized
    
    def process_dataset(
        self,
        dataset: Dataset,
        remove_columns: Optional[list[str]] = None,
    ) -> Dataset:
        """
        Process dataset for training.
        
        Args:
            dataset: Raw dataset
            remove_columns: Columns to remove after processing
        
        Returns:
            Processed dataset
        """
        logger.info("Processing dataset...")
        
        # Remove columns if specified
        if remove_columns:
            columns_to_remove = [col for col in remove_columns if col in dataset.column_names]
            if columns_to_remove:
                dataset = dataset.remove_columns(columns_to_remove)
        
        # Apply preprocessing
        processed_dataset = dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Processing dataset",
        )
        
        logger.info(f"Processed dataset: {len(processed_dataset)} samples")
        return processed_dataset
    
    def load_math_dataset(
        self,
        dataset_name: str = "gsm8k",
        split: str = "train",
        max_samples: int = -1,
    ) -> Dataset:
        """
        Load a math reasoning dataset with special formatting.
        
        Args:
            dataset_name: Name of math dataset (gsm8k, math, etc.)
            split: Dataset split
            max_samples: Maximum samples to load
        
        Returns:
            Processed math dataset
        """
        logger.info(f"Loading math dataset: {dataset_name}")
        
        # Map common dataset names to HuggingFace paths
        dataset_paths = {
            "gsm8k": "gsm8k",
            "math": "hendrycks/math",
            "mmlu": "causalLM/mmlu",
            "gpqa": "Idavidrein/gpqa",
        }
        
        hf_path = dataset_paths.get(dataset_name.lower(), dataset_name)
        dataset = self.load_dataset(hf_path, split, max_samples)
        
        # Format for math reasoning
        if dataset_name.lower() == "gsm8k":
            # GSM8K special formatting
            def format_gsm8k(example):
                question = example.get("question", "")
                answer = example.get("answer", "")
                return {
                    "instruction": f"Solve this math problem: {question}",
                    "output": f"Step-by-step solution: {answer}",
                }
            
            dataset = dataset.map(format_gsm8k)
        
        elif dataset_name.lower() == "math":
            # MATH dataset formatting
            def format_math(example):
                problem = example.get("problem", "")
                solution = example.get("solution", "")
                return {
                    "instruction": f"Solve this competition math problem: {problem}",
                    "output": f"Solution: {solution}",
                }
            
            dataset = dataset.map(format_math)
        
        return dataset
    
    def load_custom_dataset(
        self,
        data_path: str,
        data_format: str = "json",
        split: str = "train",
        max_samples: int = -1,
    ) -> Dataset:
        """
        Load a custom dataset from file.
        
        Args:
            data_path: Path to data file
            data_format: Format of data (json, csv, parquet)
            split: Dataset split
            max_samples: Maximum samples to load
        
        Returns:
            Loaded dataset
        """
        logger.info(f"Loading custom dataset from {data_path}")
        
        if data_format == "json":
            dataset = load_dataset("json", data_files=data_path, split=split)
        elif data_format == "csv":
            dataset = load_dataset("csv", data_files=data_path, split=split)
        elif data_format == "parquet":
            dataset = load_dataset("parquet", data_files=data_path, split=split)
        else:
            raise ValueError(f"Unsupported data format: {data_format}")
        
        # Limit samples if specified
        if max_samples > 0 and len(dataset) > max_samples:
            dataset = dataset.select(range(max_samples))
        
        logger.info(f"Loaded {len(dataset)} samples from custom dataset")
        return dataset


def get_dataset_processor(
    tokenizer: PreTrainedTokenizer,
    template: Optional[str] = None,
    max_length: Optional[int] = None,
) -> DatasetProcessor:
    """
    Factory function to get a dataset processor.
    
    Args:
        tokenizer: Tokenizer to use
        template: Instruction template name
        max_length: Maximum sequence length
    
    Returns:
        DatasetProcessor instance
    """
    template = template or settings.finetuning_template
    max_length = max_length or settings.finetuning_max_seq_length
    
    return DatasetProcessor(tokenizer, template, max_length)