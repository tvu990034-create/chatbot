"""
finetuning/lora_trainer.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
LoRA/QLoRA fine-tuning implementation based on research papers:

LoRA: Low-Rank Adaptation of Large Language Models (https://arxiv.org/abs/2106.09685)
QLoRA: Efficient Finetuning of Quantized LLMs (https://arxiv.org/abs/2305.14314)
AdaLoRA: Adaptive Budget Allocation (https://arxiv.org/abs/2303.10512)
DoRA: Weight-Decomposed Low-Rank Adaptation (https://arxiv.org/abs/2402.09353)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

try:
    from peft import (
        AdaLoraConfig,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from config import settings

logger = logging.getLogger(__name__)


class LoRATrainer:
    """
    LoRA fine-tuning trainer implementing the original LoRA paper.
    
    Based on: LoRA: Low-Rank Adaptation of Large Language Models
    https://arxiv.org/abs/2106.09685
    
    Mathematical formulation:
    W = W₀ + ΔW = W₀ + BA
    where B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, and r << min(d,k)
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: Optional[Path] = None,
        config: Optional[dict] = None,
    ):
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library not installed. Install with: pip install peft"
            )
        
        self.model_name = model_name
        self.output_dir = output_dir or settings.finetuning_output_dir
        self.config = config or self._get_default_config()
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def _get_default_config(self) -> dict:
        """Get default LoRA configuration from settings."""
        return {
            "r": settings.lora_r,
            "lora_alpha": settings.lora_alpha,
            "lora_dropout": settings.lora_dropout,
            "target_modules": settings.lora_target_modules,
            "bias": settings.lora_bias,
            "task_type": settings.lora_task_type,
        }
    
    def load_model_and_tokenizer(self):
        """Load base model and tokenizer for fine-tuning."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=settings.finetuning_cache_dir,
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=settings.finetuning_cache_dir,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        )
        
        logger.info(f"Model loaded with {self.model.num_parameters()} parameters")
    
    def apply_lora(self):
        """Apply LoRA adapters to the model."""
        logger.info("Applying LoRA adapters...")
        
        lora_config = LoraConfig(
            r=self.config["r"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            target_modules=self.config["target_modules"],
            bias=self.config["bias"],
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {total_params:,} || "
            f"Trainable%: {100 * trainable_params / total_params:.2f}%"
        )
    
    def prepare_trainer(
        self,
        train_dataset,
        eval_dataset=None,
        callbacks=None,
    ):
        """Prepare the Trainer with training arguments."""
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            learning_rate=settings.finetuning_learning_rate,
            per_device_train_batch_size=settings.finetuning_batch_size,
            gradient_accumulation_steps=settings.finetuning_gradient_accumulation_steps,
            num_train_epochs=settings.finetuning_num_epochs,
            warmup_steps=settings.finetuning_warmup_steps,
            lr_scheduler_type=settings.finetuning_lr_scheduler_type,
            weight_decay=settings.finetuning_weight_decay,
            max_grad_norm=settings.finetuning_max_grad_norm,
            logging_dir=str(settings.finetuning_logging_dir),
            logging_steps=10,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=settings.finetuning_eval_steps if eval_dataset else None,
            save_steps=settings.finetuning_save_steps,
            save_total_limit=settings.finetuning_save_total_limit,
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            gradient_checkpointing=True,
            optim="adamw_torch",
            ddp_find_unused_parameters=False,
            report_to=["tensorboard"],
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks or [],
        )
        
        logger.info("Trainer prepared")
    
    def train(self):
        """Start the fine-tuning process."""
        if self.trainer is None:
            raise ValueError("Trainer not prepared. Call prepare_trainer() first.")
        
        logger.info("Starting fine-tuning...")
        self.trainer.train()
        logger.info("Fine-tuning completed")
    
    def save_model(self, output_path: Optional[Path] = None):
        """Save the fine-tuned model and adapters."""
        save_path = output_path or self.output_dir
        save_path = Path(save_path)
        
        logger.info(f"Saving model to {save_path}")
        
        # Save LoRA adapters
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved successfully to {save_path}")


class QLoRATrainer(LoRATrainer):
    """
    QLoRA fine-tuning trainer implementing 4-bit quantization.
    
    Based on: QLoRA: Efficient Finetuning of Quantized LLMs
    https://arxiv.org/abs/2305.14314
    
    Key innovations:
    - 4-bit NormalFloat (NF4) quantization
    - Double quantization for additional memory savings
    - Paged optimizers for memory management
    """
    
    def _get_default_config(self) -> dict:
        """Get default QLoRA configuration."""
        config = super()._get_default_config()
        config.update({
            "bits": settings.qlora_bits,
            "quantization_type": settings.qlora_quantization_type,
            "double_quant": settings.qlora_double_quant,
            "compute_dtype": settings.qlora_compute_dtype,
            "use_gradient_checkpointing": settings.qlora_use_gradient_checkpointing,
        })
        return config
    
    def load_model_and_tokenizer(self):
        """Load quantized model and tokenizer."""
        logger.info(f"Loading quantized model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=settings.finetuning_cache_dir,
        )
        
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Configure 4-bit quantization
        compute_dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        compute_dtype = compute_dtype_map.get(
            self.config["compute_dtype"], torch.bfloat16
        )
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.config["bits"] == 4,
            load_in_8bit=self.config["bits"] == 8,
            bnb_4bit_quant_type=self.config["quantization_type"],
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.config["double_quant"],
        )
        
        # Load quantized model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            cache_dir=settings.finetuning_cache_dir,
            device_map="auto",
        )
        
        # Prepare model for k-bit training
        if self.config["use_gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable()
        
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info(f"Quantized model loaded with {self.model.num_parameters()} parameters")


class AdaLoRATrainer(LoRATrainer):
    """
    AdaLoRA fine-tuning trainer with adaptive rank allocation.
    
    Based on: AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
    https://arxiv.org/abs/2303.10512
    
    Uses SVD-based dynamic rank assignment to optimize the budget allocation.
    """
    
    def _get_default_config(self) -> dict:
        """Get default AdaLoRA configuration."""
        config = super()._get_default_config()
        config.update({
            "init_r": settings.adalora_init_r,
            "target_r": settings.adalora_target_r,
            "tota_step": settings.adalora_tota_step,
            "delta_t": settings.adalora_delta_t,
        })
        return config
    
    def apply_lora(self):
        """Apply AdaLoRA adapters to the model."""
        logger.info("Applying AdaLoRA adapters...")
        
        adalora_config = AdaLoraConfig(
            init_r=self.config["init_r"],
            target_r=self.config["target_r"],
            beta1=0.85,
            beta2=0.85,
            tinit=self.config["tota_step"] // 10,
            tfinal=self.config["tota_step"],
            deltaT=self.config["delta_t"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            target_modules=self.config["target_modules"],
            orth_reg_weight=0.5,
        )
        
        self.model = get_peft_model(self.model, adalora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"AdaLoRA - Trainable params: {trainable_params:,} || "
            f"All params: {total_params:,} || "
            f"Trainable%: {100 * trainable_params / total_params:.2f}%"
        )


class DoRATrainer(LoRATrainer):
    """
    DoRA fine-tuning trainer with weight-decomposed low-rank adaptation.
    
    Based on: DoRA: Weight-Decomposed Low-Rank Adaptation
    https://arxiv.org/abs/2402.09353
    
    Decomposes weight into magnitude and direction components for better 
    approximation of full fine-tuning.
    """
    
    def _get_default_config(self) -> dict:
        """Get default DoRA configuration."""
        config = super()._get_default_config()
        config.update({
            "use_dora": settings.dora_weight_decompose,
        })
        return config
    
    def apply_lora(self):
        """Apply DoRA adapters to the model."""
        logger.info("Applying DoRA adapters...")
        
        lora_config = LoraConfig(
            r=self.config["r"],
            lora_alpha=self.config["lora_alpha"],
            lora_dropout=self.config["lora_dropout"],
            target_modules=self.config["target_modules"],
            bias=self.config["bias"],
            task_type=TaskType.CAUSAL_LM,
            use_dora=self.config["use_dora"],
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"DoRA - Trainable params: {trainable_params:,} || "
            f"All params: {total_params:,} || "
            f"Trainable%: {100 * trainable_params / total_params:.2f}%"
        )


def get_trainer(
    method: str = "lora",
    model_name: str = None,
    output_dir: Optional[Path] = None,
    config: Optional[dict] = None,
) -> LoRATrainer:
    """
    Factory function to get the appropriate trainer based on method.
    
    Args:
        method: Fine-tuning method (lora, qlora, ada_lora, dora)
        model_name: Base model name/path
        output_dir: Output directory for adapters
        config: Optional configuration overrides
    
    Returns:
        Appropriate trainer instance
    """
    model_name = model_name or settings.local_model_name
    
    trainers = {
        "lora": LoRATrainer,
        "qlora": QLoRATrainer,
        "ada_lora": AdaLoRATrainer,
        "dora": DoRATrainer,
    }
    
    trainer_class = trainers.get(method.lower())
    if trainer_class is None:
        raise ValueError(f"Unknown fine-tuning method: {method}")
    
    return trainer_class(model_name, output_dir, config)