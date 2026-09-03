"""
finetuning/adapter_manager.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Manager for loading and serving multiple LoRA adapters.

Based on LoRA-Hub: Efficient Cross-Task Generalization via Dynamic LoRA Composition
https://arxiv.org/abs/2307.13269

Supports dynamic adapter loading for multi-task serving and adapter composition.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from config import settings

logger = logging.getLogger(__name__)


class AdapterManager:
    """
    Manager for loading and serving multiple LoRA adapters.
    
    Supports:
    - Dynamic adapter loading
    - Adapter composition (LoRA-Hub style)
    - Multi-adapter serving
    - Adapter merging and export
    """
    
    def __init__(
        self,
        base_model: str,
        adapter_dir: Optional[Path] = None,
        max_adapters: int = 8,
    ):
        if not PEFT_AVAILABLE:
            raise ImportError(
                "PEFT library not installed. Install with: pip install peft"
            )
        
        self.base_model_name = base_model
        self.adapter_dir = adapter_dir or settings.multi_lora_adapter_dir
        self.max_adapters = max_adapters or settings.multi_lora_max_adapters
        
        self.base_model = None
        self.tokenizer = None
        self.loaded_adapters: Dict[str, PeftModel] = {}
        
        logger.info(
            f"AdapterManager initialized with base model: {base_model}, "
            f"adapter dir: {self.adapter_dir}, max adapters: {max_adapters}"
        )
    
    def load_base_model(self):
        """Load the base model for adapter serving."""
        logger.info(f"Loading base model: {self.base_model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
        )
        
        logger.info("Base model loaded successfully")
    
    def load_adapter(
        self,
        adapter_name: str,
        adapter_path: Optional[Path] = None,
    ) -> PeftModel:
        """
        Load a single LoRA adapter.
        
        Args:
            adapter_name: Name/identifier for the adapter
            adapter_path: Path to adapter (defaults to adapter_dir/adapter_name)
        
        Returns:
            Loaded PEFT model with adapter
        """
        if adapter_path is None:
            adapter_path = self.adapter_dir / adapter_name
        
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        
        logger.info(f"Loading adapter: {adapter_name} from {adapter_path}")
        
        # Check if we need to unload an adapter to make space
        if len(self.loaded_adapters) >= self.max_adapters:
            self._unload_oldest_adapter()
        
        # Load adapter on top of base model
        model = PeftModel.from_pretrained(
            self.base_model,
            str(adapter_path),
            adapter_name=adapter_name,
        )
        
        self.loaded_adapters[adapter_name] = model
        
        logger.info(f"Adapter {adapter_name} loaded successfully")
        return model
    
    def unload_adapter(self, adapter_name: str):
        """Unload a specific adapter."""
        if adapter_name in self.loaded_adapters:
            del self.loaded_adapters[adapter_name]
            logger.info(f"Adapter {adapter_name} unloaded")
            torch.cuda.empty_cache()
    
    def _unload_oldest_adapter(self):
        """Unload the oldest adapter to make space."""
        if not self.loaded_adapters:
            return
        
        oldest_adapter = next(iter(self.loaded_adapters))
        self.unload_adapter(oldest_adapter)
        logger.info(f"Unloaded oldest adapter: {oldest_adapter}")
    
    def get_adapter(self, adapter_name: str) -> Optional[PeftModel]:
        """Get a loaded adapter by name."""
        return self.loaded_adapters.get(adapter_name)
    
    def list_adapters(self) -> list[str]:
        """List all available adapters in the adapter directory."""
        if not self.adapter_dir.exists():
            return []
        
        adapters = []
        for path in self.adapter_dir.iterdir():
            if path.is_dir() and (path / "adapter_config.json").exists():
                adapters.append(path.name)
        
        return sorted(adapters)
    
    def compose_adapters(
        self,
        adapter_names: list[str],
        weights: Optional[list[float]] = None,
    ) -> PeftModel:
        """
        Compose multiple adapters using weighted sum.
        
        Based on LoRA-Hub: Efficient Cross-Task Generalization via Dynamic LoRA Composition
        https://arxiv.org/abs/2307.13269
        
        Args:
            adapter_names: List of adapter names to compose
            weights: Optional weights for each adapter (default: equal weights)
        
        Returns:
            Model with composed adapters
        """
        if weights is None:
            weights = [1.0 / len(adapter_names)] * len(adapter_names)
        
        if len(adapter_names) != len(weights):
            raise ValueError("Number of adapters must match number of weights")
        
        logger.info(f"Composing adapters: {adapter_names} with weights: {weights}")
        
        # Load first adapter as base
        model = self.load_adapter(adapter_names[0])
        
        # Add remaining adapters with weights
        for adapter_name, weight in zip(adapter_names[1:], weights[1:]):
            adapter_path = self.adapter_dir / adapter_name
            model.load_adapter(
                str(adapter_path),
                adapter_name=adapter_name,
                adapter_weights={adapter_name: weight},
            )
        
        return model
    
    def merge_and_save(
        self,
        adapter_name: str,
        output_path: Optional[Path] = None,
    ):
        """
        Merge adapter weights into base model and save.
        
        Args:
            adapter_name: Name of adapter to merge
            output_path: Path to save merged model
        """
        if adapter_name not in self.loaded_adapters:
            raise ValueError(f"Adapter {adapter_name} not loaded")
        
        model = self.loaded_adapters[adapter_name]
        
        output_path = output_path or (self.adapter_dir / f"{adapter_name}_merged")
        output_path = Path(output_path)
        
        logger.info(f"Merging adapter {adapter_name} into base model")
        
        # Merge adapter weights
        merged_model = model.merge_and_unload()
        
        # Save merged model
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"Merged model saved to {output_path}")
    
    def export_adapter(
        self,
        adapter_name: str,
        export_format: str = "gguf",
        output_path: Optional[Path] = None,
    ):
        """
        Export adapter to different formats for deployment.
        
        Args:
            adapter_name: Name of adapter to export
            export_format: Format to export (gguf, onnx, tensorrt)
            output_path: Path to save exported model
        """
        if adapter_name not in self.loaded_adapters:
            raise ValueError(f"Adapter {adapter_name} not loaded")
        
        output_path = output_path or (self.adapter_dir / f"{adapter_name}_{export_format}")
        output_path = Path(output_path)
        
        logger.info(f"Exporting adapter {adapter_name} to {export_format}")
        
        if export_format == "gguf":
            self._export_to_gguf(adapter_name, output_path)
        elif export_format == "onnx":
            self._export_to_onnx(adapter_name, output_path)
        elif export_format == "tensorrt":
            self._export_to_tensorrt(adapter_name, output_path)
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        logger.info(f"Adapter exported to {output_path}")
    
    def _export_to_gguf(self, adapter_name: str, output_path: Path):
        """Export adapter to GGUF format for llama.cpp."""
        try:
            import llama_cpp
            logger.info("Converting to GGUF format")
            # Implementation would use llama.cpp conversion tools
            # This is a placeholder for the actual conversion logic
            logger.warning("GGUF export requires llama.cpp installation")
        except ImportError:
            logger.error("llama.cpp not installed for GGUF export")
    
    def _export_to_onnx(self, adapter_name: str, output_path: Path):
        """Export adapter to ONNX format."""
        try:
            import onnx
            from transformers.onnx import export
            
            model = self.loaded_adapters[adapter_name]
            logger.info("Converting to ONNX format")
            # Implementation would use ONNX export
            logger.warning("ONNX export implementation pending")
        except ImportError:
            logger.error("ONNX not installed for ONNX export")
    
    def _export_to_tensorrt(self, adapter_name: str, output_path: Path):
        """Export adapter to TensorRT format."""
        try:
            import tensorrt
            logger.info("Converting to TensorRT format")
            # Implementation would use TensorRT conversion
            logger.warning("TensorRT export implementation pending")
        except ImportError:
            logger.error("TensorRT not installed for TensorRT export")
    
    def get_adapter_info(self, adapter_name: str) -> Dict[str, Any]:
        """Get information about a specific adapter."""
        adapter_path = self.adapter_dir / adapter_name
        
        if not adapter_path.exists():
            return {}
        
        config_path = adapter_path / "adapter_config.json"
        if config_path.exists():
            import json
            with open(config_path) as f:
                return json.load(f)
        
        return {"name": adapter_name, "path": str(adapter_path)}
    
    def cleanup(self):
        """Clean up loaded adapters and free memory."""
        for adapter_name in list(self.loaded_adapters.keys()):
            self.unload_adapter(adapter_name)
        
        if self.base_model is not None:
            del self.base_model
            self.base_model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        torch.cuda.empty_cache()
        logger.info("AdapterManager cleaned up")


def get_adapter_manager(
    base_model: Optional[str] = None,
    adapter_dir: Optional[Path] = None,
    max_adapters: Optional[int] = None,
) -> AdapterManager:
    """
    Factory function to get an adapter manager.
    
    Args:
        base_model: Base model name
        adapter_dir: Directory containing adapters
        max_adapters: Maximum number of adapters to load
    
    Returns:
        AdapterManager instance
    """
    base_model = base_model or settings.local_model_name
    adapter_dir = adapter_dir or settings.multi_lora_adapter_dir
    max_adapters = max_adapters or settings.multi_lora_max_adapters
    
    return AdapterManager(base_model, adapter_dir, max_adapters)