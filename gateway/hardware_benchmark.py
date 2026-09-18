"""Lightweight hardware benchmark + local model recommendations.

Detects CPU/RAM (psutil) and GPU (torch.cuda, falling back to an estimated
lookup when a GPU is simulated) and recommends local models that fit the
machine from a small built-in catalog.

The CLI expects:

    benchmark = get_hardware_benchmark()
    benchmark.get_hardware_info()          -> {"available", "cpu", "ram_bytes", "gpus": [...]}
    benchmark.get_cpu_recommendations(top_n)
    benchmark.get_gpu_recommendations(gpu_name, top_n, speed_filter)
"""
from __future__ import annotations

import logging
import platform
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# (model_id, params_billions, quant, quality_score 0-10)
_MODEL_CATALOG: List[tuple] = [
    ("tinyllama",   1.1,  "Q4", 3.5),
    ("qwen2.5:0.5b", 0.5, "Q4", 4.0),
    ("gemma2:2b",   2.6,  "Q4", 6.0),
    ("llama3.2:3b", 3.2,  "Q4", 6.2),
    ("phi3:mini",   3.8,  "Q4", 6.5),
    ("qwen3:4b",    4.0,  "Q4", 6.8),
    ("qwen2.5:7b",  7.6,  "Q4", 7.5),
    ("llama3.1:8b", 8.0,  "Q4", 7.6),
    ("qwen2.5:14b", 14.8, "Q4", 8.2),
]

_BYTES_PER_PARAM = {"Q4": 0.55, "Q5": 0.70, "Q8": 1.00, "FP16": 2.00}

# Best-effort VRAM (bytes) for simulated GPU names.
_GPU_VRAM_GB = {
    "rtx 4090": 24, "rtx 4080": 16, "rtx 4070": 12, "rtx 4060": 8,
    "rtx 3090": 24, "rtx 3080": 10, "rtx 3070": 8, "rtx 3060": 12,
    "a100": 40, "h100": 80, "v100": 16, "t4": 16,
}
_DEFAULT_SIM_VRAM_GB = 8.0


class HardwareBenchmark:
    """Detects local hardware and recommends models that fit it."""

    def get_hardware_info(self) -> Dict[str, Any]:
        cpu = platform.processor() or platform.machine() or "Unknown"
        ram_bytes = self._ram_bytes()
        gpus = self._detect_gpus()
        available = bool(ram_bytes or gpus)
        return {
            "available": available,
            "source": "psutil+torch",
            "cpu": cpu,
            "ram_bytes": ram_bytes,
            "gpus": gpus,
        }

    def get_cpu_recommendations(self, top_n: int = 5,
                                speed_filter: str = "any") -> List[Dict[str, Any]]:
        # Reserve ~40% of system RAM for the OS.
        budget = int(self._ram_bytes() * 0.6) or 8 * 1024 ** 3
        recs = self._recommend(budget, device="cpu", speed_filter=speed_filter)
        return recs[:top_n]

    def get_gpu_recommendations(self, gpu_name: Optional[str] = None,
                                top_n: int = 5,
                                speed_filter: str = "any") -> List[Dict[str, Any]]:
        vram = self._vram_for(gpu_name)
        recs = self._recommend(vram, device="gpu", speed_filter=speed_filter)
        return recs[:top_n]

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _ram_bytes() -> int:
        try:
            import psutil
            return int(psutil.virtual_memory().total)
        except Exception:
            return 0

    @staticmethod
    def _detect_gpus() -> List[Dict[str, Any]]:
        try:
            import torch
            if torch.cuda.is_available():
                gpus = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpus.append({
                        "name": props.name,
                        "vram_bytes": int(props.total_memory),
                    })
                return gpus
        except Exception:
            pass
        return []

    def _vram_for(self, gpu_name: Optional[str]) -> int:
        if not gpu_name:
            gpus = self._detect_gpus()
            if gpus:
                return int(gpus[0]["vram_bytes"])
            return 0
        key = gpu_name.strip().lower()
        for name, gb in _GPU_VRAM_GB.items():
            if name in key:
                return int(gb * 1024 ** 3)
        return int(_DEFAULT_SIM_VRAM_GB * 1024 ** 3)

    def _recommend(self, budget_bytes: int, device: str,
                   speed_filter: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for model_id, params_b, quant, quality in _MODEL_CATALOG:
            model_bytes = int(params_b * 1e9 * _BYTES_PER_PARAM.get(quant, 0.55))
            fit = model_bytes <= budget_bytes
            if not fit:
                continue
            base = 260.0 if device == "gpu" else 22.0
            tok = base / max(params_b, 0.5)
            if speed_filter == "fast" and tok < 20:
                continue
            if speed_filter == "usable" and tok < 5:
                continue
            out.append({
                "model_id": model_id,
                "parameter_count": int(params_b * 1e9),
                "quant_type": quant,
                "estimated_tok_per_sec": round(tok, 1),
                "quality_score": quality,
                "fit_type": "full" if fit else "partial",
            })
        out.sort(key=lambda m: (m["quality_score"], m["estimated_tok_per_sec"]),
                 reverse=True)
        return out


_instance: Optional[HardwareBenchmark] = None


def get_hardware_benchmark() -> HardwareBenchmark:
    global _instance
    if _instance is None:
        _instance = HardwareBenchmark()
    return _instance
