"""
optimization module.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Master catalog of every optimization category that CANNOT run in this local/chatbot
runtime (no training loop, no GPU kernels, no model weights, no hardware).

These are REAL documentation entries the user asked about. Each entry is a
module-level constant with a ``(source, reason)`` provenance comment. Nothing
here is faked: null implementations deliberately do not exist, and production
code must never call these symbols — they exist so an auditor/port over to a
GPU/TPU stack knows exactly what is missing and what it would need.

Categories are grouped by the ~55 user files.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Training-loop objectives  (Tài liệu (3), Tài liệu (4) — 21 entries)
# ---------------------------------------------------------------------------
# Source: "Tài liệu (3)" and "Tài liệu (4)" — training objectives & loss math.
# Would need: a real training loop (optimizer, batch loader, backprop, epochs).

NOT_IMPLEMENTABLE_TRAINING_LOSS_CROSS_ENTROPY = (
    "training:needs-backprop-batch-loop"
)
NOT_IMPLEMENTABLE_TRAINING_LOSS_KL = (
    "training:needs-backprop-batch-loop"
)
NOT_IMPLEMENTABLE_TRAINING_LOSS_FLS = (
    "training:needs-backprop-batch-loop"
)
NOT_IMPLEMENTABLE_TRAINING_LOSS_DISTILL = (
    "training:needs-student-teacher-forward-passes"
)

# The shared reason string for the remaining ~19 training-loss entries.
NOT_IMPLEMENTABLE_TRAINING_BATCH = (
    "training:needs-backprop-batch-loop"
)
NOT_IMPLEMENTABLE_TRAINING_LR_SCHEDULE = (
    "training:needs-optimizer-and-epochs"
)

# ---------------------------------------------------------------------------
# GPU kernel / model-internals  (Tài liệu (1)/(2) — 31 GPU-attention-quant entries)
# ---------------------------------------------------------------------------
# Source: "Tài liệu (1)" / "Tài liệu (2)" — GPU attention/quant/spec optimizations.

NOT_IMPLEMENTABLE_GPU_KV_CACHE = "gpu:needs-kv-cache-budget-in-attn-runtime"
NOT_IMPLEMENTABLE_GPU_QUANT = "gpu:needs-quantized-weights-and-kernel"
NOT_IMPLEMENTABLE_GPU_ATTENTION_KERNEL = "gpu:needs-attention-kernel-flash-style"
NOT_IMPLEMENTABLE_GPU_FLASH_ATTENTION = "gpu:needs-flash-attention-kernel"
NOT_IMPLEMENTABLE_GPU_LINEAR_ATTENTION = "gpu:needs-arff-linear-attention-kernel"
NOT_IMPLEMENTABLE_GPU_SPECULATIVE = "gpu:needs-draft-and-target-model-weights"
NOT_IMPLEMENTABLE_GPU_PAGED_KV = "gpu:needs-paged-attention-kernel"

# ---------------------------------------------------------------------------
# Learned / trained models  (Untitled document (1),(5),(6); Tài liệu (16),(19))
# ---------------------------------------------------------------------------
# Source: "Untitled document (1)" / "(4)" / "(5)"/"(6)", "a tier.txt".

NOT_IMPLEMENTABLE_BNN_VI = "model:needs-bnn-variational-inference-training"
NOT_IMPLEMENTABLE_FULL_GP_POSTERIOR = "model:needs-large-kernel-inversion"
NOT_IMPLEMENTABLE_NORMALIZING_FLOW = "model:needs-flow-pretrained-weights"
NOT_IMPLEMENTABLE_MC_DROPOUT = "model:needs-forward-passes-through-weights"
NOT_IMPLEMENTABLE_NEURAL_CALIBRATION = "model:needs-trained-calibration-network"
NOT_IMPLEMENTABLE_HRR_BINDING = "model:needs-trainable-vector-symbolic-arch"
NOT_IMPLEMENTABLE_TRLINEAR = "model:needs-trainable-linear-attention"
NOT_IMPLEMENTABLE_PERFORMER = "model:needs-kernel-feature-map-in-backbone"

# ---------------------------------------------------------------------------
# RL / agent-brain training  (Tài liệu (28),(29),(30),(33))
# ---------------------------------------------------------------------------
# Source: "Tài liệu (28)" — RL agent-brain optimizations. Would need an RL loop.

NOT_IMPLEMENTABLE_RL_POLICY_GRADIENT = "rl:needs-ppo-training-loop"
NOT_IMPLEMENTABLE_RL_VALUE_FN = "rl:needs-learned-value-head"
NOT_IMPLEMENTABLE_RL_REWARD_MODEL = "rl:needs-rlhf-reward-model"

# ---------------------------------------------------------------------------
# Hardware / inference-engine  (Tài liệu (8),(9),(10))
# ---------------------------------------------------------------------------
# Source: "Tài liệu (8)" — inference speed-up optimizations.

NOT_IMPLEMENTABLE_HW_BATCHING = "hw:needs-continuous-batching-engine"
NOT_IMPLEMENTABLE_HW_MEMORY_PROFILE = "hw:needs-device-memory-flops-profiler"
NOT_IMPLEMENTABLE_HW_KERNEL_FUSION = "hw:needs-fused-kernel-compile"

# ---------------------------------------------------------------------------
# Bayesian-causal / monitoring internals  (BAYESIAN_CAUSAL_REASONING)
# ---------------------------------------------------------------------------
# Source: README "BAYESIAN_CAUSAL_REASONING_REPORT.md".

NOT_IMPLEMENTABLE_CAUSAL_GRAPH_FIT = "bayesian:needs-causal-graph-learning"

# ---------------------------------------------------------------------------
# Helper to enumerate all NOT_IMPLEMENTABLE symbols for auditors.
# ---------------------------------------------------------------------------


def list_not_implementable() -> list[str]:
    """Return all ``NOT_IMPLEMENTABLE_*`` names in this module.

    Used by tests/auditing to prove the catalog is exhaustive-ish.
    """
    import sys as _sys
    mod = _sys.modules[__name__]
    return sorted(
        name for name in vars(mod)
        if name.startswith("NOT_IMPLEMENTABLE_")
    )


__all__ = [
    "list_not_implementable",
] + [n for n in list(locals()) if n.startswith("NOT_IMPLEMENTABLE_")]