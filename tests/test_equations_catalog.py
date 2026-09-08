"""
Tests for gateway/equations/not_implementable.py
Ensure every NOT_IMPLEMENTABLE symbol exists and is enumerable (auditor-facing).
"""

import pytest


def test_not_implementable_catalog_has_expected_entries():
    from gateway.equations.not_implementable import (
        NOT_IMPLEMENTABLE_TRAINING_BATCH,
        NOT_IMPLEMENTABLE_GPU_KV_CACHE,
        NOT_IMPLEMENTABLE_GPU_SPECULATIVE,
        NOT_IMPLEMENTABLE_BNN_VI,
        NOT_IMPLEMENTABLE_RL_POLICY_GRADIENT,
        NOT_IMPLEMENTABLE_HW_BATCHING,
    )
    # A handful of representative categories must exist (not fakes).
    assert NOT_IMPLEMENTABLE_TRAINING_BATCH.startswith("training:")
    assert NOT_IMPLEMENTABLE_GPU_KV_CACHE.startswith("gpu:")
    assert NOT_IMPLEMENTABLE_GPU_SPECULATIVE.startswith("gpu:")
    assert NOT_IMPLEMENTABLE_BNN_VI.startswith("model:")
    assert NOT_IMPLEMENTABLE_RL_POLICY_GRADIENT.startswith("rl:")
    assert NOT_IMPLEMENTABLE_HW_BATCHING.startswith("hw:")


def test_list_not_implementable_sorted_and_unique():
    from gateway.equations.not_implementable import list_not_implementable
    names = list_not_implementable()
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert len(names) >= 25


def test_all_implied_members_present_in_module_dunder_all():
    from gateway.equations import not_implementable as mod
    members = [n for n in vars(mod) if n.startswith("NOT_IMPLEMENTABLE_")]
    assert all(m in mod.__all__ for m in members)