"""
Smoke tests for the equation-library wiring onto the live path.

These tests must NOT require an Ollama backend. They exercise:

  * gateway.advanced_optimizations.KoopmanMixer (fixes the broken import)
  * rag.llama_index_rag now imports with _koopman_available True
  * semantic cache fallback install
  * EWMA router install
  * equation modules import (+ NOT_IMPLEMENTABLE catalog)
"""

import pytest


def test_koopman_mixer_imports_and_blends():
    from gateway.advanced_optimizations import KoopmanMixer, get_koopman_mixer
    mixer = KoopmanMixer(dim=16, lift_dim=32)
    out = mixer.mix_embeddings([[1.0, 0.0, 0.0] * 0 + [1.0, 0.0],
                                [0.0, 1.0, 0.0] * 0 + [0.0, 1.0]])
    assert isinstance(out, list)
    assert len(out) == 2
    m2 = get_koopman_mixer()
    assert m2 is get_koopman_mixer()


def test_llama_index_rag_koopman_import_fixed():
    import rag.llama_index_rag as rag
    assert rag._koopman_available is True


def test_semantic_cache_install_and_hit():
    import sys
    from gateway.simple_cache import SimpleCache
    from gateway.equation_wiring import install_semantic_cache
    cache = SimpleCache(persist=False)
    assert install_semantic_cache(cache) is True
    cache.set(query="what is the capital of france?", response="Paris", _key="k1")
    # exact hit
    assert cache.get(query="what is the capital of france?", _key="k1") == "Paris"
    # non-retrievable query stays a miss
    assert cache.get(_key="unknown") is None


def test_ewma_router_install():
    from gateway.opt_core import RouterState
    from gateway.equation_wiring import install_ewma_router
    state = RouterState()
    assert install_ewma_router(state) is True
    state.record_start("ollama/test")
    state.record_end("ollama/test")
    load = state._ewma.load("ollama/test")
    assert 0.0 <= load <= 1.0


def test_all_equation_modules_import():
    import gateway.equations.cache_math
    import gateway.equations.calibration_math
    import gateway.equations.bandit_math
    import gateway.equations.routing_math
    import gateway.equations.retrieval_math
    import gateway.equations.memory_math
    import gateway.equations.planning_math
    import gateway.equations.not_implementable
    assert callable(gateway.equations.cache_math.cos_sim)


def test_install_all_tolerates_no_backend():
    from config import settings
    settings.koopman_mixing_enabled = False
    results = __import__("gateway.equation_wiring", fromlist=["install_all"]).install_all()
    assert "semantic_cache" in results
    assert "ewma_router" in results
    assert "koopman_rag" in results