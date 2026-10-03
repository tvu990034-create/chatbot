"""
Smoke tests for the optimization-library wiring onto the live path.

These tests must NOT require an Ollama backend. They exercise:

  * gateway.advanced_optimizations.KoopmanMixer (fixes the broken import)
  * rag.llama_index_rag now imports with _koopman_available True
  * semantic cache fallback install
  * EWMA router install
  * optimization modules import (+ NOT_IMPLEMENTABLE catalog)
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


def test_semantic_fallback_is_mode_scoped():
    """A near-miss must never retrieve an answer cached under a different
    performance_mode (diag: balanced chats were served speed drafts in ~1ms
    via mode-blind similarity matching)."""
    from gateway.simple_cache import SimpleCache
    from gateway import equation_wiring as wiring
    wiring._embs.clear()
    cache = SimpleCache(persist=False)
    assert wiring.install_semantic_cache(cache) is True
    ctx_speed = {"model": "m", "performance_mode": "speed"}
    ctx_bal = {"model": "m", "performance_mode": "balanced"}
    cache.set("what is the capital of france?", "SPEED-ANS",
              context=ctx_speed, _key="k-speed")
    cache.set("what is the capital of france?", "BAL-ANS",
              context=ctx_bal, _key="k-bal")
    # Same text, unknown key -> exact miss -> semantic on identical embedding
    # (sim 1.0) must return the SAME-mode answer, never the other mode's.
    assert cache.get("what is the capital of france?",
                     context=ctx_bal, _key="k-unknown") == "BAL-ANS"
    assert cache.get("what is the capital of france?",
                     context=ctx_speed, _key="k-unknown") == "SPEED-ANS"
    wiring._embs.clear()


def test_embs_index_bounded():
    """rank() scans every candidate per miss: the embedding index must stay
    bounded or long server sessions turn each miss into a ~0.5s full scan."""
    from gateway.simple_cache import SimpleCache
    from gateway import equation_wiring as wiring
    wiring._embs.clear()
    try:
        cache = SimpleCache(persist=False)
        assert wiring.install_semantic_cache(cache) is True
        for i in range(1100):
            cache.set("unique padding query number %d here" % i, "a",
                      context={"performance_mode": "speed"},
                      _key="k%d" % i)
        assert len(wiring._embs) <= 1024
    finally:
        wiring._embs.clear()


def test_embs_persisted_across_restart(tmp_path):
    """Without persistence every restart runs semantically cold until
    rewrites repopulate the index (each near-miss pays a model call)."""
    from gateway.simple_cache import SimpleCache
    from gateway import equation_wiring as wiring
    wiring._embs.clear()
    try:
        c1 = SimpleCache(cache_dir=str(tmp_path), persist=True, max_size=50)
        assert wiring.install_semantic_cache(c1) is True
        ctx = {"model": "m", "performance_mode": "balanced"}
        c1.set("what is the capital of france?", "Paris",
               context=ctx, _key="k1")
        c1.set("describe the harbor lights?", "Harbor.",
               context=ctx, _key="k2")
        wiring._save_embs(c1)
        # The responses themselves persist via the store's own save; the
        # embedding index alone is not sufficient (best-key lookup must hit).
        c1.save_now()
        assert (tmp_path / "semantic_embs.json").exists()
        # Simulate a restart: empty index, fresh instance, same dir.
        wiring._embs.clear()
        c2 = SimpleCache(cache_dir=str(tmp_path), persist=True, max_size=50)
        assert wiring.install_semantic_cache(c2) is True
        assert c2.get("what is the capital of france?",
                      context=ctx, _key="k-unknown") == "Paris"
    finally:
        wiring._embs.clear()


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