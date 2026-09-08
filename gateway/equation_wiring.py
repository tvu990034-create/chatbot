"""
gateway/equation_wiring.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Defensive, opt-in wiring of the equation library onto live-path objects.

Every `install_<domain>(target)` is idempotent, log-and-continue, and guarded by
`getattr(settings, "enable_...", False)` so it can never raise in production.
All heavy math lives in `gateway/equations/*`; this module only patches callers
(SemanticCache, RouterState, RAG provider, agent policy).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache: semantic second-stage fallback
# ---------------------------------------------------------------------------

# key -> (context_tag, embedding); the context tag scopes matches to the same
# conversation history so a near-duplicate in one session can never retrieve a
# stored answer that was computed under a different history.
_embs: Dict[str, Tuple[str, List[float]]] = {}
_embs_lock = threading.Lock()


def _cache_embed(query: str) -> List[float]:
    from gateway.opt_core import hashed_embedding
    return hashed_embedding(query)


def _ctx_tag(context: Optional[Dict]) -> Optional[str]:
    """Stable digest of the conversation so far (None = no history)."""
    msgs = (context or {}).get("history")
    if not msgs:
        return None
    try:
        blob = json.dumps(
            [{"role": m.get("role"), "content": m.get("content")} for m in msgs],
            sort_keys=True, default=str,
        )
    except Exception:
        return None
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def install_semantic_cache(cache: Any) -> bool:
    """
    Add a semantic (cosine) second stage to `SimpleCache.get()`.

    Augments the exact-match `get` so that a near-miss query returns the closest
    stored answer instead of `None`, using the equation-library
    `AdaptiveThreshold` + `SemanticCacheMath.rank` scoring (provenance:
    Untitled document (2) cache equations).

    Guarded by ``enable_performance_equations`` and ``semantic_cache_enabled``.
    If either setting is off, `get` keeps its exact-match behavior.
    """
    enabled = bool(
        getattr(settings, "enable_performance_equations", False)
        and getattr(settings, "semantic_cache_enabled", False)
    )
    if not enabled:
        logger.debug("equation_wiring: semantic cache disabled by settings")
        return False
    if cache is None or getattr(cache, "_semantic_installed", False):
        return False

    from gateway.equations.cache_math import (
        AdaptiveThreshold, SemanticCacheMath,
    )
    from gateway.equations.routing_math import majority_consensus

    try:
        threshold = AdaptiveThreshold(
            base=float(getattr(settings, "semantic_cache_threshold", 0.92)),
            mu_weight=0.5,
            sigma_mult=1.0,
            min_threshold=0.60,
            max_threshold=0.95,
        )
        math_ = SemanticCacheMath(k=3, mu_sigma_threshold=threshold)

        original_get = cache.get

        def _semantic_get(query: str = "", context: Optional[Dict] = None, *,
                          _key: Optional[str] = None) -> Optional[str]:
            exact = original_get(query, context, _key=_key)
            if exact is not None:
                # Do NOT observe here: feeding 1.0 on every exact hit pins the
                # adaptive threshold at max (mu=1, sigma=0) and makes the
                # semantic fallback dead. Track the candidate similarities that
                # actually flow through the semantic search below instead.
                return exact
            if _key is None or not query or len(query) < 8:
                return None

            tag = _ctx_tag(context)
            q_emb = _cache_embed(query)
            with _embs_lock:
                stored = {
                    k: e for k, (t, e) in _embs.items() if t == tag
                }
            if not stored:
                return None

            ranked = math_.rank(q_emb, stored, now=0.0)
            if not ranked:
                return None
            best_key, best_sim = ranked[0]
            # Feed the best candidate similarity so the gate learns the real
            # near-miss distribution of live traffic rather than staying at
            # the static base.
            threshold.observe(best_sim)
            if not threshold.hit(best_sim):
                return None
            best = original_get(_key=best_key)
            if best is None:
                return None
            logger.debug(
                "semantic cache hit query=%.40s… sim=%.3f", query, best_sim)
            cache._last_semantic_hit = (query, best_sim)
            return best

        cache.get = _semantic_get

        original_set = cache.set

        def _semantic_set(query: str, response: str,
                          metadata: Optional[Dict] = None,
                          context: Optional[Dict] = None, *, _key: Optional[str] = None):
            original_set(query, response, metadata=metadata, context=context, _key=_key)
            if _key is None or not query:
                return
            tag = _ctx_tag(context)
            with _embs_lock:
                _embs[_key] = (tag, _cache_embed(query))
                if len(_embs) > 4096:
                    # Keep the dict bounded: drop oldest key.
                    _embs.pop(next(iter(_embs)), None)

        cache.set = _semantic_set

        # Keep the semantic embedding index in lock-step with the cache store:
        # a cache.clear() must also wipe _embs or stale semantic hits survive
        # the reset.
        if not getattr(cache, "_semantic_clear_wired", False):
            orig_clear = cache.clear

            def _wrapped_clear(*args, **kwargs):
                with _embs_lock:
                    _embs.clear()
                return orig_clear(*args, **kwargs)

            cache.clear = _wrapped_clear
            cache._semantic_clear_wired = True

        cache._semantic_installed = True
        cache._semantic_stats = {
            "hits": 0, "misses": 0,
        } | getattr(cache, "_semantic_stats", {})
        logger.info("equation_wiring: semantic cache installed on %s", type(cache).__name__)
        return True
    except Exception as exc:  # noqa: BLE001 – must never break inference
        logger.warning("equation_wiring: semantic cache install failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Router: EWMA load-aware adjoint (never replaces existing steering, only adds)
# ---------------------------------------------------------------------------

def install_ewma_router(state: Any) -> bool:
    """
    Add per-model EWMA load tracking onto a `RouterState` so downstream
    `select_model` can use `state.ewma_load(model)` if wanted (provenance:
    #10 EWMA/load-aware gating).  Pure additive — does not change selection.
    """
    if state is None or getattr(state, "_ewma_installed", False):
        return False
    from gateway.equations.routing_math import LoadBalancer
    try:
        state._ewma = LoadBalancer(alpha=0.3)
        orig_start = getattr(state, "record_start", None)
        orig_end = getattr(state, "record_end", None)

        def _start(model: str) -> None:
            if orig_start is not None:
                orig_start(model)
            try:
                state._ewma.observe(model, state.queue_depth(model))
            except Exception:
                pass

        def _end(model: str) -> None:
            try:
                state._ewma.observe(model, state.queue_depth(model))
            except Exception:
                pass
            if orig_end is not None:
                orig_end(model)

        if orig_start is not None:
            state.record_start = _start
        if orig_end is not None:
            state.record_end = _end
        state._ewma_installed = True
        logger.info("equation_wiring: EWMA load tracker installed on router")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("equation_wiring: EWMA router install failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# RAG: Koopman mix post-retrieval (kills the broken-import path with real math)
# ---------------------------------------------------------------------------

def install_koopman_rag(rag: Any) -> bool:
    """
    If ``koopman_mixing_enabled`` is on and a ``retrieve()``-style method exists,
    wrap it to blend the top retrieved chunk embeddings with the Koopman mixer
    (provenance: (1)/(2) Koopman equations).  This makes the previously-import-
    failing path both importable AND functional-in-pure-Python.
    """
    enabled = bool(getattr(settings, "koopman_mixing_enabled", False))
    if not enabled or rag is None or getattr(rag, "_koopman_wired", False):
        return False
    try:
        from gateway.advanced_optimizations import get_koopman_mixer
        from gateway.opt_core import hashed_embedding

        mixer = get_koopman_mixer(
            dim=int(getattr(settings, "koopman_mix_dim", 64)),
            lift_dim=128,
        )
        orig_retrieve = getattr(rag, "retrieve", None)
        if orig_retrieve is None:
            return False

        def _retrieve(question: str) -> dict:
            result = orig_retrieve(question)
            chunks = result.get("chunks") or []
            if len(chunks) >= 2:
                vectors = [hashed_embedding(c) for c in chunks]
                try:
                    mixed = mixer.mix_embeddings(vectors)
                    result["koopman_mixed"] = mixed
                except Exception:  # noqa: BLE001
                    result["koopman_mixed"] = None
            return result

        rag.retrieve = _retrieve
        rag._koopman_wired = True
        logger.info("equation_wiring: Koopman mixing wired onto RAG provider")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("equation_wiring: Koopman RAG install failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Aggregate installer
# ---------------------------------------------------------------------------

def install_all() -> dict:
    """Install every enabled equation-wiring onto the live singletons.

    Returns a dict of {component: installed_bool} for audit/logging.
    """
    from gateway.opt_core import get_router_state
    from gateway.simple_cache import get_cache

    results: Dict[str, bool] = {}
    try:
        results["semantic_cache"] = install_semantic_cache(get_cache())
    except Exception as exc:  # noqa: BLE001
        logger.warning("semantic_cache wiring failed: %s", exc)
        results["semantic_cache"] = False

    try:
        results["ewma_router"] = install_ewma_router(get_router_state())
    except Exception as exc:  # noqa: BLE001
        logger.warning("ewma_router wiring failed: %s", exc)
        results["ewma_router"] = False

    # RAG providers are lazy; best-effort wiring if they exist already.
    try:
        from rag.llama_index_rag import get_rag
        results["koopman_rag"] = install_koopman_rag(get_rag())
    except Exception:  # noqa: BLE001 – provider may not be built yet
        results["koopman_rag"] = False

    return results


__all__ = [
    "install_semantic_cache", "install_ewma_router",
    "install_koopman_rag", "install_all",
]