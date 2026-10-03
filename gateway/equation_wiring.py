"""
gateway/equation_wiring.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Defensive, opt-in wiring of the optimization library onto live-path objects.

Every `install_<domain>(target)` is idempotent, log-and-continue, and guarded by
`getattr(settings, "enable_...", False)` so it can never raise in production.
All heavy math lives in the gateway optimization modules; this module only patches
callers (SemanticCache, RouterState, RAG provider, agent policy).
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache: semantic second-stage fallback
# ---------------------------------------------------------------------------

# key -> (context_tag, embedding, performance_mode); the mode scopes matches
# so a near-duplicate can never retrieve an answer cached under a different
# mode (speed drafts are terse by design).
_embs: Dict[str, Tuple[str, List[float], str]] = {}
_embs_lock = threading.Lock()


def _cache_embed(query: str) -> List[float]:
    from gateway.opt_core import hashed_embedding
    return hashed_embedding(query)


def _embs_path(cache: Any) -> Optional[Path]:
    """Sidecar file for the embedding index, next to the cache dir."""
    d = getattr(cache, "cache_dir", None)
    if d is None:
        return None
    try:
        return Path(d) / "semantic_embs.json"
    except Exception:  # noqa: BLE001
        return None


def _save_embs(cache: Any) -> None:
    """Persist the embedding index so semantic hits survive restarts.

    Without this every restart runs semantically cold until rewrites
    repopulate the index (every near-miss pays a full model call).
    Best-effort; never raises.
    """
    p = _embs_path(cache)
    if p is None:
        return
    try:
        with _embs_lock:
            items = list(_embs.items())[-1024:]
        blob = {}
        for k, v in items:
            try:
                blob[k] = [v[0], [float(x) for x in v[1]],
                           v[2] if len(v) == 3 else ""]
            except Exception:  # noqa: BLE001 - skip malformed entries
                continue
        p.write_text(json.dumps(blob), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("optimization wiring: embs save failed: %s", exc)


def _load_embs(cache: Any) -> None:
    """Restore a persisted embedding index at install time."""
    p = _embs_path(cache)
    if p is None:
        return
    try:
        if not p.exists():
            return
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return
        with _embs_lock:
            for k, v in list(raw.items())[-1024:]:
                # Tag is None when no conversation history was present.
                if (isinstance(v, (list, tuple)) and len(v) == 3
                        and (v[0] is None or isinstance(v[0], str))
                        and isinstance(v[2], str)):
                    try:
                        _embs[k] = (v[0], [float(x) for x in v[1]], v[2])
                    except Exception:  # noqa: BLE001 - skip bad rows
                        continue
    except Exception as exc:  # noqa: BLE001
        logger.warning("optimization wiring: embs load failed: %s", exc)


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
    stored answer instead of `None`, using the optimization-library
    `AdaptiveThreshold` + `SemanticCacheMath.rank` scoring.

    Guarded by ``enable_performance_equations`` and ``semantic_cache_enabled``.
    If either setting is off, `get` keeps its exact-match behavior.
    """
    enabled = bool(
        getattr(settings, "enable_performance_equations", False)
        and getattr(settings, "semantic_cache_enabled", False)
    )
    if not enabled:
        logger.debug("optimization wiring: semantic cache disabled by settings")
        return False
    if cache is None or getattr(cache, "_semantic_installed", False):
        return False

    from gateway.equations.cache_math import (
        AdaptiveThreshold, SemanticCacheMath,
    )

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
            # Mode scoping: a near-miss must never retrieve an answer cached
            # under a different performance_mode (speed drafts are terse by
            # design; serving one to balanced/quality corrupts their answers).
            req_mode = (context or {}).get("performance_mode", "")
            q_emb = _cache_embed(query)
            with _embs_lock:
                stored = {}
                for k, v in _embs.items():
                    t = v[0]
                    e = v[1]
                    m = v[2] if len(v) == 3 else ""
                    if t == tag and m == req_mode:
                        stored[k] = e
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
            mode = (context or {}).get("performance_mode", "")
            with _embs_lock:
                _embs[_key] = (tag, _cache_embed(query), mode)
                # Bound the index: rank() scans all candidates per miss, so
                # an unbounded index turns every miss into a ~0.5s full scan.
                # 1024 entries keeps the worst case ~0.1s; oldest evicted.
                if len(_embs) > 1024:
                    # Keep the dict bounded: drop oldest key.
                    _embs.pop(next(iter(_embs)), None)
                persist_now = len(_embs) % 100 == 0
            # Persist periodically (amortized: every 100th indexed write) so
            # restarts restore semantic hits instead of running cold.
            if persist_now:
                _save_embs(cache)

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
        # Restore a persisted index so this process serves semantic hits
        # immediately instead of waiting for rewrites to repopulate it.
        _load_embs(cache)
        # Flush-on-exit only for persistent caches: test instances are
        # transient (their dirs vanish at teardown) and must not accumulate
        # shutdown hooks.
        if getattr(cache, "_persist", False):
            try:
                atexit.register(_save_embs, cache)
            except Exception:  # noqa: BLE001 - shutdown hook is best-effort
                pass
        logger.info("optimization wiring: semantic cache installed on %s", type(cache).__name__)
        return True
    except Exception as exc:  # noqa: BLE001 – must never break inference
        logger.warning("optimization wiring: semantic cache install failed: %s", exc)
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
        logger.info("optimization wiring: EWMA load tracker installed on router")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("optimization wiring: EWMA router install failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# RAG: Koopman mix post-retrieval (kills the broken-import path with real math)
# ---------------------------------------------------------------------------

def install_koopman_rag(rag: Any) -> bool:
    """
    If ``koopman_mixing_enabled`` is on and a ``retrieve()``-style method exists,
    wrap it to blend the top retrieved chunk embeddings with the Koopman mixer.
    This makes the previously-import-
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
        logger.info("optimization wiring: Koopman mixing wired onto RAG provider")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("optimization wiring: Koopman RAG install failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Aggregate installer
# ---------------------------------------------------------------------------

def install_all() -> dict:
    """Install every enabled optimization-wiring onto the live singletons.

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

    # RAG providers are lazy AND heavy (embedding index build); only touch
    # them when Koopman mixing is actually enabled.  Constructing get_rag()
    # unconditionally made every gateway creation pay for an index build.
    try:
        if bool(getattr(settings, "koopman_mixing_enabled", False)):
            from rag.llama_index_rag import get_rag
            results["koopman_rag"] = install_koopman_rag(get_rag())
        else:
            results["koopman_rag"] = False
    except Exception:  # noqa: BLE001 – provider may not be built yet
        results["koopman_rag"] = False

    return results


__all__ = [
    "install_semantic_cache", "install_ewma_router",
    "install_koopman_rag", "install_all",
]