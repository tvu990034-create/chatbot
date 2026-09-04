"""
Two-stage cache: exact identity, then hashed-embedding similarity.
Optional sentence-transformers embeddings when installed and enabled.
"""

import hashlib
import json
import logging
import os
import threading
from typing import Dict, Optional
from pathlib import Path

from .opt_core import (
    BoundedTTLCache,
    cosine_similarity,
    hashed_embedding,
    make_cache_identity,
)

logger = logging.getLogger(__name__)


class SemanticCache:
    def __init__(
        self,
        cache_dir: str = "cache",
        similarity_threshold: float = 0.92,
        max_size: int = 1000,
        default_ttl: float = 3600.0,
        persist: bool = True,
        use_transformer: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "semantic_cache.json"
        self.similarity_threshold = similarity_threshold
        self._store = BoundedTTLCache(max_size=max_size, default_ttl=default_ttl)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._dirty = False
        self._pending_save = False
        self._max_size = max_size
        self._persist = persist
        self.embedding_model = self._load_embedding_model() if use_transformer else None
        if persist:
            self._lazy_load()

    @property
    def cache(self) -> Dict:
        return {k: v["value"] for k, v in self._store.items_snapshot()}

    def _lazy_load(self) -> None:
        if not self.cache_file.exists():
            return
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.warning("Failed to load semantic cache: %s", e)
            return
        if isinstance(raw, dict):
            for key, entry in list(raw.items())[-self._max_size:]:
                if isinstance(entry, dict):
                    self._store.set(key, entry)

    def _save_cache(self):
        if not self._persist:
            return
        try:
            self.cache_dir.mkdir(exist_ok=True)
            snapshot = {k: v["value"] for k, v in self._store.items_snapshot()}
            tmp = self.cache_file.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(snapshot, f)
            os.replace(tmp, self.cache_file)
            self._dirty = False
        except Exception as e:
            logger.warning("Failed to persist semantic cache: %s", e)

    def _schedule_save(self):
        if not self._persist:
            return

        def _run():
            self._save_cache()

        threading.Thread(target=_run, name="semantic-cache-persist", daemon=True).start()

    def _load_embedding_model(self):
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            logger.info("Transformer embeddings unavailable (%s); using hashed embeddings", e)
            return None

    def _get_embedding(self, text: str):
        if self.embedding_model is None:
            return hashed_embedding(text)
        try:
            return self.embedding_model.encode(text, show_progress_bar=False).tolist()
        except Exception as e:
            logger.warning("Transformer embed failed, falling back to hashed vectors: %s", e)
            return hashed_embedding(text)

    def _exact_key(self, query: str, context: Optional[Dict] = None) -> str:
        ctx = context or {}
        return make_cache_identity(
            query,
            model=ctx.get("model", ""),
            temperature=ctx.get("temperature"),
            max_tokens=ctx.get("max_tokens"),
            system_prompt=ctx.get("system_prompt", ""),
            messages=ctx.get("messages"),
            rag_version=ctx.get("rag_version", ""),
            rag_enabled=bool(ctx.get("rag_enabled")),
            prompt_version=ctx.get("prompt_version", "v1"),
        )

    def get(self, query: str, context: Optional[Dict] = None) -> Optional[str]:
        if not query or not query.strip():
            return None
        exact_key = self._exact_key(query, context)
        entry = self._store.get(exact_key)
        if entry is not None:
            self._hits += 1
            return entry.get("response") if isinstance(entry, dict) else entry

        # Stage 2: embedding similarity, only among unexpired entries with the same model.
        want_model = (context or {}).get("model", "")
        q_emb = self._get_embedding(query)
        best = None
        best_sim = self.similarity_threshold
        for key, wrapped in self._store.items_snapshot():
            item = wrapped.get("value") or {}
            if not isinstance(item, dict):
                continue
            if want_model and item.get("model") not in ("", want_model):
                continue
            emb = item.get("embedding")
            if not emb:
                continue
            sim = cosine_similarity(q_emb, emb)
            if sim >= best_sim:
                best_sim = sim
                best = item.get("response")
        if best is not None:
            self._hits += 1
            return best
        self._misses += 1
        return None

    def set(self, query: str, response: str, metadata: Optional[Dict] = None, context: Optional[Dict] = None):
        key = self._exact_key(query, context)
        entry = {
            "query": query,
            "response": response,
            "embedding": self._get_embedding(query),
            "model": (context or {}).get("model", ""),
            "metadata": metadata or {},
        }
        self._store.set(key, entry)
        self._writes += 1
        self._dirty = True
        if self._writes % 10 == 0:
            self._schedule_save()

    def save_now(self):
        if self._dirty:
            self._save_cache()

    def clear(self):
        self._store.clear()
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._save_cache()

    def get_stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            "total_entries": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "writes": self._writes,
            "hit_rate": self._hits / total if total else 0,
            "similarity_threshold": self.similarity_threshold,
            "embedding_model_available": True,
            "transformer_available": self.embedding_model is not None,
            "dirty": self._dirty,
        }


_semantic_cache_instance = None
_semantic_cache_lock = threading.Lock()


def get_semantic_cache() -> SemanticCache:
    global _semantic_cache_instance
    if _semantic_cache_instance is None:
        with _semantic_cache_lock:
            if _semantic_cache_instance is None:
                _semantic_cache_instance = SemanticCache()
    return _semantic_cache_instance
