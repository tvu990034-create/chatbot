"""
Simple Response Cache
In-memory LRU/TTL cache with batched, non-blocking persistence.
"""

import hashlib
import json
import logging
import os
import threading
from typing import Dict, Optional
from pathlib import Path

from .opt_core import BoundedTTLCache, make_cache_identity

logger = logging.getLogger(__name__)


class SimpleCache:
    """Bounded in-memory cache with optional async JSON persistence."""

    def __init__(self, cache_dir: str = "cache", max_size: int = 1000, default_ttl: float = 3600.0, persist: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_file = self.cache_dir / "response_cache.json"
        self._max_size = max_size
        self._store = BoundedTTLCache(max_size=max_size, default_ttl=default_ttl)
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._dirty = False
        self._pending_save = False
        self._persist = persist
        self._in_flight: Dict[str, threading.Event] = {}
        self._in_flight_results: Dict[str, str] = {}
        self._persist_thread: Optional[threading.Thread] = None
        if persist:
            self._lazy_load()

    @property
    def cache(self) -> Dict:
        """Compatibility view used by older tests."""
        return {k: v["value"] for k, v in self._store.items_snapshot()}

    def _lazy_load(self) -> None:
        if not self.cache_file.exists():
            return
        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception as e:
            logger.warning("Failed to load response cache from %s: %s", self.cache_file, e)
            return
        if not isinstance(raw, dict):
            logger.warning("Response cache file is not an object; ignoring")
            return
        # Bound what we load so startup cannot ingest an unbounded dump.
        items = list(raw.items())[-self._max_size:]
        for key, entry in items:
            if isinstance(entry, dict) and "response" in entry:
                self._store.set(key, entry)
            elif isinstance(entry, str):
                self._store.set(key, {"response": entry, "metadata": {}})

    def _save_cache(self) -> None:
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
            self._pending_save = False
        except Exception as e:
            logger.warning("Failed to persist response cache to %s: %s", self.cache_file, e)

    def _schedule_save(self) -> None:
        if not self._persist:
            return
        with self._lock:
            if self._pending_save:
                return
            self._pending_save = True

        def _run():
            try:
                self._save_cache()
            finally:
                with self._lock:
                    self._pending_save = False

        t = threading.Thread(target=_run, name="simple-cache-persist", daemon=True)
        t.start()

    def _get_key(self, query: str, context: Optional[Dict] = None) -> str:
        ctx = context or {}
        return make_cache_identity(
            query,
            model=ctx.get("model", ""),
            temperature=ctx.get("temperature"),
            max_tokens=ctx.get("max_tokens"),
            top_p=ctx.get("top_p"),
            top_k=ctx.get("top_k"),
            system_prompt=ctx.get("system_prompt", ""),
            messages=ctx.get("messages"),
            tools=ctx.get("tools"),
            rag_version=ctx.get("rag_version", ""),
            rag_enabled=bool(ctx.get("rag_enabled")),
            prompt_version=ctx.get("prompt_version", "v1"),
            language=ctx.get("language", ""),
        )

    def get(self, query: str, context: Optional[Dict] = None) -> Optional[str]:
        key = self._get_key(query, context)
        entry = self._store.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        if isinstance(entry, dict):
            return entry.get("response")
        return entry

    def get_or_wait(self, query: str, context: Optional[Dict] = None) -> Optional[str]:
        cached = self.get(query, context)
        if cached is not None:
            return cached
        key = self._get_key(query, context)
        with self._lock:
            if key in self._in_flight:
                event = self._in_flight[key]
            else:
                self._in_flight[key] = threading.Event()
                return None
        event.wait(timeout=30)
        with self._lock:
            return self._in_flight_results.get(key)

    def set_result(self, query: str, response: str, context: Optional[Dict] = None):
        if not self._is_valid_response(response):
            logger.warning("Invalid in-flight result not cached")
            key = self._get_key(query, context)
            with self._lock:
                ev = self._in_flight.pop(key, None)
                if ev:
                    ev.set()
            return
        self.set(query, response, context=context)
        key = self._get_key(query, context)
        with self._lock:
            self._in_flight_results[key] = response
            ev = self._in_flight.pop(key, None)
            if ev:
                ev.set()

    def _is_valid_response(self, response: str) -> bool:
        if not response or not isinstance(response, str) or not response.strip():
            return False
        if response.startswith("Warm cache placeholder"):
            return False
        return True

    def set(self, query: str, response: str, metadata: Optional[Dict] = None, context: Optional[Dict] = None):
        if not self._is_valid_response(response):
            logger.warning("Invalid response not cached")
            return
        key = self._get_key(query, context)
        self._store.set(key, {"response": response, "metadata": metadata or {}})
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

    def clear_benchmark_data(self):
        self.clear()

    def clear_all_for_benchmark(self):
        self.clear()
        logger.info("Cleared all cache data for benchmark mode")

    def stats(self) -> Dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "writes": self._writes,
            "size": len(self._store),
            "hit_rate": self._hits / total if total else 0,
            "dirty": self._dirty,
        }


_cache_instance = None
_cache_lock = threading.Lock()


def get_cache() -> SimpleCache:
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = SimpleCache()
    return _cache_instance
