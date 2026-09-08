"""
Regression tests for cache, concurrency, and shared-state fixes.

Covers:
  1.  Cache API: _key bypass, no double-hashing
  2.  Cache correctness: deterministic keys, no collisions
  3.  Cache concurrency: thread-safe counters under contention
  4.  Atomic persistence: no corrupted JSON after crash
  5.  get_cache() singleton lifetime
  6.  Cache statistics: real hits/misses, not cache size
  7.  Prefix cache: race condition fix
  8.  Gateway singleton: parameter-mismatch warning
  9.  No state leakage between concurrent requests
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ===================================================================
# 1. Cache API: _key bypass
# ===================================================================

class TestCacheKeyBypass:
    """get()/set() with _key must bypass internal key derivation."""

    def test_set_with_key_and_get_with_key(self):
        """Pre-computed key round-trips correctly."""
        from gateway.simple_cache import SimpleCache
        from gateway.opt_core import make_cache_identity

        cache = SimpleCache(persist=False, max_size=100)
        key = make_cache_identity("hello", model="test")
        cache.set(query="", response="world", _key=key)
        result = cache.get(_key=key)
        assert result == "world"

    def test_get_with_key_matches_set_with_key(self):
        """Same _key finds the entry even with different query text."""
        from gateway.simple_cache import SimpleCache
        from gateway.opt_core import make_cache_identity

        cache = SimpleCache(persist=False, max_size=100)
        key = make_cache_identity("actual query", model="m")
        cache.set(query="", response="answer", _key=key)
        # Pass a completely different query — but same key
        result = cache.get(query="wrong query", _key=key)
        assert result == "answer"

    def test_get_without_key_uses_query(self):
        """Normal get/set without _key still works."""
        from gateway.simple_cache import SimpleCache

        cache = SimpleCache(persist=False, max_size=100)
        cache.set("test query", "test response")
        result = cache.get("test query")
        assert result == "test response"

    def test_context_differentiates_keys(self):
        """Same query with different context produces different entries."""
        from gateway.simple_cache import SimpleCache

        cache = SimpleCache(persist=False, max_size=100)
        ctx_a = {"model": "model_a", "temperature": 0.1}
        ctx_b = {"model": "model_b", "temperature": 0.9}
        cache.set("hello", "response A", context=ctx_a)
        cache.set("hello", "response B", context=ctx_b)
        assert cache.get("hello", context=ctx_a) == "response A"
        assert cache.get("hello", context=ctx_b) == "response B"


# ===================================================================
# 2. Cache correctness: deterministic keys
# ===================================================================

class TestCacheKeyDeterminism:
    """make_cache_identity must be deterministic."""

    def test_same_input_same_key(self):
        from gateway.opt_core import make_cache_identity
        k1 = make_cache_identity("query", model="m", temperature=0.5)
        k2 = make_cache_identity("query", model="m", temperature=0.5)
        assert k1 == k2

    def test_different_model_different_key(self):
        from gateway.opt_core import make_cache_identity
        k1 = make_cache_identity("query", model="m1")
        k2 = make_cache_identity("query", model="m2")
        assert k1 != k2

    def test_different_temperature_different_key(self):
        from gateway.opt_core import make_cache_identity
        k1 = make_cache_identity("query", temperature=0.1)
        k2 = make_cache_identity("query", temperature=0.9)
        assert k1 != k2

    def test_different_system_prompt_different_key(self):
        from gateway.opt_core import make_cache_identity
        k1 = make_cache_identity("query", system_prompt="You are A")
        k2 = make_cache_identity("query", system_prompt="You are B")
        assert k1 != k2

    def test_different_rag_different_key(self):
        from gateway.opt_core import make_cache_identity
        k1 = make_cache_identity("query", rag_enabled=False)
        k2 = make_cache_identity("query", rag_enabled=True)
        assert k1 != k2


# ===================================================================
# 3. Cache concurrency: thread-safe counters
# ===================================================================

class TestCacheConcurrency:
    """Concurrent reads/writes must not lose counter updates."""

    def test_counter_accuracy_under_contention(self):
        """Parallel get/set must produce accurate hit/miss/write counts."""
        from gateway.simple_cache import SimpleCache

        cache = SimpleCache(persist=False, max_size=1000)
        n_threads = 8
        ops_per_thread = 200
        errors = []

        def writer(thread_id):
            try:
                for i in range(ops_per_thread):
                    cache.set(f"key-{thread_id}-{i}", f"val-{i}")
            except Exception as e:
                errors.append(e)

        def reader(thread_id):
            try:
                for i in range(ops_per_thread):
                    cache.get(f"key-{thread_id}-{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for t in range(n_threads):
            threads.append(threading.Thread(target=writer, args=(t,)))
            threads.append(threading.Thread(target=reader, args=(t,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"

        stats = cache.stats()
        assert stats["writes"] == n_threads * ops_per_thread
        # hits + misses should account for all reads
        assert stats["hits"] + stats["misses"] == n_threads * ops_per_thread

    def test_no_lost_writes(self):
        """Exactly N writes must be recorded after N concurrent set() calls."""
        from gateway.simple_cache import SimpleCache

        cache = SimpleCache(persist=False, max_size=10000)
        n = 500
        barrier = threading.Barrier(n)

        def do_write(i):
            barrier.wait()
            cache.set(f"k{i}", f"v{i}")

        threads = [threading.Thread(target=do_write, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert cache.stats()["writes"] == n
        assert len(cache._store) == n


# ===================================================================
# 4. Atomic persistence
# ===================================================================

class TestAtomicPersistence:
    """JSON persistence must write atomically."""

    def test_save_creates_valid_json(self):
        """Saved cache file must be valid JSON."""
        from gateway.simple_cache import SimpleCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SimpleCache(cache_dir=tmpdir, persist=True, max_size=100)
            cache.set("key1", "value1")
            cache.save_now()
            cache_file = Path(tmpdir) / "response_cache.json"
            assert cache_file.exists()
            with open(cache_file) as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_save_does_not_leave_tmp_files(self):
        """No .json.tmp files should remain after save."""
        from gateway.simple_cache import SimpleCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = SimpleCache(cache_dir=tmpdir, persist=True, max_size=100)
            for i in range(5):
                cache.set(f"key{i}", f"value{i}")
                cache.save_now()
            tmp_files = list(Path(tmpdir).glob("*.json.tmp"))
            assert len(tmp_files) == 0

    def test_atomic_write_survives_corrupt_source(self):
        """Loading a corrupt cache file should not crash."""
        from gateway.simple_cache import SimpleCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "response_cache.json"
            cache_file.write_text("NOT VALID JSON {{{")
            cache = SimpleCache(cache_dir=tmpdir, persist=True, max_size=100)
            assert len(cache._store) == 0


# ===================================================================
# 5. get_cache() singleton
# ===================================================================

class TestGetCacheSingleton:
    """get_cache() must return the same instance across calls."""

    def test_singleton_identity(self):
        from gateway.simple_cache import get_cache
        c1 = get_cache()
        c2 = get_cache()
        assert c1 is c2

    def test_singleton_thread_safety(self):
        """Multiple threads calling get_cache() must get the same instance."""
        from gateway.simple_cache import get_cache
        results = []

        def grab():
            results.append(get_cache())

        threads = [threading.Thread(target=grab) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r is results[0] for r in results)


# ===================================================================
# 6. Cache statistics accuracy
# ===================================================================

class TestCacheStats:
    """stats() must report real hits/misses, not cache size."""

    def test_hits_are_not_cache_size(self):
        """hits counts actual cache hits, not len(cache)."""
        from gateway.simple_cache import SimpleCache

        cache = SimpleCache(persist=False, max_size=100)
        # Insert 50 entries
        for i in range(50):
            cache.set(f"k{i}", f"v{i}")
        # Hit 10 of them
        for i in range(10):
            cache.get(f"k{i}")
        stats = cache.stats()
        assert stats["hits"] == 10
        assert stats["size"] == 50
        assert stats["hits"] != stats["size"]

    def test_misses_counted(self):
        from gateway.simple_cache import SimpleCache
        cache = SimpleCache(persist=False, max_size=100)
        cache.get("nonexistent")
        cache.get("also_missing")
        assert cache.stats()["misses"] == 2

    def test_hit_rate_computed(self):
        from gateway.simple_cache import SimpleCache
        cache = SimpleCache(persist=False, max_size=100)
        cache.set("k", "v")
        cache.get("k")  # hit
        cache.get("x")  # miss
        stats = cache.stats()
        assert stats["hit_rate"] == 0.5

    def test_errors_tracked(self):
        from gateway.simple_cache import SimpleCache
        cache = SimpleCache(persist=False, max_size=100)
        with cache._counter_lock:
            cache._errors += 3
        assert cache.stats()["errors"] == 3


# ===================================================================
# 7. Prefix cache race condition
# ===================================================================

class TestPrefixCacheRace:
    """_check_prefix_cache must not crash when cache is mutated concurrently."""

    def test_concurrent_check_and_write(self):
        """Concurrent _check_prefix_cache and prefix cache writes."""
        import hashlib
        import threading

        from gateway.universal_enhanced_gateway import (
            UniversalEnhancedGateway,
            _prefix_response_cache,
            _prefix_response_cache_lock,
        )

        gw = UniversalEnhancedGateway.__new__(UniversalEnhancedGateway)
        gw.prefix_response_cache = _prefix_response_cache
        gw.enable_all_optimizations = True

        errors = []

        def writer():
            try:
                for i in range(100):
                    prefix_key = hashlib.md5(f"prefix-{i}".encode()).hexdigest()
                    with _prefix_response_cache_lock:
                        if prefix_key not in _prefix_response_cache:
                            _prefix_response_cache[prefix_key] = {}
                        _prefix_response_cache[prefix_key][f"query-{i}"] = {
                            "response": f"answer-{i}",
                            "gen_identity": {"model": "test"},
                        }
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(100):
                    msgs = [
                        {"role": "system", "content": f"prefix-{i % 10}"},
                        {"role": "user", "content": f"query-{i}"},
                    ]
                    gw._check_prefix_cache(msgs, {"model": "test"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"


# ===================================================================
# 8. Gateway singleton
# ===================================================================

class TestGatewaySingleton:
    """get_universal_gateway must warn on parameter mismatch."""

    def test_singleton_identity(self):
        from gateway.universal_enhanced_gateway import get_universal_gateway
        g1 = get_universal_gateway()
        g2 = get_universal_gateway()
        assert g1 is g2

    def test_singleton_ignores_different_params(self):
        """Second call with different params returns same instance."""
        from gateway.universal_enhanced_gateway import get_universal_gateway
        g1 = get_universal_gateway()
        original_model = g1.model_name
        original_mode = g1.performance_mode
        g2 = get_universal_gateway(model_name="completely_different_model", performance_mode="quality")
        assert g1 is g2
        # Original params must NOT be overwritten
        assert g1.model_name == original_model
        assert g1.performance_mode == original_mode


# ===================================================================
# 9. No state leakage between concurrent requests
# ===================================================================

class TestNoStateLeakage:
    """Concurrent cache operations must not leak between requests."""

    def test_different_contexts_no_cross_talk(self):
        """Request A with model_a must not see Request B's model_b cache."""
        from gateway.simple_cache import SimpleCache

        cache = SimpleCache(persist=False, max_size=100)
        ctx_a = {"model": "model_a", "temperature": 0.1}
        ctx_b = {"model": "model_b", "temperature": 0.9}

        # Simulate concurrent requests
        def request_a():
            cache.set("shared question", "answer A", context=ctx_a)
            time.sleep(0.01)
            return cache.get("shared question", context=ctx_a)

        def request_b():
            cache.set("shared question", "answer B", context=ctx_b)
            time.sleep(0.01)
            return cache.get("shared question", context=ctx_b)

        t1 = threading.Thread(target=request_a)
        t2 = threading.Thread(target=request_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Each context must return its own answer
        assert cache.get("shared question", context=ctx_a) == "answer A"
        assert cache.get("shared question", context=ctx_b) == "answer B"

    def test_conistent_cache_under_parallel_writes(self):
        """Parallel writes to the same key must not corrupt the cache."""
        from gateway.simple_cache import SimpleCache

        cache = SimpleCache(persist=False, max_size=100)
        n = 200
        barrier = threading.Barrier(n)
        errors = []

        def writer(i):
            try:
                barrier.wait()
                cache.set("same_key", f"value-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        # Must return some valid value, not crash
        result = cache.get("same_key")
        assert result is not None
        assert result.startswith("value-")

    def test_cache_survives_heavy_mixed_load(self):
        """Mixed reads/writes under load must not corrupt state."""
        from gateway.simple_cache import SimpleCache

        cache = SimpleCache(persist=False, max_size=500)
        errors = []
        n = 100

        def mixed_ops(thread_id):
            try:
                for i in range(n):
                    key = f"t{thread_id}-k{i}"
                    cache.set(key, f"v{i}")
                    cache.get(key)
                    cache.get(f"missing-{thread_id}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=mixed_ops, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = cache.stats()
        assert stats["hits"] + stats["misses"] == 10 * n * 2  # 10 threads * n ops * 2 (1 hit + 1 miss)
        assert stats["writes"] == 10 * n
