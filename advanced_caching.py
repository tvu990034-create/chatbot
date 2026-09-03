"""
Advanced Caching with Redis Integration
Distributed caching support for multi-instance deployments
"""

import json
import hashlib
import time
from typing import Optional, Dict, Any
from pathlib import Path

class RedisCacheIntegration:
    """Redis integration for distributed caching."""
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_db = redis_db
        self.redis_client = None
        self.fallback_cache = {}  # Fallback to in-memory cache
        
        self._connect_redis()
    
    def _connect_redis(self):
        """Attempt to connect to Redis."""
        try:
            import redis
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2
            )
            # Test connection
            self.redis_client.ping()
            print("✅ Redis connected successfully")
        except ImportError:
            print("⚠️ Redis library not installed, using fallback cache")
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}, using fallback cache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if self.redis_client:
            try:
                value = self.redis_client.get(key)
                if value:
                    return json.loads(value)
            except:
                pass
        
        # Fallback to in-memory cache
        return self.fallback_cache.get(key)
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache with TTL."""
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(value))
                return
            except:
                pass
        
        # Fallback to in-memory cache
        self.fallback_cache[key] = value
    
    def delete(self, key: str):
        """Delete value from cache."""
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except:
                pass
        
        # Fallback to in-memory cache
        if key in self.fallback_cache:
            del self.fallback_cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "backend": "redis" if self.redis_client else "memory",
            "fallback_cache_size": len(self.fallback_cache)
        }
        
        if self.redis_client:
            try:
                info = self.redis_client.info()
                stats["redis"] = {
                    "connected_clients": info.get('connected_clients'),
                    "used_memory": info.get('used_memory_human'),
                    "total_keys": info.get('db0', {}).get('keys')
                }
            except:
                pass
        
        return stats

class CacheCompression:
    """Cache compression for large cached responses."""
    
    def __init__(self):
        self.compression_threshold = 1024  # Compress responses larger than 1KB
    
    def compress_if_needed(self, data: Any) -> tuple:
        """Compress data if it exceeds threshold."""
        serialized = json.dumps(data) if not isinstance(data, str) else data
        
        if len(serialized) > self.compression_threshold:
            try:
                import zlib
                compressed = zlib.compress(serialized.encode())
                return compressed, True
            except:
                return serialized, False
        
        return serialized, False
    
    def decompress_if_needed(self, data: bytes, is_compressed: bool) -> str:
        """Decompress data if it was compressed."""
        if is_compressed:
            try:
                import zlib
                decompressed = zlib.decompress(data).decode()
                return decompressed
            except:
                return data.decode() if isinstance(data, bytes) else str(data)
        
        return data.decode() if isinstance(data, bytes) else str(data)

class CachePersistence:
    """Cache persistence to disk for recovery across restarts."""
    
    def __init__(self, cache_file: str = "cache_persistence.json"):
        self.cache_file = Path(cache_file)
        self.cache_data = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache_data = json.load(f)
                print(f"✅ Loaded {len(self.cache_data)} cache entries from disk")
            except Exception as e:
                print(f"⚠️ Failed to load cache from disk: {e}")
    
    def save_cache(self, cache_dict: Dict[str, Any]):
        """Save cache to disk."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(cache_dict, f)
            print(f"✅ Saved {len(cache_dict)} cache entries to disk")
        except Exception as e:
            print(f"⚠️ Failed to save cache to disk: {e}")
    
    def get_persistent_cache(self) -> Dict[str, Any]:
        """Get persistent cache data."""
        return self.cache_data

# Global instances
_redis_cache = None
_cache_compression = None
_cache_persistence = None

def get_redis_cache() -> RedisCacheIntegration:
    """Get global Redis cache instance."""
    global _redis_cache
    if _redis_cache is None:
        _redis_cache = RedisCacheIntegration()
    return _redis_cache

def get_cache_compression() -> CacheCompression:
    """Get global cache compression instance."""
    global _cache_compression
    if _cache_compression is None:
        _cache_compression = CacheCompression()
    return _cache_compression

def get_cache_persistence() -> CachePersistence:
    """Get global cache persistence instance."""
    global _cache_persistence
    if _cache_persistence is None:
        _cache_persistence = CachePersistence()
    return _cache_persistence

# Integration with existing gateway
if __name__ == "__main__":
    redis_cache = get_redis_cache()
    cache_compression = get_cache_compression()
    cache_persistence = get_cache_persistence()
    
    # Test Redis cache
    redis_cache.set("test_key", {"data": "test_value"}, ttl=60)
    result = redis_cache.get("test_key")
    print(f"Redis cache test: {result}")
    
    # Test compression
    large_data = {"data": "x" * 2000}
    compressed, is_compressed = cache_compression.compress_if_needed(large_data)
    print(f"Compression test: {len(compressed)} bytes, compressed: {is_compressed}")
    
    # Test persistence
    test_cache = {"key1": "value1", "key2": "value2"}
    cache_persistence.save_cache(test_cache)
    loaded_cache = cache_persistence.get_persistent_cache()
    print(f"Persistence test: {loaded_cache}")