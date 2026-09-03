"""
Simple Response Cache
Caches common questions to improve speed for repeated queries
"""

import hashlib
import json
import logging
import os
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class SimpleCache:
    """Simple file-based cache for common questions."""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "response_cache.json"
        self.cache = self._load_cache()
    
    def _load_cache(self) -> Dict:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)
        except:
            pass
    
    def _get_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[str]:
        """Get cached response if available."""
        key = self._get_key(query)
        return self.cache.get(key)
    
    def set(self, query: str, response: str):
        """Cache a response."""
        key = self._get_key(query)
        self.cache[key] = response
        self._save_cache()
    
    def clear(self):
        """Clear all cached responses."""
        self.cache = {}
        self._save_cache()
    
    def clear_benchmark_data(self):
        """Clear any old benchmark/test data from cache."""
        # Clear entries that look like benchmark responses (very long)
        keys_to_remove = []
        for key, value in self.cache.items():
            if len(value) > 500:  # Likely benchmark data
                keys_to_remove.append(key)
        for key in keys_to_remove:
            del self.cache[key]
        if keys_to_remove:
            self._save_cache()
            logger.info(f"Cleared {len(keys_to_remove)} benchmark cache entries")
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "hits": len(self.cache),
            "size": len(self.cache),
        }


def get_cache() -> SimpleCache:
    """Get cache instance."""
    return SimpleCache()
