"""
Clear the cache to remove old benchmark data
"""

from gateway.simple_cache import get_cache

cache = get_cache()
cache.clear_benchmark_data()
cache.clear()

print("Cache cleared successfully")