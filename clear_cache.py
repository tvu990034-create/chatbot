"""
Clear the cache to remove old benchmark data
"""

from gateway.simple_cache import get_cache
from gateway.opt_core import clear_model_availability_cache

cache = get_cache()
cache.clear_benchmark_data()
cache.clear()
clear_model_availability_cache()

print("Cache cleared successfully")