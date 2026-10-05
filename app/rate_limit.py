"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

import time
import logging
from typing import Dict, Tuple
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""
    tokens: float
    last_update: float
    max_tokens: float
    refill_rate: float  # tokens per second


class RateLimiter:
    """
    In-memory token bucket rate limiter to prevent abuse.
    Limits requests per session_id.
    """
    
    def __init__(self, requests_per_minute: int = 30):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests allowed per minute per session
        """
        self.requests_per_minute = requests_per_minute
        self.max_tokens = requests_per_minute
        self.refill_rate = requests_per_minute / 60.0  # tokens per second
        self.buckets: Dict[str, TokenBucket] = {}
        self._lock = threading.Lock()
    
    def _get_bucket(self, key: str) -> TokenBucket:
        """Get or create a token bucket for the given key."""
        if key not in self.buckets:
            self.buckets[key] = TokenBucket(
                tokens=self.max_tokens,
                last_update=time.time(),
                max_tokens=self.max_tokens,
                refill_rate=self.refill_rate
            )
        return self.buckets[key]
    
    def _refill_bucket(self, bucket: TokenBucket):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - bucket.last_update
        bucket.tokens = min(
            bucket.max_tokens,
            bucket.tokens + elapsed * bucket.refill_rate
        )
        bucket.last_update = now
    
    def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        Check if a request is allowed for the given key.
        
        Args:
            key: Session ID or other identifier
            
        Returns:
            Tuple of (allowed: bool, retry_after_seconds: int)
        """
        with self._lock:
            bucket = self._get_bucket(key)
            self._refill_bucket(bucket)
            
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0
            else:
                # Calculate how long to wait for 1 token
                retry_after = int((1.0 - bucket.tokens) / bucket.refill_rate) + 1
                return False, retry_after
    
    def reset(self, key: str):
        """Reset the token bucket for a specific key."""
        with self._lock:
            if key in self.buckets:
                del self.buckets[key]
    
    def get_remaining_tokens(self, key: str) -> int:
        """Get remaining tokens for a key (for monitoring)."""
        with self._lock:
            bucket = self._get_bucket(key)
            self._refill_bucket(bucket)
            return int(bucket.tokens)
    
    def cleanup_old_buckets(self, max_age_seconds: int = 3600):
        """
        Remove buckets that haven't been used recently.
        
        Args:
            max_age_seconds: Maximum age of inactive buckets to keep
        """
        with self._lock:
            now = time.time()
            keys_to_remove = []
            for key, bucket in self.buckets.items():
                if now - bucket.last_update > max_age_seconds:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.buckets[key]
            
            if keys_to_remove:
                logger.info(f"Cleaned up {len(keys_to_remove)} inactive rate limit buckets")
