"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

"""
Security and rate limiting features for Cloud AI Chatbot
Provides request throttling, IP-based limiting, and security utilities
"""

import time
import threading
from typing import Dict, Optional, Tuple
from collections import defaultdict
from functools import wraps
import os
import hashlib


class RateLimiter:
    """
    Token bucket rate limiter for API endpoints
    Thread-safe implementation for production use
    """
    
    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self._requests: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
    
    def is_allowed(self, identifier: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a request is allowed based on rate limits
        
        Args:
            identifier: Unique identifier (IP address, session ID, etc.)
            
        Returns:
            Tuple of (allowed: bool, error_message: Optional[str])
        """
        current_time = time.time()
        
        with self._lock:
            # Get request history for this identifier
            requests = self._requests[identifier]
            
            # Remove requests older than 1 hour
            requests = [t for t in requests if current_time - t < 3600]
            self._requests[identifier] = requests
            
            # Check hourly limit
            if len(requests) >= self.requests_per_hour:
                return False, f"Rate limit exceeded: {self.requests_per_hour} requests per hour"
            
            # Check minute limit (last 60 seconds)
            minute_requests = [t for t in requests if current_time - t < 60]
            if len(minute_requests) >= self.requests_per_minute:
                return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"
            
            # Add current request
            requests.append(current_time)
            return True, None
    
    def reset(self, identifier: str):
        """Reset rate limit for a specific identifier"""
        with self._lock:
            if identifier in self._requests:
                del self._requests[identifier]


class SecurityUtils:
    """Security utility functions for input validation and sanitization"""
    
    @staticmethod
    def sanitize_input(text: str, max_length: int = 10000) -> str:
        """
        Sanitize user input to prevent injection attacks
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Truncate to max length
        text = text[:max_length]
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Basic SQL injection prevention (additional layers should be at DB level)
        dangerous_patterns = [
            "DROP TABLE",
            "DELETE FROM",
            "TRUNCATE",
            "EXEC(",
            "EXECUTE(",
            "UNION SELECT",
            "'; DROP",
            "'; DELETE",
        ]
        
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in text_lower:
                # Replace dangerous patterns with safe alternatives
                text = text.replace(pattern, "")
        
        return text
    
    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """
        Validate session ID format
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not session_id:
            return False
        
        # Check length
        if len(session_id) > 256:
            return False
        
        # Check for allowed characters (alphanumeric, hyphens, underscores)
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        return all(c in allowed_chars for c in session_id)
    
    @staticmethod
    def hash_identifier(identifier: str) -> str:
        """
        Hash an identifier for logging/analytics (privacy-preserving)
        
        Args:
            identifier: Identifier to hash
            
        Returns:
            SHA256 hash of the identifier
        """
        return hashlib.sha256(identifier.encode()).hexdigest()
    
    @staticmethod
    def check_content_type(content_type: str, allowed_types: list) -> bool:
        """
        Validate content type against allowed types
        
        Args:
            content_type: Content type header value
            allowed_types: List of allowed content types
            
        Returns:
            True if allowed, False otherwise
        """
        if not content_type:
            return False
        
        content_type_lower = content_type.lower()
        return any(allowed.lower() in content_type_lower for allowed in allowed_types)


class SecurityMiddleware:
    """
    Middleware for security checks and rate limiting
    """
    
    def __init__(self):
        self.rate_limiter = RateLimiter(
            requests_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
            requests_per_hour=int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
        )
        self.security_utils = SecurityUtils()
    
    def check_request(self, identifier: str, message: str, session_id: str) -> Tuple[bool, Optional[str]]:
        """
        Perform comprehensive security checks on a request
        
        Args:
            identifier: Request identifier (IP address, etc.)
            message: User message
            session_id: Session ID
            
        Returns:
            Tuple of (allowed: bool, error_message: Optional[str])
        """
        # Rate limiting check
        allowed, error = self.rate_limiter.is_allowed(identifier)
        if not allowed:
            return False, error
        
        # Input sanitization
        sanitized_message = self.security_utils.sanitize_input(message)
        if len(sanitized_message) != len(message):
            # Message was modified during sanitization
            # For now, we'll allow it but log the difference
            pass
        
        # Session ID validation
        if not self.security_utils.validate_session_id(session_id):
            return False, "Invalid session ID format"
        
        return True, None
    
    def get_rate_limit_info(self, identifier: str) -> Dict[str, int]:
        """
        Get rate limit information for an identifier
        
        Args:
            identifier: Request identifier
            
        Returns:
            Dictionary with rate limit info
        """
        current_time = time.time()
        
        with self.rate_limiter._lock:
            requests = self.rate_limiter._requests.get(identifier, [])
            
            # Count requests in different time windows
            minute_requests = len([t for t in requests if current_time - t < 60])
            hour_requests = len([t for t in requests if current_time - t < 3600])
            
            return {
                "requests_per_minute": minute_requests,
                "requests_per_hour": hour_requests,
                "limit_per_minute": self.rate_limiter.requests_per_minute,
                "limit_per_hour": self.rate_limiter.requests_per_hour,
                "remaining_per_minute": max(0, self.rate_limiter.requests_per_minute - minute_requests),
                "remaining_per_hour": max(0, self.rate_limiter.requests_per_hour - hour_requests),
            }


# Global security middleware instance
_global_security: Optional[SecurityMiddleware] = None


def get_security_middleware() -> SecurityMiddleware:
    """Get the global security middleware instance"""
    global _global_security
    if _global_security is None:
        _global_security = SecurityMiddleware()
    return _global_security


def rate_limit(identifier: str):
    """
    Decorator for rate limiting function calls
    
    Args:
        identifier: Unique identifier for rate limiting
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            security = get_security_middleware()
            allowed, error = security.rate_limiter.is_allowed(identifier)
            if not allowed:
                raise Exception(f"Rate limit exceeded: {error}")
            return func(*args, **kwargs)
        return wrapper
    return decorator
