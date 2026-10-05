"""
Cloud AI Chatbot - Enterprise Edition
Copyright (c) 2026. All rights reserved.

This software is proprietary and confidential. Unauthorized use,
distribution, or modification is strictly prohibited.

For licensing information, contact the development team.
"""

"""
Shared HTTP Client Factory with Connection Pooling and DNS Caching
Provides a singleton httpx.AsyncClient for reuse across all cloud engines
"""
import os
import asyncio
from typing import Optional
import httpx
import logging

logger = logging.getLogger(__name__)


class SharedHTTPClient:
    """Singleton factory for shared httpx.AsyncClient with connection pooling"""
    
    _instance: Optional['SharedHTTPClient'] = None
    _client: Optional[httpx.AsyncClient] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    async def get_client(cls) -> httpx.AsyncClient:
        """Get or create the shared async HTTP client"""
        if cls._client is not None and not cls._client.is_closed:
            return cls._client
        
        async with cls._lock:
            # Double-check after acquiring lock
            if cls._client is not None and not cls._client.is_closed:
                return cls._client
            
            # Create new client with connection pooling
            max_connections = int(os.environ.get('FASTCLOUD_HTTP_POOL_SIZE', '50'))
            keepalive_expiry = int(os.environ.get('FASTCLOUD_KEEPALIVE_SEC', '30'))
            
            limits = httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_connections,
                keepalive_expiry=keepalive_expiry
            )
            
            # DNS caching via http2 and connection reuse
            cls._client = httpx.AsyncClient(
                limits=limits,
                timeout=httpx.Timeout(30.0),
                http2=True,  # Enable HTTP/2 for multiplexing
                verify=True,  # SSL verification
            )
            
            logger.info(
                f"SharedHTTPClient created: max_connections={max_connections}, "
                f"keepalive_expiry={keepalive_expiry}s, http2=True"
            )
            
            return cls._client
    
    @classmethod
    async def close(cls) -> None:
        """Close the shared HTTP client"""
        if cls._client is not None and not cls._client.is_closed:
            await cls._client.aclose()
            cls._client = None
            logger.info("SharedHTTPClient closed")


def get_shared_client() -> httpx.AsyncClient:
    """
    Synchronous wrapper for getting the shared client.
    Note: This should be called from an async context.
    """
    return SharedHTTPClient.get_client()
