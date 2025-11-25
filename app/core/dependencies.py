from typing import Optional
from redis import asyncio as aioredis
from app.core.config import get_settings

settings = get_settings()

# Global Singleton Instances
# We use global variables to ensure only one connection/service instance exists
# throughout the application lifecycle.
_rag_service: Optional["RAGService"] = None
_redis_pool: Optional[aioredis.Redis] = None

def get_rag_service() -> "RAGService":
    """
    Singleton provider for RAGService.
    
    Note:
        We use a LAZY IMPORT (import inside the function) here.
        This is critical to prevent a circular dependency error, as
        rag_service.py imports get_redis from this file.
    
    Returns:
        RAGService: The initialized service instance.
    """
    global _rag_service
    if _rag_service is None:
        from app.services.rag_service import RAGService
        _rag_service = RAGService()
    return _rag_service

async def get_redis() -> aioredis.Redis:
    """
    Singleton provider for the Redis Connection Pool.
    
    Configured with decode_responses=True to ensure we get 
    Python strings back, not bytes.

    Returns:
        aioredis.Redis: The active Redis connection pool.
    """
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(
            settings.REDIS_URL, 
            encoding="utf8", 
            decode_responses=True
        )
    return _redis_pool