"""
Enhanced caching utility with TTL, Redis support, and invalidation
Provides both in-memory and Redis-based caching
"""

import time
from typing import Any, Optional, Dict
from collections import OrderedDict
import structlog
import json

logger = structlog.get_logger()

# Optional Redis support
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis_not_available", message="Install redis-py for Redis caching support")


class SimpleCache:
    """
    Enhanced in-memory cache with TTL support and LRU eviction
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize cache

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.ttls: Dict[str, int] = {}
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info("cache_initialized", max_size=max_size, default_ttl=default_ttl)

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self.cache:
            self.misses += 1
            logger.debug("cache_miss", key=key)
            return None

        # Check if expired
        if self._is_expired(key):
            logger.debug("cache_expired", key=key)
            self.delete(key)
            self.misses += 1
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(key)
        self.hits += 1
        logger.debug("cache_hit", key=key)
        return self.cache[key]

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = next(iter(self.cache))
            self.delete(oldest_key)
            self.evictions += 1
            logger.debug("cache_evicted", key=oldest_key)

        self.cache[key] = value
        self.timestamps[key] = time.time()
        self.ttls[key] = ttl if ttl is not None else self.default_ttl
        
        # Move to end
        self.cache.move_to_end(key)
        logger.debug("cache_set", key=key, ttl=self.ttls[key])

    def delete(self, key: str):
        """Delete key from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
            del self.ttls[key]
            logger.debug("cache_deleted", key=key)

    def invalidate_pattern(self, pattern: str):
        """
        Invalidate all keys matching pattern
        
        Args:
            pattern: Pattern to match (simple substring match)
        """
        keys_to_delete = [k for k in self.cache.keys() if pattern in k]
        for key in keys_to_delete:
            self.delete(key)
        logger.info("cache_pattern_invalidated", pattern=pattern, count=len(keys_to_delete))

    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.timestamps.clear()
        self.ttls.clear()
        logger.info("cache_cleared")

    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired"""
        if key not in self.timestamps:
            return True
        
        age = time.time() - self.timestamps[key]
        ttl = self.ttls.get(key, self.default_ttl)
        return age > ttl

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size if self.max_size > 0 else 0,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
        }


class RedisCache:
    """
    Redis-based cache for distributed caching
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        default_ttl: int = 3600,
        prefix: str = "endee:",
    ):
        """
        Initialize Redis cache

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            default_ttl: Default TTL in seconds
            prefix: Key prefix for namespacing
        """
        if not REDIS_AVAILABLE:
            raise ImportError("redis-py is required for RedisCache")
        
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.default_ttl = default_ttl
        self.prefix = prefix
        
        # Test connection
        try:
            self.client.ping()
            logger.info("redis_cache_initialized", host=host, port=port)
        except redis.ConnectionError as e:
            logger.error("redis_connection_failed", error=str(e))
            raise

    def _make_key(self, key: str) -> str:
        """Add prefix to key"""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis"""
        try:
            value = self.client.get(self._make_key(key))
            if value:
                logger.debug("redis_cache_hit", key=key)
                return json.loads(value)
            logger.debug("redis_cache_miss", key=key)
            return None
        except Exception as e:
            logger.error("redis_get_failed", key=key, error=str(e))
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis"""
        try:
            ttl = ttl if ttl is not None else self.default_ttl
            self.client.setex(
                self._make_key(key),
                ttl,
                json.dumps(value)
            )
            logger.debug("redis_cache_set", key=key, ttl=ttl)
        except Exception as e:
            logger.error("redis_set_failed", key=key, error=str(e))

    def delete(self, key: str):
        """Delete key from Redis"""
        try:
            self.client.delete(self._make_key(key))
            logger.debug("redis_cache_deleted", key=key)
        except Exception as e:
            logger.error("redis_delete_failed", key=key, error=str(e))

    def invalidate_pattern(self, pattern: str):
        """Invalidate all keys matching pattern"""
        try:
            keys = self.client.keys(f"{self.prefix}*{pattern}*")
            if keys:
                self.client.delete(*keys)
            logger.info("redis_pattern_invalidated", pattern=pattern, count=len(keys))
        except Exception as e:
            logger.error("redis_invalidate_failed", pattern=pattern, error=str(e))

    def clear(self):
        """Clear all keys with prefix"""
        try:
            keys = self.client.keys(f"{self.prefix}*")
            if keys:
                self.client.delete(*keys)
            logger.info("redis_cache_cleared", count=len(keys))
        except Exception as e:
            logger.error("redis_clear_failed", error=str(e))


# Global cache instances
_memory_cache: Optional[SimpleCache] = None
_redis_cache: Optional[RedisCache] = None


def get_cache(use_redis: bool = False) -> SimpleCache:
    """
    Get cache instance
    
    Args:
        use_redis: Use Redis cache if available
        
    Returns:
        Cache instance
    """
    global _memory_cache, _redis_cache
    
    if use_redis and REDIS_AVAILABLE:
        if _redis_cache is None:
            try:
                _redis_cache = RedisCache()
            except Exception as e:
                logger.warning("redis_cache_init_failed", error=str(e))
                # Fallback to memory cache
                if _memory_cache is None:
                    _memory_cache = SimpleCache()
                return _memory_cache
        return _redis_cache
    else:
        if _memory_cache is None:
            _memory_cache = SimpleCache()
        return _memory_cache
