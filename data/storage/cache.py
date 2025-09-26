# data/storage/cache.py

import asyncio
import logging
import pickle
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import orjson

import redis.asyncio as redis
from redis.asyncio.client import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Redis cache manager for high-performance data caching.
    Handles all caching operations for the trading bot.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client: Optional[Redis] = None
        self.is_connected = False
        
        # Cache TTL settings (in seconds)
        self.ttl_settings = {
            'market_data': 30,           # 30 seconds for real-time data
            'token_analysis': 300,        # 5 minutes for analysis results
            'honeypot_check': 600,        # 10 minutes for honeypot results
            'whale_data': 60,             # 1 minute for whale tracking
            'position_data': 10,          # 10 seconds for position updates
            'historical_data': 1800,      # 30 minutes for historical data
            'performance_metrics': 300,   # 5 minutes for performance data
            'token_metadata': 3600,       # 1 hour for token metadata
            'dex_prices': 5,              # 5 seconds for DEX prices
            'gas_prices': 15,             # 15 seconds for gas prices
        }
        
        # Pub/Sub channels
        self.channels = {
            'new_trades': 'bot:trades:new',
            'position_updates': 'bot:positions:update',
            'alerts': 'bot:alerts',
            'market_updates': 'bot:market:update',
            'signals': 'bot:signals',
        }
    
    async def connect(self) -> None:
        """Establish connection to Redis server."""
        try:
            self.redis_client = await redis.from_url(
                f"redis://{self.config.get('REDIS_HOST', 'localhost')}:"
                f"{self.config.get('REDIS_PORT', 6379)}/"
                f"{self.config.get('REDIS_DB', 0)}",
                password=self.config.get('REDIS_PASSWORD'),
                encoding='utf-8',
                decode_responses=False,  # Handle decoding ourselves
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 3,  # TCP_KEEPINTVL
                    3: 5,  # TCP_KEEPCNT
                },
                max_connections=self.config.get('REDIS_MAX_CONNECTIONS', 50),
                health_check_interval=30,
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Set up key expiration notifications
            await self._setup_keyspace_notifications()
            
            self.is_connected = True
            logger.info("Successfully connected to Redis cache")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            logger.info("Disconnected from Redis cache")
    
    async def _setup_keyspace_notifications(self) -> None:
        """Enable keyspace notifications for cache expiration events."""
        try:
            await self.redis_client.config_set('notify-keyspace-events', 'Ex')
        except Exception as e:
            logger.warning(f"Could not set keyspace notifications: {e}")
    
    # Fix get() method signature (line ~98)
    # REPLACE the existing method:

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        try:
            value = await self.redis_client.get(key)
            
            if value is None:
                return None
            
            # Try to decode as JSON first
            try:
                return orjson.loads(value)
            except:
                # Fallback to pickle for complex objects
                try:
                    return pickle.loads(value)
                except:
                    # Final fallback to string
                    return value.decode('utf-8') if isinstance(value, bytes) else value
                    
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None


    # For backward compatibility, add the extended version:
    async def get_with_options(
        self,
        key: str,
        default: Any = None,
        decode_json: bool = True
    ) -> Optional[Any]:
        """
        Get value from cache with options (internal use).
        
        Args:
            key: Cache key
            default: Default value if not found
            decode_json: Whether to attempt JSON decoding
            
        Returns:
            Cached value or default
        """
        try:
            value = await self.redis_client.get(key)
            
            if value is None:
                return default
            
            if decode_json:
                try:
                    return orjson.loads(value)
                except:
                    try:
                        return pickle.loads(value)
                    except:
                        pass
            
            return value.decode('utf-8') if isinstance(value, bytes) else value
                    
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return default
    
    # Replace the existing set() method signature to match the API spec
    # The API expects: async def set(key: str, value: Any, ttl: int = 300) -> None
    # Current implementation returns bool, should be None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 300,  # Changed: Made ttl required with default value
        cache_type: Optional[str] = None  # Keep for backward compatibility
    ) -> None:  # Changed: Return type from bool to None
        """
        Set value in cache with optional TTL.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default 300)
            cache_type: Optional cache type for predefined TTL settings
        """
        try:
            # Override ttl if cache_type is provided
            if cache_type and cache_type in self.ttl_settings:
                ttl = self.ttl_settings.get(cache_type, ttl)
            
            # Serialize value
            if isinstance(value, (dict, list)):
                serialized = orjson.dumps(value)
            elif isinstance(value, (str, int, float)):
                serialized = str(value).encode('utf-8')
            else:
                serialized = pickle.dumps(value)
            
            # Set with TTL (ttl is always provided now)
            if ttl and ttl > 0:
                await self.redis_client.setex(key, ttl, serialized)
            else:
                # If ttl is 0 or negative, set without expiration
                await self.redis_client.set(key, serialized)
            
            # No return value as per API spec
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            # Re-raise the exception to maintain API contract
            raise
    
    async def delete(self, key: Union[str, List[str]]) -> int:
        """Delete key(s) from cache."""
        try:
            if isinstance(key, str):
                return await self.redis_client.delete(key)
            else:
                return await self.redis_client.delete(*key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            return bool(await self.redis_client.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def invalidate(self, pattern: str) -> int:
        """Invalidate all keys matching pattern."""
        try:
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                return await self.redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Cache invalidate error for pattern {pattern}: {e}")
            return 0
    
    async def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        try:
            values = await self.redis_client.mget(keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = orjson.loads(value)
                    except:
                        result[key] = value.decode('utf-8') if isinstance(value, bytes) else value
            
            return result
            
        except Exception as e:
            logger.error(f"Cache get_many error: {e}")
            return {}
    
    async def set_many(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple key-value pairs."""
        try:
            # Serialize values
            serialized = {}
            for key, value in mapping.items():
                if isinstance(value, (dict, list)):
                    serialized[key] = orjson.dumps(value)
                else:
                    serialized[key] = str(value).encode('utf-8')
            
            # Use pipeline for efficiency
            async with self.redis_client.pipeline() as pipe:
                for key, value in serialized.items():
                    if ttl:
                        pipe.setex(key, ttl, value)
                    else:
                        pipe.set(key, value)
                await pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Cache set_many error: {e}")
            return False
    
    # Hash operations for structured data
    async def hget(self, name: str, key: str) -> Optional[Any]:
        """Get field value from hash."""
        try:
            value = await self.redis_client.hget(name, key)
            if value:
                try:
                    return orjson.loads(value)
                except:
                    return value.decode('utf-8') if isinstance(value, bytes) else value
            return None
        except Exception as e:
            logger.error(f"Cache hget error: {e}")
            return None
    
    async def hset(self, name: str, key: str, value: Any) -> bool:
        """Set field value in hash."""
        try:
            if isinstance(value, (dict, list)):
                serialized = orjson.dumps(value)
            else:
                serialized = str(value).encode('utf-8')
            
            await self.redis_client.hset(name, key, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache hset error: {e}")
            return False
    
    async def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all fields from hash."""
        try:
            data = await self.redis_client.hgetall(name)
            result = {}
            
            for key, value in data.items():
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                try:
                    result[key_str] = orjson.loads(value)
                except:
                    result[key_str] = value.decode('utf-8') if isinstance(value, bytes) else value
            
            return result
        except Exception as e:
            logger.error(f"Cache hgetall error: {e}")
            return {}
    
    # List operations for queues
    async def lpush(self, key: str, value: Any) -> int:
        """Push value to the left of list."""
        try:
            serialized = orjson.dumps(value) if isinstance(value, (dict, list)) else str(value)
            return await self.redis_client.lpush(key, serialized)
        except Exception as e:
            logger.error(f"Cache lpush error: {e}")
            return 0
    
    async def rpop(self, key: str) -> Optional[Any]:
        """Pop value from the right of list."""
        try:
            value = await self.redis_client.rpop(key)
            if value:
                try:
                    return orjson.loads(value)
                except:
                    return value.decode('utf-8') if isinstance(value, bytes) else value
            return None
        except Exception as e:
            logger.error(f"Cache rpop error: {e}")
            return None
    
    async def lrange(self, key: str, start: int, stop: int) -> List[Any]:
        """Get range of values from list."""
        try:
            values = await self.redis_client.lrange(key, start, stop)
            result = []
            
            for value in values:
                try:
                    result.append(orjson.loads(value))
                except:
                    result.append(value.decode('utf-8') if isinstance(value, bytes) else value)
            
            return result
        except Exception as e:
            logger.error(f"Cache lrange error: {e}")
            return []
    
    # Sorted set operations for rankings
    async def zadd(self, key: str, mapping: Dict[str, float]) -> int:
        """Add members to sorted set with scores."""
        try:
            return await self.redis_client.zadd(key, mapping)
        except Exception as e:
            logger.error(f"Cache zadd error: {e}")
            return 0
    
    async def zrange(
        self,
        key: str,
        start: int,
        stop: int,
        withscores: bool = False
    ) -> List[Any]:
        """Get range from sorted set."""
        try:
            return await self.redis_client.zrange(key, start, stop, withscores=withscores)
        except Exception as e:
            logger.error(f"Cache zrange error: {e}")
            return []
    
    # Pub/Sub operations
    async def publish(self, channel: str, message: Any) -> int:
        """Publish message to channel."""
        try:
            channel_name = self.channels.get(channel, channel)
            serialized = orjson.dumps(message) if isinstance(message, (dict, list)) else str(message)
            return await self.redis_client.publish(channel_name, serialized)
        except Exception as e:
            logger.error(f"Cache publish error: {e}")
            return 0
    
    async def subscribe(self, *channels) -> 'PubSub':
        """Subscribe to channels."""
        pubsub = self.redis_client.pubsub()
        channel_names = [self.channels.get(ch, ch) for ch in channels]
        await pubsub.subscribe(*channel_names)
        return pubsub
    
    # Atomic operations
    async def incr(self, key: str, amount: int = 1) -> int:
        """Increment value atomically."""
        try:
            return await self.redis_client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Cache incr error: {e}")
            return 0
    
    async def decr(self, key: str, amount: int = 1) -> int:
        """Decrement value atomically."""
        try:
            return await self.redis_client.decrby(key, amount)
        except Exception as e:
            logger.error(f"Cache decr error: {e}")
            return 0
    
    # TTL management
    async def expire(self, key: str, seconds: int) -> bool:
        """Set expiration time for key."""
        try:
            return await self.redis_client.expire(key, seconds)
        except Exception as e:
            logger.error(f"Cache expire error: {e}")
            return False
    
    async def ttl(self, key: str) -> int:
        """Get remaining TTL for key."""
        try:
            return await self.redis_client.ttl(key)
        except Exception as e:
            logger.error(f"Cache ttl error: {e}")
            return -1
    
    # Cache warming
    async def warm_cache(self, data_type: str, data: Any) -> None:
        """Pre-load cache with frequently accessed data."""
        try:
            if data_type == 'active_positions':
                for position in data:
                    key = f"position:{position['position_id']}"
                    await self.set(key, position, cache_type='position_data')
            
            elif data_type == 'token_metadata':
                for token in data:
                    key = f"token:{token['address']}:{token['chain']}"
                    await self.set(key, token, cache_type='token_metadata')
            
            elif data_type == 'market_data':
                for market in data:
                    key = f"market:{market['token_address']}:{market['chain']}"
                    await self.set(key, market, cache_type='market_data')
            
            logger.info(f"Cache warmed with {data_type} data")
            
        except Exception as e:
            logger.error(f"Cache warming error: {e}")
    
    # Performance monitoring
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = await self.redis_client.info()
            
            return {
                'used_memory_mb': info.get('used_memory', 0) / (1024 * 1024),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': (
                    info.get('keyspace_hits', 0) / 
                    max(info.get('keyspace_hits', 0) + info.get('keyspace_misses', 1), 1)
                ) * 100,
                'evicted_keys': info.get('evicted_keys', 0),
                'expired_keys': info.get('expired_keys', 0),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    # Distributed locking
    async def acquire_lock(
        self,
        resource: str,
        timeout: int = 10,
        blocking: bool = True
    ) -> Optional['Lock']:
        """Acquire distributed lock for resource."""
        try:
            lock = self.redis_client.lock(
                f"lock:{resource}",
                timeout=timeout,
                blocking=blocking
            )
            if await lock.acquire():
                return lock
            return None
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return None
    
    async def release_lock(self, lock: 'Lock') -> bool:
        """Release distributed lock."""
        try:
            await lock.release()
            return True
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")
            return False

    # Add this method to the CacheManager class in cache.py

    async def clear(self, pattern: str = "*") -> bool:
        """
        Clear all cache entries or those matching a specific pattern.
        
        Args:
            pattern: Redis key pattern to match (default "*" clears all)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get all keys matching pattern
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                # Delete all matching keys
                deleted_count = await self.redis_client.delete(*keys)
                logger.info(f"Cleared {deleted_count} keys from cache with pattern: {pattern}")
                return True
            else:
                logger.info(f"No keys found matching pattern: {pattern}")
                return True
                
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False