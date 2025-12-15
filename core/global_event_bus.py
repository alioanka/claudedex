#!/usr/bin/env python3
"""
Global Event Bus - Powered by Redis Pub/Sub
Facilitates real-time communication between the DEX, Futures, and Solana modules.
"""

import asyncio
import logging
import json
import redis.asyncio as redis
from typing import Callable, Any, Dict

logger = logging.getLogger(__name__)

class GlobalEventBus:
    """
    A global event bus that uses Redis Pub/Sub to enable inter-module communication.
    """

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_conn = None
        self.pubsub = None
        self.listeners: Dict[str, List[Callable]] = {}

    async def connect(self):
        """
        Establishes the connection to Redis and sets up the Pub/Sub listener.
        """
        try:
            self.redis_conn = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_conn.ping()
            self.pubsub = self.redis_conn.pubsub()
            logger.info("✅ Global Event Bus connected to Redis.")
            asyncio.create_task(self._listen())
        except Exception as e:
            logger.error(f"❌ Failed to connect Global Event Bus to Redis: {e}")
            self.redis_conn = None

    async def _listen(self):
        """
        The main listener loop that waits for messages from subscribed channels.
        """
        if not self.pubsub:
            return

        while True:
            try:
                message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message:
                    channel = message['channel']
                    try:
                        data = json.loads(message['data'])
                        if channel in self.listeners:
                            for callback in self.listeners[channel]:
                                asyncio.create_task(callback(data))
                    except json.JSONDecodeError:
                        logger.warning(f"Received non-JSON message on channel {channel}: {message['data']}")
            except asyncio.CancelledError:
                logger.info("Event bus listener task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in Global Event Bus listener: {e}")
                await asyncio.sleep(5) # Avoid rapid-fire errors

    async def subscribe(self, channel: str, callback: Callable):
        """
        Subscribe to a channel to receive events.

        Args:
            channel (str): The name of the channel to subscribe to.
            callback (Callable): An async function to be called when a message is received.
        """
        if channel not in self.listeners:
            self.listeners[channel] = []
            if self.pubsub:
                await self.pubsub.subscribe(channel)
                logger.info(f"Subscribed to global event channel: {channel}")

        self.listeners[channel].append(callback)

    async def publish(self, channel: str, data: Dict[str, Any]):
        """
        Publish an event to a specific channel.

        Args:
            channel (str): The channel to publish the message to.
            data (Dict[str, Any]): The event data, which must be JSON-serializable.
        """
        if not self.redis_conn:
            logger.error("Cannot publish event: Not connected to Redis.")
            return

        try:
            message = json.dumps(data)
            await self.redis_conn.publish(channel, message)
        except TypeError:
            logger.error(f"Failed to publish event on channel '{channel}': Data is not JSON-serializable.")
        except Exception as e:
            logger.error(f"Error publishing to Redis channel '{channel}': {e}")

    async def close(self):
        """
        Gracefully closes the Redis connection.
        """
        if self.pubsub:
            await self.pubsub.close()
        if self.redis_conn:
            await self.redis_conn.close()
        logger.info("Global Event Bus connection closed.")
