"""Redis cache-aside helpers: check cache -> miss -> caller fetches from Postgres ->
populate cache. TTL defaults to the retrain cadence (config.cache_ttl_seconds);
invalidate_prefix() exists for auto_retrain.py to call after a successful promote
(not yet wired up — auto_retrain.py doesn't exist yet, see CLAUDE.md)."""

import json

import redis.asyncio as redis

from app.backend.config import settings

redis_client = redis.from_url(settings.redis_url, decode_responses=True)


async def get_cached(key: str):
    value = await redis_client.get(key)
    return json.loads(value) if value is not None else None


async def set_cached(key: str, value, ttl: int | None = None) -> None:
    await redis_client.set(key, json.dumps(value, default=str), ex=ttl or settings.cache_ttl_seconds)


async def invalidate_prefix(prefix: str) -> None:
    """Delete every cached key starting with `prefix` (e.g. a vegetable's forecasts
    after a retrain). Uses SCAN, not KEYS, to avoid blocking Redis on a large keyspace."""
    async for key in redis_client.scan_iter(match=f"{prefix}*"):
        await redis_client.delete(key)


async def ping() -> bool:
    return await redis_client.ping()
