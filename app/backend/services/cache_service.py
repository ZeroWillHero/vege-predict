"""Cache-key conventions for the read endpoints, built on top of cache.py's generic
get/set/invalidate_prefix. Keeps key formats in one place so routers don't hand-roll
them and invalidation can target exactly what it means to."""

from datetime import date

from app.backend.cache import invalidate_prefix

PREFIX = "vegepredict"


def vegetables_key() -> str:
    return f"{PREFIX}:vegetables"


def models_key(vegetable: str | None) -> str:
    return f"{PREFIX}:models:{vegetable or 'all'}"


def prices_key(vegetable: str, start_date: date | None, end_date: date | None, limit: int, offset: int) -> str:
    return f"{PREFIX}:prices:{vegetable}:{start_date}:{end_date}:{limit}:{offset}"


def prediction_key(vegetable: str, model: str) -> str:
    return f"{PREFIX}:predictions:{vegetable}:{model}"


def future_predictions_key(vegetable: str, model: str) -> str:
    return f"{PREFIX}:predictions:future:{vegetable}:{model}"


def predictions_list_key(
    vegetable: str | None, model: str | None, year: int | None, limit: int, offset: int
) -> str:
    return f"{PREFIX}:predictions:list:{vegetable}:{model}:{year}:{limit}:{offset}"


async def invalidate_vegetable(vegetable: str) -> None:
    """Call after a successful retrain promote for this vegetable — not yet wired up,
    since auto_retrain.py doesn't exist yet (see CLAUDE.md's auto-retrain pipeline notes)."""
    await invalidate_prefix(f"{PREFIX}:models:{vegetable}")
    await invalidate_prefix(f"{PREFIX}:prices:{vegetable}")
    await invalidate_prefix(f"{PREFIX}:predictions:{vegetable}")
    await invalidate_prefix(f"{PREFIX}:predictions:list:{vegetable}")
    await invalidate_prefix(f"{PREFIX}:predictions:future:{vegetable}")


async def invalidate_all() -> None:
    """Call once after a full reseed touching every vegetable (e.g. seed.py's main()).
    invalidate_vegetable() only clears keys prefixed by one vegetable's name, which never
    matches the unfiltered list-endpoint caches (models_key(None) -> "models:all",
    predictions_list_key(None, ...) -> "predictions:list:None:..."), so those would
    otherwise keep serving stale JSON for up to the cache TTL after a full reseed."""
    await invalidate_prefix(f"{PREFIX}:models:all")
    await invalidate_prefix(f"{PREFIX}:predictions:list:None")
