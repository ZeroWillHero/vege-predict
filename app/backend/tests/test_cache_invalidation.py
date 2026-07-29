"""Regression test for a real bug hit during development: seed.py used to reseed Postgres
without touching Redis, so a retrain's new data (e.g. newly added prediction-interval
columns) stayed invisible behind the 7-day TTL until it expired on its own. Fixed by having
seed.py call invalidate_vegetable() after every successful commit — this test guards that
the invalidation helper itself actually clears what a real request populates."""

from app.backend.cache import get_cached
from app.backend.services.cache_service import invalidate_all, invalidate_vegetable, models_key, prediction_key


async def test_invalidate_vegetable_clears_cached_entries(client):
    await client.get("/models", params={"vegetable": "carrot"})
    await client.get("/predictions/carrot", params={"model": "best"})

    assert await get_cached(models_key("carrot")) is not None
    assert await get_cached(prediction_key("carrot", "best")) is not None

    await invalidate_vegetable("carrot")

    assert await get_cached(models_key("carrot")) is None
    assert await get_cached(prediction_key("carrot", "best")) is None

    # the endpoint still works after invalidation — it just repopulates from Postgres
    resp = await client.get("/models", params={"vegetable": "carrot"})
    assert resp.status_code == 200


async def test_invalidate_all_clears_unfiltered_list_cache(client):
    """Regression test for a real bug hit while adding a new model family:
    invalidate_vegetable() only clears keys prefixed by one vegetable's name, which never
    matches models_key(None) -> "models:all" — so GET /models with no vegetable filter kept
    serving a stale, pre-retrain row count after a full reseed until invalidate_all() was
    added and wired into seed.py's main()."""
    resp = await client.get("/models")
    assert resp.status_code == 200
    assert await get_cached(models_key(None)) is not None

    await invalidate_all()

    assert await get_cached(models_key(None)) is None

    resp = await client.get("/models")
    assert resp.status_code == 200
