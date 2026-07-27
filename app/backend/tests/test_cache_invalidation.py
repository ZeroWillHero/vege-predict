"""Regression test for a real bug hit during development: seed.py used to reseed Postgres
without touching Redis, so a retrain's new data (e.g. newly added prediction-interval
columns) stayed invisible behind the 7-day TTL until it expired on its own. Fixed by having
seed.py call invalidate_vegetable() after every successful commit — this test guards that
the invalidation helper itself actually clears what a real request populates."""

from app.backend.cache import get_cached
from app.backend.services.cache_service import invalidate_vegetable, models_key, prediction_key


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
