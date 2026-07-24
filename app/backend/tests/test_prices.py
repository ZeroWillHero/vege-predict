async def test_price_history(client):
    resp = await client.get(
        "/prices/carrot/history",
        params={"start_date": "2020-01-01", "end_date": "2020-12-31"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) > 0
    assert all(row["vegetable"] == "carrot" for row in body)
    assert all("2020-01-01" <= row["week_start"] <= "2020-12-31" for row in body)


async def test_price_history_unknown_vegetable(client):
    resp = await client.get("/prices/durian/history")
    assert resp.status_code == 404


async def test_price_history_limit(client):
    resp = await client.get("/prices/carrot/history", params={"limit": 5})
    assert resp.status_code == 200
    assert len(resp.json()) == 5


async def test_price_history_default_order_ascending(client):
    resp = await client.get("/prices/carrot/history", params={"limit": 20})
    assert resp.status_code == 200
    dates = [row["week_start"] for row in resp.json()]
    assert dates == sorted(dates)


async def test_price_history_start_after_end_returns_empty(client):
    resp = await client.get(
        "/prices/carrot/history",
        params={"start_date": "2020-12-31", "end_date": "2020-01-01"},
    )
    assert resp.status_code == 200
    assert resp.json() == []


async def test_price_history_offset_beyond_data_returns_empty(client):
    resp = await client.get("/prices/carrot/history", params={"offset": 1_000_000})
    assert resp.status_code == 200
    assert resp.json() == []


async def test_price_history_invalid_date_format(client):
    resp = await client.get("/prices/carrot/history", params={"start_date": "not-a-date"})
    assert resp.status_code == 422


async def test_price_history_limit_zero_rejected(client):
    resp = await client.get("/prices/carrot/history", params={"limit": 0})
    assert resp.status_code == 422


async def test_price_history_limit_too_large_rejected(client):
    resp = await client.get("/prices/carrot/history", params={"limit": 1001})
    assert resp.status_code == 422


async def test_price_history_negative_offset_rejected(client):
    resp = await client.get("/prices/carrot/history", params={"offset": -1})
    assert resp.status_code == 422


async def test_price_history_pagination_no_overlap(client):
    resp1 = await client.get("/prices/carrot/history", params={"limit": 10, "offset": 0})
    resp2 = await client.get("/prices/carrot/history", params={"limit": 10, "offset": 10})
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    weeks1 = {row["week_start"] for row in resp1.json()}
    weeks2 = {row["week_start"] for row in resp2.json()}
    assert weeks1.isdisjoint(weeks2)
