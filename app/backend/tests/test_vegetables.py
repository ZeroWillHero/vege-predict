from app.backend.constants import VEGETABLES


async def test_list_vegetables(client):
    resp = await client.get("/vegetables")
    assert resp.status_code == 200
    assert resp.json() == VEGETABLES


async def test_repeated_request_returns_same_body(client):
    r1 = await client.get("/vegetables")
    r2 = await client.get("/vegetables")
    assert r1.json() == r2.json()


async def test_post_not_allowed(client):
    resp = await client.post("/vegetables")
    assert resp.status_code == 405
