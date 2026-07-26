async def test_list_models_all(client):
    resp = await client.get("/models")
    assert resp.status_code == 200
    assert len(resp.json()) == 42


async def test_list_models_filtered(client):
    resp = await client.get("/models", params={"vegetable": "carrot"})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 7
    assert all(row["vegetable"] == "carrot" for row in body)
    rmses = [row["rmse"] for row in body]
    assert rmses == sorted(rmses)


async def test_list_models_interval_fields_are_sane(client):
    resp = await client.get("/models", params={"vegetable": "carrot"})
    assert resp.status_code == 200
    for row in resp.json():
        assert row["interval_confidence"] == 0.8
        assert 0.0 <= row["interval_coverage"] <= 1.0


async def test_list_models_unknown_vegetable(client):
    resp = await client.get("/models", params={"vegetable": "durian"})
    assert resp.status_code == 404


async def test_list_models_case_sensitive(client):
    # vegetable identifiers are lowercase-only; a mismatched case is treated as unknown
    resp = await client.get("/models", params={"vegetable": "Carrot"})
    assert resp.status_code == 404


async def test_list_models_empty_string_vegetable(client):
    resp = await client.get("/models", params={"vegetable": ""})
    assert resp.status_code == 404


async def test_list_models_injection_like_vegetable_is_safe(client):
    resp = await client.get("/models", params={"vegetable": "carrot'; DROP TABLE model_metric;--"})
    assert resp.status_code == 404
    # the table must still be intact and queryable afterwards
    follow_up = await client.get("/models", params={"vegetable": "carrot"})
    assert follow_up.status_code == 200
    assert len(follow_up.json()) == 7
