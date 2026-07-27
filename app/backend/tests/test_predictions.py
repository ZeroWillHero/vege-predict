async def test_latest_prediction_best(client):
    resp = await client.get("/predictions/carrot", params={"model": "best"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["vegetable"] == "carrot"
    assert body["model_family"] == "random_forest"


async def test_latest_prediction_has_sane_interval(client):
    resp = await client.get("/predictions/carrot", params={"model": "best"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_lower"] is not None
    assert body["predicted_upper"] is not None
    assert body["predicted_lower"] <= body["predicted_price"] <= body["predicted_upper"]


async def test_list_predictions_interval_bounds_are_consistent(client):
    resp = await client.get("/predictions", params={"vegetable": "carrot", "limit": 10})
    assert resp.status_code == 200
    for row in resp.json():
        assert row["predicted_lower"] <= row["predicted_price"] <= row["predicted_upper"]


async def test_latest_prediction_specific_model(client):
    resp = await client.get("/predictions/carrot", params={"model": "sarimax"})
    assert resp.status_code == 200
    assert resp.json()["model_family"] == "sarimax"


async def test_latest_prediction_unknown_vegetable(client):
    resp = await client.get("/predictions/durian")
    assert resp.status_code == 404


async def test_list_predictions_filtered(client):
    resp = await client.get("/predictions", params={"vegetable": "carrot", "year": 2026, "limit": 3})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    assert all(row["vegetable"] == "carrot" for row in body)
    assert all(row["forecast_date"].startswith("2026") for row in body)


async def test_list_predictions_pagination(client):
    resp1 = await client.get("/predictions", params={"vegetable": "carrot", "limit": 5, "offset": 0})
    resp2 = await client.get("/predictions", params={"vegetable": "carrot", "limit": 5, "offset": 5})
    assert resp1.status_code == 200
    assert resp2.status_code == 200
    ids1 = {(r["model_family"], r["forecast_date"]) for r in resp1.json()}
    ids2 = {(r["model_family"], r["forecast_date"]) for r in resp2.json()}
    assert ids1.isdisjoint(ids2)


async def test_latest_prediction_unknown_model_is_404_not_500(client):
    resp = await client.get("/predictions/carrot", params={"model": "not_a_real_model"})
    assert resp.status_code == 404


async def test_latest_prediction_case_sensitive_vegetable(client):
    resp = await client.get("/predictions/CARROT")
    assert resp.status_code == 404


async def test_list_predictions_unknown_vegetable(client):
    resp = await client.get("/predictions", params={"vegetable": "durian"})
    assert resp.status_code == 404


async def test_list_predictions_unknown_model_returns_empty_not_error(client):
    # unlike the vegetable filter, an unrecognized model family is not validated —
    # it simply matches zero rows
    resp = await client.get("/predictions", params={"vegetable": "carrot", "model": "not_a_real_model"})
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_predictions_nonexistent_year_returns_empty(client):
    resp = await client.get("/predictions", params={"vegetable": "carrot", "year": 1900})
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_predictions_limit_zero_rejected(client):
    resp = await client.get("/predictions", params={"limit": 0})
    assert resp.status_code == 422


async def test_list_predictions_limit_too_large_rejected(client):
    resp = await client.get("/predictions", params={"limit": 1001})
    assert resp.status_code == 422


async def test_list_predictions_negative_offset_rejected(client):
    resp = await client.get("/predictions", params={"offset": -1})
    assert resp.status_code == 422


async def test_best_model_matches_lowest_rmse_in_models_endpoint(client):
    models_resp = await client.get("/models", params={"vegetable": "brinjal"})
    assert models_resp.status_code == 200
    lowest_rmse_family = models_resp.json()[0]["model_family"]

    best_resp = await client.get("/predictions/brinjal", params={"model": "best"})
    assert best_resp.status_code == 200
    assert best_resp.json()["model_family"] == lowest_rmse_family
