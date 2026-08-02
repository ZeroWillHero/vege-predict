from app.backend.tests.conftest import auth_headers


async def test_login_success(client, superadmin_user):
    _, email, password = superadmin_user
    resp = await client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] > 0
    assert body["access_token"]


async def test_login_wrong_password(client, superadmin_user):
    _, email, _ = superadmin_user
    resp = await client.post("/auth/login", json={"email": email, "password": "wrong-password"})
    assert resp.status_code == 401


async def test_login_nonexistent_email(client):
    resp = await client.post("/auth/login", json={"email": "nobody@test.vegepredict.com", "password": "x"})
    assert resp.status_code == 401


async def test_login_deactivated_account_fails(client, farmer_user, superadmin_token):
    farmer_id, email, password = farmer_user
    resp = await client.patch(
        f"/users/{farmer_id}", json={"is_active": False}, headers=auth_headers(superadmin_token)
    )
    assert resp.status_code == 200

    resp = await client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 401


async def test_me_with_valid_token(client, farmer_token, farmer_user):
    _, email, _ = farmer_user
    resp = await client.get("/auth/me", headers=auth_headers(farmer_token))
    assert resp.status_code == 200
    assert resp.json()["email"] == email
    assert resp.json()["role"] == "farmer"


async def test_me_without_token(client):
    resp = await client.get("/auth/me")
    assert resp.status_code == 401


async def test_me_with_garbage_token(client):
    resp = await client.get("/auth/me", headers=auth_headers("not-a-real-token"))
    assert resp.status_code == 401
