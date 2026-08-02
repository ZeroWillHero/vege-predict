from app.backend.tests.conftest import auth_headers


def _payload(role: str, suffix: str) -> dict:
    return {
        "first_name": "New",
        "last_name": "User",
        "email": f"new-{suffix}@test.vegepredict.com",
        "password": "NewPass123",
        "role": role,
    }


async def test_superadmin_can_create_any_role(client, superadmin_token):
    for role in ("farmer", "admin", "superadmin"):
        resp = await client.post("/users", json=_payload(role, role), headers=auth_headers(superadmin_token))
        assert resp.status_code == 201, resp.text
        assert resp.json()["role"] == role


async def test_admin_can_create_farmer(client, admin_token):
    resp = await client.post("/users", json=_payload("farmer", "af"), headers=auth_headers(admin_token))
    assert resp.status_code == 201


async def test_admin_cannot_create_admin(client, admin_token):
    resp = await client.post("/users", json=_payload("admin", "aa"), headers=auth_headers(admin_token))
    assert resp.status_code == 403


async def test_admin_cannot_create_superadmin(client, admin_token):
    resp = await client.post("/users", json=_payload("superadmin", "as"), headers=auth_headers(admin_token))
    assert resp.status_code == 403


async def test_create_user_duplicate_email_conflicts(client, superadmin_token, farmer_user):
    _, email, _ = farmer_user
    payload = _payload("farmer", "dup")
    payload["email"] = email
    resp = await client.post("/users", json=payload, headers=auth_headers(superadmin_token))
    assert resp.status_code == 409


async def test_farmer_cannot_create_users(client, farmer_token):
    resp = await client.post("/users", json=_payload("farmer", "fc"), headers=auth_headers(farmer_token))
    assert resp.status_code == 403


async def test_create_user_requires_auth(client):
    resp = await client.post("/users", json=_payload("farmer", "noauth"))
    assert resp.status_code == 401


async def test_superadmin_sees_all_users(client, superadmin_token, admin_user, farmer_user):
    resp = await client.get("/users", headers=auth_headers(superadmin_token))
    assert resp.status_code == 200
    ids = {u["id"] for u in resp.json()}
    assert admin_user[0] in ids
    assert farmer_user[0] in ids


async def test_admin_sees_only_farmers_and_self(client, admin_token, admin_user, farmer_user, superadmin_user):
    resp = await client.get("/users", headers=auth_headers(admin_token))
    assert resp.status_code == 200
    ids = {u["id"] for u in resp.json()}
    assert admin_user[0] in ids  # self
    assert farmer_user[0] in ids
    assert superadmin_user[0] not in ids


async def test_farmer_cannot_list_users(client, farmer_token):
    resp = await client.get("/users", headers=auth_headers(farmer_token))
    assert resp.status_code == 403


async def test_admin_get_other_admin_returns_404_not_403(client, admin_token, superadmin_user):
    target_id, _, _ = superadmin_user
    resp = await client.get(f"/users/{target_id}", headers=auth_headers(admin_token))
    assert resp.status_code == 404


async def test_admin_can_get_farmer(client, admin_token, farmer_user):
    target_id, _, _ = farmer_user
    resp = await client.get(f"/users/{target_id}", headers=auth_headers(admin_token))
    assert resp.status_code == 200


async def test_get_nonexistent_user_returns_404(client, superadmin_token):
    resp = await client.get("/users/999999999", headers=auth_headers(superadmin_token))
    assert resp.status_code == 404


async def test_farmer_can_edit_own_name(client, farmer_token, farmer_user):
    target_id, _, _ = farmer_user
    resp = await client.patch(
        f"/users/{target_id}", json={"first_name": "Updated"}, headers=auth_headers(farmer_token)
    )
    assert resp.status_code == 200
    assert resp.json()["first_name"] == "Updated"


async def test_farmer_cannot_change_own_role(client, farmer_token, farmer_user):
    target_id, _, _ = farmer_user
    resp = await client.patch(
        f"/users/{target_id}", json={"role": "admin"}, headers=auth_headers(farmer_token)
    )
    assert resp.status_code == 403


async def test_farmer_cannot_deactivate_self(client, farmer_token, farmer_user):
    target_id, _, _ = farmer_user
    resp = await client.patch(
        f"/users/{target_id}", json={"is_active": False}, headers=auth_headers(farmer_token)
    )
    assert resp.status_code == 403


async def test_admin_cannot_edit_another_admin(client, admin_token, superadmin_token):
    # Create a second admin via superadmin, then have the first admin try to edit them.
    resp = await client.post(
        "/users", json=_payload("admin", "second"), headers=auth_headers(superadmin_token)
    )
    other_admin_id = resp.json()["id"]
    resp = await client.patch(
        f"/users/{other_admin_id}", json={"first_name": "Hacked"}, headers=auth_headers(admin_token)
    )
    assert resp.status_code in (403, 404)


async def test_admin_can_deactivate_farmer(client, admin_token, farmer_user):
    target_id, _, _ = farmer_user
    resp = await client.patch(
        f"/users/{target_id}", json={"is_active": False}, headers=auth_headers(admin_token)
    )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False


async def test_admin_cannot_change_role(client, admin_token, farmer_user):
    target_id, _, _ = farmer_user
    resp = await client.patch(
        f"/users/{target_id}", json={"role": "admin"}, headers=auth_headers(admin_token)
    )
    assert resp.status_code == 403


async def test_superadmin_can_promote_farmer_to_admin(client, superadmin_token, farmer_user):
    target_id, _, _ = farmer_user
    resp = await client.patch(
        f"/users/{target_id}", json={"role": "admin"}, headers=auth_headers(superadmin_token)
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "admin"


async def test_update_nonexistent_user_returns_404(client, superadmin_token):
    resp = await client.patch(
        "/users/999999999", json={"first_name": "X"}, headers=auth_headers(superadmin_token)
    )
    assert resp.status_code == 404
