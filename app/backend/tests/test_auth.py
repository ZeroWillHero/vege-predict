from datetime import datetime, timedelta, timezone

from jose import jwt

from app.backend.config import settings
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


async def test_me_with_non_numeric_sub_claim_is_401_not_500(client):
    """Regression test: a validly-signed token whose `sub` claim isn't an integer used to
    crash get_current_user's bare int(user_id) with an uncaught ValueError -> 500."""
    expire = datetime.now(timezone.utc) + timedelta(minutes=5)
    token = jwt.encode({"sub": "not-a-number", "role": "farmer", "exp": expire}, settings.jwt_secret, algorithm="HS256")
    resp = await client.get("/auth/me", headers=auth_headers(token))
    assert resp.status_code == 401


async def test_me_with_expired_token(client):
    expired = datetime.now(timezone.utc) - timedelta(minutes=5)
    token = jwt.encode({"sub": "1", "role": "farmer", "exp": expired}, settings.jwt_secret, algorithm="HS256")
    resp = await client.get("/auth/me", headers=auth_headers(token))
    assert resp.status_code == 401


async def test_me_with_wrong_signature(client, farmer_token):
    # Flip a character in the middle of the signature segment, not the last char - base64url's
    # final character of a 32-byte SHA256 digest only encodes 2 significant bits (the rest is
    # padding), so tampering it can round-trip to the same decoded bytes and not actually
    # invalidate the signature.
    mid = len(farmer_token) * 3 // 4
    flipped_char = "A" if farmer_token[mid] != "A" else "B"
    tampered = farmer_token[:mid] + flipped_char + farmer_token[mid + 1 :]
    resp = await client.get("/auth/me", headers=auth_headers(tampered))
    assert resp.status_code == 401


async def test_login_with_oversized_password_is_401_not_500(client, superadmin_user):
    """Regression test: bcrypt raises ValueError on secrets over 72 bytes - verify_password()
    used to let that propagate uncaught -> 500 instead of a normal failed-login 401."""
    _, email, _ = superadmin_user
    resp = await client.post("/auth/login", json={"email": email, "password": "x" * 200})
    assert resp.status_code == 401


async def test_login_missing_fields_is_422(client):
    resp = await client.post("/auth/login", json={"email": "someone@test.vegepredict.com"})
    assert resp.status_code == 422


async def test_login_malformed_email_is_422(client):
    resp = await client.post("/auth/login", json={"email": "not-an-email", "password": "x"})
    assert resp.status_code == 422
