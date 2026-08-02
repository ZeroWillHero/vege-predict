"""Router test fixtures. Hits the real vegepredict Postgres/Redis (already seeded
by seed.py) rather than mocking infra — this is a research-project integration
suite exercising the actual read paths, not a unit-test suite with test doubles."""

import uuid

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete

from app.backend.database import async_session_factory
from app.backend.main import app
from app.backend.models import User
from app.backend.security import hash_password


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


async def _make_user(role: str, password: str = "TestPass123") -> User:
    """Inserts a User row directly via the DB, bypassing the API (there's no
    self-registration to call), with a known plaintext password for login tests."""
    email = f"{role}-{uuid.uuid4().hex[:8]}@test.vegepredict.com"
    async with async_session_factory() as session:
        user = User(
            first_name=role.capitalize(),
            last_name="Tester",
            email=email,
            hashed_password=hash_password(password),
            role=role,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        # detach fields we need after the session closes
        user_id, user_email = user.id, user.email
    return user_id, user_email, password


@pytest.fixture
async def superadmin_user():
    return await _make_user("superadmin")


@pytest.fixture
async def admin_user():
    return await _make_user("admin")


@pytest.fixture
async def farmer_user():
    return await _make_user("farmer")


async def _login(client, email: str, password: str) -> str:
    resp = await client.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture
async def superadmin_token(client, superadmin_user):
    _, email, password = superadmin_user
    return await _login(client, email, password)


@pytest.fixture
async def admin_token(client, admin_user):
    _, email, password = admin_user
    return await _login(client, email, password)


@pytest.fixture
async def farmer_token(client, farmer_user):
    _, email, password = farmer_user
    return await _login(client, email, password)


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(autouse=True)
async def _cleanup_test_users():
    """Removes every user this test session created (matched by the test-only email
    domain) after each test, so test runs stay independent and don't pollute the
    real users table."""
    yield
    async with async_session_factory() as session:
        await session.execute(delete(User).where(User.email.like("%@test.vegepredict.com")))
        await session.commit()
