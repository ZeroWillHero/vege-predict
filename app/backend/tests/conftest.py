"""Router test fixtures. Hits the real vegepredict Postgres/Redis (already seeded
by seed.py) rather than mocking infra — this is a research-project integration
suite exercising the actual read paths, not a unit-test suite with test doubles."""

import pytest
from httpx import ASGITransport, AsyncClient

from app.backend.main import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
