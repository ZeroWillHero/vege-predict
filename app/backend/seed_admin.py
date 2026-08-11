"""Automatically seeds a default superadmin account on API startup, so a fresh deployment
always has a working admin login without running create_superadmin.py by hand. Idempotent -
safe to run on every startup; never touches the account again once it exists, so changing
the password after first login sticks.

SECURITY NOTE: DEFAULT_ADMIN_PASSWORD below is a hardcoded, committed-to-source-control
default - anyone with repo access has this password. Override it via
VEGEPREDICT_DEFAULT_ADMIN_PASSWORD in any shared/non-local environment, and change the
password (PATCH /users/{id} after logging in) immediately after first login wherever this
account has real access to anything sensitive.
"""

import os

from sqlalchemy import select

from app.backend.database import async_session_factory
from app.backend.models import User
from app.backend.schemas.user import UserRole
from app.backend.security import hash_password

DEFAULT_ADMIN_EMAIL = "admin@vegepredict.com"
DEFAULT_ADMIN_FIRST_NAME = "Super"
DEFAULT_ADMIN_LAST_NAME = "Admin"
DEFAULT_ADMIN_PASSWORD = os.environ.get("VEGEPREDICT_DEFAULT_ADMIN_PASSWORD", "VegePredict@2026")


async def seed_default_admin() -> None:
    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.email == DEFAULT_ADMIN_EMAIL))
        if result.scalar_one_or_none() is not None:
            return  # already seeded on a prior startup - never overwrite (password may have changed since)

        user = User(
            first_name=DEFAULT_ADMIN_FIRST_NAME,
            last_name=DEFAULT_ADMIN_LAST_NAME,
            email=DEFAULT_ADMIN_EMAIL,
            hashed_password=hash_password(DEFAULT_ADMIN_PASSWORD),
            role=UserRole.SUPERADMIN.value,
        )
        session.add(user)
        await session.commit()
