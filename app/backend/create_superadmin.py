"""One-off script to create (or promote) the first superadmin account.

There is no public self-registration endpoint - POST /users requires an existing admin/
superadmin caller, which is a chicken-and-egg problem for the very first account. Run this
script directly against the target database instead.

Idempotent: if the email already exists, promotes that account to superadmin and reactivates
it rather than erroring; otherwise inserts a new one.

Usage:
    python create_superadmin.py --email you@example.com --first-name Ada --last-name Lovelace
    (prompts for a password interactively)

    VEGEPREDICT_SUPERADMIN_PASSWORD=... python create_superadmin.py --email you@example.com \\
        --first-name Ada --last-name Lovelace
    (non-interactive, e.g. for CI/scripted setup - password never appears as a CLI arg either
    way, to avoid shell-history leakage)
"""

import argparse
import asyncio
import getpass
import os
import sys
from pathlib import Path

from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.backend.database import async_session_factory  # noqa: E402
from app.backend.models import User  # noqa: E402
from app.backend.schemas.user import UserRole  # noqa: E402
from app.backend.security import hash_password  # noqa: E402


async def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", required=True)
    parser.add_argument("--first-name", required=True)
    parser.add_argument("--last-name", required=True)
    args = parser.parse_args()

    password = os.environ.get("VEGEPREDICT_SUPERADMIN_PASSWORD")
    if not password:
        password = getpass.getpass("Password: ")
        confirm = getpass.getpass("Confirm password: ")
        if password != confirm:
            print("Passwords do not match.", file=sys.stderr)
            sys.exit(1)
    if len(password) < 8:
        print("Password must be at least 8 characters.", file=sys.stderr)
        sys.exit(1)

    async with async_session_factory() as session:
        result = await session.execute(select(User).where(User.email == args.email))
        user = result.scalar_one_or_none()

        if user is None:
            user = User(
                first_name=args.first_name,
                last_name=args.last_name,
                email=args.email,
                hashed_password=hash_password(password),
                role=UserRole.SUPERADMIN.value,
            )
            session.add(user)
            action = "Created"
        else:
            user.first_name = args.first_name
            user.last_name = args.last_name
            user.hashed_password = hash_password(password)
            user.role = UserRole.SUPERADMIN.value
            user.is_active = True
            action = "Promoted"

        await session.commit()
        print(f"{action} superadmin: {args.email}")


if __name__ == "__main__":
    asyncio.run(main())
