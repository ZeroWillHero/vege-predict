"""Password hashing, JWT issuing/verification, and the auth/role FastAPI dependencies used by
routers/auth.py and routers/users.py. All other routers stay unauthenticated - see CLAUDE.md's
"User Management & Auth" section for which endpoints require what."""

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession

from app.backend.config import settings
from app.backend.database import get_db
from app.backend.models import User
from app.backend.schemas.user import UserRole

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# tokenUrl only drives Swagger's "Authorize" button - the actual login endpoint takes a JSON
# LoginRequest body, not an OAuth2 password-grant form.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except ValueError:
        # bcrypt rejects secrets over 72 bytes with a ValueError rather than just failing the
        # comparison - treat that the same as a wrong password instead of letting it 500.
        return False


def create_access_token(user: User) -> tuple[str, int]:
    expires_in = settings.jwt_expire_minutes * 60
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    claims = {"sub": str(user.id), "role": user.role, "exp": expire}
    token = jwt.encode(claims, settings.jwt_secret, algorithm="HS256")
    return token, expires_in


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    token: str | None = Depends(oauth2_scheme), session: AsyncSession = Depends(get_db)
) -> User:
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    claims = _decode_token(token)
    user_id = claims.get("sub")
    if user_id is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    user = await session.get(User, user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or inactive user")
    return user


def require_role(*roles: UserRole):
    async def _dependency(user: User = Depends(get_current_user)) -> User:
        if user.role not in {r.value for r in roles}:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return _dependency
