from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.backend.database import get_db
from app.backend.models import User
from app.backend.schemas.user import LoginRequest, TokenOut, UserOut
from app.backend.security import create_access_token, get_current_user
from app.backend.services import user_service

router = APIRouter()


@router.post(
    "/auth/login",
    response_model=TokenOut,
    summary="Log in",
    description=(
        "Exchanges an email + password for a JWT bearer token. Takes a JSON body (not an OAuth2 "
        "password-grant form) - the `Authorize` button in this Swagger UI is wired up for "
        "convenience but the actual request contract is `LoginRequest`. Pass the returned "
        "`access_token` as `Authorization: Bearer <token>` on subsequent requests to `/auth/me` "
        "and `/users*`. Deliberately returns the same generic error for a nonexistent email, a "
        "wrong password, or a deactivated account, to avoid leaking which of those is true."
    ),
    response_description="A bearer token and its expiry.",
    responses={401: {"description": "Invalid email/password, or the account is deactivated."}},
)
async def login(payload: LoginRequest, session: AsyncSession = Depends(get_db)):
    user = await user_service.authenticate(session, payload.email, payload.password)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")
    token, expires_in = create_access_token(user)
    return TokenOut(access_token=token, token_type="bearer", expires_in=expires_in)


@router.get(
    "/auth/me",
    response_model=UserOut,
    summary="Current user profile",
    description="Returns the profile of the user identified by the bearer token on this request.",
    response_description="The caller's own user record.",
    responses={401: {"description": "Missing, invalid, or expired token; or the account was deactivated."}},
)
async def me(user: User = Depends(get_current_user)):
    return user
