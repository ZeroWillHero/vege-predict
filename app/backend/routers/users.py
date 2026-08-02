from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.backend.database import get_db
from app.backend.models import User
from app.backend.schemas.user import UserCreate, UserOut, UserRole, UserUpdate
from app.backend.security import get_current_user, require_role
from app.backend.services import user_service
from app.backend.services.user_service import DuplicateEmail, PermissionDenied

router = APIRouter()

_ADMIN_OR_SUPERADMIN = require_role(UserRole.SUPERADMIN, UserRole.ADMIN)


@router.post(
    "/users",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user account",
    description=(
        "Creates a new user account. Requires an authenticated superadmin or admin caller. "
        "Superadmins may create an account with any role; admins may only create `farmer` "
        "accounts (creating an `admin` or `superadmin` account as an admin caller returns 403). "
        "There is no public self-registration endpoint - account creation always goes through "
        "this endpoint, by an existing admin/superadmin."
    ),
    response_description="The newly created user.",
    responses={
        401: {"description": "Missing/invalid/expired token."},
        403: {"description": "Caller is not admin/superadmin, or an admin tried to assign a non-farmer role."},
        409: {"description": "A user with this email already exists."},
    },
)
async def create_user(
    payload: UserCreate,
    actor: User = Depends(_ADMIN_OR_SUPERADMIN),
    session: AsyncSession = Depends(get_db),
):
    try:
        user = await user_service.create_user(session, actor, payload)
        await session.commit()
        return user
    except PermissionDenied as e:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except DuplicateEmail as e:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))


@router.get(
    "/users",
    response_model=list[UserOut],
    summary="List users",
    description=(
        "Lists user accounts visible to the caller. Superadmins see every user; admins see only "
        "farmer accounts plus their own record (other admins/superadmins are not returned, and "
        "not distinguishable from not existing - see `GET /users/{user_id}`)."
    ),
    response_description="Users visible to the caller.",
    responses={401: {"description": "Missing/invalid/expired token."}, 403: {"description": "Caller is not admin/superadmin."}},
)
async def list_users(actor: User = Depends(_ADMIN_OR_SUPERADMIN), session: AsyncSession = Depends(get_db)):
    return await user_service.list_users(session, actor)


@router.get(
    "/users/{user_id}",
    response_model=UserOut,
    summary="Get a user by ID",
    description=(
        "Returns a single user account, subject to the same visibility rule as `GET /users`. "
        "An admin requesting an admin/superadmin account (other than their own) gets a 404, not "
        "a 403 - this deliberately avoids confirming whether that account exists."
    ),
    response_description="The requested user.",
    responses={
        401: {"description": "Missing/invalid/expired token."},
        403: {"description": "Caller is not admin/superadmin."},
        404: {"description": "No such user, or the user exists but isn't visible to the caller."},
    },
)
async def get_user(
    user_id: int = Path(description="User ID.", examples=[1]),
    actor: User = Depends(_ADMIN_OR_SUPERADMIN),
    session: AsyncSession = Depends(get_db),
):
    user = await user_service.get_user(session, actor, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.patch(
    "/users/{user_id}",
    response_model=UserOut,
    summary="Update a user",
    description=(
        "Updates a user account. Any authenticated user may update their own `first_name`/"
        "`last_name`/`password`. Only a superadmin may change `role` (403 for anyone else, even "
        "on their own account). `is_active` may be set by a superadmin for any account, or by an "
        "admin for a farmer account only. An admin may not edit an admin/superadmin account other "
        "than their own, and even on their own account cannot change `role` or `is_active`."
    ),
    response_description="The updated user.",
    responses={
        401: {"description": "Missing/invalid/expired token."},
        403: {"description": "The caller isn't permitted to make this specific change (e.g. a non-superadmin setting `role`)."},
        404: {"description": "No such user, or the user exists but isn't visible to the caller."},
    },
)
async def update_user(
    payload: UserUpdate,
    user_id: int = Path(description="User ID.", examples=[1]),
    actor: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_db),
):
    try:
        user = await user_service.update_user(session, actor, user_id, payload)
    except PermissionDenied as e:
        await session.rollback()
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    await session.commit()
    return user
