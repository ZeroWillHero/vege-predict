"""User CRUD + auth queries. Fine-grained authorization (who can see/create/edit whom) lives
here, not just at the router's role dependency - several rules depend on the *relationship*
between the caller and the target (e.g. an admin may only touch farmer accounts), which a
router-level require_role() check alone can't express. See CLAUDE.md's permission matrix."""

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.backend.models import User
from app.backend.schemas.user import UserCreate, UserRole, UserUpdate
from app.backend.security import hash_password, verify_password


class PermissionDenied(Exception):
    pass


class DuplicateEmail(Exception):
    pass


async def authenticate(session: AsyncSession, email: str, password: str) -> User | None:
    stmt = select(User).where(User.email == email)
    result = await session.execute(stmt)
    user = result.scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(password, user.hashed_password):
        return None
    return user


async def create_user(session: AsyncSession, actor: User, payload: UserCreate) -> User:
    if actor.role == UserRole.ADMIN.value and payload.role != UserRole.FARMER:
        raise PermissionDenied("Admins may only create farmer accounts")

    existing = await session.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none() is not None:
        raise DuplicateEmail(f"Email already registered: {payload.email}")

    user = User(
        first_name=payload.first_name,
        last_name=payload.last_name,
        email=payload.email,
        hashed_password=hash_password(payload.password),
        role=payload.role.value,
    )
    session.add(user)
    await session.flush()
    await session.refresh(user)
    return user


def _visible_to(actor: User):
    """Query filter expressing "what can this actor see": superadmin sees everyone; admin
    sees farmers plus themselves; farmer sees only themselves (only used by get_user, since
    farmers have no list endpoint per the permission matrix)."""
    if actor.role == UserRole.SUPERADMIN.value:
        return None
    if actor.role == UserRole.ADMIN.value:
        return or_(User.role == UserRole.FARMER.value, User.id == actor.id)
    return User.id == actor.id


async def list_users(session: AsyncSession, actor: User) -> list[User]:
    stmt = select(User)
    visibility = _visible_to(actor)
    if visibility is not None:
        stmt = stmt.where(visibility)
    stmt = stmt.order_by(User.id)
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def get_user(session: AsyncSession, actor: User, target_id: int) -> User | None:
    stmt = select(User).where(User.id == target_id)
    visibility = _visible_to(actor)
    if visibility is not None:
        stmt = stmt.where(visibility)
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def update_user(session: AsyncSession, actor: User, target_id: int, payload: UserUpdate) -> User | None:
    target = await get_user(session, actor, target_id)
    if target is None:
        return None

    is_self = target.id == actor.id
    is_superadmin = actor.role == UserRole.SUPERADMIN.value
    is_admin = actor.role == UserRole.ADMIN.value

    if payload.role is not None and not is_superadmin:
        raise PermissionDenied("Only a superadmin may change a user's role")
    if payload.is_active is not None and not (is_superadmin or (is_admin and target.role == UserRole.FARMER.value)):
        raise PermissionDenied("Not permitted to change this account's active status")
    if is_admin and not is_self and target.role != UserRole.FARMER.value:
        raise PermissionDenied("Admins may only edit farmer accounts")

    if payload.first_name is not None:
        target.first_name = payload.first_name
    if payload.last_name is not None:
        target.last_name = payload.last_name
    if payload.password is not None:
        target.hashed_password = hash_password(payload.password)
    if payload.role is not None:
        target.role = payload.role.value
    if payload.is_active is not None:
        target.is_active = payload.is_active

    await session.flush()
    await session.refresh(target)
    return target
