from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class UserRole(str, Enum):
    SUPERADMIN = "superadmin"
    ADMIN = "admin"
    FARMER = "farmer"


class UserBase(BaseModel):
    first_name: str = Field(description="User's first name.", examples=["Nimal"])
    last_name: str = Field(description="User's last name.", examples=["Perera"])
    email: EmailStr = Field(description="Login email, unique across all users.", examples=["nimal@example.com"])


class UserCreate(UserBase):
    password: str = Field(min_length=8, description="Plaintext password (min 8 chars) - never stored as-is.")
    role: UserRole = Field(
        description="Role to assign. Superadmins may assign any role; admins may only assign `farmer` "
        "(a 403 is returned otherwise)."
    )


class UserUpdate(BaseModel):
    first_name: str | None = Field(default=None, description="New first name, if changing it.")
    last_name: str | None = Field(default=None, description="New last name, if changing it.")
    password: str | None = Field(
        default=None, min_length=8, description="New plaintext password, if changing it."
    )
    role: UserRole | None = Field(
        default=None,
        description="New role. Only a superadmin may set this field - any other caller supplying "
        "it gets a 403, even if the rest of the update would otherwise be allowed.",
    )
    is_active: bool | None = Field(
        default=None, description="Set to false to deactivate the account (soft delete)."
    )


class UserOut(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(description="User ID.")
    role: UserRole = Field(description="Assigned role.")
    is_active: bool = Field(description="Whether the account can currently log in.")
    created_at: datetime = Field(description="When the account was created (UTC).")
    updated_at: datetime = Field(description="When the account was last modified (UTC).")


class LoginRequest(BaseModel):
    email: EmailStr = Field(description="Account email.", examples=["nimal@example.com"])
    password: str = Field(description="Account password.")


class TokenOut(BaseModel):
    access_token: str = Field(description="JWT bearer token - pass as `Authorization: Bearer <token>`.")
    token_type: str = Field(default="bearer", description="Always `bearer`.")
    expires_in: int = Field(description="Seconds until the token expires.")
