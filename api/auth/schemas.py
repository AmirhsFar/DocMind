import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field

# ── Inbound ──────────────────────────────────────────────────────────────────


class UserCreate(BaseModel):
    """Body for POST /auth/signup."""

    email: EmailStr
    password: str = Field(min_length=8, description="At least 8 characters")


class LoginRequest(BaseModel):
    """Body for POST /auth/login."""

    email: EmailStr
    password: str


# ── Outbound ─────────────────────────────────────────────────────────────────


class UserOut(BaseModel):
    """User fields safe to expose to the client (no hashed_password)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    is_active: bool
    created_at: datetime


class Token(BaseModel):
    """JWT response returned on successful login."""

    access_token: str
    token_type: str = "bearer"
