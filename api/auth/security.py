"""Auth primitives: password hashing and JWT token handling.

We use:
  - `bcrypt` directly (not passlib) — passlib has a known compatibility issue
    with bcrypt >=4.0 and hasn't been updated since 2023.
  - `PyJWT` — actively maintained, well-audited JWT library.

Algorithm: HS256 (symmetric, signing key = SECRET_KEY from settings).
For production you'd want RS256 (asymmetric), but HS256 is fine for an MVP
where a single service both issues and validates tokens.
"""

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt
from jwt.exceptions import InvalidTokenError  # noqa: F401 — re-exported for callers

from api.core.config import get_settings

settings = get_settings()

ALGORITHM = "HS256"


# ── Passwords ─────────────────────────────────────────────────────────────────


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*. The salt is embedded in the output."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the stored *hashed* value."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── JWT ───────────────────────────────────────────────────────────────────────


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    """Encode a JWT with `sub` set to *subject* (typically the user's email).

    The token is signed with `settings.secret_key` using HS256.
    """
    expire = datetime.now(UTC) + (
        expires_delta or timedelta(minutes=settings.access_token_expire_minutes)
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> str:
    """Decode *token* and return its `sub` claim (the user email).

    Raises `jwt.exceptions.InvalidTokenError` (or a subclass) on any problem:
    expired, tampered, malformed, or missing `sub`.
    """
    payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    subject: str | None = payload.get("sub")
    if subject is None:
        raise InvalidTokenError("Token has no 'sub' claim")
    return subject
