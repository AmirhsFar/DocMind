"""Reusable FastAPI dependencies for authentication.

Usage in any protected route:

    from api.auth.dependencies import get_current_user
    from api.auth.models import User

    @router.get("/protected")
    async def protected(current_user: User = Depends(get_current_user)):
        ...
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwt.exceptions import InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.crud import get_user_by_email
from api.auth.models import User
from api.auth.security import decode_access_token
from api.core.database import get_db

# tokenUrl is only used by the Swagger UI "Authorize" button; our actual
# /auth/login endpoint accepts JSON (better for the React frontend).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

_CREDENTIALS_EXCEPTION = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Validate the Bearer token and return the corresponding active User.

    Raises HTTP 401 if the token is missing, expired, tampered, or belongs
    to a user that no longer exists or has been deactivated.
    """
    try:
        email = decode_access_token(token)
    except InvalidTokenError:
        raise _CREDENTIALS_EXCEPTION from None

    user = await get_user_by_email(db, email)
    if user is None:
        raise _CREDENTIALS_EXCEPTION

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    return user
