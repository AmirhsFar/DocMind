import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.crud import get_user_by_email
from api.auth.dependencies import get_current_user
from api.auth.models import User
from api.auth.schemas import LoginRequest, Token, UserCreate, UserOut
from api.auth.security import create_access_token, hash_password, verify_password
from api.core.database import get_db

logger = logging.getLogger("docmind.auth")

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/signup",
    response_model=UserOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new account",
)
async def signup(body: UserCreate, db: AsyncSession = Depends(get_db)) -> User:
    """Register a new user with *email* and *password*.

    Returns the created user (without the hashed password).
    Responds 400 if the email is already taken.
    """
    existing = await get_user_by_email(db, body.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email already exists",
        )

    user = User(email=body.email, hashed_password=hash_password(body.password))
    db.add(user)
    await db.commit()
    await db.refresh(user)

    logger.info("New user registered: %s", user.email)
    return user


@router.post(
    "/login",
    response_model=Token,
    summary="Obtain a JWT access token",
)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)) -> Token:
    """Exchange *email* + *password* for a signed JWT.

    The token should be sent as `Authorization: Bearer <token>` on all
    subsequent requests to protected endpoints.

    Responds 401 (with a deliberately vague message) for both wrong password
    and unknown email — never reveal which one failed.
    """
    _AUTH_ERROR = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect email or password",
        headers={"WWW-Authenticate": "Bearer"},
    )

    user = await get_user_by_email(db, body.email)
    if user is None or not verify_password(body.password, user.hashed_password):
        raise _AUTH_ERROR

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )

    token = create_access_token(subject=user.email)
    logger.info("User logged in: %s", user.email)
    return Token(access_token=token)


@router.get(
    "/me",
    response_model=UserOut,
    summary="Return the currently authenticated user",
)
async def me(current_user: User = Depends(get_current_user)) -> User:
    """Requires a valid JWT in the `Authorization: Bearer` header.

    Useful for the frontend to fetch session info on page load.
    """
    return current_user
