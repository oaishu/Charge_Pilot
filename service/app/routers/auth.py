"""Auth endpoints (login only) for the standalone ML service.

JSON login against this service's own users table (seeded by
``scripts_setup_db.py``). Access tokens are signed with this service's own
JWT secret — a token issued anywhere else carries no authority here.
Refresh-token rotation stays with the full backend.
"""
import logging

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.errors import AppError
from app.core.security import create_access_token, get_current_user, verify_password
from app.models import User
from app.schemas.auth import LoginJSONRequest, TokenPairResponse, UserBrief

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger("ml.auth")


@router.post("/login", response_model=TokenPairResponse)
async def login(payload: LoginJSONRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(User).where(
            (User.username == payload.username) | (User.email == payload.username.lower()),
            User.is_active.is_(True),
        )
    )
    user = result.scalar_one_or_none()
    if user is None or not verify_password(payload.password, user.password_hash):
        logger.warning("ml-service login failed username=%s", payload.username)
        raise AppError(
            "INVALID_CREDENTIALS",
            "Invalid username or password",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    access_token, expires_in = create_access_token(user)
    await db.commit()
    logger.info("ml-service login success username=%s role=%s", user.username, user.role)
    return TokenPairResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
        user=UserBrief.model_validate(user),
    )


@router.get("/me", response_model=UserBrief)
async def me(user: User = Depends(get_current_user)):
    return UserBrief.model_validate(user)
