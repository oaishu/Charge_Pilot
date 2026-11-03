"""Minimal JWT auth for the standalone ML service.

Fully self-contained: the users live in this service's own database and the
JWT secret/algorithm come from this service's own config — tokens issued
elsewhere (including the main backend) mean nothing here. Claims: ``sub``,
``role``, ``exp``, ``iat``, ``jti``.

Out of scope on purpose: refresh-token rotation and login lockout stay with
the full backend — this service only needs bcrypt login + role guards for its
operator endpoints.
"""
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models import User

bearer_scheme = HTTPBearer(auto_error=False)


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


# --------------------------------------------------------------------------- #
# Access tokens (this service's own secret; not shared with anything else)
# --------------------------------------------------------------------------- #
def create_access_token(user: User):
    """Returns (token, expires_in_seconds)."""
    expires_delta = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": str(user.id),
        "role": user.role,
        "exp": now + expires_delta,
        "iat": now,
        "jti": str(uuid.uuid4()),
    }
    token = jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return token, int(expires_delta.total_seconds())


def decode_access_token(token: str) -> Dict[str, Any]:
    """Decode & validate an access token. Raises 401 HTTPException on failure."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
    except JWTError as exc:
        if "expire" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        raise credentials_exception from exc

    if payload.get("sub") is None or payload.get("role") is None:
        raise credentials_exception
    return payload


# --------------------------------------------------------------------------- #
# Current-user dependencies
# --------------------------------------------------------------------------- #
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    result = await db.execute(select(User).where(User.id == uuid.UUID(payload["sub"])))
    user = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # trust the verified JWT role claim as the authorization source of truth
    user.role = payload["role"]
    return user


def require_role(required_role: str):
    """Composable role guard: 403 when the token's role claim is insufficient."""

    async def role_dependency(user: User = Depends(get_current_user)) -> User:
        if user.role != required_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires '{required_role}' role",
            )
        return user

    return role_dependency
