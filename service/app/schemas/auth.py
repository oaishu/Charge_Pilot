"""Auth schemas for the standalone ML service (login only; register/reset stay
with the main backend)."""
import uuid
from typing import Optional

from pydantic import BaseModel, ConfigDict


class LoginJSONRequest(BaseModel):
    username: str
    password: str


class UserBrief(BaseModel):
    id: uuid.UUID
    username: str
    role: str

    model_config = ConfigDict(from_attributes=True)


class TokenPairResponse(BaseModel):
    # refresh_token is None here: the ML service issues access tokens only;
    # refresh rotation remains a backend capability. Clients that just need
    # the access token see the same shape as the backend's login response.
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    user: UserBrief
