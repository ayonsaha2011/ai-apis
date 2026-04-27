from __future__ import annotations

import secrets

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from openai_multi_backend.config import Settings, get_settings
from openai_multi_backend.errors import OpenAIHTTPException

bearer = HTTPBearer(auto_error=False)


async def require_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    settings: Settings = Depends(get_settings),
) -> None:
    if settings.environment in {"development", "test"} and not settings.api_keys:
        return
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise OpenAIHTTPException(
            status_code=401,
            message="Missing bearer token",
            error_type="authentication_error",
            code="missing_api_key",
        )
    if not any(secrets.compare_digest(credentials.credentials, key) for key in settings.api_keys):
        raise OpenAIHTTPException(
            status_code=401,
            message="Invalid bearer token",
            error_type="authentication_error",
            code="invalid_api_key",
        )
