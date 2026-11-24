"""API-key authentication.

Machine-to-machine services (inspection stations, PLCs, line controllers) are
the primary clients here, so a shared API key passed in the ``X-API-Key`` header
is the appropriate mechanism. Keys are configured via ``ICVQI_API_KEYS``.

If no keys are configured the dependency allows all requests but the service logs
a loud warning on startup — acceptable for local development, never for
production.
"""

import logging
import secrets

from fastapi import Header, HTTPException, status

from backend.app.core.config import settings

logger = logging.getLogger("icvqi.security")

API_KEY_HEADER = "X-API-Key"


async def require_api_key(x_api_key: str | None = Header(default=None)) -> str:
    """FastAPI dependency enforcing a valid API key when auth is enabled."""
    if not settings.auth_enabled:
        return "auth-disabled"

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    # Constant-time comparison against every configured key.
    for key in settings.api_keys:
        if secrets.compare_digest(x_api_key, key):
            return x_api_key

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key.",
    )
