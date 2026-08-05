"""Admin authentication dependency."""

from fastapi import Header, HTTPException, status

from config import settings


async def require_admin_api_key(x_admin_api_key: str | None = Header(None)) -> None:
    """Require a valid admin API key for protected endpoints."""
    if not settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API key is not configured",
        )
    if x_admin_api_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin API key",
        )
