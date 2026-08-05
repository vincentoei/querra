"""FastAPI middleware for request logging and in-memory rate limiting."""

import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from config import settings

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, duration, and client."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = time.perf_counter()
        client = request.client.host if request.client else "unknown"
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.perf_counter() - start
            logger.info(
                "%s %s - %s - %.3fs",
                request.method,
                request.url.path,
                status_code,
                duration,
                extra={"client": client},
            )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter keyed by client IP.

    Disabled by default. Enable with RATE_LIMIT_ENABLED=true.
    """

    def __init__(self, app: Any, max_requests: int = 60, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, deque[float]] = {}

    def _is_allowed(self, client: str) -> bool:
        now = time.perf_counter()
        window = self._requests.setdefault(client, deque())
        while window and window[0] < now - self.window_seconds:
            window.popleft()
        if len(window) >= self.max_requests:
            return False
        window.append(now)
        return True

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not settings.rate_limit_enabled:
            return await call_next(request)

        # Skip rate limiting for health checks and CORS preflight.
        if request.method == "OPTIONS" or request.url.path == "/health":
            return await call_next(request)

        client = request.client.host if request.client else "unknown"
        if not self._is_allowed(client):
            return Response(
                content="Rate limit exceeded",
                status_code=429,
                media_type="text/plain",
            )
        return await call_next(request)
