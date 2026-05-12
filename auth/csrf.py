"""
CSRF Protection — double-submit-cookie pattern (MB-27).

Auth in the dashboard is cookie-based, which makes every POST/PUT/DELETE
vulnerable to CSRF unless requests prove they have read the cookie. We use
the stateless double-submit pattern: set a random token in a non-HTTPOnly
cookie, require the same token in the X-CSRF-Token header on state-changing
requests, and 403 on any mismatch.
"""
import os
import secrets
from typing import Callable

from aiohttp import web

CSRF_COOKIE = 'csrf_token'
CSRF_HEADER = 'X-CSRF-Token'

# Methods that mutate state — POST/PUT/DELETE/PATCH always need a token.
_PROTECTED_METHODS = frozenset(('POST', 'PUT', 'DELETE', 'PATCH'))

# Endpoints exempt from CSRF. /api/auth/login is the entrypoint (no cookie
# yet); /api/auth/logout is safe to allow either way.
_EXEMPT_PATHS = frozenset((
    '/api/auth/login',
    '/api/auth/logout',
))


def _new_token() -> str:
    return secrets.token_urlsafe(32)


async def csrf_middleware_factory(app: web.Application, handler: Callable) -> Callable:
    async def middleware(request: web.Request):
        method = request.method.upper()
        path = request.path

        # State-changing requests must present a matching token unless exempt.
        if method in _PROTECTED_METHODS and path not in _EXEMPT_PATHS:
            cookie_token = request.cookies.get(CSRF_COOKIE)
            header_token = request.headers.get(CSRF_HEADER)
            if not cookie_token or not header_token or not secrets.compare_digest(
                cookie_token, header_token
            ):
                return web.json_response(
                    {'error': 'CSRF token missing or invalid'},
                    status=403,
                )

        response = await handler(request)

        # Make sure the client always has a csrf_token cookie. Reuse the
        # existing one if present so a single tab keeps one token per session.
        existing = request.cookies.get(CSRF_COOKIE)
        if not existing and hasattr(response, 'set_cookie'):
            secure = os.getenv('DASHBOARD_HTTPS', 'true').lower() not in (
                'false', '0', 'no'
            )
            response.set_cookie(
                CSRF_COOKIE,
                _new_token(),
                httponly=False,  # JS must read this to echo it in the header
                secure=secure,
                samesite='Lax',
                max_age=3600,
            )
        return response

    return middleware
