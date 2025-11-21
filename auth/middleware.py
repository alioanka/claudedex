"""
Authentication Middleware
Provides route protection decorators and middleware functions
"""
from functools import wraps
from aiohttp import web
from typing import Callable, Optional
import logging

from .models import UserRole
from .auth_service import AuthService

logger = logging.getLogger(__name__)


def require_auth(handler: Callable) -> Callable:
    """
    Decorator to require authentication for a route
    Usage: @require_auth
    """
    @wraps(handler)
    async def middleware(request: web.Request):
        # Check if auth_service is available
        if not hasattr(request.app, 'auth_service'):
            logger.error("AuthService not initialized in app")
            return web.json_response({'error': 'Authentication system not available'}, status=500)

        auth_service: AuthService = request.app['auth_service']

        # Get session_id from cookie
        session_id = request.cookies.get('session_id')

        if not session_id:
            # Return login redirect for HTML pages, 401 for API
            if request.path.startswith('/api/'):
                return web.json_response({'error': 'Authentication required'}, status=401)
            else:
                return web.HTTPFound('/login')

        # Validate session
        user = await auth_service.validate_session(session_id)

        if not user:
            # Session invalid or expired
            if request.path.startswith('/api/'):
                return web.json_response({'error': 'Session expired or invalid'}, status=401)
            else:
                response = web.HTTPFound('/login')
                response.del_cookie('session_id')
                return response

        # Attach user to request for access in handlers
        request['user'] = user

        # Call the actual handler
        return await handler(request)

    return middleware


def require_admin(handler: Callable) -> Callable:
    """
    Decorator to require admin role
    Must be used together with @require_auth
    Usage:
        @require_auth
        @require_admin
    """
    @wraps(handler)
    async def middleware(request: web.Request):
        user = request.get('user')

        if not user:
            return web.json_response({'error': 'Authentication required'}, status=401)

        if user.role != UserRole.ADMIN:
            return web.json_response({'error': 'Admin access required'}, status=403)

        return await handler(request)

    return middleware


def require_role(required_role: UserRole) -> Callable:
    """
    Decorator to require specific role
    Usage:
        @require_auth
        @require_role(UserRole.OPERATOR)
    """
    def decorator(handler: Callable) -> Callable:
        @wraps(handler)
        async def middleware(request: web.Request):
            user = request.get('user')

            if not user:
                return web.json_response({'error': 'Authentication required'}, status=401)

            # Admin has access to everything
            if user.role == UserRole.ADMIN:
                return await handler(request)

            if user.role != required_role:
                return web.json_response(
                    {'error': f'Role {required_role.value} required'},
                    status=403
                )

            return await handler(request)

        return middleware

    return decorator


async def auth_middleware_factory(app: web.Application, handler: Callable) -> Callable:
    """
    Global middleware to enforce authentication on all routes
    Actively blocks unauthenticated requests except for public routes
    """
    async def middleware(request: web.Request):
        # Public routes that don't require authentication
        public_routes = [
            '/login',
            '/api/auth/login',
            '/api/auth/logout',
        ]

        # Check if this is a public route
        is_public = (
            request.path in public_routes or
            request.path.startswith('/static/')
        )

        if is_public:
            return await handler(request)

        # For all other routes, authentication is required
        if not hasattr(app, 'auth_service'):
            # Auth service not initialized - this is a critical error
            logger.error(f"🚨 SECURITY: Auth service not initialized, blocking access to {request.path}")
            if request.path.startswith('/api/'):
                return web.json_response({
                    'error': 'Authentication system not initialized',
                    'message': 'Server is starting up. Please wait and try again.'
                }, status=503)
            else:
                return web.Response(
                    text='<h1>Server Starting Up</h1><p>Authentication system is initializing. Please wait a moment and refresh.</p>',
                    content_type='text/html',
                    status=503
                )

        auth_service: AuthService = app['auth_service']
        session_id = request.cookies.get('session_id')

        if not session_id:
            # No session cookie - redirect to login
            if request.path.startswith('/api/'):
                return web.json_response({'error': 'Authentication required'}, status=401)
            else:
                return web.HTTPFound('/login')

        # Validate session
        user = await auth_service.validate_session(session_id)

        if not user:
            # Invalid or expired session - redirect to login
            if request.path.startswith('/api/'):
                return web.json_response({'error': 'Session expired or invalid'}, status=401)
            else:
                response = web.HTTPFound('/login')
                response.del_cookie('session_id')
                return response

        # Session valid - attach user to request and proceed
        request['user'] = user
        return await handler(request)

    return middleware


def get_client_ip(request: web.Request) -> str:
    """Get client IP address from request"""
    # Check for X-Forwarded-For header (if behind proxy)
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()

    # Check for X-Real-IP header
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip

    # Fall back to remote address
    peername = request.transport.get_extra_info('peername')
    if peername:
        return peername[0]

    return 'unknown'


def get_user_agent(request: web.Request) -> str:
    """Get user agent from request"""
    return request.headers.get('User-Agent', 'unknown')
