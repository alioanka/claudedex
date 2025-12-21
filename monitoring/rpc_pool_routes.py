"""
RPC/API Pool Management Routes for Dashboard

API endpoints for managing RPC and API endpoints through the web interface
"""

import asyncio
import logging
import time
import aiohttp
from aiohttp import web
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RPCPoolRoutes:
    """
    RPC/API Pool management API routes

    Provides endpoints for:
    - Viewing all endpoints
    - Adding/editing/deleting endpoints
    - Testing endpoints
    - Running health checks
    - Viewing usage statistics
    """

    def __init__(self, pool_engine=None, jinja_env=None):
        """
        Initialize RPC pool routes

        Args:
            pool_engine: PoolEngine instance
            jinja_env: Jinja2 environment for templates
        """
        self.pool_engine = pool_engine
        self.jinja_env = jinja_env
        self.logger = logging.getLogger("RPCPoolRoutes")

    def setup_routes(self, app: web.Application):
        """
        Setup RPC pool routes on the application

        Args:
            app: aiohttp web application
        """
        # HTML page
        app.router.add_get('/settings/rpc-api', self.rpc_api_settings_page)

        # API endpoints
        app.router.add_get('/api/rpc-pool/endpoints', self.get_endpoints)
        app.router.add_post('/api/rpc-pool/endpoints', self.add_endpoint)
        app.router.add_put('/api/rpc-pool/endpoints/{endpoint_id}', self.update_endpoint)
        app.router.add_delete('/api/rpc-pool/endpoints/{endpoint_id}', self.delete_endpoint)
        app.router.add_post('/api/rpc-pool/endpoints/{endpoint_id}/test', self.test_endpoint)
        app.router.add_get('/api/rpc-pool/provider-types', self.get_provider_types)
        app.router.add_get('/api/rpc-pool/stats', self.get_usage_stats)
        app.router.add_post('/api/rpc-pool/health-check', self.run_health_check)

        self.logger.info("RPC Pool routes registered")

    async def set_pool_engine(self, pool_engine):
        """Set the pool engine after initialization"""
        self.pool_engine = pool_engine

    async def rpc_api_settings_page(self, request: web.Request) -> web.Response:
        """
        Render RPC/API settings page

        Args:
            request: HTTP request

        Returns:
            web.Response: HTML response
        """
        try:
            if not self.jinja_env:
                return web.Response(text="Template engine not available", status=500)

            template = self.jinja_env.get_template('settings_rpc_api.html')
            html = template.render()
            return web.Response(text=html, content_type='text/html')

        except Exception as e:
            self.logger.error(f"Error rendering RPC/API settings page: {e}", exc_info=True)
            return web.Response(text=f"Error: {str(e)}", status=500)

    async def get_endpoints(self, request: web.Request) -> web.Response:
        """
        Get all RPC/API endpoints

        Args:
            request: HTTP request

        Returns:
            web.Response: JSON response with endpoints
        """
        try:
            if not self.pool_engine:
                return web.json_response({
                    'success': False,
                    'error': 'Pool engine not initialized'
                }, status=500)

            endpoints = await self.pool_engine.get_all_endpoints_data()

            return web.json_response({
                'success': True,
                'endpoints': endpoints
            })

        except Exception as e:
            self.logger.error(f"Error getting endpoints: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def add_endpoint(self, request: web.Request) -> web.Response:
        """
        Add a new RPC/API endpoint

        Args:
            request: HTTP request with endpoint data

        Returns:
            web.Response: JSON response
        """
        try:
            if not self.pool_engine:
                return web.json_response({
                    'success': False,
                    'error': 'Pool engine not initialized'
                }, status=500)

            data = await request.json()

            # Validate required fields
            required = ['provider_type', 'name', 'url']
            for field in required:
                if not data.get(field):
                    return web.json_response({
                        'success': False,
                        'error': f'Missing required field: {field}'
                    }, status=400)

            # Determine chain from provider type
            chain = None
            if '_RPC' in data['provider_type'] or '_WS' in data['provider_type']:
                chain = data['provider_type'].replace('_RPC', '').replace('_WS', '').lower()

            endpoint_id = await self.pool_engine.add_endpoint(
                provider_type=data['provider_type'],
                name=data['name'],
                url=data['url'],
                api_key=data.get('api_key'),
                chain=chain,
                priority=data.get('priority', 100)
            )

            if endpoint_id:
                return web.json_response({
                    'success': True,
                    'endpoint_id': endpoint_id,
                    'message': 'Endpoint added successfully'
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': 'Failed to add endpoint'
                }, status=500)

        except Exception as e:
            self.logger.error(f"Error adding endpoint: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def update_endpoint(self, request: web.Request) -> web.Response:
        """
        Update an existing endpoint

        Args:
            request: HTTP request with endpoint data

        Returns:
            web.Response: JSON response
        """
        try:
            if not self.pool_engine:
                return web.json_response({
                    'success': False,
                    'error': 'Pool engine not initialized'
                }, status=500)

            endpoint_id = int(request.match_info['endpoint_id'])
            data = await request.json()

            success = await self.pool_engine.update_endpoint(endpoint_id, data)

            if success:
                return web.json_response({
                    'success': True,
                    'message': 'Endpoint updated successfully'
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': 'Failed to update endpoint'
                }, status=500)

        except Exception as e:
            self.logger.error(f"Error updating endpoint: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def delete_endpoint(self, request: web.Request) -> web.Response:
        """
        Delete an endpoint

        Args:
            request: HTTP request

        Returns:
            web.Response: JSON response
        """
        try:
            if not self.pool_engine:
                return web.json_response({
                    'success': False,
                    'error': 'Pool engine not initialized'
                }, status=500)

            endpoint_id = int(request.match_info['endpoint_id'])

            success = await self.pool_engine.delete_endpoint(endpoint_id)

            if success:
                return web.json_response({
                    'success': True,
                    'message': 'Endpoint deleted successfully'
                })
            else:
                return web.json_response({
                    'success': False,
                    'error': 'Failed to delete endpoint'
                }, status=500)

        except Exception as e:
            self.logger.error(f"Error deleting endpoint: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def test_endpoint(self, request: web.Request) -> web.Response:
        """
        Test an endpoint's connectivity and response time

        Args:
            request: HTTP request

        Returns:
            web.Response: JSON response with test results
        """
        try:
            if not self.pool_engine:
                return web.json_response({
                    'success': False,
                    'error': 'Pool engine not initialized'
                }, status=500)

            endpoint_id = int(request.match_info['endpoint_id'])

            # Find the endpoint
            endpoints = await self.pool_engine.get_all_endpoints_data()
            endpoint = next((e for e in endpoints if e['id'] == endpoint_id), None)

            if not endpoint:
                return web.json_response({
                    'success': False,
                    'error': 'Endpoint not found'
                }, status=404)

            # Perform test based on endpoint type
            url = endpoint['url']
            provider_type = endpoint['provider_type']

            start_time = time.time()
            success = False
            error_message = None

            try:
                is_rate_limited = False

                # Helper function to check if error message indicates rate limit
                def check_rate_limit_message(msg):
                    if not msg:
                        return False
                    msg_lower = str(msg).lower()
                    rate_limit_keywords = [
                        'rate limit', 'ratelimit', 'rate-limit',
                        'too many requests', 'too many request',
                        'request limit', 'exceeded', 'throttl',
                        'quota', 'limit exceeded', 'capacity'
                    ]
                    return any(kw in msg_lower for kw in rate_limit_keywords)

                if 'RPC' in provider_type or 'WS' in provider_type:
                    # Test RPC endpoint
                    if 'solana' in provider_type.lower():
                        payload = {"jsonrpc": "2.0", "id": 1, "method": "getHealth"}
                    else:
                        payload = {"jsonrpc": "2.0", "id": 1, "method": "eth_blockNumber", "params": []}

                    async with aiohttp.ClientSession() as session:
                        async with session.post(
                            url,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            response_text = await response.text()

                            # Check for rate limit (HTTP 429)
                            if response.status == 429:
                                is_rate_limited = True
                                error_message = "Rate limited (HTTP 429)"
                            elif response.status == 401:
                                # Check if 401 is actually rate limit (some providers do this)
                                if check_rate_limit_message(response_text):
                                    is_rate_limited = True
                                    error_message = "Rate limited (detected in response)"
                                else:
                                    error_message = "Authentication failed (HTTP 401)"
                            elif response.status == 403:
                                # Some providers use 403 for rate limits
                                if check_rate_limit_message(response_text):
                                    is_rate_limited = True
                                    error_message = "Rate limited (HTTP 403)"
                                else:
                                    error_message = "Forbidden (HTTP 403)"
                            elif response.status == 200:
                                try:
                                    data = await response.json(content_type=None)
                                    if 'result' in data:
                                        success = True
                                    elif 'error' in data:
                                        error_obj = data.get('error', {})
                                        if isinstance(error_obj, dict):
                                            error_message = error_obj.get('message', 'Unknown error')
                                            error_code = error_obj.get('code', 0)
                                        else:
                                            error_message = str(error_obj)
                                            error_code = 0

                                        # Check for rate limit in JSON response
                                        # Common rate limit error codes: -32005, -32097, -32098
                                        rate_limit_codes = [-32005, -32097, -32098, -32099]
                                        if error_code in rate_limit_codes or check_rate_limit_message(error_message):
                                            is_rate_limited = True
                                    else:
                                        # No error field, consider it success
                                        success = True
                                except Exception as json_err:
                                    # Check if response text indicates rate limit
                                    if check_rate_limit_message(response_text):
                                        is_rate_limited = True
                                        error_message = "Rate limited (detected in response)"
                                    else:
                                        error_message = f"Invalid JSON response: {str(json_err)[:50]}"
                            else:
                                error_message = f"HTTP {response.status}"
                                # Check response body for rate limit info
                                if check_rate_limit_message(response_text):
                                    is_rate_limited = True
                else:
                    # Test API endpoint with simple GET
                    async with aiohttp.ClientSession() as session:
                        async with session.get(
                            url,
                            timeout=aiohttp.ClientTimeout(total=10)
                        ) as response:
                            response_text = await response.text()

                            # Check for rate limit (HTTP 429)
                            if response.status == 429:
                                is_rate_limited = True
                                error_message = "Rate limited (HTTP 429)"
                            elif response.status in [200, 201]:
                                success = True
                            elif response.status == 401:
                                if check_rate_limit_message(response_text):
                                    is_rate_limited = True
                                    error_message = "Rate limited (detected in response)"
                                else:
                                    error_message = "Authentication failed (HTTP 401)"
                            elif response.status == 403:
                                if check_rate_limit_message(response_text):
                                    is_rate_limited = True
                                    error_message = "Rate limited (HTTP 403)"
                                else:
                                    error_message = "Forbidden (HTTP 403)"
                            else:
                                error_message = f"HTTP {response.status}"
                                if check_rate_limit_message(response_text):
                                    is_rate_limited = True

            except asyncio.TimeoutError:
                error_message = "Connection timeout"
            except aiohttp.ClientError as e:
                error_message = f"Connection error: {str(e)}"
            except Exception as e:
                error_message = str(e)

            latency_ms = int((time.time() - start_time) * 1000)

            # Report result to pool engine
            if success:
                await self.pool_engine.report_success(provider_type, url, latency_ms)
            elif is_rate_limited:
                # Report rate limit with 5 minute cooldown
                await self.pool_engine.report_rate_limit(provider_type, url, duration_seconds=300, error_message=error_message)
            else:
                await self.pool_engine.report_failure(provider_type, url, 'test_failure', error_message)

            return web.json_response({
                'success': success,
                'latency_ms': latency_ms,
                'error': error_message,
                'rate_limited': is_rate_limited
            })

        except Exception as e:
            self.logger.error(f"Error testing endpoint: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_provider_types(self, request: web.Request) -> web.Response:
        """
        Get all available provider types

        Args:
            request: HTTP request

        Returns:
            web.Response: JSON response with provider types
        """
        try:
            if not self.pool_engine:
                return web.json_response({
                    'success': False,
                    'error': 'Pool engine not initialized'
                }, status=500)

            provider_types = await self.pool_engine.get_provider_types()

            return web.json_response({
                'success': True,
                'provider_types': provider_types
            })

        except Exception as e:
            self.logger.error(f"Error getting provider types: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def get_usage_stats(self, request: web.Request) -> web.Response:
        """
        Get usage statistics

        Args:
            request: HTTP request

        Returns:
            web.Response: JSON response with stats
        """
        try:
            if not self.pool_engine:
                return web.json_response({
                    'success': False,
                    'error': 'Pool engine not initialized'
                }, status=500)

            hours = int(request.query.get('hours', 24))
            stats = await self.pool_engine.get_usage_stats(hours)

            return web.json_response({
                'success': True,
                'stats': stats
            })

        except Exception as e:
            self.logger.error(f"Error getting usage stats: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def run_health_check(self, request: web.Request) -> web.Response:
        """
        Run health checks on all endpoints

        Args:
            request: HTTP request

        Returns:
            web.Response: JSON response with results
        """
        try:
            if not self.pool_engine:
                return web.json_response({
                    'success': False,
                    'error': 'Pool engine not initialized'
                }, status=500)

            results = await self.pool_engine.run_health_checks()

            return web.json_response({
                'success': True,
                'results': results
            })

        except Exception as e:
            self.logger.error(f"Error running health check: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)
