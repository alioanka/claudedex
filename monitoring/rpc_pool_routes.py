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

from auth.middleware import require_auth, require_admin, require_operator

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

        # API endpoints. Wave-F6 RBAC (adjudication finding #4): endpoint
        # CRUD carries API-key-bearing URLs and changes what every on-chain
        # module broadcasts through - admin only. Test/health-check are
        # bounded ops (can burn provider quota) - operator or admin.
        # GET /endpoints is admin too: get_all_endpoints_data returns the
        # full URLs, which embed provider API keys (same precedent as
        # /api/settings/sensitive/*). Stats/provider-types stay viewer.
        app.router.add_get('/api/rpc-pool/endpoints', require_auth(require_admin(self.get_endpoints)))
        app.router.add_post('/api/rpc-pool/endpoints', require_auth(require_admin(self.add_endpoint)))
        app.router.add_put('/api/rpc-pool/endpoints/{endpoint_id}', require_auth(require_admin(self.update_endpoint)))
        app.router.add_delete('/api/rpc-pool/endpoints/{endpoint_id}', require_auth(require_admin(self.delete_endpoint)))
        app.router.add_post('/api/rpc-pool/endpoints/{endpoint_id}/test', require_auth(require_operator(self.test_endpoint)))
        app.router.add_post('/api/rpc-pool/test-all', require_auth(require_operator(self.test_all_endpoints)))
        app.router.add_get('/api/rpc-pool/provider-types', self.get_provider_types)
        app.router.add_get('/api/rpc-pool/stats', self.get_usage_stats)
        # Wave-F7: per-(module, provider) governor consumption + throttle
        # state. Read-only aggregates (no URLs/keys) — viewer-safe, same
        # precedent as /stats.
        app.router.add_get('/api/rpc-pool/governor', self.get_governor_status)
        app.router.add_post('/api/rpc-pool/health-check', require_auth(require_operator(self.run_health_check)))

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

    async def _probe_endpoint(self, endpoint: Dict, session: Optional[aiohttp.ClientSession] = None) -> Dict:
        """
        Probe a single endpoint's connectivity + latency and report the result
        back to the pool engine. Returns a result dict. Shared by the
        single-endpoint test route and the TEST ALL action so behaviour is
        identical. Fail-soft: any exception is captured into the result dict,
        never raised.

        Args:
            endpoint: endpoint data dict (must have 'url', 'provider_type', 'id')
            session: optional shared aiohttp session (TEST ALL reuses one)

        Returns:
            dict with id, name, provider_type, success, latency_ms, error,
            rate_limited
        """
        url = endpoint['url']
        provider_type = endpoint['provider_type']

        start_time = time.time()
        success = False
        error_message = None
        is_rate_limited = False

        # When no session is supplied, manage our own (single-test path).
        own_session = session is None
        if own_session:
            session = aiohttp.ClientSession()
        try:
            try:
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

            # Report result to pool engine (fail-soft).
            try:
                if success:
                    await self.pool_engine.report_success(provider_type, url, latency_ms)
                elif is_rate_limited:
                    # Report rate limit with 5 minute cooldown
                    await self.pool_engine.report_rate_limit(provider_type, url, duration_seconds=300, error_message=error_message)
                else:
                    await self.pool_engine.report_failure(provider_type, url, 'test_failure', error_message)
            except Exception as report_err:
                self.logger.debug(f"Pool-engine report failed during probe: {report_err}")

            return {
                'id': endpoint.get('id'),
                'name': endpoint.get('name'),
                'provider_type': provider_type,
                'chain': endpoint.get('chain'),
                'success': success,
                'latency_ms': latency_ms,
                'error': error_message,
                'rate_limited': is_rate_limited,
            }
        finally:
            if own_session and session is not None:
                await session.close()

    async def test_endpoint(self, request: web.Request) -> web.Response:
        """
        Test a single endpoint's connectivity and response time.

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
            endpoints = await self.pool_engine.get_all_endpoints_data()
            endpoint = next((e for e in endpoints if e['id'] == endpoint_id), None)
            if not endpoint:
                return web.json_response({
                    'success': False,
                    'error': 'Endpoint not found'
                }, status=404)

            result = await self._probe_endpoint(endpoint)
            return web.json_response(result)

        except Exception as e:
            self.logger.error(f"Error testing endpoint: {e}", exc_info=True)
            return web.json_response({
                'success': False,
                'error': str(e)
            }, status=500)

    async def test_all_endpoints(self, request: web.Request) -> web.Response:
        """
        FEATURE 5 — TEST ALL: probe every endpoint and report connectivity +
        latency. Runs with bounded concurrency (default 8) over ONE shared
        aiohttp session so a large pool (85 endpoints) finishes quickly without
        opening 85 sockets at once or stampeding any single provider.

        Returns a per-endpoint result list plus a summary tally.
        """
        try:
            if not self.pool_engine:
                return web.json_response({
                    'success': False,
                    'error': 'Pool engine not initialized'
                }, status=500)

            endpoints = await self.pool_engine.get_all_endpoints_data()
            if not endpoints:
                return web.json_response({
                    'success': True, 'results': [], 'summary': {
                        'total': 0, 'passed': 0, 'failed': 0, 'rate_limited': 0
                    }
                })

            sem = asyncio.Semaphore(8)

            async def _bounded(session, ep):
                async with sem:
                    try:
                        return await self._probe_endpoint(ep, session=session)
                    except Exception as e:
                        return {
                            'id': ep.get('id'), 'name': ep.get('name'),
                            'provider_type': ep.get('provider_type'),
                            'chain': ep.get('chain'), 'success': False,
                            'latency_ms': 0, 'error': str(e), 'rate_limited': False,
                        }

            async with aiohttp.ClientSession() as session:
                results = await asyncio.gather(
                    *[_bounded(session, ep) for ep in endpoints]
                )

            passed = sum(1 for r in results if r.get('success'))
            rate_limited = sum(1 for r in results if r.get('rate_limited'))
            failed = len(results) - passed - rate_limited
            return web.json_response({
                'success': True,
                'results': results,
                'summary': {
                    'total': len(results),
                    'passed': passed,
                    'failed': failed,
                    'rate_limited': rate_limited,
                },
            })

        except Exception as e:
            self.logger.error(f"Error in test-all: {e}", exc_info=True)
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

    async def get_governor_status(self, request: web.Request) -> web.Response:
        """
        Wave-F7: per-(module, provider) RPC consumption + throttle state.

        Rows come from the rpc_governor_status table (each module subprocess
        flushes its own governor snapshot every ~30s) overlaid with this
        process's live snapshot. Fail-soft: pre-migration-151 or no DB
        returns an empty list, never an error the page can't render.
        """
        try:
            if not self.pool_engine:
                return web.json_response({'success': True, 'rows': [],
                                          'note': 'pool engine not initialized'})
            rows = await self.pool_engine.get_governor_status()
            return web.json_response({'success': True, 'rows': rows})
        except Exception as e:
            self.logger.error(f"Error getting governor status: {e}", exc_info=True)
            return web.json_response({'success': True, 'rows': [],
                                      'note': str(e)})

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
