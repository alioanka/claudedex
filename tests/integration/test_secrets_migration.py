"""Regression and contract tests for the secrets-encryption migration sweep."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

MIGRATION_MATRIX: Dict[str, Dict[str, List[str]]] = {
    'modules/dex_trading/main_dex.py': {
        'sensitive_keys': [
            'PRIVATE_KEY', 'WEB3_PROVIDER_URL',
            'WEB3_BACKUP_PROVIDER_1', 'WEB3_BACKUP_PROVIDER_2',
            'DEXSCREENER_API_KEY', 'TWITTER_API_KEY', 'TWITTER_API_SECRET',
        ],
        'markers': [],
    },
    'modules/solana_trading/main_solana.py': {
        'sensitive_keys': [],
        'markers': ['secrets.initialize('],
    },
    'modules/solana_trading/config/solana_config_manager.py': {
        'sensitive_keys': [],
        'markers': ['SENSITIVE_KEYS', 'from security.secrets_manager import secrets'],
    },
    'modules/sniper/main_sniper.py': {
        'sensitive_keys': [],
        'markers': ['secrets.initialize('],
    },
    'modules/sniper/core/trade_executor.py': {
        'sensitive_keys': [],
        'markers': ['_resolve_jupiter_base', 'JUPITER_API_URL'],
    },
    'modules/arbitrage/main_arbitrage.py': {
        'sensitive_keys': [],
        'markers': ['secrets.initialize('],
    },
    'modules/arbitrage/triangular_engine.py': {
        'sensitive_keys': ['ETHERSCAN_API_KEY', 'WALLET_ADDRESS'],
        'markers': [],
    },
    'modules/arbitrage/solana_engine.py': {
        'sensitive_keys': ['JITO_BLOCK_ENGINE_URL', 'JITO_TIP_ACCOUNT'],
        'markers': [],
    },
    'modules/copy_trading/main_copy.py': {
        'sensitive_keys': ['ETHERSCAN_API_KEY', 'HELIUS_API_KEY'],
        'markers': ['secrets.initialize('],
    },
    'modules/futures_trading/main_futures.py': {
        'sensitive_keys': [],
        'markers': ['secrets.initialize('],
    },
    'modules/futures_trading/config/futures_config.py': {
        'sensitive_keys': ['BINANCE_API_KEY', 'BYBIT_API_KEY'],
        'markers': [],
    },
    'modules/futures_trading/config/futures_config_manager.py': {
        'sensitive_keys': [],
        'markers': ['SENSITIVE_KEYS'],
    },
    'modules/ai_analysis/core/sentiment_engine.py': {
        'sensitive_keys': [],
        'markers': ['secrets.get_async', 'from security.secrets_manager import secrets'],
    },
}


def _sensitive_key_cases() -> List[Tuple[str, str]]:
    cases = []
    for file_path, spec in MIGRATION_MATRIX.items():
        for key in spec['sensitive_keys']:
            cases.append(pytest.param(file_path, key, id=f"{file_path}::{key}"))
    return cases


def _marker_cases() -> List[Tuple[str, str]]:
    cases = []
    for file_path, spec in MIGRATION_MATRIX.items():
        for marker in spec['markers']:
            cases.append(pytest.param(file_path, marker, id=f"{file_path}::{marker}"))
    return cases


@pytest.mark.integration
class TestSecretsMigrationStatic:

    @pytest.mark.parametrize("file_path,key", _sensitive_key_cases())
    def test_sensitive_key_routes_through_secrets_manager(self, file_path: str, key: str):
        contents = (REPO_ROOT / file_path).read_text()
        candidates = (
            f"secrets.get('{key}'",
            f'secrets.get("{key}"',
            f"secrets.get_async('{key}'",
            f'secrets.get_async("{key}"',
        )
        assert any(c in contents for c in candidates), (
            f"{file_path} no longer routes {key} through secrets_manager "
            f"(expected one of {candidates})"
        )

    @pytest.mark.parametrize("file_path,marker", _marker_cases())
    def test_file_contains_marker(self, file_path: str, marker: str):
        contents = (REPO_ROOT / file_path).read_text()
        assert marker in contents, (
            f"{file_path} is missing migration marker substring: {marker!r}"
        )


@pytest.mark.integration
class TestDexSecretsHotPath:
    """Regression checks for the DEX module hot-path PoolEngine routing
    introduced after the secrets sweep (follow-up to a21ec41)."""

    DEX_MAIN = REPO_ROOT / 'modules/dex_trading/main_dex.py'

    @pytest.mark.parametrize("needle", [
        pytest.param(
            "RPCProvider.get_rpc_sync('ETHEREUM_RPC')",
            id="test_web3_connection_consults_rpc_provider_first",
        ),
    ])
    def test_test_web3_connection_consults_rpc_provider_first(self, needle: str):
        contents = self.DEX_MAIN.read_text()
        assert needle in contents, (
            f"main_dex.py no longer routes test_web3_connection through "
            f"RPCProvider; missing substring: {needle!r}"
        )

    def test_validate_environment_falls_through_secrets_then_env(self):
        contents = self.DEX_MAIN.read_text()
        for needle in (
            "RPCProvider.get_rpc_sync",
            "secrets.get('WEB3_PROVIDER_URL'",
            "os.getenv('WEB3_PROVIDER_URL')",
        ):
            assert needle in contents, (
                f"main_dex.py is missing 3-tier WEB3_PROVIDER_URL resolution "
                f"substring: {needle!r}"
            )

    def test_rpc_urls_scan_consults_pool_engine_first(self):
        contents = self.DEX_MAIN.read_text()
        for needle in ("KNOWN_CHAINS", "get_rpcs_sync", "endswith('_RPC_URLS')"):
            assert needle in contents, (
                f"main_dex.py _RPC_URLS scan no longer prefers PoolEngine; "
                f"missing substring: {needle!r}"
            )


@pytest.mark.integration
class TestSecretsManagerContract:

    @pytest.fixture(autouse=True)
    def reset_secrets_singleton(self):
        from security.secrets_manager import SecureSecretsManager
        SecureSecretsManager._instance = None
        yield
        SecureSecretsManager._instance = None

    def test_bootstrap_mode_falls_through_to_env(self, monkeypatch):
        from security.secrets_manager import SecureSecretsManager
        key = f"TEST_BOOT_KEY_{uuid.uuid4().hex[:8]}"
        monkeypatch.setenv(key, "env-value")
        mgr = SecureSecretsManager.get_instance()
        mgr.initialize()
        assert mgr._bootstrap_mode is True
        assert mgr.get(key) == "env-value"

    def test_cache_hit_short_circuits_db_and_env(self, monkeypatch):
        from security.secrets_manager import SecureSecretsManager
        key = f"CACHED_KEY_{uuid.uuid4().hex[:8]}"
        monkeypatch.delenv(key, raising=False)
        mgr = SecureSecretsManager.get_instance()
        mgr.initialize()
        mgr._cache[key] = "cached-value"
        assert mgr.get(key) == "cached-value"

    def test_missing_key_returns_default(self):
        from security.secrets_manager import SecureSecretsManager
        mgr = SecureSecretsManager.get_instance()
        mgr.initialize()
        missing = f"DEFINITELY_NOT_SET_{uuid.uuid4().hex}"
        assert mgr.get(missing, default="fallback") == "fallback"

    def test_log_access_false_does_not_leak_value(self, caplog, monkeypatch):
        from security.secrets_manager import SecureSecretsManager
        key = f"PRIVATE_KEY_TEST_{uuid.uuid4().hex[:8]}"
        monkeypatch.delenv(key, raising=False)
        mgr = SecureSecretsManager.get_instance()
        mgr.initialize()
        secret_value = "leaked-value-xyz"
        mgr._cache[key] = secret_value
        with caplog.at_level(logging.DEBUG):
            result = mgr.get(key, log_access=False)
        assert result == secret_value
        assert secret_value not in caplog.text

    def test_cache_encrypted_value_triggers_refetch_with_warning(
        self, caplog, monkeypatch
    ):
        """If encrypted ciphertext ever leaks into _cache (bug, test
        pollution), get() must NOT return it — it must skip the cache,
        re-fetch, and emit a warning so the violation is noisy."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            pytest.skip("cryptography package not installed")
        from security.secrets_manager import SecureSecretsManager
        fernet_key = Fernet.generate_key()
        fernet = Fernet(fernet_key)
        ciphertext = fernet.encrypt(b"the-real-secret").decode()
        key = f"POLLUTED_KEY_{uuid.uuid4().hex[:8]}"
        # Populate env with the SAME ciphertext so the re-fetch path
        # finds a valid encrypted value that fernet can decrypt.
        monkeypatch.setenv(key, ciphertext)
        mgr = SecureSecretsManager.get_instance()
        mgr.initialize()
        mgr._fernet = fernet
        # Pollute the cache directly with ciphertext — simulates the bug
        # the safety net is defending against.
        mgr._cache[key] = ciphertext
        with caplog.at_level(logging.WARNING):
            result = mgr.get(key)
        # Re-fetch path resolved from env, decrypted, and returned plaintext
        assert result == "the-real-secret", (
            f"get() returned ciphertext or wrong value: {result!r}"
        )
        # Warning surfaced so a real future regression would be noisy
        assert "Cache invariant violation" in caplog.text or \
               any("encrypted value found" in r.message for r in caplog.records), \
            f"Expected cache-invariant warning, got: {caplog.text}"

    def test_fernet_encrypted_value_decrypts_on_read(self, monkeypatch):
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            pytest.skip("cryptography package not installed")
        from security.secrets_manager import SecureSecretsManager
        fernet_key = Fernet.generate_key()
        fernet = Fernet(fernet_key)
        token = fernet.encrypt(b"plain-secret").decode()
        env_key = f"ENC_KEY_TEST_{uuid.uuid4().hex[:8]}"
        monkeypatch.setenv(env_key, token)
        mgr = SecureSecretsManager.get_instance()
        mgr.initialize()
        mgr._fernet = fernet
        assert mgr.get(env_key) == "plain-secret"


@pytest.mark.integration
class TestReconcileContract:
    """Locks the BaseModule reconcile contract and its FUTURES/DEX overrides
    plus the dashboard alert surface (3981ffd, 24241e3, 592cb1b)."""

    BASE_MODULE = REPO_ROOT / 'modules/base_module.py'
    FUTURES_MODULE = REPO_ROOT / 'modules/futures_trading/futures_module.py'
    DEX_MODULE = REPO_ROOT / 'modules/dex_trading/dex_module.py'
    FUTURES_ENGINE = REPO_ROOT / 'modules/futures_trading/core/futures_engine.py'
    FUTURES_RISK_MGR = REPO_ROOT / 'modules/futures_trading/futures_risk_manager.py'

    def test_base_module_defines_reconcile_hook(self):
        contents = self.BASE_MODULE.read_text()
        for needle in (
            "async def reconcile_open_positions",
            "_evaluate_restart_alert",
            "self.last_reconcile_at",
            "self.last_reconcile_count",
            "self.last_restart_alert",
        ):
            assert needle in contents, (
                f"base_module.py missing reconcile contract marker: {needle!r}"
            )

    def test_start_wrap_auto_calls_reconcile(self):
        contents = self.BASE_MODULE.read_text()
        # The __init_subclass__ wrap should call reconcile after start succeeds
        assert "await self.reconcile_open_positions()" in contents, (
            "base_module.py __init_subclass__ no longer auto-calls "
            "reconcile_open_positions after start()"
        )

    def test_futures_module_overrides_reconcile(self):
        contents = self.FUTURES_MODULE.read_text()
        for needle in (
            "async def reconcile_open_positions",
            "_sync_positions",
            "_evaluate_restart_alert",
        ):
            assert needle in contents, (
                f"futures_module.py reconcile override missing: {needle!r}"
            )

    def test_dex_module_overrides_reconcile(self):
        contents = self.DEX_MODULE.read_text()
        for needle in (
            "async def reconcile_open_positions",
            "position_tracker",
            "_evaluate_restart_alert",
        ):
            assert needle in contents, (
                f"dex_module.py reconcile override missing: {needle!r}"
            )

    def test_futures_engine_stamps_reconcile_observability(self):
        contents = self.FUTURES_ENGINE.read_text()
        for needle in (
            "self.last_reconcile_at",
            "self.last_reconcile_count",
            "RESTART OVER-CAP",
            "check_reconciled_capacity",
        ):
            assert needle in contents, (
                f"futures_engine.py reconcile observability missing: {needle!r}"
            )

    def test_futures_risk_manager_exposes_capacity_check(self):
        contents = self.FUTURES_RISK_MGR.read_text()
        assert "def check_reconciled_capacity" in contents, (
            "futures_risk_manager.py no longer exposes check_reconciled_capacity"
        )


@pytest.mark.integration
class TestSubprocessDiscovery:
    """Locks ModuleManager subprocess-discovery surface (7808ed0)."""

    MODULE_MGR = REPO_ROOT / 'core/module_manager.py'

    def test_subprocess_constants_present(self):
        contents = self.MODULE_MGR.read_text()
        for needle in (
            "SUBPROCESS_MODULE_DIRS",
            "SUBPROCESS_LIVENESS_THRESHOLD_SECONDS",
            "_discover_subprocess_modules",
        ):
            assert needle in contents, (
                f"module_manager.py subprocess discovery surface missing: {needle!r}"
            )

    def test_status_summary_unions_both_sources(self):
        contents = self.MODULE_MGR.read_text()
        for needle in (
            "'source': 'in_process'",
            "'source': 'subprocess'",
            "in_process_count",
            "subprocess_count",
        ):
            assert needle in contents, (
                f"module_manager.py get_status_summary not unioning sources: {needle!r}"
            )


@pytest.mark.integration
class TestDashboardReconcileSurface:
    """Locks dashboard last_reconcile_at + RESTART OVER-CAP banner (592cb1b)."""

    MODULES_TEMPLATE = REPO_ROOT / 'dashboard/templates/modules.html'
    BASE_TEMPLATE = REPO_ROOT / 'dashboard/templates/base.html'

    def test_modules_template_renders_reconcile_tile(self):
        contents = self.MODULES_TEMPLATE.read_text()
        for needle in (
            "Last Reconcile",
            "last_reconcile_at",
            "last_restart_alert",
            "restart-alert-badge",
        ):
            assert needle in contents, (
                f"modules.html missing reconcile-surface element: {needle!r}"
            )

    def test_base_template_has_restart_alert_banner(self):
        contents = self.BASE_TEMPLATE.read_text()
        for needle in (
            "restart-alert-banner",
            "last_restart_alert",
            "/api/modules",
        ):
            assert needle in contents, (
                f"base.html missing restart-alert banner element: {needle!r}"
            )


@pytest.mark.integration
class TestPositionNormalizer:
    """Locks the Binance/Bybit position/balance normalizer (ed350d0)."""

    NORMALIZERS = REPO_ROOT / 'modules/futures_trading/exchanges/_normalizers.py'
    RISK_MGR = REPO_ROOT / 'modules/futures_trading/futures_risk_manager.py'

    def test_normalizer_module_exists(self):
        assert self.NORMALIZERS.exists(), (
            "modules/futures_trading/exchanges/_normalizers.py missing"
        )
        contents = self.NORMALIZERS.read_text()
        for needle in (
            "def normalize_position",
            "def normalize_balance",
            "CANONICAL_POSITION_KEYS",
        ):
            assert needle in contents, (
                f"_normalizers.py missing surface: {needle!r}"
            )

    def test_risk_manager_auto_normalizes_when_liq_price_missing(self):
        contents = self.RISK_MGR.read_text()
        assert "normalize_position" in contents, (
            "futures_risk_manager.py no longer auto-normalizes position dicts"
        )

    def test_bybit_shape_canonicalizes_with_liq_price_recovery(self):
        try:
            from modules.futures_trading.exchanges import normalize_position
        except ImportError as e:
            pytest.skip(f"normalize_position not importable: {e}")
        raw = {
            'symbol': 'BTCUSDT', 'side': 'LONG', 'size': '0.5',
            'entry_price': '50000', 'mark_price': '50100',
            'unrealised_pnl': '5', 'leverage': '10',
            'position_value': '25050',
            'raw': {'liqPrice': '45000', 'tradeMode': 1},
        }
        result = normalize_position(raw, 'bybit')
        assert result is not None
        assert result['liquidation_price'] == 45000.0, result
        assert result['margin_type'] == 'ISOLATED', result
        assert result['unrealized_pnl'] == 5.0, result
        assert result['size'] == 0.5, result
        assert result['notional_value'] == 25050.0, result

    def test_binance_shape_canonicalizes(self):
        try:
            from modules.futures_trading.exchanges import normalize_position
        except ImportError as e:
            pytest.skip(f"normalize_position not importable: {e}")
        raw = {
            'symbol': 'BTCUSDT', 'side': 'LONG', 'position_amt': 0.5,
            'entry_price': 50000, 'mark_price': 50100,
            'unrealized_pnl': 5, 'leverage': 10,
            'notional_value': 25050, 'liquidation_price': 45000,
            'margin_type': 'ISOLATED',
        }
        result = normalize_position(raw, 'binance')
        assert result is not None
        assert result['liquidation_price'] == 45000.0
        assert result['size'] == 0.5
        assert result['unrealized_pnl'] == 5.0
        assert result['notional_value'] == 25050.0

    def test_normalize_position_handles_none(self):
        try:
            from modules.futures_trading.exchanges import normalize_position
        except ImportError as e:
            pytest.skip(f"normalize_position not importable: {e}")
        assert normalize_position(None, 'bybit') is None
        assert normalize_position({}, 'binance') is None


@pytest.mark.integration
class TestPoolEngineSweep:
    """Locks pool_engine enforcement across the 10 migrated files
    (a21ec41 main sweep + helper-module follow-ups)."""

    @pytest.mark.parametrize("file_path", [
        pytest.param(p, id=p) for p in [
            'modules/arbitrage/main_arbitrage.py',
            'modules/arbitrage/triangular_engine.py',
            'modules/arbitrage/solana_engine.py',
            'modules/sniper/core/trade_executor.py',
            'modules/sniper/core/evm_listener.py',
            'modules/sniper/core/solana_listener.py',
            'modules/copy_trading/copy_engine.py',
            'modules/solana_trading/config/solana_config.py',
            'modules/solana_strategies/jupiter_helper.py',
            'modules/solana_strategies/drift_helper.py',
        ]
    ])
    def test_file_uses_rpc_provider(self, file_path: str):
        contents = (REPO_ROOT / file_path).read_text()
        # At least one of get_rpc / get_rpc_sync / get_rpcs / get_rpcs_sync
        assert any(needle in contents for needle in (
            "RPCProvider.get_rpc",
            "RPCProvider.get_rpcs",
        )), (
            f"{file_path} no longer routes RPC reads through RPCProvider — "
            f"pool_engine enforcement sweep (a21ec41) regressed"
        )


@pytest.mark.integration
class TestMigrationScriptCoverage:
    """Locks parity between secrets_manager.get()-requested keys
    (across all modules) and the migration script's CREDENTIAL_MAPPINGS.
    If a module starts asking for a new key, this test fails — operator
    runs the migration script and the new key gets seeded too.

    Allow-list: keys that modules request but are intentionally
    bootstrap-only (ENCRYPTION_KEY) or have a known migration alias."""

    MIGRATE_SCRIPT = REPO_ROOT / 'scripts/migrate_credentials_to_db.py'
    EXCLUDED_KEYS = {
        # Bootstrap-only — must come from .env / .encryption_key file
        'ENCRYPTION_KEY',
        # Add other allow-listed keys here with a comment explaining why
    }

    def _migration_script_keys(self) -> set:
        """Parse the script's CREDENTIAL_MAPPINGS keys via regex.
        AST-level parse avoids executing the script (which imports DB)."""
        import re
        src = self.MIGRATE_SCRIPT.read_text()
        # Match top-level keys inside CREDENTIAL_MAPPINGS dict literal
        block_match = re.search(
            r"CREDENTIAL_MAPPINGS\s*=\s*\{(.*?)\n\}",
            src, re.DOTALL,
        )
        if not block_match:
            return set()
        body = block_match.group(1)
        return set(re.findall(r"^\s*'([A-Z_0-9]+)'\s*:\s*\{", body, re.MULTILINE))

    def _module_requested_keys(self) -> set:
        """Grep all module sources for secrets.get(...) / get_async(...)
        and extract the key names."""
        import re
        keys = set()
        modules_dir = REPO_ROOT / 'modules'
        for py in modules_dir.rglob('*.py'):
            try:
                src = py.read_text()
            except Exception:
                continue
            for m in re.finditer(
                r"secrets\.(?:get|get_async)\(\s*['\"]([A-Z_0-9]+)['\"]",
                src,
            ):
                keys.add(m.group(1))
        return keys

    def test_every_requested_key_is_migrated(self):
        requested = self._module_requested_keys()
        migrated = self._migration_script_keys()
        missing = requested - migrated - self.EXCLUDED_KEYS
        assert not missing, (
            f"Modules request {len(missing)} keys that the migration "
            f"script doesn't seed — operators would silently fall back "
            f"to .env for these: {sorted(missing)}"
        )
