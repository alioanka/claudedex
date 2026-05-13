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
