# tests/unit/test_dry_run.py
"""Smoke tests for core.dry_run."""
import pytest

from core import dry_run
from core.dry_run import (
    is_global_kill_switch,
    resolve_dry_run_env,
    set_global_kill_switch,
    should_skip_live,
)


@pytest.fixture(autouse=True)
def _reset_kill_switch():
    set_global_kill_switch(False)
    yield
    set_global_kill_switch(False)


@pytest.mark.unit
def test_resolve_dry_run_env_truthy(monkeypatch):
    for v in ("true", "TRUE", "1", "yes", "on", " True "):
        monkeypatch.setenv("DRY_RUN", v)
        assert resolve_dry_run_env() is True


@pytest.mark.unit
def test_resolve_dry_run_env_falsy(monkeypatch):
    for v in ("false", "FALSE", "0", "no", "off"):
        monkeypatch.setenv("DRY_RUN", v)
        assert resolve_dry_run_env() is False


@pytest.mark.unit
def test_resolve_dry_run_env_missing(monkeypatch):
    monkeypatch.delenv("DRY_RUN", raising=False)
    assert resolve_dry_run_env() is True  # safe default
    assert resolve_dry_run_env(default=False) is False


@pytest.mark.unit
def test_resolve_dry_run_env_typo(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "ture")  # typo -> safe default
    assert resolve_dry_run_env() is True
    assert resolve_dry_run_env(default=False) is False


@pytest.mark.unit
def test_should_skip_live_honors_module_flag():
    assert should_skip_live(True) is True
    assert should_skip_live(False) is False
    assert should_skip_live(True, module="solana", account="abc") is True


@pytest.mark.unit
def test_should_skip_live_honors_global_kill_switch():
    assert should_skip_live(False) is False
    set_global_kill_switch(True)
    assert should_skip_live(False) is True
    assert should_skip_live(False, module="dex") is True


@pytest.mark.unit
def test_global_kill_switch_toggle():
    assert is_global_kill_switch() is False
    set_global_kill_switch(True)
    assert is_global_kill_switch() is True
    set_global_kill_switch(False)
    assert is_global_kill_switch() is False


@pytest.mark.unit
def test_module_state_round_trip():
    # ensure import surface stable
    assert hasattr(dry_run, "should_skip_live")
    assert hasattr(dry_run, "resolve_dry_run_env")
