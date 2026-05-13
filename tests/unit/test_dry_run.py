# tests/unit/test_dry_run.py
"""Smoke tests for core.dry_run."""
import asyncio
import logging
from pathlib import Path

import pytest

from core import dry_run
from core.dry_run import (
    is_global_kill_switch,
    is_module_paused,
    resolve_dry_run_env,
    set_global_kill_switch,
    set_module_pause,
    should_skip_live,
    start_killswitch_poller,
    stop_killswitch_poller,
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
    assert hasattr(dry_run, "start_killswitch_poller")
    assert hasattr(dry_run, "stop_killswitch_poller")


@pytest.fixture
def _reset_poller():
    """Tear the per-process poller down between tests."""
    yield
    stop_killswitch_poller()
    set_global_kill_switch(False)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poller_flips_switch_when_file_present(tmp_path, _reset_poller):
    flag = tmp_path / ".killswitch"
    flag.write_text('{"reason": "test"}')
    assert is_global_kill_switch() is False
    start_killswitch_poller(flag, interval=0.05)
    await asyncio.sleep(0.2)
    assert is_global_kill_switch() is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poller_no_flip_when_file_absent(tmp_path, _reset_poller):
    flag = tmp_path / "does_not_exist"
    start_killswitch_poller(flag, interval=0.05)
    await asyncio.sleep(0.2)
    assert is_global_kill_switch() is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poller_is_idempotent(tmp_path, caplog, _reset_poller):
    p1 = tmp_path / ".killswitch"
    p2 = tmp_path / "other.killswitch"
    t1 = start_killswitch_poller(p1, interval=0.05)
    t2 = start_killswitch_poller(p1, interval=0.05)
    assert t1 is t2
    with caplog.at_level(logging.WARNING, logger="core.dry_run"):
        t3 = start_killswitch_poller(p2, interval=0.05)
    assert t3 is t1
    assert any("already running" in r.message for r in caplog.records)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_poller_survives_read_errors(tmp_path, _reset_poller):
    # Make the killswitch path a directory: exists() True, read_text() fails.
    flag_dir = tmp_path / ".killswitch"
    flag_dir.mkdir()
    task = start_killswitch_poller(flag_dir, interval=0.05)
    await asyncio.sleep(0.2)
    assert is_global_kill_switch() is True
    assert not task.done()


@pytest.mark.unit
def test_is_module_paused_no_flag(tmp_path, monkeypatch):
    """No flag file => not paused."""
    monkeypatch.setattr(dry_run, "_PAUSE_FLAG_DIR", tmp_path)
    assert is_module_paused("dex_trading") is False
    assert is_module_paused("") is False


@pytest.mark.unit
def test_is_module_paused_with_flag(tmp_path, monkeypatch):
    """Flag file present => paused."""
    monkeypatch.setattr(dry_run, "_PAUSE_FLAG_DIR", tmp_path)
    (tmp_path / ".pause_solana_strategies").write_text("")
    assert is_module_paused("solana_strategies") is True
    assert is_module_paused("dex_trading") is False  # unrelated module


@pytest.mark.unit
def test_set_module_pause_writes_and_deletes(tmp_path, monkeypatch):
    monkeypatch.setattr(dry_run, "_PAUSE_FLAG_DIR", tmp_path)
    assert set_module_pause("futures_trading", True) is True
    assert (tmp_path / ".pause_futures_trading").exists()
    assert is_module_paused("futures_trading") is True
    assert set_module_pause("futures_trading", False) is True
    assert not (tmp_path / ".pause_futures_trading").exists()
    assert is_module_paused("futures_trading") is False


@pytest.mark.unit
def test_should_skip_live_honors_module_pause(tmp_path, monkeypatch):
    """MB-30: should_skip_live returns True when paused even if module_dry_run=False."""
    monkeypatch.setattr(dry_run, "_PAUSE_FLAG_DIR", tmp_path)
    assert should_skip_live(False, module="dex_trading") is False
    (tmp_path / ".pause_dex_trading").write_text("")
    assert should_skip_live(False, module="dex_trading") is True
    # Unrelated modules unaffected
    assert should_skip_live(False, module="solana_strategies") is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_base_module_start_wires_poller(_reset_poller):
    """Concrete BaseModule subclasses get the killswitch poller for free."""
    from modules.base_module import BaseModule, ModuleConfig, ModuleType

    class _TestModule(BaseModule):
        async def initialize(self): return True
        async def start(self):
            self.started_flag = True
            return True
        async def stop(self): return True
        async def process_opportunity(self, opp): return None
        async def get_positions(self): return []
        async def get_metrics(self): return self.metrics

    cfg = ModuleConfig(name="test_wiring", module_type=ModuleType.CUSTOM)
    inst = _TestModule(cfg)
    assert await inst.start() is True
    assert inst.started_flag is True
    # The poller task should now be live on this loop.
    assert dry_run._POLLER_TASK is not None
    assert not dry_run._POLLER_TASK.done()
