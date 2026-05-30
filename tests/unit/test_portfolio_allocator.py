"""Unit tests for modules.portfolio_allocator.core.allocator.

Pure-function tests — no DB. Each test pins one decision path.
"""

from __future__ import annotations

import pytest

from modules.portfolio_allocator.core.allocator import (
    AllocationInput,
    DEFAULT_MODULE_CEILING_PCT,
    DEFAULT_MODULE_FLOOR_PCT,
    DEFAULT_RESERVE_PCT,
    allocate,
)


def _sum_alloc_pct(report):
    return sum(p.pct_of_book for p in report.proposals)


# ----- edge cases -----

def test_no_enabled_modules_reserves_everything():
    r = allocate([AllocationInput('m', False, 1.0)], 1000.0)
    assert r.reserve_pct == 100.0
    assert all(p.pct_of_book == 0.0 for p in r.proposals)


def test_empty_input_reserves_everything():
    r = allocate([], 1000.0)
    assert r.reserve_pct == 100.0
    assert r.proposals == []


def test_single_enabled_module_strong_sharpe_hits_ceiling():
    r = allocate(
        [AllocationInput('sniper', True, 5.0)],   # well above max_sharpe
        1000.0,
    )
    sniper = next(p for p in r.proposals if p.module == 'sniper')
    # ceiling caps at 40 (default), and post-clip re-normalize stays at ceiling
    # because there are no other modules to share the budget with.
    assert sniper.pct_of_book == DEFAULT_MODULE_CEILING_PCT


def test_single_enabled_zero_sharpe_gets_floor():
    r = allocate(
        [AllocationInput('arbitrage', True, 0.0)],
        1000.0,
    )
    arb = next(p for p in r.proposals if p.module == 'arbitrage')
    # zero sharpe → equal floor allocation path
    assert arb.pct_of_book >= DEFAULT_MODULE_FLOOR_PCT


def test_disabled_modules_get_zero_proposal():
    r = allocate(
        [
            AllocationInput('sniper', True, 1.5),
            AllocationInput('futures', False, None),
            AllocationInput('ai', False, 99.0),  # high sharpe but disabled
        ],
        1000.0,
    )
    futures = next(p for p in r.proposals if p.module == 'futures')
    ai = next(p for p in r.proposals if p.module == 'ai')
    assert futures.pct_of_book == 0.0
    assert ai.pct_of_book == 0.0


# ----- normalization -----

def test_normalization_sum_plus_reserve_equals_100():
    r = allocate(
        [
            AllocationInput('sniper', True, 1.5),
            AllocationInput('arbitrage', True, 0.5),
            AllocationInput('copy_trading', True, 1.0),
        ],
        1000.0,
    )
    total = _sum_alloc_pct(r) + r.reserve_pct
    assert 99.0 <= total <= 101.0, f"unbalanced: {total}"


def test_sharpe_none_treated_as_zero():
    r = allocate(
        [
            AllocationInput('sniper', True, None),
            AllocationInput('arbitrage', True, 1.5),
        ],
        1000.0,
    )
    sniper = next(p for p in r.proposals if p.module == 'sniper')
    arb = next(p for p in r.proposals if p.module == 'arbitrage')
    # Arb should be allocated MORE than sniper (sniper has no sharpe data)
    assert arb.pct_of_book >= sniper.pct_of_book


def test_proportional_split_with_two_equal_sharpe():
    r = allocate(
        [
            AllocationInput('sniper', True, 1.0),
            AllocationInput('arbitrage', True, 1.0),
        ],
        1000.0,
    )
    sniper = next(p for p in r.proposals if p.module == 'sniper')
    arb = next(p for p in r.proposals if p.module == 'arbitrage')
    # Equal sharpe → equal allocation
    assert abs(sniper.pct_of_book - arb.pct_of_book) < 0.5


def test_usd_amount_tracks_pct():
    r = allocate(
        [AllocationInput('sniper', True, 1.0)],
        2000.0,
    )
    p = next(p for p in r.proposals if p.module == 'sniper')
    # usd_amount should be pct_of_book / 100 * total_book
    expected = p.pct_of_book / 100.0 * 2000.0
    assert abs(p.usd_amount - expected) < 0.5


def test_floor_ceiling_clip_holds_under_extreme_input():
    """Even with an absurd Sharpe spread, no module exceeds ceiling
    and no enabled module goes below floor."""
    r = allocate(
        [
            AllocationInput('sniper', True, 999.0),       # huge
            AllocationInput('arbitrage', True, 0.001),    # tiny
            AllocationInput('copy_trading', True, 0.001), # tiny
        ],
        1000.0,
    )
    for p in r.proposals:
        if p.module == 'sniper':
            assert p.pct_of_book <= DEFAULT_MODULE_CEILING_PCT + 0.5
        else:
            assert p.pct_of_book >= DEFAULT_MODULE_FLOOR_PCT - 0.5


def test_custom_max_sharpe_affects_allocation():
    base_inputs = [
        AllocationInput('sniper', True, 0.5),
        AllocationInput('arbitrage', True, 0.5),
    ]
    r1 = allocate(base_inputs, 1000.0, max_sharpe=2.0)
    r2 = allocate(base_inputs, 1000.0, max_sharpe=1.0)
    # Smaller max_sharpe → same inputs map to higher raw_f → higher
    # pre-norm allocation. Both should sum to risk_budget, but the
    # raw_kelly_f component should differ.
    sniper1 = next(p for p in r1.proposals if p.module == 'sniper')
    sniper2 = next(p for p in r2.proposals if p.module == 'sniper')
    # raw_kelly_f changes; normalized pct may be the same after clip
    assert sniper2.raw_kelly_f >= sniper1.raw_kelly_f
