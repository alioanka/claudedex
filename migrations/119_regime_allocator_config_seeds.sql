-- Migration 119: regime_allocator — config seeds (config_type='regime_allocator')
--
-- Every knob defaults safe: the module is ADVISORY ONLY, default DISABLED via
-- REGIME_ALLOCATOR_MODULE_ENABLED (env, default false), and the only optional
-- side write (mirror_to_portfolio_allocations) seeds FALSE. Applying this
-- migration changes no behavior.
--
-- Single-quoted SQL literals only ('' escapes apostrophes). Idempotent.

BEGIN;

INSERT INTO config_settings (config_type, key, value, value_type, description, created_at, updated_at)
VALUES
    ('regime_allocator', 'tick_interval_seconds', '3600', 'int',
     'Seconds between regime classification cycles (default hourly). Regimes '
     'move slowly; values below ~900 add API load with no signal benefit.',
     NOW(), NOW()),

    ('regime_allocator', 'kline_interval', '4h', 'string',
     'Candle size for the BTC/ETH series: 1h, 4h or 1d (default 4h).',
     NOW(), NOW()),

    ('regime_allocator', 'kline_limit', '180', 'int',
     'Bars fetched per asset per tick (180 x 4h = 30 days of context).',
     NOW(), NOW()),

    ('regime_allocator', 'short_vol_bars', '24', 'int',
     'Recent realized-vol window in bars (24 x 4h = 4 days).', NOW(), NOW()),

    ('regime_allocator', 'long_vol_bars', '96', 'int',
     'Baseline realized-vol window in bars (96 x 4h = 16 days).', NOW(), NOW()),

    ('regime_allocator', 'trend_bars', '42', 'int',
     'Kaufman efficiency-ratio window in bars (42 x 4h = 7 days).', NOW(), NOW()),

    ('regime_allocator', 'vol_expand_ratio', '1.15', 'float',
     'short/long realized-vol ratio at or above which vol is judged EXPANSION.',
     NOW(), NOW()),

    ('regime_allocator', 'vol_compress_ratio', '0.85', 'float',
     'Ratio at or below which vol is judged COMPRESSION. The gap to '
     'vol_expand_ratio is anti-flap hysteresis.', NOW(), NOW()),

    ('regime_allocator', 'er_trend', '0.35', 'float',
     'Efficiency ratio at or above which the tape is judged TRENDING.',
     NOW(), NOW()),

    ('regime_allocator', 'er_range', '0.20', 'float',
     'Efficiency ratio at or below which the tape is judged RANGEBOUND.',
     NOW(), NOW()),

    ('regime_allocator', 'btc_weight', '0.6', 'float',
     'BTC vote weight in the cross-asset blend (ETH gets 1 minus this).',
     NOW(), NOW()),

    ('regime_allocator', 'reserve_pct', '10.0', 'float',
     'Always-unallocated share of the book in every proposal batch.',
     NOW(), NOW()),

    ('regime_allocator', 'chop_extra_reserve_pct', '10.0', 'float',
     'Extra reserve added in chop_expansion (confidence-scaled) — expanding '
     'vol without direction is where momentum books bleed.', NOW(), NOW()),

    ('regime_allocator', 'min_confidence', '0.25', 'float',
     'Below this regime confidence, tilts are fully suppressed and proposals '
     'fall back to base weights (a low-conviction read must not move capital).',
     NOW(), NOW()),

    ('regime_allocator', 'regime_tilts', '', 'json',
     'Optional JSON override of the regime->module tilt matrix, e.g. '
     '{''trend_expansion'': {''futures'': 1.5}}. Partial overrides merge over '
     'the code defaults; empty = defaults.', NOW(), NOW()),

    ('regime_allocator', 'base_weights', '', 'json',
     'Optional JSON per-module base weights before the regime tilt, e.g. '
     '{''futures'': 2.0, ''dex'': 1.0}. Empty = equal weights.', NOW(), NOW()),

    ('regime_allocator', 'mirror_to_portfolio_allocations', 'false', 'bool',
     'When true, also write each proposal batch into portfolio_allocations '
     'with proposed_by=''regime'' so the existing allocator dashboard panel '
     'shows both authorities side-by-side. Default false (own tables only).',
     NOW(), NOW())
ON CONFLICT (config_type, key) DO NOTHING;

COMMIT;
