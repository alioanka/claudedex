"""AI-Q-07 — LSTM rolling buffer regression tests.

Verifies that `EnsemblePredictor`:
  * defaults `lstm_rolling_buffer_enabled` to False (no behavior change).
  * when enabled, pushes per-(token, chain) feature rows into the buffer
    and emits a real (1, seq_len, F) tensor — never the degenerate
    (1, 1, F) shape the legacy code path produced.
  * caps memory via LRU eviction once
    `lstm_buffer_max_tokens` is exceeded.
  * edge-pads when the buffer is shorter than seq_len.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from ml.models.ensemble_model import EnsemblePredictor  # noqa: E402


def test_buffer_off_by_default():
    """Default constructor must NOT enable the rolling buffer."""
    p = EnsemblePredictor(config={})
    assert p.lstm_rolling_buffer_enabled is False
    assert p.lstm_sequence_length == 20
    assert p._lstm_feature_buffer == {}


def test_buffer_push_grows_then_trims_to_seq_len():
    """Push N>seq_len rows; window must always equal last seq_len rows."""
    p = EnsemblePredictor(config={
        'lstm_rolling_buffer_enabled': True,
        'lstm_sequence_length': 4,
    })
    key = 'ethereum:0xABC'
    rows = [np.arange(3, dtype=np.float32) + i for i in range(7)]
    for r in rows:
        win = p._lstm_push_features(key, r.reshape(1, -1))
    # After 7 pushes with seq_len=4, the window should be the last 4 rows.
    expected = np.stack(rows[-4:])
    assert win.shape == (4, 3)
    assert np.allclose(win, expected)


def test_buffer_build_input_pads_short_window():
    """Window shorter than seq_len must be edge-padded to (1, T, F)."""
    p = EnsemblePredictor(config={
        'lstm_rolling_buffer_enabled': True,
        'lstm_sequence_length': 5,
    })
    short = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=np.float32)
    out = p._lstm_build_input(short)
    assert out.shape == (1, 5, 3)
    # Edge pad reuses the first row.
    np_out = out.numpy()[0]
    assert np.allclose(np_out[0], np_out[1])
    assert np.allclose(np_out[1], np_out[2])
    # The trailing two rows should be the original input.
    assert np.allclose(np_out[-2:], short)


def test_buffer_build_input_passthrough_when_full():
    """Full window must NOT be padded; shape (1, seq_len, F) preserved."""
    p = EnsemblePredictor(config={
        'lstm_rolling_buffer_enabled': True,
        'lstm_sequence_length': 3,
    })
    full = np.arange(9, dtype=np.float32).reshape(3, 3)
    out = p._lstm_build_input(full)
    assert out.shape == (1, 3, 3)
    assert np.allclose(out.numpy()[0], full)


def test_buffer_lru_eviction():
    """Once tokens exceed the cap, the oldest is evicted."""
    p = EnsemblePredictor(config={
        'lstm_rolling_buffer_enabled': True,
        'lstm_sequence_length': 2,
        'lstm_buffer_max_tokens': 3,
    })
    for i in range(5):
        p._lstm_push_features(f'ethereum:0x{i:040x}',
                              np.zeros((1, 4), dtype=np.float32))
    # Only the last 3 keys survive.
    assert len(p._lstm_feature_buffer) == 3
    assert 'ethereum:0x' + '0' * 39 + '0' not in p._lstm_feature_buffer
    assert 'ethereum:0x' + '0' * 39 + '4' in p._lstm_feature_buffer


def test_buffer_per_token_isolation():
    """Pushing rows under different keys must NOT cross-contaminate."""
    p = EnsemblePredictor(config={
        'lstm_rolling_buffer_enabled': True,
        'lstm_sequence_length': 5,
    })
    a_row = np.ones((1, 3), dtype=np.float32)
    b_row = np.full((1, 3), 7.0, dtype=np.float32)
    p._lstm_push_features('ethereum:A', a_row)
    p._lstm_push_features('ethereum:B', b_row)
    p._lstm_push_features('ethereum:A', a_row)
    win_a = np.asarray(p._lstm_feature_buffer['ethereum:A'])
    win_b = np.asarray(p._lstm_feature_buffer['ethereum:B'])
    assert win_a.shape == (2, 3)
    assert win_b.shape == (1, 3)
    assert np.allclose(win_a, 1.0)
    assert np.allclose(win_b, 7.0)
