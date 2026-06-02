"""
signal_engine — multi-layer composite signal for the advisor module.

TRANSPARENCY NOTICE
-------------------
This is a deterministic, rule-based multi-indicator scoring system.
It is NOT a trained model, does NOT make promises about future returns,
and does NOT have predictive confidence in the statistical sense.
"Confidence" here means *internal agreement between layers*: how many
independent technical indicators point in the same direction.
Regime labels are heuristics derived from published TA definitions
(Wilder ADX, Bollinger, Murphy price structure). They can be wrong.
The operator is responsible for all investment decisions.

ADVICE-ONLY — no orders are placed from this module.

Layer architecture
------------------
1. Trend     : EMA20/50/200 alignment + ADX-14  -> trend_score [0-100]
2. Momentum  : RSI-14 + ROC-10 + MFI-14         -> momentum_score [0-100]
3. Volatility: ATR-14 + BB width + hist-vol      -> volatility_score [0-100]
4. Volume    : OBV direction + rel-volume +      -> volume_score [0-100]
               volume delta
5. Regime    : ADX + ATR + price structure       -> market_regime enum
6. Risk      : drawdown + vol-based              -> risk_score [0-100] higher=riskier
7. Composite : weighted sum of directional       -> action enum + confidence [0-100]
               layer scores; weights tunable via DB keys (migration 061)

Output schema (stable contract for dashboard + rationale_helper)
----------------------------------------------------------------
{
  "trend_score":        int  [0-100],
  "trend_label":        str  (STRONG_UP / WEAK_UP / NEUTRAL / WEAK_DOWN / STRONG_DOWN),
  "momentum_score":     int  [0-100],
  "volatility_score":   int  [0-100],
  "volatility_regime":  str  (LOW / NORMAL / ELEVATED / EXTREME),
  "volume_score":       int  [0-100],
  "market_regime":      str  (TRENDING / RANGING / PANIC / EUPHORIA /
                              ACCUMULATION / DISTRIBUTION),
  "risk_score":         int  [0-100],   # higher = riskier
  "ai_forecast":        str | None,     # Kronos signal label if available
  "confidence":         int  [0-100],   # layer-agreement derived
  "action":             str  (STRONG_BUY / WEAK_BUY / HOLD / WEAK_SELL / STRONG_SELL),
  "_layer_details":     dict            # internal breakdown for debugging
}

DB weight keys (migration 061, config_type='advisor_config')
-----------------------------------------------------------
  signal_weight_trend       float  default 0.30
  signal_weight_momentum    float  default 0.25
  signal_weight_volume      float  default 0.20
  signal_weight_volatility  float  default 0.15
  signal_weight_regime      float  default 0.10

If weights do not sum to 1.0, they are normalised at runtime.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

logger = logging.getLogger("advisor.signal_engine")

# ---------------------------------------------------------------------------
# Default weights (overridable via DB — see migration 061)
# ---------------------------------------------------------------------------
_DEFAULT_WEIGHTS = {
    "signal_weight_trend":      0.30,
    "signal_weight_momentum":   0.25,
    "signal_weight_volume":     0.20,
    "signal_weight_volatility": 0.15,
    "signal_weight_regime":     0.10,
}

# Action thresholds — composite directional score [-100, +100] -> action label.
# Positive = bullish. Negative = bearish.
_ACTION_THRESHOLDS = [
    (60,  "STRONG_BUY"),
    (20,  "WEAK_BUY"),
    (-20, "HOLD"),
    (-60, "WEAK_SELL"),
]  # below -60 -> STRONG_SELL


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_composite_signal(
    ohlcv_df,                          # pandas DataFrame [open,high,low,close,volume]
    market: str = "",                  # market label (informational only)
    config: Optional[dict] = None,     # advisor_config dict for weight overrides
    kronos_signal: Optional[float] = None,  # Kronos output: +ve=bullish, -ve=bearish
) -> dict:
    """
    Compute a multi-layer composite signal from OHLCV data.

    Parameters
    ----------
    ohlcv_df      : pandas DataFrame, columns = [open, high, low, close, volume],
                    DatetimeIndex ascending. Requires >= 20 rows for useful output.
    market        : str label used only for logging.
    config        : advisor_config dict (from AdvisorConfigManager); used to
                    load DB weight overrides. May be None (defaults applied).
    kronos_signal : Kronos forecaster output (positive=bullish, negative=bearish,
                    None if not loaded). Stored in the ai_forecast slot.

    Returns
    -------
    dict — stable output schema described in module docstring.
    Falls back to neutral/safe defaults on any computation error. Never raises.
    """
    try:
        return _compute(ohlcv_df, market, config or {}, kronos_signal)
    except Exception as exc:
        logger.warning(
            "[signal_engine] compute_composite_signal failed for market=%s: %s. "
            "Returning neutral defaults.",
            market, exc,
        )
        return _neutral_defaults(kronos_signal)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute(df, market: str, config: dict, kronos_signal: Optional[float]) -> dict:
    """Inner computation. May raise — caller wraps in try/except."""
    # Lazy import — pandas is not required at module load time.
    import pandas as pd  # noqa: F401

    if df is None or len(df) < 20:
        return _neutral_defaults(kronos_signal)

    df = df.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = 0.0
    df = df.dropna(subset=["close"])
    if len(df) < 20:
        return _neutral_defaults(kronos_signal)

    close  = df["close"].astype(float)
    high   = df["high"].astype(float)
    low    = df["low"].astype(float)
    volume = df["volume"].astype(float)

    # --- Layer 1: Trend ---
    trend_dir, trend_score, trend_label = _trend_layer(close, high, low)

    # --- Layer 2: Momentum ---
    mom_dir, momentum_score = _momentum_layer(close, high, low, volume)

    # --- Layer 3: Volatility ---
    vol_score, volatility_regime, atr14 = _volatility_layer(close, high, low)

    # --- Layer 4: Volume ---
    vol_layer_dir, volume_score = _volume_layer(close, volume)

    # --- Layer 5: Regime ---
    market_regime = _regime_layer(close, high, low, atr14)

    # --- Layer 6: Risk ---
    risk_score = _risk_layer(close, atr14, volatility_regime)

    # --- Layer 7: Composite ---
    weights = _load_weights(config)

    # Convert each directional layer to a signed score in [-100, +100].
    trend_signed    = trend_dir    * trend_score
    momentum_signed = mom_dir      * momentum_score
    volume_signed   = vol_layer_dir * volume_score

    # Volatility contribution: direction-neutral risk penalty.
    # EXTREME regime applies a negative adjustment (uncertainty dampening).
    vol_regime_adj = {"LOW": 0.0, "NORMAL": 0.0, "ELEVATED": -15.0, "EXTREME": -30.0}
    volatility_adj = vol_regime_adj.get(volatility_regime, 0.0)

    # Regime contribution: TRENDING amplifies; non-trending regimes dampen.
    regime_adj_map = {
        "TRENDING":      15.0,
        "RANGING":      -10.0,
        "PANIC":        -25.0,
        "EUPHORIA":     -15.0,
        "ACCUMULATION":  10.0,
        "DISTRIBUTION": -10.0,
    }
    regime_adj = regime_adj_map.get(market_regime, 0.0)

    # Normalise weights.
    w_sum = sum(weights.values())
    if w_sum <= 0:
        w_sum = 1.0
    wn = {k: v / w_sum for k, v in weights.items()}

    composite = (
        wn["signal_weight_trend"]      * trend_signed
        + wn["signal_weight_momentum"] * momentum_signed
        + wn["signal_weight_volume"]   * volume_signed
        + wn["signal_weight_volatility"] * volatility_adj
        + wn["signal_weight_regime"]   * regime_adj
    )
    composite = max(-100.0, min(100.0, composite))

    confidence = _compute_confidence(
        composite, trend_dir, mom_dir, vol_layer_dir,
        trend_score, momentum_score, volume_score,
    )

    action = _score_to_action(composite)

    if kronos_signal is not None:
        direction_word = "bullish" if kronos_signal > 0 else "bearish"
        ai_forecast = f"{direction_word} ({kronos_signal:.3f})"
    else:
        ai_forecast = "unavailable"

    return {
        "trend_score":       int(round(trend_score)),
        "trend_label":       trend_label,
        "momentum_score":    int(round(momentum_score)),
        "volatility_score":  int(round(vol_score)),
        "volatility_regime": volatility_regime,
        "volume_score":      int(round(volume_score)),
        "market_regime":     market_regime,
        "risk_score":        int(round(risk_score)),
        "ai_forecast":       ai_forecast,
        "confidence":        int(round(confidence)),
        "action":            action,
        "_layer_details": {
            "trend_dir":       trend_dir,
            "trend_score_raw": trend_score,
            "mom_dir":         mom_dir,
            "vol_layer_dir":   vol_layer_dir,
            "composite_raw":   composite,
            "weights_used":    wn,
        },
    }


# ---------------------------------------------------------------------------
# Layer implementations
# ---------------------------------------------------------------------------

def _trend_layer(close, high, low):
    """
    Trend layer: EMA20 / EMA50 / EMA200 alignment + ADX-14.

    Alignment heuristic:
      EMA20 > EMA50 > EMA200  => full uptrend     (align_score=100, dir=+1)
      EMA20 > EMA50           => partial uptrend  (align_score=60,  dir=+1)
      EMA20 > EMA200 only     => weak uptrend     (align_score=40,  dir=+1)
      Symmetric for downtrend.

    ADX multiplier:
      ADX >= 40 -> 1.00 (strong trend confirmation)
      ADX >= 25 -> 0.85
      ADX >= 20 -> 0.65
      ADX <  20 -> 0.40 (likely ranging; dampen trend signal)

    Returns: (trend_dir, trend_score, trend_label)
      trend_dir in {-1, 0, +1}; trend_score in [0, 100].
    """
    n = len(close)
    ema20  = close.ewm(span=20,  adjust=False).mean()
    ema50  = close.ewm(span=50,  adjust=False).mean() if n >= 50  else ema20
    ema200 = close.ewm(span=200, adjust=False).mean() if n >= 200 else ema50

    e20  = ema20.iloc[-1]
    e50  = ema50.iloc[-1]
    e200 = ema200.iloc[-1]

    if e20 > e50 > e200:
        align_score, trend_dir = 100, 1
    elif e20 > e50:
        align_score, trend_dir = 60, 1
    elif e20 > e200:
        align_score, trend_dir = 40, 1
    elif e20 < e50 < e200:
        align_score, trend_dir = 100, -1
    elif e20 < e50:
        align_score, trend_dir = 60, -1
    elif e20 < e200:
        align_score, trend_dir = 40, -1
    else:
        align_score, trend_dir = 20, 0

    adx = _compute_adx(high, low, close, period=14)
    if adx >= 40:
        adx_mult = 1.00
    elif adx >= 25:
        adx_mult = 0.85
    elif adx >= 20:
        adx_mult = 0.65
    else:
        adx_mult = 0.40

    trend_score = align_score * adx_mult

    if trend_dir > 0:
        label = "STRONG_UP" if trend_score >= 70 else "WEAK_UP"
    elif trend_dir < 0:
        label = "STRONG_DOWN" if trend_score >= 70 else "WEAK_DOWN"
    else:
        label = "NEUTRAL"

    return trend_dir, trend_score, label


def _momentum_layer(close, high, low, volume):
    """
    Momentum layer: RSI-14 + ROC-10 + MFI-14.

    Each sub-indicator is mapped to a directional vote in [-1, +1]:
      RSI <30 -> oversold (+1); >70 -> overbought (-1); else proportional
      ROC >+3% -> bullish (+1); <-3% -> bearish (-1); else proportional
      MFI <20 -> oversold (+1); >80 -> overbought (-1); else proportional

    Average vote magnitude -> momentum_score [0-100].

    Returns: (mom_dir, momentum_score)
    """
    rsi = _rsi(close, 14)

    roc_period = min(10, len(close) - 1)
    if roc_period > 0 and close.iloc[-1 - roc_period] != 0:
        roc = (close.iloc[-1] / close.iloc[-1 - roc_period] - 1.0) * 100.0
    else:
        roc = 0.0

    mfi = _mfi(high, low, close, volume, 14)

    rsi_vote = _rsi_to_dir(rsi)
    roc_vote = 1.0 if roc > 3 else (-1.0 if roc < -3 else roc / 3.0)
    mfi_vote = _mfi_to_dir(mfi)

    avg_vote = (rsi_vote + roc_vote + mfi_vote) / 3.0
    mom_dir = 1 if avg_vote > 0.15 else (-1 if avg_vote < -0.15 else 0)
    momentum_score = abs(avg_vote) * 100.0

    return mom_dir, momentum_score


def _volatility_layer(close, high, low):
    """
    Volatility layer: ATR-14 as % of close + Bollinger Band width + 20d hist-vol.

    ATR% thresholds:     <1%=low, 1-3%=normal, 3-6%=elevated, >6%=extreme
    BB width% thresholds:<5%=low, 5-15%=normal, 15-30%=elevated, >30%=extreme
    Hist-vol% thresholds:<20%=low, 20-50%=normal, 50-80%=elevated, >80%=extreme

    Each mapped linearly to [0-100], then averaged.

    Returns: (vol_score, volatility_regime, atr14_abs)
    """
    import pandas as pd

    atr_abs = _atr(high, low, close, 14)
    last_close = close.iloc[-1]
    atr_pct = (atr_abs / max(last_close, 1e-10)) * 100.0

    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_width_pct = (4.0 * std20.iloc[-1] / max(sma20.iloc[-1], 1e-10)) * 100.0

    log_ret = close.pct_change().rolling(20).std() * (252 ** 0.5) * 100.0
    hist_vol = log_ret.iloc[-1]
    if math.isnan(hist_vol):
        hist_vol = 30.0

    atr_score = min(atr_pct / 6.0 * 100.0, 100.0)
    bb_score  = min(bb_width_pct / 30.0 * 100.0, 100.0)
    hv_score  = min(hist_vol / 80.0 * 100.0, 100.0)

    vol_score = (atr_score + bb_score + hv_score) / 3.0

    if vol_score < 25:
        regime = "LOW"
    elif vol_score < 50:
        regime = "NORMAL"
    elif vol_score < 75:
        regime = "ELEVATED"
    else:
        regime = "EXTREME"

    return vol_score, regime, atr_abs


def _volume_layer(close, volume):
    """
    Volume layer: OBV EMA crossover + relative volume + volume delta.

    OBV direction: 5-period EMA vs 20-period EMA of OBV.
    Relative volume: last bar vs 20-day average.
    Volume delta: mean(last 3 bars) vs mean(prior 3 bars).

    Majority of three sub-votes -> vol_layer_dir.
    Relative volume magnitude -> volume_score [0-100].

    Returns: (vol_layer_dir, volume_score)
    """
    sign_series = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (volume * sign_series).cumsum()
    obv_ema5  = obv.ewm(span=5,  adjust=False).mean()
    obv_ema20 = obv.ewm(span=20, adjust=False).mean()
    obv_dir = 1 if obv_ema5.iloc[-1] > obv_ema20.iloc[-1] else -1

    avg_vol = volume.rolling(20).mean().iloc[-1]
    rel_vol = volume.iloc[-1] / max(avg_vol, 1.0)
    rel_vol_score = min(rel_vol / 2.0 * 100.0, 100.0)  # 2x avg = 100

    if len(volume) >= 6:
        recent = volume.iloc[-3:].mean()
        prior  = volume.iloc[-6:-3].mean()
        delta_ratio = recent / max(prior, 1.0)
        accel_dir = 1 if delta_ratio > 1.1 else (-1 if delta_ratio < 0.9 else 0)
    else:
        accel_dir = 0

    rel_vol_dir = 1 if rel_vol > 1.2 else (-1 if rel_vol < 0.8 else 0)
    votes = obv_dir + rel_vol_dir + accel_dir
    vol_layer_dir = 1 if votes > 0 else (-1 if votes < 0 else 0)
    volume_score = (abs(votes) / 3.0) * rel_vol_score

    return vol_layer_dir, volume_score


def _regime_layer(close, high, low, atr14: float) -> str:
    """
    Market regime classifier.

    Rules (transparent, documented):

    EUPHORIA     : price > 2x 200d-EMA AND RSI > 75
                   (crowd euphoria, historically unsustainable extension)
    PANIC        : price < 0.85x 200d-EMA AND RSI < 25
                   (capitulation; oversold on long-term basis)
    TRENDING     : ADX > 25 AND EMA20 != EMA50
                   (directional momentum confirmed by Wilder's ADX)
    ACCUMULATION : ADX < 20 AND price within 3% of 200d-EMA AND
                   5d-MA > 20d-MA (price recovering quietly near base)
    DISTRIBUTION : ADX < 20 AND price within 3% of 200d-EMA AND
                   5d-MA < 20d-MA (price declining quietly from peak)
    RANGING      : ADX < 20 (default non-directional state)
    """
    n = len(close)
    ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1] if n >= 200 else close.mean()
    ema50  = close.ewm(span=50,  adjust=False).mean().iloc[-1] if n >= 50  else close.mean()
    ema20  = close.ewm(span=20,  adjust=False).mean().iloc[-1]
    rsi    = _rsi(close, 14)
    adx    = _compute_adx(high, low, close, period=14)
    last_close = close.iloc[-1]

    if last_close > ema200 * 2.0 and rsi > 75:
        return "EUPHORIA"

    if last_close < ema200 * 0.85 and rsi < 25:
        return "PANIC"

    if adx > 25 and abs(ema20 - ema50) > 0:
        return "TRENDING"

    near_ema200 = abs(last_close - ema200) / max(ema200, 1e-10) < 0.03
    if adx < 20 and near_ema200:
        ma5  = close.rolling(5).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        return "ACCUMULATION" if ma5 > ma20 else "DISTRIBUTION"

    return "RANGING"


def _risk_layer(close, atr14: float, volatility_regime: str) -> float:
    """
    Risk score [0-100], higher = riskier.

    Components:
    - Max drawdown over the full window (max 40 pts)
    - Current drawdown from rolling high (max 25 pts)
    - ATR as % of close (max 20 pts)
    - Volatility regime bonus (max 45 pts — total sum capped at 100)
    """
    rolling_max = close.cummax()
    drawdown_series = (close - rolling_max) / rolling_max.replace(0, 1.0)
    max_dd_pct = abs(drawdown_series.min()) * 100.0
    current_dd = abs(drawdown_series.iloc[-1]) * 100.0

    last_close = close.iloc[-1]
    atr_pct = (atr14 / max(last_close, 1e-10)) * 100.0

    regime_bonus = {"LOW": 0, "NORMAL": 10, "ELEVATED": 25, "EXTREME": 45}
    reg_bonus = regime_bonus.get(volatility_regime, 10)

    risk_score = (
        min(max_dd_pct * 2.0, 40.0)
        + min(current_dd * 2.0, 25.0)
        + min(atr_pct * 5.0, 20.0)
        + reg_bonus
    )
    return min(risk_score, 100.0)


# ---------------------------------------------------------------------------
# Composite & confidence helpers
# ---------------------------------------------------------------------------

def _compute_confidence(
    composite: float,
    trend_dir: int,
    mom_dir: int,
    vol_dir: int,
    trend_score: float,
    momentum_score: float,
    volume_score: float,
) -> float:
    """
    Confidence = internal layer-agreement measure.

    NOT a predictive probability. Interpretation:
      - Near-neutral composite (|composite| < 5) -> low confidence (25).
      - Agreement ratio: fraction of directional layer score-weight agreeing
        with the composite direction.
      - Magnitude factor: abs(composite)/100 captures signal strength.

    Formula: confidence = (0.6 * agreement_ratio + 0.4 * magnitude) * 100
    Clamped to [5, 95] — never 0 (always some uncertainty) or 100 (no guarantee).
    """
    composite_dir = 1 if composite > 5 else (-1 if composite < -5 else 0)
    if composite_dir == 0:
        return 25.0

    agreements = 0.0
    total_weight = 0.0
    for (d, s) in [(trend_dir, trend_score), (mom_dir, momentum_score), (vol_dir, volume_score)]:
        total_weight += max(s, 1.0)
        if d == composite_dir:
            agreements += s

    agreement_ratio = agreements / max(total_weight, 1.0)
    magnitude_factor = abs(composite) / 100.0
    confidence = (0.6 * agreement_ratio + 0.4 * magnitude_factor) * 100.0
    return min(max(confidence, 5.0), 95.0)


def _score_to_action(composite: float) -> str:
    """Map composite score [-100, +100] to action label."""
    for threshold, label in _ACTION_THRESHOLDS:
        if composite >= threshold:
            return label
    return "STRONG_SELL"


def _load_weights(config: dict) -> dict:
    """Load layer weights from config dict, falling back to module defaults."""
    weights = {}
    for key, default in _DEFAULT_WEIGHTS.items():
        try:
            weights[key] = float(config.get(key, default))
        except (TypeError, ValueError):
            weights[key] = default
    return weights


def _neutral_defaults(kronos_signal: Optional[float]) -> dict:
    """Return neutral/safe defaults when data is insufficient or computation fails."""
    if kronos_signal is not None:
        direction_word = "bullish" if kronos_signal > 0 else "bearish"
        ai_forecast = f"{direction_word} ({kronos_signal:.3f})"
    else:
        ai_forecast = "unavailable"
    return {
        "trend_score":       50,
        "trend_label":       "NEUTRAL",
        "momentum_score":    50,
        "volatility_score":  50,
        "volatility_regime": "NORMAL",
        "volume_score":      50,
        "market_regime":     "RANGING",
        "risk_score":        50,
        "ai_forecast":       ai_forecast,
        "confidence":        25,
        "action":            "HOLD",
        "_layer_details":    {"source": "neutral_default"},
    }


# ---------------------------------------------------------------------------
# TA primitives (pure pandas — no TA-Lib dependency required)
# ---------------------------------------------------------------------------

def _rsi(series, period: int = 14) -> float:
    """RSI-n, last value. Returns 50.0 on insufficient data."""
    delta = series.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs    = gain / (loss + 1e-10)
    rsi_s = 100.0 - (100.0 / (1.0 + rs))
    val   = rsi_s.iloc[-1]
    return val if not math.isnan(val) else 50.0


def _mfi(high, low, close, volume, period: int = 14) -> float:
    """Money Flow Index, last value. Returns 50.0 on insufficient data."""
    typical = (high + low + close) / 3.0
    raw_mf  = typical * volume
    diff    = typical.diff()
    pos_mf  = raw_mf.where(diff > 0, 0.0).rolling(period).sum()
    neg_mf  = raw_mf.where(diff < 0, 0.0).rolling(period).sum()
    mfi_s   = 100.0 - (100.0 / (1.0 + pos_mf / (neg_mf + 1e-10)))
    val     = mfi_s.iloc[-1]
    return val if not math.isnan(val) else 50.0


def _atr(high, low, close, period: int = 14) -> float:
    """Average True Range, last value (in price units). Returns 0.0 on failure."""
    prev_close = close.shift(1)
    tr = (
        (high - low).abs()
        .combine((high - prev_close).abs(), max)
        .combine((low - prev_close).abs(), max)
    )
    atr_s = tr.rolling(period).mean()
    val   = atr_s.iloc[-1]
    if val and not math.isnan(val):
        return float(val)
    return 0.0


def _compute_adx(high, low, close, period: int = 14) -> float:
    """
    Simplified ADX (Wilder). Returns last ADX value [0, 100].
    Values > 25 indicate a trending market (Wilder's threshold).
    Returns 0.0 on insufficient data.
    """
    up_move   = high.diff()
    down_move = (-low.diff())

    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr14 = _atr(high, low, close, period)
    if atr14 < 1e-10:
        return 0.0

    plus_di  = (plus_dm.rolling(period).mean()  / atr14) * 100.0
    minus_di = (minus_dm.rolling(period).mean() / atr14) * 100.0

    di_sum  = (plus_di + minus_di).replace(0, 1e-10)
    dx      = ((plus_di - minus_di).abs() / di_sum) * 100.0
    adx_s   = dx.rolling(period).mean()

    val = adx_s.iloc[-1]
    if val and not math.isnan(val):
        return float(val)
    return 0.0


def _rsi_to_dir(rsi: float) -> float:
    """Map RSI value to directional vote in [-1.0, +1.0]."""
    if rsi < 30:
        return 1.0
    if rsi < 40:
        return 0.5
    if rsi < 60:
        return 0.0
    if rsi < 70:
        return -0.5
    return -1.0


def _mfi_to_dir(mfi: float) -> float:
    """Map MFI value to directional vote in [-1.0, +1.0]."""
    if mfi < 20:
        return 1.0
    if mfi < 40:
        return 0.5
    if mfi < 60:
        return 0.0
    if mfi < 80:
        return -0.5
    return -1.0
