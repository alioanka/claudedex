"""
Futures Risk Manager

Specialized risk management for leverage trading:
- Liquidation price monitoring
- Leverage limits
- Position size validation
- Cross-position exposure tracking
- Auto-deleveraging on drawdown
"""

import logging
from collections import deque
from typing import Deque, Dict, List, Optional
from datetime import datetime, timedelta


class FuturesRiskManager:
    """
    Risk management for futures trading

    Key features:
    - Prevent over-leverage
    - Monitor liquidation risk
    - Enforce position limits
    - Track total exposure (long + short)
    - Auto-reduce leverage on losses
    """

    def __init__(self, config: Dict):
        """
        Initialize futures risk manager

        Args:
            config: Risk configuration
        """
        self.config = config
        self.logger = logging.getLogger("FuturesRiskManager")

        # Risk parameters
        self.max_leverage = config.get('max_leverage', 3)
        self.max_positions = config.get('max_positions', 3)
        self.max_total_exposure = config.get('max_total_exposure', 500.0)  # USD
        self.liquidation_buffer = config.get('liquidation_buffer', 0.20)  # 20% from liq price
        self.max_drawdown = config.get('max_drawdown', 0.10)  # 10%

        # FUT-RM-05: funding-rate directional gate. Units = basis points
        # of the per-interval funding rate (1 bp = 0.0001 fraction).
        # Zero disables the gate on that side.
        self.skip_long_funding_bps = float(
            config.get('skip_long_funding_bps', 0.0) or 0.0
        )
        self.skip_short_funding_bps = float(
            config.get('skip_short_funding_bps', 0.0) or 0.0
        )

        # FUT-RM-08 (Wave 3): per-symbol leverage cap overrides. Normalized to
        # uppercased, slash-stripped form so 'btc/usdt', 'BTC/USDT', and
        # 'BTCUSDT' all hit the same entry. An override of 0 / None / negative
        # is treated as "no override" and we fall back to the global cap.
        raw_overrides = config.get('max_leverage_overrides') or {}
        self.max_leverage_overrides: Dict[str, int] = {}
        if isinstance(raw_overrides, dict):
            for sym, lev in raw_overrides.items():
                try:
                    norm = self._normalize_symbol(sym)
                    lev_i = int(lev)
                    if norm and lev_i > 0:
                        self.max_leverage_overrides[norm] = lev_i
                except (TypeError, ValueError):
                    # Bad row in the JSON map — skip rather than block boot.
                    continue

        # State tracking
        self.consecutive_losses = 0
        self.total_realized_pnl = 0.0

        # FUT-RM-17 (Wave 5): per-symbol consecutive-loss cool-off.
        # After N consecutive losses on the same symbol, refuse new entries
        # on that symbol for `post_loss_cooloff_minutes` minutes. Tracks
        # symbol -> int (consecutive losses) and symbol -> datetime
        # (cool-off expiry; absent means no cool-off in effect).
        self.post_loss_cooloff_minutes = int(
            config.get('post_loss_cooloff_minutes', 240) or 0
        )
        self.post_loss_cooloff_threshold = int(
            config.get('post_loss_cooloff_threshold', 2) or 0
        )
        self.per_symbol_consec_losses: Dict[str, int] = {}
        self.per_symbol_cooloff_until: Dict[str, datetime] = {}

        # FUT-RM-27 (Wave 25): per-symbol tiering + rolling performance gate.
        # Two layers, both fail-open:
        #   1. STATIC tier weights (`symbol_size_weights`, operator-curated,
        #      seeded by migration 088 from a week of live data): weight 0
        #      disables a symbol entirely; 0 < w < 1 trades it at reduced
        #      size; missing symbol = 1.0 (full size).
        #   2. ROLLING gate: per-symbol trailing-N net PnL + win rate. When
        #      BOTH fall below thresholds, the symbol is auto-BENCHED (no new
        #      entries) for `rolling_gate_bench_minutes`. On expiry the
        #      window is cleared and the symbol re-enters on PROBATION at
        #      `rolling_gate_probation_weight` × static weight until it has
        #      `rolling_gate_min_trades` fresh trades — so a still-losing
        #      symbol re-benches quickly at reduced cost. This keeps the
        #      tiering current without manual curation (the ZEC failure
        #      mode: 42 trades/week through a 4h cool-off that kept expiring).
        self.symbol_tiering_enabled = bool(
            config.get('symbol_tiering_enabled', True)
        )
        raw_weights = config.get('symbol_size_weights') or {}
        self.symbol_size_weights: Dict[str, float] = {}
        if isinstance(raw_weights, dict):
            for sym, w in raw_weights.items():
                try:
                    norm = self._normalize_symbol(sym)
                    w_f = float(w)
                    if norm and w_f >= 0.0:
                        self.symbol_size_weights[norm] = w_f
                except (TypeError, ValueError):
                    continue  # bad row — skip rather than block boot
        self.rolling_gate_enabled = bool(config.get('rolling_gate_enabled', True))
        self.rolling_gate_window = int(config.get('rolling_gate_window', 20) or 0)
        self.rolling_gate_min_trades = int(
            config.get('rolling_gate_min_trades', 10) or 0
        )
        self.rolling_gate_max_net_pnl_usd = float(
            config.get('rolling_gate_max_net_pnl_usd', -5.0)
        )
        self.rolling_gate_max_win_rate = float(
            config.get('rolling_gate_max_win_rate', 0.45)
        )
        self.rolling_gate_bench_minutes = int(
            config.get('rolling_gate_bench_minutes', 1440) or 0
        )
        self.rolling_gate_probation_weight = float(
            config.get('rolling_gate_probation_weight', 0.5)
        )
        self._rolling_pnl: Dict[str, Deque[float]] = {}
        self._rolling_benched_until: Dict[str, datetime] = {}
        self._probation_trades_left: Dict[str, int] = {}

    @staticmethod
    def _normalize_symbol(sym: str) -> str:
        """Canonical form for override lookup: uppercase, no slash, no spaces."""
        if not sym:
            return ''
        return str(sym).strip().upper().replace('/', '').replace(' ', '')

    def resolve_max_leverage(self, symbol: str) -> int:
        """FUT-RM-08: return the effective per-symbol leverage cap.

        Lookup order:
          1. max_leverage_overrides[normalize(symbol)]
          2. self.max_leverage (global)

        Defensive: never returns < 1. Callers use the result as the cap to
        compare requested leverage against in validate_new_position.
        """
        try:
            norm = self._normalize_symbol(symbol)
            override = self.max_leverage_overrides.get(norm)
            if override is not None and override > 0:
                return int(override)
        except Exception:
            pass
        return max(1, int(self.max_leverage))

    def validate_new_position(
        self,
        symbol: str,
        side: str,
        size_usd: float,
        leverage: int,
        current_positions: List[Dict],
        available_capital: float
    ) -> Dict:
        """
        Validate if new position can be opened

        Args:
            symbol: Trading symbol
            side: 'LONG' or 'SHORT'
            size_usd: Position size in USD
            leverage: Requested leverage
            current_positions: List of current positions
            available_capital: Available capital

        Returns:
            Dict: Validation result with 'allowed' and 'reason'
        """
        try:
            # Check position count
            if len(current_positions) >= self.max_positions:
                return {
                    'allowed': False,
                    'reason': f'Max positions reached ({self.max_positions})'
                }

            # Check leverage limit. FUT-RM-08: per-symbol override > global.
            effective_max = self.resolve_max_leverage(symbol)
            if leverage > effective_max:
                # Make the reason explicit about which cap fired so the
                # operator can tell an override-block from a global block.
                src = 'override' if effective_max != self.max_leverage else 'global'
                return {
                    'allowed': False,
                    'reason': f'Leverage {leverage}x exceeds max {effective_max}x ({src})',
                    'suggested_leverage': effective_max,
                    'effective_max_leverage': effective_max,
                    'cap_source': src,
                }

            # Check capital availability
            required_margin = size_usd / leverage
            if required_margin > available_capital:
                return {
                    'allowed': False,
                    'reason': f'Insufficient capital (need ${required_margin:.2f}, have ${available_capital:.2f})'
                }

            # Check total exposure
            current_exposure = sum(
                abs(p.get('notional_value', 0)) for p in current_positions
            )
            total_exposure = current_exposure + size_usd

            if total_exposure > self.max_total_exposure:
                return {
                    'allowed': False,
                    'reason': f'Total exposure ${total_exposure:.2f} exceeds limit ${self.max_total_exposure:.2f}'
                }

            # All checks passed
            return {
                'allowed': True,
                'reason': 'Position validated',
                'required_margin': required_margin,
                'total_exposure_after': total_exposure
            }

        except Exception as e:
            self.logger.error(f"Error validating position: {e}")
            return {
                'allowed': False,
                'reason': f'Validation error: {str(e)}'
            }

    def should_skip_for_funding(
        self,
        side: str,
        funding_rate: Optional[float],
    ) -> Dict:
        """FUT-RM-05: directional funding-rate gate.

        Args:
            side: 'LONG' or 'SHORT' (case-insensitive)
            funding_rate: per-interval funding rate as a FRACTION
                (e.g. 0.0005 = 0.05% = 5 bps for one funding interval).
                None when the rate is unavailable -> gate is bypassed
                (we never block on missing data; the caller logs).

        Returns:
            dict with:
              skip: bool
              reason: human-readable string
              rate_bps: float (funding rate converted to bps; 0 if missing)
              threshold_bps: float (which side's threshold applied)
        """
        try:
            if funding_rate is None:
                return {
                    'skip': False,
                    'reason': 'funding rate unavailable',
                    'rate_bps': 0.0,
                    'threshold_bps': 0.0,
                }
            rate_bps = float(funding_rate) * 10000.0
            side_u = (side or '').upper()
            if side_u == 'LONG':
                threshold = self.skip_long_funding_bps
                if threshold > 0 and rate_bps > threshold:
                    return {
                        'skip': True,
                        'reason': (
                            f'funding {rate_bps:+.2f} bps > +{threshold:.2f} bps '
                            f'(long would pay funding above cap)'
                        ),
                        'rate_bps': rate_bps,
                        'threshold_bps': threshold,
                    }
            elif side_u == 'SHORT':
                threshold = self.skip_short_funding_bps
                # For shorts the bad direction is NEGATIVE funding (shorts pay).
                if threshold > 0 and rate_bps < -threshold:
                    return {
                        'skip': True,
                        'reason': (
                            f'funding {rate_bps:+.2f} bps < -{threshold:.2f} bps '
                            f'(short would pay funding above cap)'
                        ),
                        'rate_bps': rate_bps,
                        'threshold_bps': threshold,
                    }
            return {
                'skip': False,
                'reason': 'funding within tolerance',
                'rate_bps': rate_bps,
                'threshold_bps': (
                    self.skip_long_funding_bps if side_u == 'LONG'
                    else self.skip_short_funding_bps
                ),
            }
        except Exception as e:
            self.logger.warning(f"should_skip_for_funding errored: {e}")
            # Fail-open: never block trade on validator bug.
            return {
                'skip': False,
                'reason': f'gate error: {e}',
                'rate_bps': 0.0,
                'threshold_bps': 0.0,
            }

    def check_reconciled_capacity(self, current_positions: List[Dict]) -> Dict:
        """Restart-time sanity check: returns whether the reconciled position
        count already meets or exceeds max_positions. Caller logs/alerts on
        over_cap=True and should refuse to open new positions until count drops."""
        try:
            count = len(current_positions)
            return {
                'at_cap': count >= self.max_positions,
                'over_cap': count > self.max_positions,
                'count': count,
                'max_positions': self.max_positions,
            }
        except Exception as e:
            self.logger.error(f"check_reconciled_capacity error: {e}")
            return {'at_cap': False, 'over_cap': False, 'count': 0,
                    'max_positions': self.max_positions}

    def check_liquidation_risk(
        self,
        position: Dict,
        current_price: float
    ) -> Dict:
        """
        Check if position is at risk of liquidation. Accepts either
        Binance or Bybit position shape (auto-normalized).

        Args:
            position: Position info with liquidation_price
            current_price: Current market price

        Returns:
            Dict: Risk assessment
        """
        try:
            # Auto-normalize so this method works regardless of source executor.
            # If 'liquidation_price' is missing but 'raw' contains Bybit-style
            # liqPrice, the normalizer recovers it.
            if 'liquidation_price' not in position or position.get('liquidation_price') is None:
                try:
                    from modules.futures_trading.exchanges import normalize_position
                    src = position.get('source') or ('bybit' if 'size' in position else 'binance')
                    normalized = normalize_position(position, src)
                    if normalized:
                        position = normalized
                except Exception:
                    pass

            liq_price = position.get('liquidation_price', 0)
            if liq_price == 0:
                return {'risk_level': 'unknown'}

            side = position.get('side', 'LONG')

            # Calculate distance to liquidation
            if side == 'LONG':
                # For longs, liquidation price is below current
                distance_pct = (current_price - liq_price) / current_price
            else:
                # For shorts, liquidation price is above current
                distance_pct = (liq_price - current_price) / current_price

            # Assess risk level
            if distance_pct < 0:
                risk_level = 'CRITICAL'  # Already past liquidation
            elif distance_pct < 0.05:
                risk_level = 'EXTREME'  # < 5% from liquidation
            elif distance_pct < 0.10:
                risk_level = 'HIGH'  # < 10% from liquidation
            elif distance_pct < self.liquidation_buffer:
                risk_level = 'MEDIUM'  # < buffer from liquidation
            else:
                risk_level = 'LOW'  # Safe distance

            return {
                'risk_level': risk_level,
                'distance_pct': distance_pct * 100,
                'current_price': current_price,
                'liquidation_price': liq_price,
                'should_reduce': risk_level in ['CRITICAL', 'EXTREME', 'HIGH']
            }

        except Exception as e:
            self.logger.error(f"Error checking liquidation risk: {e}")
            return {'risk_level': 'unknown'}

    def should_auto_deleverage(
        self,
        total_pnl: float,
        total_capital: float
    ) -> bool:
        """
        Check if positions should be auto-deleveraged due to losses

        Args:
            total_pnl: Total realized + unrealized PnL
            total_capital: Total capital

        Returns:
            bool: True if should deleverage
        """
        try:
            if total_capital == 0:
                return False

            # Calculate drawdown
            drawdown_pct = abs(total_pnl / total_capital)

            # Auto-deleverage if drawdown exceeds threshold
            if drawdown_pct > self.max_drawdown:
                self.logger.warning(
                    f"⚠️ Auto-deleveraging triggered: "
                    f"Drawdown {drawdown_pct*100:.1f}% > {self.max_drawdown*100:.1f}%"
                )
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking auto-deleverage: {e}")
            return False

    def calculate_position_size(
        self,
        capital: float,
        risk_per_trade: float,
        leverage: int,
        stop_loss_pct: float
    ) -> float:
        """
        Calculate appropriate position size

        Args:
            capital: Available capital
            risk_per_trade: Risk per trade (e.g., 0.02 = 2%)
            leverage: Leverage to use
            stop_loss_pct: Stop loss percentage (e.g., 0.03 = 3%)

        Returns:
            float: Position size in USD
        """
        try:
            # Risk amount
            risk_amount = capital * risk_per_trade

            # Position size = risk / stop_loss_pct
            position_size = risk_amount / stop_loss_pct

            # Account for leverage (can open larger position with same margin)
            leveraged_size = position_size * leverage

            # Cap at max exposure
            leveraged_size = min(leveraged_size, self.max_total_exposure)

            return leveraged_size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0

    def update_on_trade_close(self, pnl: float, symbol: Optional[str] = None):
        """
        Update risk state after trade closes

        Args:
            pnl: Trade profit/loss
            symbol: trading symbol (enables per-symbol cool-off tracking)
        """
        try:
            self.total_realized_pnl += pnl

            # Track consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

            # Log if multiple consecutive losses
            if self.consecutive_losses >= 3:
                self.logger.warning(
                    f"⚠️ {self.consecutive_losses} consecutive losses in futures"
                )

            # FUT-RM-17 (Wave 5): per-symbol consecutive-loss cool-off.
            if symbol:
                norm = self._normalize_symbol(symbol)
                # FUT-RM-27: feed the rolling per-symbol performance gate.
                self._record_rolling_trade(norm, symbol, pnl)
                if pnl < 0:
                    self.per_symbol_consec_losses[norm] = (
                        self.per_symbol_consec_losses.get(norm, 0) + 1
                    )
                    threshold = self.post_loss_cooloff_threshold
                    if (
                        threshold > 0
                        and self.per_symbol_consec_losses[norm] >= threshold
                        and self.post_loss_cooloff_minutes > 0
                    ):
                        from datetime import timedelta
                        until = datetime.now() + timedelta(
                            minutes=self.post_loss_cooloff_minutes
                        )
                        self.per_symbol_cooloff_until[norm] = until
                        self.logger.warning(
                            f"🛑 FUT-RM-17 cool-off armed for {symbol}: "
                            f"{self.per_symbol_consec_losses[norm]} consecutive "
                            f"losses; refusing entries until {until.isoformat()}"
                        )
                else:
                    # Win resets per-symbol loss streak and clears cool-off.
                    if norm in self.per_symbol_consec_losses:
                        self.per_symbol_consec_losses[norm] = 0
                    if norm in self.per_symbol_cooloff_until:
                        del self.per_symbol_cooloff_until[norm]

        except Exception as e:
            self.logger.error(f"Error updating trade close: {e}")

    def should_skip_for_cooloff(self, symbol: str) -> Dict:
        """FUT-RM-17 (Wave 5): per-symbol consecutive-loss cool-off gate.

        Returns dict {skip: bool, reason: str, expires_at: Optional[str]}.
        Fail-open on any error so a validator bug never silently blocks
        every trade.
        """
        try:
            if self.post_loss_cooloff_minutes <= 0:
                return {'skip': False, 'reason': 'cooloff disabled'}
            norm = self._normalize_symbol(symbol)
            until = self.per_symbol_cooloff_until.get(norm)
            if until is None:
                return {'skip': False, 'reason': 'no cooloff'}
            now = datetime.now()
            if now >= until:
                # Cool-off expired — clear it and reset the per-symbol counter
                # so the symbol gets a fresh start without immediately re-arming
                # on the next loss.
                del self.per_symbol_cooloff_until[norm]
                self.per_symbol_consec_losses[norm] = 0
                return {'skip': False, 'reason': 'cooloff expired'}
            remaining_min = int((until - now).total_seconds() / 60)
            return {
                'skip': True,
                'reason': (
                    f'{self.per_symbol_consec_losses.get(norm, 0)} consecutive '
                    f'losses; cool-off ~{remaining_min} min remaining'
                ),
                'expires_at': until.isoformat(),
            }
        except Exception as e:
            self.logger.warning(f"should_skip_for_cooloff errored: {e}")
            return {'skip': False, 'reason': f'gate error: {e}'}

    # ------------------------------------------------------------------
    # FUT-RM-27 (Wave 25): per-symbol tiering + rolling performance gate
    # ------------------------------------------------------------------

    def _record_rolling_trade(self, norm: str, symbol: str, pnl: float) -> None:
        """Append a closed trade to the symbol's trailing window and evaluate
        the bench condition. Loudly logs ONCE per BENCH transition. Never
        raises (called from update_on_trade_close's try block anyway)."""
        if not (
            self.symbol_tiering_enabled
            and self.rolling_gate_enabled
            and self.rolling_gate_window > 0
        ):
            return
        dq = self._rolling_pnl.get(norm)
        if dq is None or dq.maxlen != self.rolling_gate_window:
            dq = deque(dq or [], maxlen=self.rolling_gate_window)
            self._rolling_pnl[norm] = dq
        dq.append(float(pnl))

        # Probation bookkeeping: count down fresh post-unbench trades.
        if norm in self._probation_trades_left:
            left = self._probation_trades_left[norm] - 1
            if left <= 0:
                del self._probation_trades_left[norm]
                self.logger.info(
                    f"FUT-RM-27 probation complete for {symbol}: "
                    f"restored to full tier weight"
                )
            else:
                self._probation_trades_left[norm] = left

        if norm in self._rolling_benched_until:
            return  # already benched; nothing more to evaluate
        min_trades = max(1, self.rolling_gate_min_trades)
        if len(dq) < min_trades:
            return
        net = sum(dq)
        wins = sum(1 for x in dq if x > 0)
        win_rate = wins / len(dq)
        if (
            net < self.rolling_gate_max_net_pnl_usd
            and win_rate < self.rolling_gate_max_win_rate
        ):
            until = datetime.now() + timedelta(
                minutes=max(1, self.rolling_gate_bench_minutes)
            )
            self._rolling_benched_until[norm] = until
            self.logger.warning(
                f"🪑 FUT-RM-27 rolling gate BENCHED {symbol}: trailing "
                f"{len(dq)} trades net ${net:.2f} < "
                f"${self.rolling_gate_max_net_pnl_usd:.2f} and win rate "
                f"{win_rate:.0%} < {self.rolling_gate_max_win_rate:.0%}; "
                f"no new entries until {until.isoformat()} "
                f"(existing positions unaffected)"
            )

    def seed_symbol_history(self, symbol: str, pnls: List[float]) -> None:
        """Warm the rolling window from persisted trades, OLDEST FIRST, so
        the gate is effective immediately after a restart instead of needing
        rolling_gate_min_trades fresh closes. Bench evaluation runs exactly
        as if the trades closed live (one loud log per benched symbol)."""
        try:
            norm = self._normalize_symbol(symbol)
            if not norm:
                return
            for pnl in pnls:
                self._record_rolling_trade(norm, symbol, float(pnl))
        except Exception as e:
            self.logger.warning(f"seed_symbol_history({symbol}) errored: {e}")

    def should_skip_for_symbol(self, symbol: str) -> Dict:
        """FUT-RM-27 entry gate. Returns {skip, reason, source[, expires_at]}.
        Checks the static tier weight (0 = operator-disabled) then the
        rolling bench. Handles auto-UNBENCH on expiry (logged once, symbol
        re-enters on probation with a cleared window). Fail-open on error."""
        try:
            if not self.symbol_tiering_enabled:
                return {'skip': False, 'reason': 'tiering disabled'}
            norm = self._normalize_symbol(symbol)
            w = self.symbol_size_weights.get(norm)
            if w is not None and w <= 0.0:
                return {
                    'skip': True,
                    'reason': 'symbol disabled (tier weight 0 — operator '
                              'can re-enable via futures_risk.'
                              'symbol_size_weights)',
                    'source': 'static',
                }
            if self.rolling_gate_enabled:
                until = self._rolling_benched_until.get(norm)
                if until is not None:
                    now = datetime.now()
                    if now >= until:
                        # UNBENCH transition: clear bench + window so the
                        # symbol needs fresh evidence to re-bench; start
                        # probation at reduced size.
                        del self._rolling_benched_until[norm]
                        self._rolling_pnl.pop(norm, None)
                        if 0.0 < self.rolling_gate_probation_weight < 1.0:
                            self._probation_trades_left[norm] = max(
                                1, self.rolling_gate_min_trades
                            )
                        self.logger.info(
                            f"✅ FUT-RM-27 rolling gate UNBENCHED {symbol}: "
                            f"probation for next "
                            f"{self._probation_trades_left.get(norm, 0)} "
                            f"trades at "
                            f"{self.rolling_gate_probation_weight:.2f}x size"
                        )
                    else:
                        remaining_min = int((until - now).total_seconds() / 60)
                        return {
                            'skip': True,
                            'reason': (
                                f'rolling-gate bench ~{remaining_min} min '
                                f'remaining'
                            ),
                            'source': 'rolling',
                            'expires_at': until.isoformat(),
                        }
            return {'skip': False, 'reason': 'symbol ok'}
        except Exception as e:
            self.logger.warning(f"should_skip_for_symbol errored: {e}")
            return {'skip': False, 'reason': f'gate error: {e}'}

    def resolve_symbol_size_weight(self, symbol: str) -> float:
        """FUT-RM-27 sizing multiplier: static tier weight (default 1.0)
        × probation factor when in a post-unbench probation window.
        Clamped to [0, 5]; returns 1.0 on any error (fail-open)."""
        try:
            if not self.symbol_tiering_enabled:
                return 1.0
            norm = self._normalize_symbol(symbol)
            w = float(self.symbol_size_weights.get(norm, 1.0))
            if norm in self._probation_trades_left and \
                    self.rolling_gate_probation_weight > 0.0:
                w *= float(self.rolling_gate_probation_weight)
            return max(0.0, min(w, 5.0))
        except Exception as e:
            self.logger.warning(f"resolve_symbol_size_weight errored: {e}")
            return 1.0

    def get_symbol_gate_state(self) -> Dict:
        """Observability snapshot for stats/dashboard surfaces."""
        try:
            return {
                'tiering_enabled': self.symbol_tiering_enabled,
                'rolling_gate_enabled': self.rolling_gate_enabled,
                'static_weights': dict(self.symbol_size_weights),
                'benched': {
                    s: t.isoformat()
                    for s, t in self._rolling_benched_until.items()
                },
                'probation': dict(self._probation_trades_left),
                'window_counts': {
                    s: len(d) for s, d in self._rolling_pnl.items()
                },
            }
        except Exception:
            return {}

    def get_adjusted_leverage(
        self,
        base_leverage: int,
        volatility: float,
        win_rate: float
    ) -> int:
        """
        Adjust leverage based on market conditions and performance

        Args:
            base_leverage: Base leverage to use
            volatility: Current market volatility (%)
            win_rate: Current win rate (0-1)

        Returns:
            int: Adjusted leverage
        """
        try:
            adjusted = base_leverage

            # Reduce leverage in high volatility
            if volatility > 100:
                adjusted = max(1, adjusted - 1)
            elif volatility > 50:
                # Keep same
                pass
            else:
                # Can slightly increase in low volatility
                adjusted = min(self.max_leverage, adjusted + 1)

            # Reduce leverage if poor performance
            if win_rate < 0.4:
                adjusted = max(1, adjusted - 1)

            # Reduce leverage if consecutive losses
            if self.consecutive_losses >= 3:
                adjusted = 1  # Drop to 1x after 3 losses

            return max(1, min(adjusted, self.max_leverage))

        except Exception as e:
            self.logger.error(f"Error adjusting leverage: {e}")
            return 1

    def get_risk_summary(self, positions: List[Dict]) -> Dict:
        """
        Get overall risk summary

        Args:
            positions: List of current positions

        Returns:
            Dict: Risk summary
        """
        try:
            total_long = sum(
                p.get('notional_value', 0)
                for p in positions
                if p.get('side') == 'LONG'
            )

            total_short = sum(
                abs(p.get('notional_value', 0))
                for p in positions
                if p.get('side') == 'SHORT'
            )

            net_exposure = total_long - total_short
            total_exposure = total_long + total_short

            # Count positions at risk
            positions_at_risk = sum(
                1 for p in positions
                if self.check_liquidation_risk(p, p.get('mark_price', 0))['risk_level'] in ['HIGH', 'EXTREME', 'CRITICAL']
            )

            return {
                'total_positions': len(positions),
                'total_long_exposure': total_long,
                'total_short_exposure': total_short,
                'net_exposure': net_exposure,
                'total_exposure': total_exposure,
                'positions_at_risk': positions_at_risk,
                'consecutive_losses': self.consecutive_losses,
                'total_realized_pnl': self.total_realized_pnl
            }

        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {}
