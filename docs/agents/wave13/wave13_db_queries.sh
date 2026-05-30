#!/usr/bin/env bash
# =============================================================================
# ClaudeDex Wave-13 — Consolidated VPS DB-query script
# =============================================================================
# Run this ON THE VPS from ~/claudedex. It is READ-ONLY except the clearly
# marked optional cleanup blocks (which are transaction-wrapped and default to
# ROLLBACK). Each module section leads with schema-discovery (\d / information_
# schema) because the LIVE schema can drift from migrations/.
#
# Usage:
#   bash docs/agents/wave13/wave13_db_queries.sh            # run everything
#   bash docs/agents/wave13/wave13_db_queries.sh solana     # one section
# Sections: solana dex futures sniper copy ai infra arb
# =============================================================================

PG() { docker exec trading-postgres psql -U "$(docker exec trading-postgres cat /run/secrets/db_user)" -d tradingbot -P pager=off "$@"; }

section() { echo; echo "############################################################"; echo "## $1"; echo "############################################################"; }

run_solana() {
  section "SOLANA (Agent 4)"
  PG -c "SELECT column_name, data_type FROM information_schema.columns WHERE table_name='solana_trades' AND column_name IN ('entry_time','exit_time','created_at','pnl_pct','pnl_sol','pnl_usd') ORDER BY column_name;"
  PG -c "SELECT COUNT(*) AS inflated_count, MAX(pnl_pct) AS max_pnl_pct, MIN(pnl_pct) AS min_pnl_pct FROM solana_trades WHERE pnl_pct > 2000 OR pnl_pct < -100;"
  PG -c "SELECT id, token_symbol, strategy, pnl_pct, pnl_sol, pnl_usd, entry_price, exit_price, exit_reason, entry_time FROM solana_trades WHERE pnl_pct > 2000 OR pnl_pct < -100 ORDER BY pnl_pct DESC LIMIT 50;"
  PG -c "SELECT strategy, is_simulated, COUNT(*) AS trades, ROUND(100.0*SUM(CASE WHEN pnl_sol>0 THEN 1 ELSE 0 END)/COUNT(*),1) AS win_rate_pct, ROUND(SUM(pnl_sol)::numeric,4) AS total_pnl_sol FROM solana_trades WHERE pnl_pct BETWEEN -100 AND 2000 GROUP BY strategy, is_simulated ORDER BY strategy, is_simulated;"
  # OPTIONAL CLEANUP (review the SELECT above first, then change ROLLBACK->COMMIT):
  # PG -c "BEGIN; DELETE FROM solana_trades WHERE pnl_pct > 2000 OR pnl_pct < -100; ROLLBACK;"
}

run_dex() {
  section "DEX (Agent 2)"
  PG -c "\d trades"
  PG -c "SELECT chain, COUNT(*) AS n, ROUND(AVG(profit_loss_percentage)::numeric,2) AS avg_pnl, SUM(CASE WHEN profit_loss_percentage>=20 THEN 1 ELSE 0 END) AS pump_pos, SUM(CASE WHEN profit_loss_percentage<=-50 OR metadata->>'close_reason'='stop_loss_rapid' THEN 1 ELSE 0 END) AS rug_pos FROM trades WHERE status='closed' AND metadata IS NOT NULL GROUP BY chain ORDER BY n DESC;"
  PG -c "SELECT metadata->>'close_reason' AS reason, COUNT(*) AS n, ROUND(AVG(profit_loss_percentage)::numeric,2) AS avg_pnl FROM trades WHERE status='closed' GROUP BY reason ORDER BY n DESC LIMIT 20;"
  PG -c "SELECT key, value, updated_at FROM config_settings WHERE config_type='ml_models';"
  PG -c "SELECT ROUND((CAST(metadata->'pair'->>'volume_24h' AS numeric)/NULLIF(CAST(metadata->'pair'->>'liquidity_usd' AS numeric),0))::numeric,2) AS vol_liq_ratio, COUNT(*) AS n, ROUND(AVG(profit_loss_percentage)::numeric,2) AS avg_pnl FROM trades WHERE status='closed' AND metadata->'pair'->>'volume_24h' IS NOT NULL GROUP BY 1 ORDER BY 1 LIMIT 30;"
  PG -c "SELECT config_type, key, value FROM config_settings WHERE config_type IN ('trading','risk_management') ORDER BY config_type, key;"
  PG -c "SELECT chain, COUNT(*) AS n, SUM(CASE WHEN profit_loss_percentage>0 THEN 1 ELSE 0 END) AS wins, ROUND(AVG(profit_loss_percentage)::numeric,2) AS avg_pnl, MAX(profit_loss_percentage) AS best, MIN(profit_loss_percentage) AS worst FROM trades WHERE status='closed' AND created_at>NOW()-INTERVAL '30 days' GROUP BY chain ORDER BY n DESC;"
}

run_futures() {
  section "FUTURES (Agent 3)"
  PG -c "\d futures_trades"
  # NOTE: live schema uses net_pnl/pnl/fees/duration_seconds/exit_time (no pnl_usd/funding columns)
  PG -c "SELECT symbol, count(*) AS trades, round(avg(net_pnl),2) AS avg_net_pnl, round(sum(net_pnl),2) AS total_net_pnl, round(100.0*sum(case when net_pnl>0 then 1 end)/count(*),1) AS win_pct, round(avg(fees),4) AS avg_fees, round(avg(duration_seconds)/60.0,1) AS avg_hold_min, round(avg(leverage),1) AS avg_lev, round(sum(net_pnl) FILTER (WHERE is_simulated),2) AS sim_pnl, round(sum(net_pnl) FILTER (WHERE NOT is_simulated),2) AS live_pnl FROM futures_trades WHERE exit_time>=now()-interval '30 days' GROUP BY symbol ORDER BY total_net_pnl DESC;"
}

run_sniper() {
  section "SNIPER (Agent 6)"
  PG -c "\d sniper_trades"
  PG -c "\d sniper_runtime_stats"
  PG -c "SELECT chain, status, COUNT(*) FROM sniper_trades GROUP BY chain, status ORDER BY chain, status;"
  PG -c "SELECT COUNT(*) AS absurd_rows, MAX(profit_loss_pct) AS max_pct, MIN(profit_loss_pct) AS min_pct FROM sniper_trades WHERE profit_loss_pct > 200 OR profit_loss_pct < -100;"
  PG -c "SELECT chain, COUNT(*) AS closed, ROUND(100.0*SUM(CASE WHEN profit_loss_pct>0 THEN 1 ELSE 0 END)/NULLIF(COUNT(*),0),1) AS win_pct, ROUND(AVG(profit_loss_pct)::numeric,2) AS avg_pct FROM sniper_trades WHERE status='closed' AND profit_loss_pct BETWEEN -100 AND 200 GROUP BY chain ORDER BY closed DESC;"
}

run_copy() {
  section "COPY TRADING (Agent 7)"
  PG -c "\d copytrading_trades"
  PG -c "\d copy_leader_scores"
  PG -c "\d copy_slippage_observations"
  PG -c "SELECT chain, status, COUNT(*) FROM copytrading_trades GROUP BY chain, status ORDER BY chain, status;"
  PG -c "SELECT COUNT(*) AS stablecoin_rows FROM copytrading_trades WHERE token_address='EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v';"
  PG -c "SELECT chain, wallet_address, probation_until, probation_reason FROM copy_leader_scores WHERE on_probation=true AND probation_until>NOW();"
  PG -c "SELECT chain, COUNT(*), percentile_cont(0.5) WITHIN GROUP (ORDER BY delta_ms) AS p50_ms, percentile_cont(0.95) WITHIN GROUP (ORDER BY delta_ms) AS p95_ms FROM copy_slippage_observations WHERE recorded_at>NOW()-INTERVAL '7 days' GROUP BY chain;"
}

run_ai() {
  section "AI ADVISOR (Agent 8)"
  PG -c "SELECT key, value FROM config_settings WHERE config_type='ai_config' AND key IN ('claude_model','openai_model','ai_provider','confidence_threshold');"
  # If 'claude-3-5-haiku-latest' shows up above, fix it:
  # PG -c "UPDATE config_settings SET value='claude-3-5-sonnet-20241022' WHERE config_type='ai_config' AND key='claude_model';"
  PG -c "SELECT COUNT(*), MIN(score), MAX(score), AVG(ABS(score)) AS avg_abs FROM sentiment_logs WHERE timestamp>NOW()-INTERVAL '7 days';"
  PG -c "SELECT COUNT(*), MIN(entry_timestamp), MAX(entry_timestamp) FROM ai_trades WHERE entry_timestamp>NOW()-INTERVAL '30 days';"
  PG -c "SELECT updated_at, stats->>'has_anthropic' AS has_claude, stats->>'direct_trading' AS direct_trading, stats->>'last_skip_reason' AS last_skip FROM ai_runtime_stats WHERE id=1;"
}

run_infra() {
  section "INFRA — Pool engine + secrets (Agent 9)"
  PG -c "SELECT provider_type, name, LEFT(url,40)||'...' AS url_prefix, priority, status, is_enabled, rate_limit_count FROM rpc_api_pool ORDER BY provider_type, priority;"
  PG -c "SELECT category, COUNT(*) AS total, COUNT(*) FILTER (WHERE encrypted_value!='PLACEHOLDER') AS configured FROM secure_credentials WHERE is_active=TRUE GROUP BY category;"
}

run_arb() {
  section "ARBITRAGE (Agent 5)"
  PG -c "\d arbitrage_runtime_stats"
  PG -c "SELECT chain, updated_at, NOW()-updated_at AS age, stats->>'last_error' AS last_error FROM arbitrage_runtime_stats ORDER BY chain;"
  PG -c "SELECT chain, dex_pair, pair_symbol, sample_count, median_pct, p90_pct FROM arb_realized_slippage ORDER BY chain, sample_count DESC LIMIT 30;"
  PG -c "SELECT chain, buy_dex, sell_dex, spread_pct, profit_loss_pct, entry_timestamp, status, is_simulated FROM arbitrage_trades WHERE entry_timestamp>NOW()-INTERVAL '24h' ORDER BY entry_timestamp DESC LIMIT 50;"
}

if [ $# -eq 0 ]; then
  run_solana; run_dex; run_futures; run_sniper; run_copy; run_ai; run_infra; run_arb
else
  for s in "$@"; do "run_$s"; done
fi
