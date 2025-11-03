# ClaudeDex Static Audit & Enhancement Plan

_Auto-generated on:_

- Project Root: `/mnt/data/ClaudeDex_audit/ClaudeDex`
- Python files: 142
- HTML/JS/CSS: 9/5/4
- YAML/SQL/SH/MD: 8/2/20

## 1) High-Level Architecture Overview

Detected key directories and their approximate roles:

- `.` — ~1 Python modules
- `analysis` — ~1 Python modules
- `config` — ~1 Python modules
- `core` — ~1 Python modules
- `data` — ~1 Python modules
- `data/collectors` — ~1 Python modules
- `data/processors` — ~1 Python modules
- `data/storage` — ~1 Python modules
- `ml` — ~1 Python modules
- `ml/models` — ~1 Python modules
- `ml/optimization` — ~1 Python modules
- `ml/training` — ~1 Python modules
- `monitoring` — ~1 Python modules
- `scripts` — ~1 Python modules
- `security` — ~1 Python modules
- `tests` — ~1 Python modules
- `tests/fixtures` — ~1 Python modules
- `tests/integration` — ~1 Python modules
- `tests/performance` — ~1 Python modules
- `tests/security` — ~1 Python modules
- `tests/smoke` — ~1 Python modules
- `tests/unit` — ~1 Python modules
- `trading` — ~1 Python modules
- `trading/chains` — ~1 Python modules
- `trading/chains/solana` — ~1 Python modules
- `trading/executors` — ~1 Python modules
- `trading/orders` — ~1 Python modules
- `trading/strategies` — ~1 Python modules
- `utils` — ~1 Python modules

## 2) Module Coupling (Import Graph Snapshot)

- `create_tree.py` imports: argparse, pathlib, textwrap
- `generate_file_tree.py` imports: argparse, pathlib, typing
- `main.py` imports: argparse, asyncio, config, core, cryptography, data, datetime, dotenv, logging, ml, monitoring, os, pathlib, scripts, security, signal, sys, trading, typing, web3
- `main_log_analyzer.py` imports: collections, json, re
- `setup.py` imports: os, setuptools
- `setup_env_keys.py` imports: cryptography, os, pathlib, secrets, sys
- `trade_analyzer.py` imports: json
- `verify_references.py` imports: argparse, ast, collections, dataclasses, fnmatch, json, os, re, typing
- `xref_symbol_db.py` imports: argparse, ast, collections, dataclasses, fnmatch, json, os, typing
- `analysis/dev_analyzer.py` imports: __future__, aiohttp, asyncio, dataclasses, datetime, decimal, enum, eth_utils, hashlib, json, loguru, typing, utils, web3
- `analysis/liquidity_monitor.py` imports: aiohttp, asyncio, config, core, data, dataclasses, datetime, decimal, logging, monitoring, numpy, pandas, typing
- `analysis/market_analyzer.py` imports: aiohttp, asyncio, core, data, dataclasses, datetime, decimal, logging, ml, numpy, pandas, scipy, sklearn, typing
- `analysis/pump_predictor.py` imports: analysis, asyncio, core, data, dataclasses, datetime, decimal, logging, ml, numpy, pandas, scipy, typing, warnings
- `analysis/rug_detector.py` imports: aiohttp, asyncio, datetime, decimal, eth_account, json, logging, typing, web3
- `analysis/smart_contract_analyzer.py` imports: aiohttp, asyncio, dataclasses, datetime, decimal, enum, eth_utils, hashlib, json, loguru, re, typing, utils, web3
- `analysis/token_scorer.py` imports: analysis, asyncio, core, data, dataclasses, datetime, decimal, logging, ml, numpy, pandas, scipy, typing
- `config/config_manager.py` imports: __future__, aiofiles, asyncio, dataclasses, datetime, enum, hashlib, json, jsonschema, logging, os, pathlib, pydantic, security, typing, yaml
- `config/consecutive_losses_config.py` imports: datetime
- `config/settings.py` imports: decimal, enum, os, pathlib, typing
- `config/validation.py` imports: datetime, decimal, ipaddress, json, logging, pathlib, pydantic, re, typing, urllib, web3, yaml
- `core/decision_maker.py` imports: asyncio, dataclasses, datetime, decimal, enum, monitoring, numpy, typing
- `core/engine.py` imports: asyncio, collections, config, core, data, dataclasses, datetime, decimal, enum, json, logging, ml, monitoring, numpy, os, security, time, traceback, trading, typing, uuid
- `core/event_bus.py` imports: asyncio, collections, dataclasses, datetime, enum, json, typing, uuid
- `core/pattern_analyzer.py` imports: asyncio, dataclasses, datetime, enum, numpy, pandas, scipy, sklearn, talib, typing
- `core/portfolio_manager.py` imports: asyncio, collections, config, dataclasses, datetime, decimal, enum, monitoring, numpy, pandas, typing, uuid
- `core/risk_manager.py` imports: asyncio, core, data, dataclasses, datetime, decimal, enum, json, monitoring, numpy, security, traceback, typing, utils
- `data/collectors/chain_data.py` imports: aiohttp, asyncio, collections, dataclasses, datetime, decimal, eth_account, json, typing, web3
- `data/collectors/dexscreener.py` imports: aiohttp, asyncio, collections, dataclasses, datetime, json, numpy, time, traceback, typing
- `data/collectors/honeypot_checker.py` imports: aiohttp, asyncio, decimal, json, logging, re, solders, typing, utils, web3
- `data/collectors/mempool_monitor.py` imports: asyncio, collections, dataclasses, datetime, json, typing, web3
- `data/collectors/social_data.py` imports: aiohttp, asyncio, dataclasses, datetime, decimal, enum, hashlib, loguru, numpy, re, textblob, typing, utils
- `data/collectors/token_sniffer.py` imports: aiohttp, asyncio, dataclasses, datetime, decimal, enum, hashlib, json, loguru, re, typing, utils
- `data/collectors/volume_analyzer.py` imports: aiohttp, asyncio, collections, dataclasses, datetime, decimal, enum, loguru, numpy, statistics, typing, utils
- `data/collectors/whale_tracker.py` imports: aiohttp, asyncio, datetime, decimal, logging, numpy, typing, utils, web3
- `data/processors/aggregator.py` imports: asyncio, collections, datetime, decimal, loguru, numpy, pandas, statistics, typing
- `data/processors/feature_extractor.py` imports: datetime, decimal, loguru, numpy, pandas, scipy, sklearn, talib, typing
- `data/processors/normalizer.py` imports: dataclasses, datetime, decimal, enum, loguru, numpy, pandas, re, typing
- `data/processors/validator.py` imports: dataclasses, datetime, decimal, enum, loguru, numpy, re, typing
- `data/storage/cache.py` imports: asyncio, datetime, logging, orjson, pickle, redis, typing
- `data/storage/database.py` imports: asyncio, asyncpg, contextlib, datetime, decimal, logging, orjson, typing, urllib
- `data/storage/models.py` imports: datetime, decimal, enum, sqlalchemy, typing, uuid
- `ml/models/ensemble_model.py` imports: asyncio, concurrent, config, data, dataclasses, datetime, joblib, json, lightgbm, numpy, pandas, pathlib, sklearn, sys, torch, typing, xgboost
- `ml/models/pump_predictor.py` imports: asyncio, datetime, joblib, lightgbm, logging, numpy, pandas, pathlib, pickle, sklearn, tensorflow, typing, xgboost
- `ml/models/rug_classifier.py` imports: asyncio, datetime, joblib, lightgbm, logging, numpy, pandas, pathlib, pickle, sklearn, typing, xgboost
- `ml/models/volume_validator.py` imports: dataclasses, datetime, decimal, joblib, lightgbm, loguru, numpy, pandas, pickle, sklearn, typing, warnings, xgboost
- `ml/optimization/hyperparameter.py` imports: dataclasses, json, loguru, numpy, typing
- `ml/optimization/reinforcement.py` imports: collections, dataclasses, loguru, numpy, typing
- `monitoring/alerts.py` imports: aiohttp, asyncio, collections, dataclasses, datetime, decimal, enum, hashlib, hmac, json, logging, typing
- `monitoring/dashboard.py` imports: aiohttp, aiohttp_cors, aiohttp_sse, asyncio, dataclasses, datetime, decimal, enum, jinja2, json, logging, socketio, typing
- `monitoring/enhanced_dashboard.py` imports: aiohttp, aiohttp_cors, aiohttp_sse, asyncio, collections, config, csv, dataclasses, datetime, decimal, enum, io, jinja2, json, logging, pandas, socketio, statistics, typing
- `monitoring/logger.py` imports: asyncio, collections, dataclasses, datetime, decimal, enum, json, logging, numpy, pandas, pathlib, sqlite3, statistics, sys, traceback, typing, uuid
- `monitoring/performance.py` imports: asyncio, collections, dataclasses, datetime, decimal, enum, json, logging, numpy, pandas, pathlib, sqlite3, statistics, sys, traceback, typing
- `scripts/analyze_strategy.py` imports: argparse, asyncio, asyncpg, datetime, numpy, os, typing
- `scripts/check_balance.py` imports: asyncio, decimal, os, web3
- `scripts/close_all_positions.py` imports: argparse, asyncio, asyncpg, datetime, os, pathlib, sys, web3
- `scripts/daily_report.py` imports: asyncio, asyncpg, datetime, decimal, json, os
- `scripts/dev_autofix_imports.py` imports: argparse, ast, os, pathlib, re, sys, typing
- `scripts/emergency_stop.py` imports: asyncio, asyncpg, datetime, os, redis
- `scripts/export_trades.py` imports: argparse, asyncio, asyncpg, datetime, json, os, pandas
- `scripts/fix_illegal_relatives.py` imports: argparse, ast, dataclasses, os, pathlib, typing
- `scripts/generate_report.py` imports: argparse, asyncio, asyncpg, datetime, json, pandas, pathlib
- `scripts/generate_solana_wallet.py` imports: base58, solders, sys
- `scripts/health_check.py` imports: aiohttp, asyncio, asyncpg, datetime, redis, sys
- `scripts/init_config.py` imports: cryptography, json, os, pathlib, secrets
- `scripts/migrate_database.py` imports: asyncio, asyncpg, datetime, os, pathlib
- `scripts/optimize_db.py` imports: asyncio, asyncpg, datetime, os
- `scripts/overnight_summary.py` imports: asyncio, asyncpg, datetime, os
- `scripts/post_update_check.py` imports: asyncio, asyncpg, os, pathlib, redis, sys, web3
- `scripts/reset_db_sequences.py` imports: asyncio, asyncpg, os
- `scripts/reset_nonce.py` imports: asyncio, os, web3
- `scripts/retrain_models.py` imports: argparse, asyncio, asyncpg, datetime, ml, numpy, os, pathlib, sys
- `scripts/run_tests.py` imports: argparse, json, os, pathlib, psycopg2, redis, subprocess, sys, time, typing
- `scripts/security_audit.py` imports: asyncio, datetime, hashlib, json, os, pathlib
- `scripts/setup_database.py` imports: asyncio, asyncpg, os, pathlib, sys
- `scripts/solana_wallet_balance.py` imports: trading
- `scripts/strategy_analysis.py` imports: argparse, asyncio, asyncpg, collections, datetime, numpy, os
- `scripts/test_alerts.py` imports: asyncio, monitoring, os, pathlib, sys
- `scripts/test_apis.py` imports: aiohttp, asyncio, datetime, os
- `scripts/test_solana_setup.py` imports: asyncio, trading
- `scripts/update_blacklists.py` imports: aiohttp, asyncio, datetime, json, os, pathlib
- `scripts/update_models.py` imports: argparse, asyncio, datetime, json, ml, os, pathlib, sys
- `scripts/verify_claudedex_plus.py` imports: argparse, ast, collections, dataclasses, glob, json, pathlib, re, typing
- `scripts/verify_claudedex_plus2.py` imports: argparse, ast, collections, dataclasses, glob, json, pathlib, re, typing
- `scripts/verify_claudedex_plus3.py` imports: argparse, ast, collections, dataclasses, difflib, glob, json, pathlib, re, typing
- `scripts/weekly_report.py` imports: asyncio, asyncpg, datetime, numpy, os
- `scripts/withdraw_funds.py` imports: argparse, asyncio, decimal, eth_account, os, web3
- `security/api_security.py` imports: collections, datetime, hashlib, hmac, ipaddress, jwt, logging, time, typing
- `security/audit_logger.py` imports: aiofiles, asyncio, cryptography, dataclasses, datetime, enum, gzip, hashlib, hmac, json, logging, os, pathlib, typing, uuid
- `security/encryption.py` imports: base64, cryptography, datetime, hashlib, json, logging, os, secrets, typing
- `security/wallet_security.py` imports: asyncio, base64, cryptography, dataclasses, datetime, decimal, enum, eth_account, hashlib, hdwallet, hmac, json, logging, os, secrets, security, trading, typing, web3
- `tests/conftest.py` imports: aioredis, asyncio, asyncpg, config, core, data, datetime, decimal, json, os, pathlib, pytest, security, sys, typing, unittest
- `tests/test_all.py` imports: pathlib, pytest, sys
- `tests/fixtures/mock_data.py` imports: datetime, decimal, random, string, typing
- `tests/fixtures/test_helpers.py` imports: asyncio, datetime, decimal, typing, unittest
- `tests/integration/test_data_integration.py` imports: aiohttp, asyncio, data, datetime, decimal, pytest, unittest
- `tests/integration/test_dexscreener.py` imports: asyncio, data, traceback
- `tests/integration/test_ml_integration.py` imports: data, datetime, decimal, ml, numpy, pandas, pytest
- `tests/integration/test_trading_integration.py` imports: datetime, decimal, pytest, trading, unittest
- `tests/performance/test_performance.py` imports: asyncio, concurrent, core, data, datetime, decimal, memory_profiler, ml, numpy, os, psutil, pytest, time, trading
- `tests/security/test_security.py` imports: asyncio, config, datetime, hashlib, hmac, pytest, secrets, security, unittest

## 3) Configuration & Hardcode Scan

### print_debug
Count: 770
- **create_tree.py** → `print(`  
  Snippet: `}.  Replace this content with your real implementation. \"\"\"  def main():     print("This is a placeholder for {name}.")  if __name__ == "__main__":     main() """ `
- **create_tree.py** → `print(`  
  Snippet: `solve()     created, skipped = create_tree(root, overwrite=args.overwrite)      print(f"\nRoot: {root}\n")     if created:         print("Created files:")         for`
- **create_tree.py** → `print(`  
  Snippet: `erwrite=args.overwrite)      print(f"\nRoot: {root}\n")     if created:         print("Created files:")         for f in created:             print("  +", f)     else`
- **create_tree.py** → `print(`  
  Snippet: ` created:         print("Created files:")         for f in created:             print("  +", f)     else:         print("No files were created.")      if skipped:    `
- **create_tree.py** → `print(`  
  Snippet: `iles:")         for f in created:             print("  +", f)     else:         print("No files were created.")      if skipped:         print("\nSkipped (already exi`
- **create_tree.py** → `print(`  
  Snippet: `, f)     else:         print("No files were created.")      if skipped:         print("\nSkipped (already existed):")         for f in skipped:             print("  -`
- **create_tree.py** → `print(`  
  Snippet: `    print("\nSkipped (already existed):")         for f in skipped:             print("  -", f)      print("\nDone.")  if __name__ == "__main__":     main() `
- **create_tree.py** → `print(`  
  Snippet: `(already existed):")         for f in skipped:             print("  -", f)      print("\nDone.")  if __name__ == "__main__":     main() `
- **generate_file_tree.py** → `print(`  
  Snippet: `ok=True)     out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")     print(f"Wrote {len(lines)} lines to: {out_path}")  def main():     p = argparse.Argume`
- **generate_file_tree.py** → `print(`  
  Snippet: `_args()      root = Path(args.root).resolve()     if not root.exists():         print(f"ERROR: root path does not exist: {root}")         return      default_out = ro`
- **main.py** → `print(`  
  Snippet: `   enabled_chains = [c.strip() for c in enabled_chains_str.split(',')]          print(f"🌐 Enabled chains: {', '.join(enabled_chains)}")          # Multi-chain configu`
- **main.py** → `print(`  
  Snippet: `_hours': int(os.getenv(f'{chain_upper}_MAX_AGE_HOURS', '24'))         }         print(f"  └─ {chain}: min_liq=${chains_config[chain]['min_liquidity']:,.0f}")         `
- **main.py** → `print(`  
  Snippet: `     await asyncio.sleep(60)             except Exception as e:                 print(f"Health check error: {e}")     class TradingBotApplication:     """Main applica`
- **main.py** → `print(`  
  Snippet: `validate_config_at_startup(config)              except ValueError as e:         print(f"❌ Configuration validation failed:")         print(f"   {e}")         sys.exit`
- **main.py** → `print(`  
  Snippet: `t ValueError as e:         print(f"❌ Configuration validation failed:")         print(f"   {e}")         sys.exit(1)     except Exception as e:         print(f"❌ Erro`
- **main.py** → `print(`  
  Snippet: `        print(f"   {e}")         sys.exit(1)     except Exception as e:         print(f"❌ Error validating configuration: {e}")         sys.exit(1)     # ============`
- **main.py** → `print(`  
  Snippet: `  )          try:         await app.run()     except KeyboardInterrupt:         print("\n👋 Goodbye!")     except Exception as e:         print(f"❌ Fatal error: {e}") `
- **main.py** → `print(`  
  Snippet: `oardInterrupt:         print("\n👋 Goodbye!")     except Exception as e:         print(f"❌ Fatal error: {e}")         sys.exit(1)  if __name__ == "__main__":     # Che`
- **main.py** → `print(`  
  Snippet: `__main__":     # Check Python version     if sys.version_info < (3, 9):         print("Python 3.9+ required")         sys.exit(1)              # Run the application  `
- **main_log_analyzer.py** → `print(`  
  Snippet: `es = defaultdict(int)     cooldowns = defaultdict(int)     volatility = []      print(f"Analyzing log file: {log_file}")      with open(log_file, 'r') as f:         f`
- **main_log_analyzer.py** → `print(`  
  Snippet: `except json.JSONDecodeError:                 pass  # Ignore non-JSON lines      print("\n--- Main Log Analysis Results ---")     for key, value in stats.items():     `
- **main_log_analyzer.py** → `print(`  
  Snippet: `-- Main Log Analysis Results ---")     for key, value in stats.items():         print(f"{key.replace('_', ' ').title()}: {value}")      print("\n--- Cooldown Reasons `
- **main_log_analyzer.py** → `print(`  
  Snippet: ` stats.items():         print(f"{key.replace('_', ' ').title()}: {value}")      print("\n--- Cooldown Reasons ---")     for reason, count in cooldowns.items():       `
- **main_log_analyzer.py** → `print(`  
  Snippet: `n--- Cooldown Reasons ---")     for reason, count in cooldowns.items():         print(f"{reason.replace('_', ' ').title()}: {count}")      if volatility:         prin`
- **main_log_analyzer.py** → `print(`  
  Snippet: `int(f"{reason.replace('_', ' ').title()}: {count}")      if volatility:         print("\n--- Volatility Analysis ---")         print(f"Average Volatility Warning: {su`
- **main_log_analyzer.py** → `print(`  
  Snippet: `t}")      if volatility:         print("\n--- Volatility Analysis ---")         print(f"Average Volatility Warning: {sum(volatility) / len(volatility):.2f}%")        `
- **main_log_analyzer.py** → `print(`  
  Snippet: `"Average Volatility Warning: {sum(volatility) / len(volatility):.2f}%")         print(f"Max Volatility Warning: {max(volatility):.2f}%")   if __name__ == "__main__": `
- **setup_env_keys.py** → `print(`  
  Snippet: `t(private_key.encode())          return encrypted_key.decode()  def main():     print("🔐 ClaudeDex Environment Key Setup")     print("=" * 40)          # Generate enc`
- **setup_env_keys.py** → `print(`  
  Snippet: `ed_key.decode()  def main():     print("🔐 ClaudeDex Environment Key Setup")     print("=" * 40)          # Generate encryption key     encryption_key = generate_encry`
- **setup_env_keys.py** → `print(`  
  Snippet: `   # Generate encryption key     encryption_key = generate_encryption_key()     print(f"✅ Generated ENCRYPTION_KEY: {encryption_key}")          # Generate JWT secret `
- **setup_env_keys.py** → `print(`  
  Snippet: `ey}")          # Generate JWT secret     jwt_secret = generate_jwt_secret()     print(f"✅ Generated JWT_SECRET: {jwt_secret}")          # Get private key from user   `
- **setup_env_keys.py** → `print(`  
  Snippet: `✅ Generated JWT_SECRET: {jwt_secret}")          # Get private key from user     print("\n📝 Please enter your wallet private key (64 hex characters, no 0x prefix):")  `
- **setup_env_keys.py** → `print(`  
  Snippet: ` Key: ").strip()          if not private_key or len(private_key) != 64:         print("❌ Invalid private key format. Must be 64 hexadecimal characters.")         retu`
- **setup_env_keys.py** → `print(`  
  Snippet: `ncrypted_private_key = encrypt_private_key(private_key, encryption_key)         print(f"✅ Encrypted PRIVATE_KEY: {encrypted_private_key}")     except Exception as e: `
- **setup_env_keys.py** → `print(`  
  Snippet: `ypted PRIVATE_KEY: {encrypted_private_key}")     except Exception as e:         print(f"❌ Error encrypting private key: {e}")         return          # Generate .env `
- **setup_env_keys.py** → `print(`  
  Snippet: `env file     with open('.env', 'w') as f:         f.write(env_content)          print(f"\n✅ Created .env file with encrypted keys")     print(f"🔒 Your private key has`
- **setup_env_keys.py** → `print(`  
  Snippet: `e(env_content)          print(f"\n✅ Created .env file with encrypted keys")     print(f"🔒 Your private key has been encrypted and stored securely")     print(f"⚠️  Re`
- **setup_env_keys.py** → `print(`  
  Snippet: `s")     print(f"🔒 Your private key has been encrypted and stored securely")     print(f"⚠️  Remember to add .env to your .gitignore file!")          # Create .gitigno`
- **setup_env_keys.py** → `print(`  
  Snippet: `th open('.gitignore', 'w') as f:             f.write(gitignore_content)         print(f"✅ Created .gitignore file")  if __name__ == "__main__":     main() `
- **trade_analyzer.py** → `print(`  
  Snippet: `sses     win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0      print(f"Total trades: {total_trades}")     print(f"Wins: {wins}")     print(f"Losses: `
- **trade_analyzer.py** → `print(`  
  Snippet: ` 100 if total_trades > 0 else 0      print(f"Total trades: {total_trades}")     print(f"Wins: {wins}")     print(f"Losses: {losses}")     print(f"Win rate: {win_rate:`
- **trade_analyzer.py** → `print(`  
  Snippet: `se 0      print(f"Total trades: {total_trades}")     print(f"Wins: {wins}")     print(f"Losses: {losses}")     print(f"Win rate: {win_rate:.2f}%")     print(f"Total P`
- **trade_analyzer.py** → `print(`  
  Snippet: ` {total_trades}")     print(f"Wins: {wins}")     print(f"Losses: {losses}")     print(f"Win rate: {win_rate:.2f}%")     print(f"Total P&L: ${total_pnl:.2f}")      ret`
- **trade_analyzer.py** → `print(`  
  Snippet: `s}")     print(f"Losses: {losses}")     print(f"Win rate: {win_rate:.2f}%")     print(f"Total P&L: ${total_pnl:.2f}")      return {         "total_trades": total_trad`
- **verify_references.py** → `print(`  
  Snippet: `res))     total=sum(len(v) for k,v in res.items() if not k.startswith("_"))     print(f"[verify_references] Analyzed {len(py)} Python files and {len(cfg)} config file`
- **verify_references.py** → `print(`  
  Snippet: `_references] Analyzed {len(py)} Python files and {len(cfg)} config files.")     print(f"[verify_references] Modules indexed: {len(az.modules)} | Calls analyzed: {len(`
- **verify_references.py** → `print(`  
  Snippet: `ces] Modules indexed: {len(az.modules)} | Calls analyzed: {len(az.calls)}")     print(f"[verify_references] Skipped external: {az.skipped_external} | Skipped unknown:`
- **verify_references.py** → `print(`  
  Snippet: `d external: {az.skipped_external} | Skipped unknown: {az.skipped_unknown}")     print(f"[verify_references] Findings: {total} (see {args.md_out} / {args.json_out})") `
- **xref_symbol_db.py** → `print(`  
  Snippet: `2,ensure_ascii=False)         write_symbol_md(root, db, "symbol_db.md")         print(f"[build] Files: {len(files)} | Modules: {len(db)-1} | Collisions(classes:{len(d`
- **xref_symbol_db.py** → `print(`  
  Snippet: `th open(args.md_out,"w",encoding="utf-8") as f: f.write("".join(lines))         print(f"[verify] Problems: {total} (see {args.md_out} / {args.json_out})")         ret`
- ... and 720 more.
### http_urls
Count: 109
- **main.py** → `https://api.dexscreener.com/latest`  
  Snippet: `ey': os.getenv('DEXSCREENER_API_KEY', ''),                         'base_url': 'https://api.dexscreener.com/latest',                         'rate_limit': 300,                         'chains': `
- **main.py** → `https://api.mainnet-beta.solana.com`  
  Snippet: `ue                 self.config['solana_rpc_url'] = os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com')                 self.config['solana_private_key'] = os.getenv('SOLANA_PRIVATE`
- **main.py** → `https://api.mainnet-beta.solana.com`  
  Snippet: `   'enabled': True,                     'rpc_url': os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com'),                     'private_key': os.getenv('SOLANA_PRIVATE_KEY'),         `
- **setup.py** → `https://github.com/claudedex/trading-bot`  
  Snippet: `n=long_description,     long_description_content_type="text/markdown",     url="https://github.com/claudedex/trading-bot",     project_urls={         "Bug Tracker": "https://github.com/claudedex/tradi`
- **setup.py** → `https://github.com/claudedex/trading-bot/issues`  
  Snippet: `//github.com/claudedex/trading-bot",     project_urls={         "Bug Tracker": "https://github.com/claudedex/trading-bot/issues",         "Documentation": "https://docs.claudedex.io",         "Source"`
- **setup.py** → `https://docs.claudedex.io`  
  Snippet: `": "https://github.com/claudedex/trading-bot/issues",         "Documentation": "https://docs.claudedex.io",         "Source": "https://github.com/claudedex/trading-bot",     },     clas`
- **setup.py** → `https://github.com/claudedex/trading-bot`  
  Snippet: `sues",         "Documentation": "https://docs.claudedex.io",         "Source": "https://github.com/claudedex/trading-bot",     },     classifiers=[         "Development Status :: 4 - Beta",         "I`
- **setup_env_keys.py** → `https://eth-mainnet.g.alchemy.com/v2/your_alchemy_key`  
  Snippet: `cryption_key} PRIVATE_KEY={encrypted_private_key}  # Blockchain RPC ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/your_alchemy_key BSC_RPC_URL=https://bsc-dataseed.binance.org/  # Monitoring TELEGR`
- **setup_env_keys.py** → `https://bsc-dataseed.binance.org/`  
  Snippet: `C ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/your_alchemy_key BSC_RPC_URL=https://bsc-dataseed.binance.org/  # Monitoring TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here TELEGRAM_CHAT_ID=`
- **analysis/liquidity_monitor.py** → `https://api.dexscreener.com/latest/dex`  
  Snippet: `tr, List] = {}                  # API endpoints         self.dexscreener_api = "https://api.dexscreener.com/latest/dex"         self.defined_api = "https://api.defined.fi"                  # Running`
- **analysis/liquidity_monitor.py** → `https://api.defined.fi`  
  Snippet: `ener_api = "https://api.dexscreener.com/latest/dex"         self.defined_api = "https://api.defined.fi"                  # Running tasks         self._monitoring_tasks: List[asyncio.`
- **analysis/rug_detector.py** → `https://api.etherscan.io/api`  
  Snippet: `        # Simplified for example         explorers = {             'ethereum': 'https://api.etherscan.io/api',             'bsc': 'https://api.bscscan.com/api',             'polygon': 'htt`
- **analysis/rug_detector.py** → `https://api.bscscan.com/api`  
  Snippet: `= {             'ethereum': 'https://api.etherscan.io/api',             'bsc': 'https://api.bscscan.com/api',             'polygon': 'https://api.polygonscan.com/api'         }           `
- **analysis/rug_detector.py** → `https://api.polygonscan.com/api`  
  Snippet: `api',             'bsc': 'https://api.bscscan.com/api',             'polygon': 'https://api.polygonscan.com/api'         }                  api_url = explorers.get(chain)         if not api_u`
- **config/settings.py** → `https://eth-mainnet.alchemyapi.io/v2/your-api-key`  
  Snippet: `um',             chain_id=1,             rpc_url=os.getenv('ETHEREUM_RPC_URL', 'https://eth-mainnet.alchemyapi.io/v2/your-api-key'),             explorer_url='https://etherscan.io',             native`
- **config/settings.py** → `https://etherscan.io`  
  Snippet: `'https://eth-mainnet.alchemyapi.io/v2/your-api-key'),             explorer_url='https://etherscan.io',             native_token='ETH'         ),         'bsc': ChainConfig(        `
- **config/settings.py** → `https://bsc-dataseed1.binance.org`  
  Snippet: ` Chain',             chain_id=56,             rpc_url=os.getenv('BSC_RPC_URL', 'https://bsc-dataseed1.binance.org'),             explorer_url='https://bscscan.com',             native_token='BN`
- **config/settings.py** → `https://bscscan.com`  
  Snippet: `('BSC_RPC_URL', 'https://bsc-dataseed1.binance.org'),             explorer_url='https://bscscan.com',             native_token='BNB'         ),         'polygon': ChainConfig(    `
- **config/settings.py** → `https://polygon-rpc.com`  
  Snippet: `n',             chain_id=137,             rpc_url=os.getenv('POLYGON_RPC_URL', 'https://polygon-rpc.com'),             explorer_url='https://polygonscan.com',             native_token`
- **config/settings.py** → `https://polygonscan.com`  
  Snippet: `getenv('POLYGON_RPC_URL', 'https://polygon-rpc.com'),             explorer_url='https://polygonscan.com',             native_token='MATIC'         ),         'arbitrum': ChainConfig( `
- **config/settings.py** → `https://arb1.arbitrum.io/rpc`  
  Snippet: `             chain_id=42161,             rpc_url=os.getenv('ARBITRUM_RPC_URL', 'https://arb1.arbitrum.io/rpc'),             explorer_url='https://arbiscan.io',             native_token='ET`
- **config/settings.py** → `https://arbiscan.io`  
  Snippet: `('ARBITRUM_RPC_URL', 'https://arb1.arbitrum.io/rpc'),             explorer_url='https://arbiscan.io',             native_token='ETH'         ),         'avalanche': ChainConfig(  `
- **config/settings.py** → `https://api.avax.network/ext/bc/C/rpc`  
  Snippet: `            chain_id=43114,             rpc_url=os.getenv('AVALANCHE_RPC_URL', 'https://api.avax.network/ext/bc/C/rpc'),             explorer_url='https://snowtrace.io',             native_token='A`
- **config/settings.py** → `https://snowtrace.io`  
  Snippet: `E_RPC_URL', 'https://api.avax.network/ext/bc/C/rpc'),             explorer_url='https://snowtrace.io',             native_token='AVAX'         ),         'fantom': ChainConfig(    `
- **config/settings.py** → `https://rpc.fantom.network`  
  Snippet: `om',             chain_id=250,             rpc_url=os.getenv('FANTOM_RPC_URL', 'https://rpc.fantom.network'),             explorer_url='https://ftmscan.com',             native_token='FT`
- **config/settings.py** → `https://ftmscan.com`  
  Snippet: `tenv('FANTOM_RPC_URL', 'https://rpc.fantom.network'),             explorer_url='https://ftmscan.com',             native_token='FTM'         ),         'base': ChainConfig(       `
- **config/settings.py** → `https://mainnet.base.org`  
  Snippet: `ase',             chain_id=8453,             rpc_url=os.getenv('BASE_RPC_URL', 'https://mainnet.base.org'),             explorer_url='https://basescan.org',             native_token='E`
- **config/settings.py** → `https://basescan.org`  
  Snippet: `s.getenv('BASE_RPC_URL', 'https://mainnet.base.org'),             explorer_url='https://basescan.org',             native_token='ETH'         )     }          # Testnet Chains (for`
- **config/settings.py** → `https://eth-goerli.alchemyapi.io/v2/your-api-key`  
  Snippet: `               chain_id=5,                 rpc_url=os.getenv('GOERLI_RPC_URL', 'https://eth-goerli.alchemyapi.io/v2/your-api-key'),                 explorer_url='https://goerli.etherscan.io',         `
- **config/settings.py** → `https://goerli.etherscan.io`  
  Snippet: `tps://eth-goerli.alchemyapi.io/v2/your-api-key'),                 explorer_url='https://goerli.etherscan.io',                 native_token='ETH',                 is_testnet=True          `
- **config/settings.py** → `https://data-seed-prebsc-1-s1.binance.org:8545`  
  Snippet: `         chain_id=97,                 rpc_url=os.getenv('BSC_TESTNET_RPC_URL', 'https://data-seed-prebsc-1-s1.binance.org:8545'),                 explorer_url='https://testnet.bscscan.com',           `
- **config/settings.py** → `https://testnet.bscscan.com`  
  Snippet: `https://data-seed-prebsc-1-s1.binance.org:8545'),                 explorer_url='https://testnet.bscscan.com',                 native_token='BNB',                 is_testnet=True          `
- **data/collectors/dexscreener.py** → `https://api.dexscreener.com`  
  Snippet: `onfig         self.api_key = config.get('api_key', '')         self.base_url = "https://api.dexscreener.com"                  # Rate limiting         self.rate_limit = config.get('rate_li`
- **data/collectors/honeypot_checker.py** → `https://api.rugcheck.xyz/v1/tokens/{address}/report/summary`  
  Snippet: `dress": address[:16] + "..."                 }                          url = f"https://api.rugcheck.xyz/v1/tokens/{address}/report/summary"                          async with self.session.get(url, t`
- **data/collectors/honeypot_checker.py** → `https://api.honeypot.is/v2/IsHoneypot`  
  Snippet: `      try:             chain_id = self._get_chain_id(chain)             url = f"https://api.honeypot.is/v2/IsHoneypot"             params = {                 "address": address,                 "ch`
- **data/collectors/honeypot_checker.py** → `https://tokensniffer.com/api/v2/tokens/{chain_id}/{address}`  
  Snippet: `      try:             chain_id = self._get_chain_id(chain)             url = f"https://tokensniffer.com/api/v2/tokens/{chain_id}/{address}"             headers = {"Authorization": f"Bearer {self.api_`
- **data/collectors/honeypot_checker.py** → `https://api.gopluslabs.io/api/v1/token_security/{chain_id}`  
  Snippet: `      try:             chain_id = self._get_chain_id(chain)             url = f"https://api.gopluslabs.io/api/v1/token_security/{chain_id}"             params = {"contract_addresses": address}        `
- **data/collectors/social_data.py** → `https://api.twitter.com/2/tweets/search/recent`  
  Snippet: `          async with aiohttp.ClientSession() as session:                 url = "https://api.twitter.com/2/tweets/search/recent"                 async with session.get(url, headers=headers, params=para`
- **data/collectors/social_data.py** → `https://www.reddit.com/api/v1/access_token`  
  Snippet: `n() as session:                 # Get access token                 token_url = "https://www.reddit.com/api/v1/access_token"                 token_data = {"grant_type": "client_credentials"}           `
- **data/collectors/social_data.py** → `https://oauth.reddit.com/search`  
  Snippet: `     headers["Authorization"] = f"Bearer {token}"                 search_url = "https://oauth.reddit.com/search"                 params = {                     "q": symbol,                   `
- **data/collectors/social_data.py** → `https://api.twitter.com/2/tweets/search/recent`  
  Snippet: `      async with aiohttp.ClientSession() as session:                     url = "https://api.twitter.com/2/tweets/search/recent"                     params = {"query": "test", "max_results": 10}       `
- **data/collectors/token_sniffer.py** → `https://tokensniffer.com/api/v2`  
  Snippet: `fig = config                  # API endpoints         self.token_sniffer_api = "https://tokensniffer.com/api/v2"         self.goplus_api = "https://api.gopluslabs.io/api/v1"         self.hone`
- **data/collectors/token_sniffer.py** → `https://api.gopluslabs.io/api/v1`  
  Snippet: `oken_sniffer_api = "https://tokensniffer.com/api/v2"         self.goplus_api = "https://api.gopluslabs.io/api/v1"         self.honeypot_api = "https://api.honeypot.is/v2"         self.dextools`
- **data/collectors/token_sniffer.py** → `https://api.honeypot.is/v2`  
  Snippet: `lf.goplus_api = "https://api.gopluslabs.io/api/v1"         self.honeypot_api = "https://api.honeypot.is/v2"         self.dextools_api = "https://api.dextools.io/v1"                  # AP`
- **data/collectors/token_sniffer.py** → `https://api.dextools.io/v1`  
  Snippet: `  self.honeypot_api = "https://api.honeypot.is/v2"         self.dextools_api = "https://api.dextools.io/v1"                  # API keys         self.token_sniffer_key = config.get("token`
- **monitoring/alerts.py** → `https://api.telegram.org/bot{bot_token}/sendMessage`  
  Snippet: `                         # Send to Telegram with MarkdownV2             url = f"https://api.telegram.org/bot{bot_token}/sendMessage"             payload = {                 'chat_id': chat_id,        `
- **monitoring/dashboard.py** → `http://{self.host}:{self.port}`  
  Snippet: `elf.port)         await site.start()         logger.info(f"Dashboard running on http://{self.host}:{self.port}")          def create_portfolio_chart(self) -> str:         """Create portfolio`
- **monitoring/enhanced_dashboard.py** → `http://{self.host}:{self.port}`  
  Snippet: `         await site.start()         logger.info(f"Enhanced dashboard running on http://{self.host}:{self.port}")                  # Keep running         await asyncio.Event().wait()`
- **scripts/generate_solana_wallet.py** → `https://www.coinbase.com/`  
  Snippet: `$300+)")     print("")     print("Where to buy SOL:")     print("   • Coinbase: https://www.coinbase.com/")     print("   • Binance: https://www.binance.com/")     print("   • Phantom W`
- **scripts/generate_solana_wallet.py** → `https://www.binance.com/`  
  Snippet: `     print("   • Coinbase: https://www.coinbase.com/")     print("   • Binance: https://www.binance.com/")     print("   • Phantom Wallet: https://phantom.app/")     print("")         `
- ... and 59 more.
### env_usage
Count: 38
- **main.py** → `os.getenv('WEB3_PROVIDER_URL')`  
  Snippet: `     _logger = logging.getLogger(__name__)          try:         provider_url = os.getenv('WEB3_PROVIDER_URL')         if not provider_url:             _logger.warning("No WEB3_PROVIDER_URL `
- **main.py** → `os.getenv('TWITTER_API_KEY')`  
  Snippet: `': os.getenv('TWITTER_API_SECRET', ''),                         'enabled': bool(os.getenv('TWITTER_API_KEY'))                     }                 }                                  # ✅ A`
- **main.py** → `os.getenv('WEB3_PROVIDER_URL')`  
  Snippet: `ig:                 self.config['web3'] = {                     'provider_url': os.getenv('WEB3_PROVIDER_URL'),                     'backup_providers': [                         os.getenv('W`
- **main.py** → `os.getenv('WEB3_BACKUP_PROVIDER_1')`  
  Snippet: `OVIDER_URL'),                     'backup_providers': [                         os.getenv('WEB3_BACKUP_PROVIDER_1'),                         os.getenv('WEB3_BACKUP_PROVIDER_2')                   `
- **main.py** → `os.getenv('WEB3_BACKUP_PROVIDER_2')`  
  Snippet: `                   os.getenv('WEB3_BACKUP_PROVIDER_1'),                         os.getenv('WEB3_BACKUP_PROVIDER_2')                     ],                     'chain_id': int(os.getenv('CHAIN_ID'`
- **main.py** → `os.getenv('TELEGRAM_BOT_TOKEN')`  
  Snippet: `id': os.getenv('TELEGRAM_CHAT_ID', ''),                         'enabled': bool(os.getenv('TELEGRAM_BOT_TOKEN'))                     },                     'discord': {                       `
- **main.py** → `os.getenv('DISCORD_WEBHOOK_URL')`  
  Snippet: `: os.getenv('DISCORD_WEBHOOK_URL', ''),                         'enabled': bool(os.getenv('DISCORD_WEBHOOK_URL'))                     },                     # ✅ CORRECT - Dict mapping AlertPri`
- **main.py** → `os.getenv('TELEGRAM_BOT_TOKEN')`  
  Snippet: ` {                         AlertPriority.LOW: [NotificationChannel.TELEGRAM] if os.getenv('TELEGRAM_BOT_TOKEN') else [],                         AlertPriority.MEDIUM: [NotificationChannel.TEL`
- **main.py** → `os.getenv('TELEGRAM_BOT_TOKEN')`  
  Snippet: `                        AlertPriority.MEDIUM: [NotificationChannel.TELEGRAM] if os.getenv('TELEGRAM_BOT_TOKEN') else [],                         AlertPriority.HIGH: [NotificationChannel.TELEG`
- **main.py** → `os.getenv('TELEGRAM_BOT_TOKEN')`  
  Snippet: `,                         AlertPriority.HIGH: [NotificationChannel.TELEGRAM] if os.getenv('TELEGRAM_BOT_TOKEN') else [],                         AlertPriority.CRITICAL: [NotificationChannel.T`
- **main.py** → `os.getenv('TELEGRAM_BOT_TOKEN')`  
  Snippet: `                      AlertPriority.CRITICAL: [NotificationChannel.TELEGRAM] if os.getenv('TELEGRAM_BOT_TOKEN') else []                     },                     # ✅ Also add priority_cooldo`
- **main.py** → `os.getenv('PRIVATE_KEY')`  
  Snippet: `LWAYS set/override security config from environment             encrypted_key = os.getenv('PRIVATE_KEY')             encryption_key = os.getenv('ENCRYPTION_KEY')              # Decrypt`
- **main.py** → `os.getenv('ENCRYPTION_KEY')`  
  Snippet: `          encrypted_key = os.getenv('PRIVATE_KEY')             encryption_key = os.getenv('ENCRYPTION_KEY')              # Decrypt private key if encrypted             decrypted_key = enc`
- **main.py** → `os.getenv('SOLANA_PRIVATE_KEY')`  
  Snippet: `i.mainnet-beta.solana.com')                 self.config['solana_private_key'] = os.getenv('SOLANA_PRIVATE_KEY')                 self.config['jupiter_max_slippage_bps'] = int(os.getenv('JUPITE`
- **main.py** → `os.getenv('SOLANA_PRIVATE_KEY')`  
  Snippet: `RL', 'https://api.mainnet-beta.solana.com'),                     'private_key': os.getenv('SOLANA_PRIVATE_KEY'),                     'max_slippage_bps': int(os.getenv('JUPITER_MAX_SLIPPAGE_BP`
- **config/config_manager.py** → `os.getenv('MIN_OPPORTUNITY_SCORE')`  
  Snippet: `FOR TRADING CONFIG         if config_type == ConfigType.TRADING:             if os.getenv('MIN_OPPORTUNITY_SCORE'):                 env_data['min_opportunity_score'] = float(os.getenv('MIN_OPPOR`
- **config/config_manager.py** → `os.getenv('MIN_OPPORTUNITY_SCORE')`  
  Snippet: `_OPPORTUNITY_SCORE'):                 env_data['min_opportunity_score'] = float(os.getenv('MIN_OPPORTUNITY_SCORE'))             if os.getenv('MAX_POSITION_SIZE_PERCENT'):                 env_dat`
- **config/config_manager.py** → `os.getenv('MAX_POSITION_SIZE_PERCENT')`  
  Snippet: `_opportunity_score'] = float(os.getenv('MIN_OPPORTUNITY_SCORE'))             if os.getenv('MAX_POSITION_SIZE_PERCENT'):                 env_data['max_position_size'] = float(os.getenv('MAX_POSITION_`
- **config/config_manager.py** → `os.getenv('MAX_POSITION_SIZE_PERCENT')`  
  Snippet: `_POSITION_SIZE_PERCENT'):                 env_data['max_position_size'] = float(os.getenv('MAX_POSITION_SIZE_PERCENT')) / 100             if os.getenv('MAX_SLIPPAGE'):                 env_data['max_`
- **config/config_manager.py** → `os.getenv('MAX_SLIPPAGE')`  
  Snippet: `ion_size'] = float(os.getenv('MAX_POSITION_SIZE_PERCENT')) / 100             if os.getenv('MAX_SLIPPAGE'):                 env_data['max_slippage'] = float(os.getenv('MAX_SLIPPAGE'))   `
- **config/config_manager.py** → `os.getenv('MAX_SLIPPAGE')`  
  Snippet: ` if os.getenv('MAX_SLIPPAGE'):                 env_data['max_slippage'] = float(os.getenv('MAX_SLIPPAGE'))                  # ✅ ADD THIS: Parse ENABLED_CHAINS from .env         if confi`
- **config/config_manager.py** → `os.getenv('ENABLED_CHAINS')`  
  Snippet: `e.API:  # Store chains config here temporarily             enabled_chains_str = os.getenv('ENABLED_CHAINS')             if enabled_chains_str:                 env_data['enabled_chains'] =`
- **scripts/check_balance.py** → `os.getenv("ETH_RPC_URL")`  
  Snippet: `allet balance"""          # Connect to Ethereum     w3 = Web3(Web3.HTTPProvider(os.getenv("ETH_RPC_URL")))          if not w3.is_connected():         print("❌ Failed to connect to Ethe`
- **scripts/check_balance.py** → `os.getenv("WALLET_ADDRESS")`  
  Snippet: `)     # In production, this should decrypt the private key     wallet_address = os.getenv("WALLET_ADDRESS")          if not wallet_address:         print("❌ Wallet address not configured"`
- **scripts/close_all_positions.py** → `os.getenv("ETH_RPC_URL")`  
  Snippet: `positions\n")                  # Setup Web3         w3 = Web3(Web3.HTTPProvider(os.getenv("ETH_RPC_URL")))         if not w3.is_connected():             print("Error: Cannot connect to`
- **scripts/post_update_check.py** → `os.getenv("ETH_RPC_URL")`  
  Snippet: `\n3. Checking Web3 connection...")     try:         w3 = Web3(Web3.HTTPProvider(os.getenv("ETH_RPC_URL")))         if w3.is_connected():             block = w3.eth.block_number        `
- **scripts/reset_nonce.py** → `os.getenv("ETH_RPC_URL")`  
  Snippet: `nonce():     """Reset transaction nonce"""          w3 = Web3(Web3.HTTPProvider(os.getenv("ETH_RPC_URL")))          if not w3.is_connected():         print("❌ Failed to connect to Ethe`
- **scripts/reset_nonce.py** → `os.getenv("WALLET_ADDRESS")`  
  Snippet: `ailed to connect to Ethereum network")         return          wallet_address = os.getenv("WALLET_ADDRESS")     if not wallet_address:         print("❌ Wallet address not configured")    `
- **scripts/security_audit.py** → `os.getenv('DB_SSL_MODE')`  
  Snippet: `in']:         warnings.append("Using privileged database user")          if not os.getenv('DB_SSL_MODE'):         warnings.append("Database SSL not configured")          # Check API ra`
- **scripts/security_audit.py** → `os.getenv('RATE_LIMIT_ENABLED')`  
  Snippet: `   # Check API rate limiting     print("\nChecking API security...")     if not os.getenv('RATE_LIMIT_ENABLED'):         warnings.append("Rate limiting not enabled")          # Check wallet s`
- **scripts/test_alerts.py** → `os.getenv("TELEGRAM_BOT_TOKEN")`  
  Snippet: ` channels"""          config = {         "telegram": {             "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),             "chat_id": os.getenv("TELEGRAM_CHAT_ID")         },         "disco`
- **scripts/test_alerts.py** → `os.getenv("TELEGRAM_CHAT_ID")`  
  Snippet: `           "bot_token": os.getenv("TELEGRAM_BOT_TOKEN"),             "chat_id": os.getenv("TELEGRAM_CHAT_ID")         },         "discord": {             "webhook_url": os.getenv("DISCORD_W`
- **scripts/test_alerts.py** → `os.getenv("DISCORD_WEBHOOK_URL")`  
  Snippet: `("TELEGRAM_CHAT_ID")         },         "discord": {             "webhook_url": os.getenv("DISCORD_WEBHOOK_URL")         }     }          alerts = AlertsSystem(config)          print("🔔 Testin`
- **scripts/test_apis.py** → `os.getenv("ETH_RPC_URL")`  
  Snippet: `est RPC endpoints         print("🔍 Testing RPC endpoints...")         rpc_url = os.getenv("ETH_RPC_URL")         if rpc_url:             try:                 payload = {               `
- **scripts/update_blacklists.py** → `os.getenv("GOPLUS_API_KEY")`  
  Snippet: `  print(f"\n🔍 Fetching from external sources...")                  goplus_key = os.getenv("GOPLUS_API_KEY")         if goplus_key:             try:                 headers = {"Authorizati`
- **scripts/withdraw_funds.py** → `os.getenv("ETH_RPC_URL")`  
  Snippet: `tion: {to_address}")          # Connect to Web3     w3 = Web3(Web3.HTTPProvider(os.getenv("ETH_RPC_URL")))     if not w3.is_connected():         print("Error: Cannot connect to Ethereu`
- **scripts/withdraw_funds.py** → `os.getenv("WALLET_PRIVATE_KEY")`  
  Snippet: `Ethereum network")         return False          # Get wallet     private_key = os.getenv("WALLET_PRIVATE_KEY")     if not private_key:         print("Error: WALLET_PRIVATE_KEY not set")     `
- **trading/chains/solana/jupiter_executor.py** → `os.getenv('ENCRYPTION_KEY')`  
  Snippet: ` import Fernet                 encryption_key = config.get('encryption_key') or os.getenv('ENCRYPTION_KEY')                 if encryption_key:                     cipher = Fernet(encrypti`
### dotenv_load
Count: 2
- **main.py** → `load_dotenv`  
  Snippet: ` import signal import sys import os from pathlib import Path from dotenv import load_dotenv import logging from datetime import datetime import argparse from typing import`
- **main.py** → `load_dotenv`  
  Snippet: `Manager from monitoring.alerts import AlertsSystem # Load environment variables load_dotenv()  # Global logger logger = None  # ===========================================`
### logger_usage
Count: 1288
- **main.py** → `logger.`  
  Snippet: `, signum, frame):         """Handle shutdown signals gracefully"""         self.logger.warning(f"Received signal {signum}, initiating graceful shutdown...")         se`
- **main.py** → `logger.`  
  Snippet: `ze(self):         """Initialize all components"""         try:             self.logger.info("=" * 80)             self.logger.info("🚀 DexScreener Trading Bot Starting.`
- **main.py** → `logger.`  
  Snippet: `ponents"""         try:             self.logger.info("=" * 80)             self.logger.info("🚀 DexScreener Trading Bot Starting...")             self.logger.info(f"Mod`
- **main.py** → `logger.`  
  Snippet: `     self.logger.info("🚀 DexScreener Trading Bot Starting...")             self.logger.info(f"Mode: {self.mode}")             self.logger.info(f"Time: {datetime.now().`
- **main.py** → `logger.`  
  Snippet: `arting...")             self.logger.info(f"Mode: {self.mode}")             self.logger.info(f"Time: {datetime.now().isoformat()}")             self.logger.info("=" * 8`
- **main.py** → `logger.`  
  Snippet: `       self.logger.info(f"Time: {datetime.now().isoformat()}")             self.logger.info("=" * 80)                          # Load configuration asynchronously     `
- **main.py** → `logger.`  
  Snippet: `)                          # Load configuration asynchronously             self.logger.info("Loading configuration...")             await self.config_manager.initializ`
- **main.py** → `logger.`  
  Snippet: `                 # Initialize security using EncryptionManager             self.logger.info("Initializing security manager...")             security_manager = Encrypti`
- **main.py** → `logger.`  
  Snippet: `tionManager({})                          # Initialize database             self.logger.info("Connecting to database...")             await self.db_manager.connect()   `
- **main.py** → `logger.`  
  Snippet: `     trading_config = self.config_manager.get_trading_config()             self.logger.info(f"📊 Min Opportunity Score: {trading_config.min_opportunity_score}")        `
- **main.py** → `logger.`  
  Snippet: `ypted_key = f.decrypt(encrypted_key.encode()).decode()                     self.logger.info("✅ Successfully decrypted private key")                 except Exception as`
- **main.py** → `logger.`  
  Snippet: `d private key")                 except Exception as e:                     self.logger.error(f"Failed to decrypt private key: {e}")                     raise ValueErro`
- **main.py** → `logger.`  
  Snippet: `ey,                 'private_key': decrypted_key             }             self.logger.info(f"DEBUG: Set security config with private_key: {decrypted_key[:10] if decry`
- **main.py** → `logger.`  
  Snippet: `     # Right after line 295 (after creating flat_config), add:             self.logger.info(f"DEBUG: flat_config keys: {flat_config.keys()}")             self.logger.i`
- **main.py** → `logger.`  
  Snippet: `.logger.info(f"DEBUG: flat_config keys: {flat_config.keys()}")             self.logger.info(f"DEBUG: flat_config has private_key: {'private_key' in flat_config}")     `
- **main.py** → `logger.`  
  Snippet: `g:                 # Only show first 10 chars for security                 self.logger.info(f"DEBUG: private_key value: {str(flat_config.get('private_key'))[:10]}...")`
- **main.py** → `logger.`  
  Snippet: `nterval'] = int(os.getenv('DISCOVERY_INTERVAL_SECONDS', 300))              self.logger.info("✅ Chain configuration loaded:")             self.logger.info(f"  Enabled c`
- **main.py** → `logger.`  
  Snippet: `             self.logger.info("✅ Chain configuration loaded:")             self.logger.info(f"  Enabled chains: {self.config['chains']['enabled']}")             self.l`
- **main.py** → `logger.`  
  Snippet: `.info(f"  Enabled chains: {self.config['chains']['enabled']}")             self.logger.info(f"  Chain settings: {self.config['chains']}")              # ✅ ADD THIS ENT`
- **main.py** → `logger.`  
  Snippet: `'false').lower() == 'true'              if solana_enabled:                 self.logger.info("🟣 Configuring Solana integration...")                                  # A`
- **main.py** → `logger.`  
  Snippet: `_MAX_AGE_HOURS', '24'))                 }                                  self.logger.info("  ✅ Solana enabled")                 self.logger.info(f"     RPC: {self.co`
- **main.py** → `logger.`  
  Snippet: `                    self.logger.info("  ✅ Solana enabled")                 self.logger.info(f"     RPC: {self.config['solana_rpc_url'][:50]}...")                 self.`
- **main.py** → `logger.`  
  Snippet: `info(f"     RPC: {self.config['solana_rpc_url'][:50]}...")                 self.logger.info(f"     Min Liquidity: ${self.config['solana_min_liquidity']:,.0f}")        `
- **main.py** → `logger.`  
  Snippet: `n Liquidity: ${self.config['solana_min_liquidity']:,.0f}")                 self.logger.info(f"     Max Slippage: {self.config['jupiter_max_slippage_bps']/100:.2f}%")  `
- **main.py** → `logger.`  
  Snippet: `['jupiter_max_slippage_bps']/100:.2f}%")             else:                 self.logger.info("ℹ️  Solana integration disabled")                 self.config['solana_enab`
- **main.py** → `logger.`  
  Snippet: `tialize trading engine             # Initialize trading engine             self.logger.info("Initializing trading engine...")              # ✅ CRITICAL: Merge Solana c`
- **main.py** → `logger.`  
  Snippet: `g engine             if self.config.get('solana_enabled'):                 self.logger.info("🔧 Merging Solana config into flat_config...")                 flat_config[`
- **main.py** → `logger.`  
  Snippet: `     flat_config['solana'] = self.config.get('solana', {})                 self.logger.info(f"   ✅ Solana config merged: {list(flat_config.get('solana', {}).keys())}")`
- **main.py** → `logger.`  
  Snippet: `ager,                 db_manager=self.db_manager             )             self.logger.info("Enhanced dashboard initialized")                          # Initialize hea`
- **main.py** → `logger.`  
  Snippet: `ialized")                          # Initialize health checker             self.logger.info("Starting health monitoring...")             self.health_checker = HealthCh`
- **main.py** → `logger.`  
  Snippet: `s             await self._perform_system_checks()                          self.logger.info("✅ Initialization complete!")             self.logger.info("=" * 80)       `
- **main.py** → `logger.`  
  Snippet: `                self.logger.info("✅ Initialization complete!")             self.logger.info("=" * 80)                      except Exception as e:             self.logg`
- **main.py** → `logger.`  
  Snippet: `ger.info("=" * 80)                      except Exception as e:             self.logger.error(f"Failed to initialize: {e}", exc_info=True)             raise            `
- **main.py** → `logger.`  
  Snippet: ` optional_vars.items():             if not os.getenv(var):                 self.logger.warning(f"Optional API key missing: {var} - {description}")       # And add this`
- **main.py** → `logger.`  
  Snippet: `                                if positions_info:                         self.logger.info(f"📊 Active Positions: {', '.join(positions_info)}")                        `
- **main.py** → `logger.`  
  Snippet: `econds                              except Exception as e:                 self.logger.error(f"Error in position monitor: {e}")                 await asyncio.sleep(60)`
- **main.py** → `logger.`  
  Snippet: `    for check_name, check_func in checks:             try:                 self.logger.info(f"Checking {check_name}...")                 await check_func()            `
- **main.py** → `logger.`  
  Snippet: `cking {check_name}...")                 await check_func()                 self.logger.info(f"✔ {check_name} OK")             except Exception as e:                 se`
- **main.py** → `logger.`  
  Snippet: `o(f"✔ {check_name} OK")             except Exception as e:                 self.logger.error(f"✗ {check_name} failed: {e}")                 if self.mode == "production`
- **main.py** → `logger.`  
  Snippet: `nc def run(self):         """Main application loop"""         try: #            logger.info("Starting DexScreener Trading Bot...")              # To this:             `
- **main.py** → `logger.`  
  Snippet: `"Starting DexScreener Trading Bot...")              # To this:             self.logger.info("Starting DexScreener Trading Bot...")             # Initialize components `
- **main.py** → `logger.`  
  Snippet: `                                    # Start the trading engine             self.logger.info("🎯 Starting trading engine...")              # Log multi-chain status      `
- **main.py** → `logger.`  
  Snippet: `                 chains = self.config['chains']['enabled']                 self.logger.info(f"🌐 Multi-chain mode: {len(chains)} chains enabled")                 self.l`
- **main.py** → `logger.`  
  Snippet: `.info(f"🌐 Multi-chain mode: {len(chains)} chains enabled")                 self.logger.info(f"   Chains: {', '.join(chains)}")             else:                 self.l`
- **main.py** → `logger.`  
  Snippet: `.info(f"   Chains: {', '.join(chains)}")             else:                 self.logger.warning("⚠️ Single-chain mode (multi-chain not configured)")                    `
- **main.py** → `logger.`  
  Snippet: `for task in done:                 if task.exception():                     self.logger.error(f"Task failed: {task.exception()}")                                  # Can`
- **main.py** → `logger.`  
  Snippet: ` task.cancel()                          except Exception as e:             self.logger.error(f"Critical error in main loop: {e}", exc_info=True)                      f`
- **main.py** → `logger.`  
  Snippet: `                 stats = await self.engine.get_stats()                     self.logger.info(f"📊 Status: {stats}")                                      await asyncio.sl`
- **main.py** → `logger.`  
  Snippet: `minute                              except Exception as e:                 self.logger.error(f"Error in status reporter: {e}")                 await asyncio.sleep(60) `
- **main.py** → `logger.`  
  Snippet: `or for shutdown signal"""         await self.shutdown_event.wait()         self.logger.info("Shutdown signal received")              async def shutdown(self):         `
- ... and 1238 more.
### hardcoded_wallet
Count: 104
- **analysis/dev_analyzer.py** → `0x0000000000000000000000000000000000000000`  
  Snippet: `eturn deployer address         # For now, returning placeholder         return "0x0000000000000000000000000000000000000000"          async def _check_liquidity_status(         self,         project_ad`
- **analysis/rug_detector.py** → `0x0000000000000000000000000000000000000000`  
  Snippet: `  # Check if renounced (owner is 0x0 or dead address)             if owner in ['0x0000000000000000000000000000000000000000',                         '0x000000000000000000000000000000000000dEaD']:     `
- **analysis/rug_detector.py** → `0x000000000000000000000000000000000000dEaD`  
  Snippet: `wner in ['0x0000000000000000000000000000000000000000',                         '0x000000000000000000000000000000000000dEaD']:                 ownership['is_renounced'] = True                          `
- **analysis/rug_detector.py** → `0x360894a13ba1a3210667c828492db98dca3e2076`  
  Snippet: `slots             # EIP-1967 proxy implementation slot             impl_slot = '0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc'             implementation = web3.eth.get_storage_at`
- **analysis/rug_detector.py** → `0xb53127684a568b3173ae13b9f8a6016e243e63b6`  
  Snippet: `mmon proxy patterns             # EIP-1967 admin slot             admin_slot = '0xb53127684a568b3173ae13b9f8a6016e243e63b6e8ee1178d6a717850b5d6103'             admin = web3.eth.get_storage_at(address,`
- **analysis/smart_contract_analyzer.py** → `0x0000000000000000000000000000000000000000`  
  Snippet: `t.functions, func_name)().call()                         if owner and owner != "0x0000000000000000000000000000000000000000":                             return owner                     except (Networ`
- **analysis/smart_contract_analyzer.py** → `0x0000000000000000000000000000000000000000`  
  Snippet: `    score += 5                  # Centralization         if owner and owner != "0x0000000000000000000000000000000000000000":             owner_func_count = len(permissions.get("owner_functions", [])) `
- **analysis/smart_contract_analyzer.py** → `0x0000000000000000000000000000000000000000`  
  Snippet: `, deep_analysis=False)                  return (             analysis.owner == "0x0000000000000000000000000000000000000000" or             analysis.owner == "0x000000000000000000000000000000000000dEaD`
- **analysis/smart_contract_analyzer.py** → `0x000000000000000000000000000000000000dEaD`  
  Snippet: ` "0x0000000000000000000000000000000000000000" or             analysis.owner == "0x000000000000000000000000000000000000dEaD"         )          async def estimate_gas_usage(         self,         addre`
- **config/settings.py** → `0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D`  
  Snippet: `uniswap_v2': {             'name': 'Uniswap V2',             'router_address': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',             'factory_address': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f'`
- **config/settings.py** → `0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f`  
  Snippet: `: '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',             'factory_address': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',             'chains': ['ethereum', 'goerli'],             'fee': 0.003  # 0`
- **config/settings.py** → `0xE592427A0AEce92De3Edee1F18E0157C05861564`  
  Snippet: `uniswap_v3': {             'name': 'Uniswap V3',             'router_address': '0xE592427A0AEce92De3Edee1F18E0157C05861564',             'factory_address': '0x1F98431c8aD98523631AE4a59f267346ea31F984'`
- **config/settings.py** → `0x1F98431c8aD98523631AE4a59f267346ea31F984`  
  Snippet: `: '0xE592427A0AEce92De3Edee1F18E0157C05861564',             'factory_address': '0x1F98431c8aD98523631AE4a59f267346ea31F984',             'chains': ['ethereum', 'polygon', 'arbitrum', 'goerli'],       `
- **config/settings.py** → `0x10ED43C718714eb63d5aA57B78B54704E256024E`  
  Snippet: `ncakeswap': {             'name': 'PancakeSwap',             'router_address': '0x10ED43C718714eb63d5aA57B78B54704E256024E',             'factory_address': '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73'`
- **config/settings.py** → `0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73`  
  Snippet: `: '0x10ED43C718714eb63d5aA57B78B54704E256024E',             'factory_address': '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73',             'chains': ['bsc', 'bsc_testnet'],             'fee': 0.0025  # `
- **config/settings.py** → `0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F`  
  Snippet: ` 'sushiswap': {             'name': 'SushiSwap',             'router_address': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',             'factory_address': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac'`
- **config/settings.py** → `0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac`  
  Snippet: `: '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F',             'factory_address': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac',             'chains': ['ethereum', 'polygon', 'arbitrum', 'fantom'],       `
- **config/settings.py** → `0xA0b86a33E6441c8fb7e61b9e5E4F8b23C3b7b54c`  
  Snippet: `n Lists     STABLECOIN_ADDRESSES = {         'ethereum': {             'USDC': '0xA0b86a33E6441c8fb7e61b9e5E4F8b23C3b7b54c',             'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',          `
- **config/settings.py** → `0xdAC17F958D2ee523a2206206994597C13D831ec7`  
  Snippet: `     'USDC': '0xA0b86a33E6441c8fb7e61b9e5E4F8b23C3b7b54c',             'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',             'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',           `
- **config/settings.py** → `0x6B175474E89094C44Da98b954EedeAC495271d0F`  
  Snippet: `      'USDT': '0xdAC17F958D2ee523a2206206994597C13D831ec7',             'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',             'BUSD': '0x4Fabb145d64652a948d72533023f6E7A623C7C53'         },`
- **config/settings.py** → `0x4Fabb145d64652a948d72533023f6E7A623C7C53`  
  Snippet: `      'DAI': '0x6B175474E89094C44Da98b954EedeAC495271d0F',             'BUSD': '0x4Fabb145d64652a948d72533023f6E7A623C7C53'         },         'bsc': {             'USDC': '0x8AC76a51cc950d9822D68b83f`
- **config/settings.py** → `0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d`  
  Snippet: `652a948d72533023f6E7A623C7C53'         },         'bsc': {             'USDC': '0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d',             'USDT': '0x55d398326f99059fF775485246999027B3197955',          `
- **config/settings.py** → `0x55d398326f99059fF775485246999027B3197955`  
  Snippet: `     'USDC': '0x8AC76a51cc950d9822D68b83fE1Ad97B32Cd580d',             'USDT': '0x55d398326f99059fF775485246999027B3197955',             'BUSD': '0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56',          `
- **config/settings.py** → `0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56`  
  Snippet: `     'USDT': '0x55d398326f99059fF775485246999027B3197955',             'BUSD': '0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56',             'DAI': '0x1AF3F329e8BE154074D8769D1FFa4eE058B1DBc3'         }, `
- **config/settings.py** → `0x1AF3F329e8BE154074D8769D1FFa4eE058B1DBc3`  
  Snippet: `      'BUSD': '0xe9e7CEA3DedcA5984780Bafc599bD69ADd087D56',             'DAI': '0x1AF3F329e8BE154074D8769D1FFa4eE058B1DBc3'         },         'polygon': {             'USDC': '0x2791Bca1f2de4661ED88A`
- **config/settings.py** → `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174`  
  Snippet: `074D8769D1FFa4eE058B1DBc3'         },         'polygon': {             'USDC': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',             'USDT': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',          `
- **config/settings.py** → `0xc2132D05D31c914a87C6611C10748AEb04B58e8F`  
  Snippet: `     'USDC': '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174',             'USDT': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',             'DAI': '0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063'         }  `
- **config/settings.py** → `0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063`  
  Snippet: `      'USDT': '0xc2132D05D31c914a87C6611C10748AEb04B58e8F',             'DAI': '0x8f3Cf7ad23Cd3CaDbD9735AFf958023239c6A063'         }     }          # Blacklisted Tokens (Known scams/rugs)     BLACKLI`
- **config/settings.py** → `0x0000000000000000000000000000000000000000`  
  Snippet: `# Blacklisted Tokens (Known scams/rugs)     BLACKLISTED_TOKENS = set([         '0x0000000000000000000000000000000000000000',  # Null address         # Add known scam tokens here     ])          # Gas `
- **core/engine.py** → `0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D`  
  Snippet: `ax_retries': 3,             'retry_delay': 1,             'uniswap_v2_router': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',  # Mainnet router             '1inch_api_key': config.get('api', {}).get('1`
- **data/collectors/chain_data.py** → `0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D`  
  Snippet: `      self.routers = {             'ethereum': {                 'uniswap_v2': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',                 'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',`
- **data/collectors/chain_data.py** → `0xE592427A0AEce92De3Edee1F18E0157C05861564`  
  Snippet: `': '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D',                 'uniswap_v3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',                 'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F'  `
- **data/collectors/chain_data.py** → `0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F`  
  Snippet: `3': '0xE592427A0AEce92De3Edee1F18E0157C05861564',                 'sushiswap': '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F'             },             'bsc': {                 'pancakeswap_v2': '0x10E`
- **data/collectors/chain_data.py** → `0x10ED43C718714eb63d5aA57B78B54704E256024E`  
  Snippet: `C378B9F'             },             'bsc': {                 'pancakeswap_v2': '0x10ED43C718714eb63d5aA57B78B54704E256024E',                 'pancakeswap_v3': '0x13f4EA83D0bd40E75C8222255bc855a974568D`
- **data/collectors/chain_data.py** → `0x13f4EA83D0bd40E75C8222255bc855a974568Dd4`  
  Snippet: `0x10ED43C718714eb63d5aA57B78B54704E256024E',                 'pancakeswap_v3': '0x13f4EA83D0bd40E75C8222255bc855a974568Dd4'             },             'polygon': {                 'quickswap': '0xa5E0`
- **data/collectors/chain_data.py** → `0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff`  
  Snippet: `74568Dd4'             },             'polygon': {                 'quickswap': '0xa5E0829CaCEd8fFDD4De3c43696c57F7D7A678ff'             }         }                  # Known factory addresses         s`
- **data/collectors/chain_data.py** → `0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f`  
  Snippet: `    self.factories = {             'ethereum': {                 'uniswap_v2': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',                 'sushiswap': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac'  `
- **data/collectors/chain_data.py** → `0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac`  
  Snippet: `2': '0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f',                 'sushiswap': '0xC0AEe478e3658e2610c5F7A4A2E1777cE9e4f2Ac'             },             'bsc': {                 'pancakeswap_v2': '0xcA1`
- **data/collectors/chain_data.py** → `0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73`  
  Snippet: `9e4f2Ac'             },             'bsc': {                 'pancakeswap_v2': '0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73'             }         }                  # Suspicious function signatures   `
- **data/collectors/chain_data.py** → `0x0000000000000000000000000000000000000000`  
  Snippet: `  # Check if owner is zero address (renounced)                 return owner == '0x0000000000000000000000000000000000000000'             except:                 # No owner function or error            `
- **data/collectors/mempool_monitor.py** → `0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D`  
  Snippet: `          # Router addresses for decoding         self.routers = {             '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D': 'Uniswap V2',             '0xE592427A0AEce92De3Edee1F18E0157C05861564': 'Un`
- **data/collectors/mempool_monitor.py** → `0xE592427A0AEce92De3Edee1F18E0157C05861564`  
  Snippet: `       '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D': 'Uniswap V2',             '0xE592427A0AEce92De3Edee1F18E0157C05861564': 'Uniswap V3',             '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F': 'Su`
- **data/collectors/mempool_monitor.py** → `0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F`  
  Snippet: `       '0xE592427A0AEce92De3Edee1F18E0157C05861564': 'Uniswap V3',             '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F': 'SushiSwap',             '0x10ED43C718714eb63d5aA57B78B54704E256024E': 'Pan`
- **data/collectors/mempool_monitor.py** → `0x10ED43C718714eb63d5aA57B78B54704E256024E`  
  Snippet: `        '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F': 'SushiSwap',             '0x10ED43C718714eb63d5aA57B78B54704E256024E': 'PancakeSwap V2'         }                  # Method signatures         sel`
- **data/collectors/mempool_monitor.py** → `0x00000000000006B7e42F8b5C3Fd2e5BDC0d7a0AC`  
  Snippet: `"         # These would be loaded from a database         return {             '0x00000000000006B7e42F8b5C3Fd2e5BDC0d7a0AC',  # Example MEV bot             '0xa69babEF1cA67A37Ffaf7a485DfFF3382056e78C'`
- **data/collectors/mempool_monitor.py** → `0xa69babEF1cA67A37Ffaf7a485DfFF3382056e78C`  
  Snippet: `  '0x00000000000006B7e42F8b5C3Fd2e5BDC0d7a0AC',  # Example MEV bot             '0xa69babEF1cA67A37Ffaf7a485DfFF3382056e78C',  # Example sandwich bot             # Add more known bots         }        `
- **data/collectors/whale_tracker.py** → `0x0000000000000000000000000000000000000000`  
  Snippet: `        # Example addresses (would be loaded from database/config)             "0x0000000000000000000000000000000000000000",  # Placeholder         ])                  # Load whale labels         self`
- **data/collectors/whale_tracker.py** → `0x0000000000000000000000000000000000000000`  
  Snippet: ` labels         self.whale_labels = {             # Example labels             "0x0000000000000000000000000000000000000000": "Major Fund #1",         }              async def close(self):         """C`
- **data/collectors/whale_tracker.py** → `0x0000000000000000000000000000000000000000`  
  Snippet: `lue = event['args']['value']                                  if to_address != "0x0000000000000000000000000000000000000000":                     balances[to_address] = balances.get(to_address, 0) + va`
- **data/collectors/whale_tracker.py** → `0x0000000000000000000000000000000000000000`  
  Snippet: `ress] = balances.get(to_address, 0) + value                 if from_address != "0x0000000000000000000000000000000000000000":                     balances[from_address] = balances.get(from_address, 0) `
- ... and 54 more.
### chain_ids
Count: 9
- **config/settings.py** → `chain_id=1`  
  Snippet: `] = {         'ethereum': ChainConfig(             name='Ethereum',             chain_id=1,             rpc_url=os.getenv('ETHEREUM_RPC_URL', 'https://eth-mainnet.alchemy`
- **config/settings.py** → `chain_id=56`  
  Snippet: `        'bsc': ChainConfig(             name='Binance Smart Chain',             chain_id=56,             rpc_url=os.getenv('BSC_RPC_URL', 'https://bsc-dataseed1.binance.or`
- **config/settings.py** → `chain_id=137`  
  Snippet: `     ),         'polygon': ChainConfig(             name='Polygon',             chain_id=137,             rpc_url=os.getenv('POLYGON_RPC_URL', 'https://polygon-rpc.com'),  `
- **config/settings.py** → `chain_id=42161`  
  Snippet: `,         'arbitrum': ChainConfig(             name='Arbitrum One',             chain_id=42161,             rpc_url=os.getenv('ARBITRUM_RPC_URL', 'https://arb1.arbitrum.io/rp`
- **config/settings.py** → `chain_id=43114`  
  Snippet: ` ),         'avalanche': ChainConfig(             name='Avalanche',             chain_id=43114,             rpc_url=os.getenv('AVALANCHE_RPC_URL', 'https://api.avax.network/e`
- **config/settings.py** → `chain_id=250`  
  Snippet: `       ),         'fantom': ChainConfig(             name='Fantom',             chain_id=250,             rpc_url=os.getenv('FANTOM_RPC_URL', 'https://rpc.fantom.network'),`
- **config/settings.py** → `chain_id=8453`  
  Snippet: `M'         ),         'base': ChainConfig(             name='Base',             chain_id=8453,             rpc_url=os.getenv('BASE_RPC_URL', 'https://mainnet.base.org'),    `
- **config/settings.py** → `chain_id=5`  
  Snippet: ` 'goerli': ChainConfig(                 name='Ethereum Goerli',                 chain_id=5,                 rpc_url=os.getenv('GOERLI_RPC_URL', 'https://eth-goerli.alchem`
- **config/settings.py** → `chain_id=97`  
  Snippet: `'bsc_testnet': ChainConfig(                 name='BSC Testnet',                 chain_id=97,                 rpc_url=os.getenv('BSC_TESTNET_RPC_URL', 'https://data-seed-pr`
### sleep_calls
Count: 1
- **scripts/run_tests.py** → `time.sleep(2)`  
  Snippet: ` "redis-test",             "-p", "6379:6379", "redis:alpine"         ])         time.sleep(2)  # Wait for Redis to start          def run_tests(         self,         test_t`
### api_key_like
Count: 5
- **tests/integration/test_data_integration.py** → `token = "0x1234567890123456789012345678901234567890"`  
  Snippet: `ker.initialize()         checker.cache_manager = cache_manager                  token = "0x1234567890123456789012345678901234567890"                  with patch('aiohttp.ClientSession.get') as mock_ge`
- **tests/integration/test_data_integration.py** → `token = "0x1234567890123456789012345678901234567890"`  
  Snippet: `ger = db_manager         tracker.cache_manager = cache_manager                  token = "0x1234567890123456789012345678901234567890"                  # Mock whale movements         whale_data = {     `
- **tests/integration/test_data_integration.py** → `token = "0x1234567890123456789012345678901234567890"`  
  Snippet: `nager = cache_manager                  # Simulate multiple data sources         token = "0x1234567890123456789012345678901234567890"                  # Mock data from various sources         await db_`
- **tests/integration/test_ml_integration.py** → `token = "0x1234567890123456789012345678901234567890"`  
  Snippet: `] = PumpPredictor(mock_config)                  # Mock some predictions         token = "0x1234567890123456789012345678901234567890"                  with pytest.mock.patch.object(ensemble.rug_classif`
- **tests/security/test_security.py** → `token = "invalid.token.here"`  
  Snippet: `in payload["permissions"]                  # Test invalid token         invalid_token = "invalid.token.here"         payload = await api_security.validate_token(invalid_token)         asse`
### todo
Count: 9
- **trading/executors/base_executor.py** → `TODO: Implement Uniswap V3 integration`  
  Snippet: `TradeOrder) -> Dict:         """Get Uniswap V3 quote - placeholder"""         # TODO: Implement Uniswap V3 integration         return {}          async def _get_paraswap_quote(self, order: TradeOrde`
- **trading/executors/base_executor.py** → `TODO: Implement Paraswap integration`  
  Snippet: `: TradeOrder) -> Dict:         """Get Paraswap quote - placeholder"""         # TODO: Implement Paraswap integration         return {}          async def _get_toxisol_quote(self, order: TradeOrder`
- **trading/executors/base_executor.py** → `TODO: Implement ToxiSol integration`  
  Snippet: `r: TradeOrder) -> Dict:         """Get ToxiSol quote - placeholder"""         # TODO: Implement ToxiSol integration         return {}                  async def _execute_with_retry(self, order: T`
- **trading/executors/base_executor.py** → `TODO: Implement Uniswap V3 execution`  
  Snippet: `utionResult:         """Execute trade via Uniswap V3 - placeholder"""         # TODO: Implement Uniswap V3 execution         return ExecutionResult(             success=False,             tx_hash=`
- **trading/executors/base_executor.py** → `TODO: Implement Paraswap execution`  
  Snippet: `ecutionResult:         """Execute trade via Paraswap - placeholder"""         # TODO: Implement Paraswap execution         return ExecutionResult(             success=False,             tx_hash=`
- **trading/executors/base_executor.py** → `TODO: Implement ToxiSol execution`  
  Snippet: `xecutionResult:         """Execute trade via ToxiSol - placeholder"""         # TODO: Implement ToxiSol execution         return ExecutionResult(             success=False,             tx_hash=`
- **trading/executors/base_executor.py** → `TODO: Implement direct execution`  
  Snippet: `> ExecutionResult:         """Execute trade directly - placeholder"""         # TODO: Implement direct execution         return ExecutionResult(             success=False,             tx_hash=`
- **trading/executors/base_executor.py** → `TODO: Implement Flashbots integration`  
  Snippet: `ction through private mempool"""         if self.flashbots_relay:             # TODO: Implement Flashbots integration             pass                  # Fallback to public mempool         return s`
- **trading/executors/base_executor.py** → `TODO: Load from JSON files in production`  
  Snippet: `(self, contract_type: str) -> List:         """Load contract ABI"""         # ✅ TODO: Load from JSON files in production         # For now, return minimal ABIs                  if contract_type == 'un`
### hardcoded_path
Count: 1
- **docker-compose.yml** → `/var/`  
  Snippet: `t_user       POSTGRES_PASSWORD: bot_password     volumes:       - postgres-data:/var/lib/postgresql/data       - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.`

## 4) Files Likely Involved in Real Trading

- analysis/dev_analyzer.py
- analysis/liquidity_monitor.py
- analysis/market_analyzer.py
- analysis/pump_predictor.py
- analysis/smart_contract_analyzer.py
- analysis/token_scorer.py
- config/config_manager.py
- config/settings.py
- config/validation.py
- core/decision_maker.py
- core/engine.py
- core/event_bus.py
- core/pattern_analyzer.py
- core/portfolio_manager.py
- core/risk_manager.py
- data/collectors/chain_data.py
- data/collectors/dexscreener.py
- data/collectors/honeypot_checker.py
- data/collectors/mempool_monitor.py
- data/collectors/social_data.py
- data/collectors/token_sniffer.py
- data/collectors/volume_analyzer.py
- data/collectors/whale_tracker.py
- data/processors/aggregator.py
- data/processors/feature_extractor.py
- data/processors/normalizer.py
- data/processors/validator.py
- data/storage/cache.py
- data/storage/database.py
- data/storage/models.py
- main.py
- ml/models/ensemble_model.py
- ml/models/pump_predictor.py
- ml/models/rug_classifier.py
- ml/optimization/hyperparameter.py
- ml/optimization/reinforcement.py
- monitoring/alerts.py
- monitoring/dashboard.py
- monitoring/enhanced_dashboard.py
- monitoring/logger.py
- scripts/analyze_strategy.py
- scripts/check_balance.py
- scripts/close_all_positions.py
- scripts/daily_report.py
- scripts/export_trades.py
- scripts/generate_report.py
- scripts/generate_solana_wallet.py
- scripts/migrate_database.py
- scripts/optimize_db.py
- scripts/overnight_summary.py
- scripts/reset_nonce.py
- scripts/retrain_models.py
- scripts/security_audit.py
- scripts/setup_database.py
- scripts/solana_wallet_balance.py
- scripts/strategy_analysis.py
- scripts/test_solana_setup.py
- scripts/update_blacklists.py
- scripts/verify_claudedex_plus.py
- scripts/weekly_report.py
- scripts/withdraw_funds.py
- security/api_security.py
- security/audit_logger.py
- security/encryption.py
- security/wallet_security.py
- setup.py
- setup_env_keys.py
- tests/conftest.py
- tests/fixtures/mock_data.py
- tests/fixtures/test_helpers.py
- tests/integration/test_data_integration.py
- tests/integration/test_dexscreener.py
- tests/integration/test_ml_integration.py
- tests/integration/test_trading_integration.py
- tests/performance/test_performance.py
- tests/security/test_security.py
- tests/smoke/test_smoke.py
- tests/unit/test_engine.py
- tests/unit/test_risk_manager.py
- trade_analyzer.py
- trading/chains/solana/__init__.py
- trading/chains/solana/jupiter_executor.py
- trading/chains/solana/solana_client.py
- trading/chains/solana/spl_token_handler.py
- trading/executors/base_executor.py
- trading/executors/direct_dex.py
- trading/executors/mev_protection.py
- trading/executors/toxisol_api.py
- trading/orders/order_manager.py
- trading/orders/position_tracker.py
- trading/strategies/ai_strategy.py
- trading/strategies/base_strategy.py
- trading/strategies/momentum.py
- trading/strategies/scalping copy.py
- trading/strategies/scalping.py
- utils/constants.py
- utils/errors.py
- utils/helpers.py

## 5) TODO / FIXME Mentions

- **trading/executors/base_executor.py** → `TradeOrder) -> Dict:         """Get Uniswap V3 quote - placeholder"""         # TODO: Implement Uniswap V3 integration         return {}          async def _get_paraswap_quote(self, order: TradeOrde`
- **trading/executors/base_executor.py** → `: TradeOrder) -> Dict:         """Get Paraswap quote - placeholder"""         # TODO: Implement Paraswap integration         return {}          async def _get_toxisol_quote(self, order: TradeOrder`
- **trading/executors/base_executor.py** → `r: TradeOrder) -> Dict:         """Get ToxiSol quote - placeholder"""         # TODO: Implement ToxiSol integration         return {}                  async def _execute_with_retry(self, order: T`
- **trading/executors/base_executor.py** → `utionResult:         """Execute trade via Uniswap V3 - placeholder"""         # TODO: Implement Uniswap V3 execution         return ExecutionResult(             success=False,             tx_hash=`
- **trading/executors/base_executor.py** → `ecutionResult:         """Execute trade via Paraswap - placeholder"""         # TODO: Implement Paraswap execution         return ExecutionResult(             success=False,             tx_hash=`
- **trading/executors/base_executor.py** → `xecutionResult:         """Execute trade via ToxiSol - placeholder"""         # TODO: Implement ToxiSol execution         return ExecutionResult(             success=False,             tx_hash=`
- **trading/executors/base_executor.py** → `> ExecutionResult:         """Execute trade directly - placeholder"""         # TODO: Implement direct execution         return ExecutionResult(             success=False,             tx_hash=`
- **trading/executors/base_executor.py** → `ction through private mempool"""         if self.flashbots_relay:             # TODO: Implement Flashbots integration             pass                  # Fallback to public mempool         return s`
- **trading/executors/base_executor.py** → `(self, contract_type: str) -> List:         """Load contract ABI"""         # ✅ TODO: Load from JSON files in production         # For now, return minimal ABIs                  if contract_type == 'un`

## 6) Immediate Risks & Quick Wins

- Centralize all configuration into `.env` and `config/config_manager.py`; replace any direct literals for RPC URLs, chain IDs, slippage, gas, secrets.

- Replace `time.sleep()` polling in critical paths with async tasks or event-driven queues; where keeping sleep, gate behind config.

- Ensure wallet usage always goes through `security/wallet_security.py` with signing isolation and MEV protection.

- Add per-chain risk guards (rug/honeypot checks) to pre-trade validation; enforce via `core/risk_manager.py`.

- Promote consistent logging (structured JSON) and disable bare `print()` in production paths.
