# Centralized Configuration System

## Overview

The trading bot now uses a **database-backed centralized configuration system** that allows you to manage all settings from the dashboard UI. Changes made in the dashboard will immediately affect the running bot.

## Architecture

### Configuration Priority (highest to lowest):
1. **Database** (config_settings table) - Highest priority
2. Environment variables
3. YAML config files
4. Pydantic model defaults

### Sensitive Configuration
Sensitive values (API keys, private keys, passwords) are:
- Stored in `config_sensitive` table
- Encrypted using AES-256-GCM
- Managed through the Sensitive Config tab in the dashboard

## Initial Setup

### 1. Import Existing .env Secrets to Database

After your first deployment, run the import script to migrate your .env secrets to the database:

```bash
# Inside the Docker container
docker exec -it trading-bot python scripts/import_env_secrets.py

# OR from the host machine
cd /path/to/claudedex
python scripts/import_env_secrets.py
```

This will:
- Read all sensitive keys from your .env file
- Encrypt them using your ENCRYPTION_KEY
- Store them in the config_sensitive table
- Make them visible/manageable in the Sensitive Config tab

### 2. Verify Configuration Loading

Check the bot logs on startup. You should see:
```
✅ Config manager now reading from database
✅ Configs reloaded from database
```

## Using the Dashboard

### Managing Regular Settings

1. Navigate to **Settings** page in the dashboard
2. Select the appropriate tab (General, Portfolio, Risk Management, etc.)
3. Modify settings as needed
4. Click **Save Changes**
5. Changes take effect immediately (unless marked "Requires Restart")

### Managing Sensitive Configs

1. Go to **Settings** → **Sensitive** tab
2. View existing sensitive configurations (values are hidden)
3. Click **Add Sensitive Config** to add new secrets
4. Enter:
   - Key (e.g., `NEW_API_KEY`)
   - Value (will be encrypted)
   - Description
   - Rotation interval (days)
5. Click **Save**

**Note**: You can also delete or update sensitive configs from this tab.

## How It Works

### At Startup:
1. ConfigManager loads defaults from Pydantic models
2. Loads and merges YAML config files
3. Loads and merges environment variables
4. **Database connection established**
5. **Database configs loaded** (override all previous values)
6. Bot starts with final merged configuration

### When You Change Settings in Dashboard:
1. Settings API receives your changes
2. Values are validated and saved to database
3. Audit log created in config_history table
4. ConfigManager reloads the changed config type
5. **Bot immediately uses new values** (unless restart required)

### Accessing Configs in Code:

```python
# Via get_X_config() methods (recommended)
risk_config = self.config_manager.get_risk_management_config()
max_risk = risk_config.max_risk_per_trade

# Via dictionary access (for backward compatibility)
api_key = self.config_manager.get('DEXSCREENER_API_KEY')

# Via sensitive config (encrypted)
private_key = await self.config_manager.get_sensitive_config('PRIVATE_KEY')
```

## Configuration Types

The system supports these configuration categories:
- **general**: Bot mode, ML enabled, dry run
- **portfolio**: Balance, position sizing, limits
- **risk_management**: Stop loss, take profit, circuit breakers
- **trading**: Slippage, strategies, opportunity scoring
- **chain**: Enabled chains, liquidity minimums
- **position_management**: Stop loss, trailing stops, exits
- **api**: API server configuration
- **monitoring**: Logging, metrics, alerting
- **ml_models**: ML model parameters
- **feature_flags**: Experimental features toggles
- **sensitive**: API keys, private keys, secrets (encrypted)

## Database Tables

### config_settings
Stores non-sensitive configuration values:
- `config_type`: Category (general, portfolio, etc.)
- `key`: Setting name
- `value`: Setting value (stored as text)
- `value_type`: Data type (string, int, float, bool, json)
- `description`: Human-readable description
- `is_editable`: Can be changed via dashboard
- `requires_restart`: Needs bot restart to take effect
- `updated_by`: User who last modified
- `updated_at`: Last modification timestamp

### config_sensitive
Stores encrypted sensitive configuration:
- `key`: Secret key name
- `encrypted_value`: AES-256-GCM encrypted value
- `encryption_method`: Encryption algorithm used
- `description`: What this secret is for
- `last_rotated`: When the secret was last changed
- `rotation_interval_days`: Recommended rotation frequency
- `is_active`: Whether this secret is currently active
- `updated_by`: User who last modified

### config_history
Audit log of all configuration changes:
- `config_type`: Which config category changed
- `key`: Which setting changed
- `old_value`: Previous value
- `new_value`: New value
- `change_source`: How it was changed (api, database, file)
- `changed_by`: User ID who made the change
- `changed_by_username`: Username who made the change
- `reason`: Optional reason for the change
- `timestamp`: When the change occurred

## Security Considerations

1. **Encryption Key**: Keep your `ENCRYPTION_KEY` safe. Without it, you cannot decrypt sensitive configs.
2. **Database Access**: Limit access to the database. Raw encrypted values are stored there.
3. **Audit Logs**: All configuration changes are logged with user attribution.
4. **Admin Only**: Sensitive config management requires admin role.
5. **Values Never Logged**: Actual secret values are never written to logs.

## Troubleshooting

### Settings UI shows empty tabs
- Check that migrations have run successfully
- Verify config_settings table has data
- Check browser console for API errors

### Sensitive tab shows "No sensitive configs found"
- Run the import script: `python scripts/import_env_secrets.py`
- Check that ENCRYPTION_KEY is set in .env
- Verify config_sensitive table exists

### Changes in dashboard don't affect bot
- Check logs for "Config manager now reading from database"
- Verify db_pool was passed to ConfigManager
- Some settings require bot restart (check "Requires Restart" badge)

### Import script fails
- Verify ENCRYPTION_KEY is set in .env
- Check database connection (DATABASE_URL or individual DB_* vars)
- Ensure config_sensitive table exists (run migrations)

## Migration from Old System

If you're upgrading from the YAML-only configuration system:

1. **Backup your existing config files**:
   ```bash
   cp -r config/ config.backup/
   ```

2. **Run database migrations**:
   ```bash
   # Migrations run automatically on bot startup
   # OR manually: python scripts/migrate_database.py
   ```

3. **Import .env secrets to database**:
   ```bash
   python scripts/import_env_secrets.py
   ```

4. **Restart the bot**:
   ```bash
   docker-compose restart trading-bot
   ```

5. **Verify in dashboard**:
   - Check all settings tabs have values
   - Check Sensitive tab shows your secrets (names only, values hidden)

6. **(Optional) Keep YAML files as backup**:
   - The bot will still read YAML files
   - Database values override YAML values
   - You can keep YAML files as documentation

## Best Practices

1. **Use Dashboard for Changes**: Instead of editing YAML files or .env, use the dashboard UI for all configuration changes.

2. **Document Changes**: When updating sensitive configs, add a meaningful description.

3. **Review Audit Logs**: Periodically check config_history table to see who changed what.

4. **Rotate Secrets**: Use the rotation_interval_days to track when API keys should be rotated.

5. **Backup Database**: Include config_settings and config_sensitive tables in your database backups.

6. **Test in Dry Run**: After major config changes, test in dry_run mode before live trading.

## Example Workflows

### Adding a New API Key
1. Go to Settings → Sensitive tab
2. Click "Add Sensitive Config"
3. Key: `COINGECKO_API_KEY`
4. Value: `your-api-key-here`
5. Description: `CoinGecko Pro API key for price data`
6. Rotation: `90` days
7. Click Save
8. Access in code: `await config_mgr.get_sensitive_config('COINGECKO_API_KEY')`

### Changing Risk Parameters
1. Go to Settings → Risk Management tab
2. Modify `max_risk_per_trade` from 0.10 to 0.08
3. Modify `stop_loss_pct` from 0.12 to 0.10
4. Click "Save Changes"
5. Bot immediately uses new risk parameters (no restart needed)

### Enabling a New Chain
1. Go to Settings → Chains tab
2. Toggle `polygon_enabled` to ON
3. Set `polygon_min_liquidity` to desired value
4. Click "Save Changes"
5. Restart bot (required for chain initialization)
