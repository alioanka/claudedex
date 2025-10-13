To Switch to REAL Trading Later:
When you're ready for real trades:

Update .env:

env   DRY_RUN=false

Ensure you have real funds in your wallet
Lower position sizes for safety:

env   MIN_TRADE_SIZE_USD=50   # Start small!
   MAX_POSITION_SIZE_PERCENT=2  # Only 2% of portfolio per trade

Restart and monitor closely!