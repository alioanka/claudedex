# test_solana_setup.py
import asyncio
from trading.chains.solana import JupiterExecutor

async def test():
    config = {
        'jupiter_url': 'https://quote-api.jup.ag/v6',
        'solana_rpc_url': 'https://api.mainnet-beta.solana.com',
        'solana_private_key': 'YOUR_KEY_HERE',
        'max_slippage': 0.05
    }
    
    executor = JupiterExecutor(config)
    await executor.initialize()
    
    # Test quote (SOL → USDC)
    quote = await executor.get_quote(
        input_mint='So11111111111111111111111111111111111111112',  # SOL
        output_mint='EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
        amount_lamports=100000000,  # 0.1 SOL
    )
    
    if quote:
        print(f"✅ Quote received!")
        print(f"Input: {quote.in_amount / 1e9} SOL")
        print(f"Output: {quote.out_amount / 1e6} USDC")
        print(f"Price Impact: {quote.price_impact_pct:.2f}%")
    else:
        print("❌ Failed to get quote")
    
    await executor.cleanup()

asyncio.run(test())