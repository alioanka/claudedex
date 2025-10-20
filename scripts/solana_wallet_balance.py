from trading.chains.solana.solana_client import SolanaClient

async def check_balance():
    client = SolanaClient(['https://mainnet.helius-rpc.com/?api-key=a78df2b2-9cfb-421b-86cf-801dddff77c4'])
    await client.initialize()
    
    balance = await client.get_balance('125QXNdNdjKZZjag1FTuYKHDGsszthbVnpgVrt6wbpTc')
    print(f"SOL Balance: {balance} SOL")
    
    await client.cleanup()