"""Static map of blue-chip Solana mint addresses -> Pyth Hermes price-feed-ids.

Wave-3 SNIPER enhancement (PM mission item: Pyth-feed wiring).

Pump.fun memecoins do NOT have a Pyth feed-id, so resolution must
gracefully fall through to Jupiter / Birdeye. This map is intentionally
small: only mints that have a published Pyth Hermes feed AND that a
sniper might realistically hold (post-graduation tokens, or where the
operator widened scope to include established mints for routing).

Feed-ids are the 32-byte Pyth-network price-feed identifiers, hex with
the 0x prefix, as listed at https://pyth.network/developers/price-feed-ids.
Hermes accepts them in this exact form.

Map keys are the canonical mainnet mint base58 addresses. EVM tokens
intentionally NOT included here — current sniper price path for EVM
uses DexScreener, and the Pyth feed-ids for ETH/WBTC are listed below
as commented-out examples for any future EVM-side wiring.
"""

# Solana blue-chip mints -> Pyth Hermes feed-id (Crypto.<SYMBOL>/USD).
# Sources: https://pyth.network/developers/price-feed-ids#solana-mainnet-beta
# Validated 2026-05-19. Operator can extend at runtime via DB without
# touching this file (see _resolve_pyth_feed_id for the precedence order).
SOLANA_MINT_TO_PYTH_FEED_ID = {
    # SOL (native, wrapped)
    'So11111111111111111111111111111111111111112':
        '0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d',
    # USDC (Circle)
    'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v':
        '0xeaa020c61cc479712813461ce153894a96a6c00b21ed0cfc2798d1f9a9e9c94a',
    # USDT (Tether)
    'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB':
        '0x2b89b9dc8fdf9f34709a5b106b472f0f39bb6ca9ce04b0fd7f2e971688e2e53b',
    # ETH (Wormhole-wrapped, 8 decimals on Solana)
    '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs':
        '0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace',
    # WBTC (Wormhole / Sollet legacy — operator may swap to portal mint)
    '3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh':
        '0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43',
    # JUP (Jupiter governance)
    'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN':
        '0x0a0408d619e9380abad35060f9192039ed5042fa6f82301d0e48bb52be830996',
    # WIF (dogwifhat)
    'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm':
        '0x4ca4beeca86f0d164160323817a4e42b10010a724c2217c6ee41b54cd4cc61fc',
    # BONK
    'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263':
        '0x72b021217ca3fe68922a19aaf990109cb9d84e9ad004b4d2025ad6f529314419',
    # PYTH (native governance)
    'HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3':
        '0x0bbf28e9a841a1cc788f6a361b17ca072d0ea3098a1e5df1c3922d06719579ff',
    # RAY (Raydium)
    '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R':
        '0x91568baa8beb53db23eb3fb7f22c6e8bd303d103919e19733f2bb642d3e7987a',
    # ORCA
    'orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE':
        '0x37505261e557e251290b8c8899453064e8d760ed5c65a779726f2490980da74c',
    # JTO (Jito governance)
    'jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL':
        '0xb43660a5f790c69354b0729a5ef9d50d68f1df92107540210b9cccba1f947cc2',
    # JLP (Jupiter Liquidity Provider — used in lending markets)
    '27G8MtK7VtTcCHkpASjSDdkWWYfoqT6ggEuKidVJidD4':
        '0xc811abc82b4bad1f9bd711a2773ccaa935b03ecef974236942cec5e0eb845a3a',
}

# Pyth Hermes base URL. Free public endpoint, no API key required.
# Latest-prices endpoint: /v2/updates/price/latest?ids[]=<feed_id>...
PYTH_HERMES_BASE_URL = 'https://hermes.pyth.network'


def get_pyth_feed_id(token_address: str) -> str | None:
    """Return the Pyth Hermes feed-id for a Solana mint, or None.

    Pump.fun and other freshly-minted memecoins will return None and
    the caller falls through to Jupiter / Birdeye. This is the only
    public entrypoint — callers should not import the map directly so
    we can swap the resolution strategy (e.g. DB-backed) later without
    a per-caller change.
    """
    if not token_address:
        return None
    return SOLANA_MINT_TO_PYTH_FEED_ID.get(token_address)
