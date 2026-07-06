#!/usr/bin/env python3
"""
RPC / API key verify-and-import tool (Wave-F5 follow-up).

The operator has ~119 endpoint/key combos (Ankr / dRPC / Alchemy per EVM
chain, Helius / Chainstack / Alchemy for Solana, Infura wss, plus
Etherscan / Birdeye / GoPlus API keys) in a spreadsheet and doesn't know
which still work. This tool:

  1. Collects candidates from: --file (xlsx/csv spreadsheet), .env vars,
     existing rpc_api_pool DB rows, and secrets_manager keys.
  2. Probes each one with EXACTLY ONE cheap read-only call chain
     (eth_chainId/eth_blockNumber/1-block eth_getLogs for EVM https;
     eth_chainId over ws for EVM wss; getSlot+getGenesisHash for Solana
     https; slotSubscribe for Solana wss; provider-specific cheap calls
     for bare API keys).
  3. Prints an aligned console report grouped by provider/chain and
     writes a redacted JSON report to logs/rpc_verify_report.json.
  4. With --apply: imports verified-OK endpoints into rpc_api_pool
     (existing pool_engine taxonomy), stores verified bare API keys
     ENCRYPTED via secrets_manager in the numbered Wave-F5 slots, and
     disables (never deletes) DB rows that failed hard.

Default is DRY-RUN (report only). Read-only probes only; no
state-changing RPC methods are ever sent. Keys are redacted to their
first 6 characters in ALL output (console + JSON).

Run inside the bot container:
    docker compose exec trading-bot python scripts/verify_and_import_rpcs.py \
        --file /app/data/rpc_keys.csv            # dry-run report
    docker compose exec trading-bot python scripts/verify_and_import_rpcs.py \
        --file /app/data/rpc_keys.csv --apply    # import the working set

Self-test (no network, no DB):
    python scripts/verify_and_import_rpcs.py --mock

HONESTY NOTE: rpc_api_pool URLs (which may embed provider keys, e.g.
Helius/Alchemy-style key-in-URL) are stored PLAINTEXT in Postgres —
exactly like the rows the dashboard /settings/rpc-api page already
writes. Only standalone API keys (HELIUS/ETHERSCAN/BIRDEYE/GOPLUS) go
through the Fernet-encrypted secure_credentials store. Full at-rest
encryption of pool URLs is a possible future hardening, tracked in
docs/RPC_API_KEYS_GUIDE.md.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow `python scripts/verify_and_import_rpcs.py` from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:  # fail-soft: container may rely on real process env
    from dotenv import load_dotenv
    for _env in ('/app/.env', '.env'):
        if Path(_env).exists():
            load_dotenv(_env, override=False)
            break
except Exception:
    pass

logging.basicConfig(level=logging.WARNING, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('verify_rpcs')

# =========================================================================
# Constants — MUST stay aligned with config/pool_engine.py + migration 011
# =========================================================================

# Solana mainnet-beta genesis hash. Anything else (e.g. the operator's
# Ankr solana_DEVNET rows, genesis EtWTRABZ...) is WRONG_NETWORK.
SOLANA_MAINNET_GENESIS = '5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d'

# chain -> expected eth_chainId. Monad is SOFT (accept-any-with-warning):
# expected mainnet id 143, but if the live id differs we warn instead of
# rejecting so a chain-id migration doesn't silently drop every endpoint.
EVM_CHAIN_IDS: Dict[str, int] = {
    'ethereum': 1,
    'bsc': 56,
    'polygon': 137,
    'arbitrum': 42161,
    'base': 8453,
    'avalanche': 43114,
    'monad': 143,
    'fantom': 250,
    'cronos': 25,
    'pulsechain': 369,
}
SOFT_CHAIN_ID_CHAINS = {'monad'}

# chain -> rpc_api_pool provider_type (taxonomy from migration 011,
# identical to what the dashboard /settings/rpc-api page groups by).
CHAIN_TO_RPC_PROVIDER: Dict[str, str] = {
    'ethereum': 'ETHEREUM_RPC', 'bsc': 'BSC_RPC', 'polygon': 'POLYGON_RPC',
    'arbitrum': 'ARBITRUM_RPC', 'base': 'BASE_RPC', 'avalanche': 'AVALANCHE_RPC',
    'monad': 'MONAD_RPC', 'fantom': 'FANTOM_RPC', 'cronos': 'CRONOS_RPC',
    'pulsechain': 'PULSECHAIN_RPC', 'solana': 'SOLANA_RPC',
}
# Only these WS provider types exist in rpc_api_provider_types (mig 011).
# wss URLs for other chains are reported but SKIPPED on --apply rather
# than inventing a new provider_type nothing consumes.
CHAIN_TO_WS_PROVIDER: Dict[str, str] = {
    'ethereum': 'ETHEREUM_WS', 'bsc': 'BSC_WS',
    'arbitrum': 'ARBITRUM_WS', 'solana': 'SOLANA_WS',
}

# Bare-API-key kinds -> (provider_type, chain, secrets base env var)
KEY_KINDS: Dict[str, Tuple[str, Optional[str], str]] = {
    'helius_key':    ('HELIUS_API', 'solana', 'HELIUS_API_KEY'),
    'etherscan_key': ('ETHERSCAN_API', 'ethereum', 'ETHERSCAN_API_KEY'),
    'birdeye_key':   ('BIRDEYE_API', 'solana', 'BIRDEYE_API_KEY'),
    'goplus_key':    ('GOPLUS_API', None, 'GOPLUS_API_KEY'),
}

MAX_NUMBERED_KEYS = 9  # same ceiling as pool_engine

# Classification values
OK = 'OK'
AUTH_FAIL = 'AUTH_FAIL'
RATE_LIMITED = 'RATE_LIMITED'
QUOTA_EXHAUSTED = 'QUOTA_EXHAUSTED'
WRONG_NETWORK = 'WRONG_NETWORK'
TIMEOUT = 'TIMEOUT'
DEAD = 'DEAD'
UNTESTED = 'UNTESTED'

HARD_FAIL = {AUTH_FAIL, DEAD, WRONG_NETWORK}

NON_MAINNET_RE = re.compile(r'devnet|testnet|goerli|sepolia|holesky|mumbai|amoy|fuji|chapel', re.I)

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_CONCURRENCY = 12
DEFAULT_JSON_OUT = 'logs/rpc_verify_report.json'

USDC_ETH = '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48'  # GoPlus/getLogs sample token
WSOL_MINT = 'So11111111111111111111111111111111111111112'


# =========================================================================
# Redaction — never print a full key anywhere (console, JSON, logs)
# =========================================================================

_KEY_QUERY_PARAMS = {'api-key', 'apikey', 'api_key', 'dkey', 'key', 'token', 'auth', 'access-token'}
_KEYISH_SEGMENT = re.compile(r'^[A-Za-z0-9_-]{16,}$')


def redact_secret(value: Optional[str]) -> str:
    if not value:
        return ''
    return value[:6] + '…' if len(value) > 8 else '…'


def redact_url(url: Optional[str]) -> str:
    """Redact key-looking query params and path segments; keep host readable."""
    if not url:
        return ''
    try:
        import urllib.parse as up
        parts = up.urlsplit(url)
        # Query params
        q_out = []
        for k, v in up.parse_qsl(parts.query, keep_blank_values=True):
            if k.lower() in _KEY_QUERY_PARAMS and v:
                v = redact_secret(v)
            q_out.append((k, v))
        query = up.urlencode(q_out, safe='…')
        # Path segments that look like keys (require a digit or mixed case
        # so plain words like 'transactions' survive)
        segs = []
        for seg in parts.path.split('/'):
            if (_KEYISH_SEGMENT.match(seg)
                    and (any(c.isdigit() for c in seg)
                         or (seg.lower() != seg and seg.upper() != seg))):
                seg = redact_secret(seg)
            segs.append(seg)
        return up.urlunsplit((parts.scheme, parts.netloc, '/'.join(segs), query, ''))
    except Exception:
        return url[:24] + '…'


# =========================================================================
# Data model
# =========================================================================

@dataclass
class Candidate:
    """One endpoint/key to verify. Deduped on identity()."""
    kind: str                      # evm_rpc | evm_ws | solana_rpc | solana_ws | <x>_key | other_api
    url: Optional[str] = None      # full probe URL (RPC/WS kinds)
    key: Optional[str] = None      # bare API key (key kinds)
    chain: Optional[str] = None    # ethereum/bsc/.../solana; None = unknown
    provider_type: Optional[str] = None
    name: str = ''
    account: str = ''              # account email when known (spreadsheet col D)
    sources: List[str] = field(default_factory=list)
    db_id: Optional[int] = None    # existing rpc_api_pool row id
    db_enabled: Optional[bool] = None
    secret_slot: Optional[str] = None  # existing secrets slot name holding this key
    # probe outcome
    status: str = UNTESTED
    latency_ms: Optional[int] = None
    getlogs_ok: Optional[bool] = None
    observed: str = ''             # chainId / genesis observed
    note: str = ''

    def identity(self) -> Tuple[str, str]:
        if self.key and self.kind in KEY_KINDS:
            return (self.kind, self.key)
        return (self.kind, (self.url or '').rstrip('/'))

    def looks_non_mainnet(self) -> bool:
        return bool(self.url and NON_MAINNET_RE.search(self.url)) or bool(NON_MAINNET_RE.search(self.name or ''))

    def effective_provider_type(self) -> Optional[str]:
        if self.provider_type:
            return self.provider_type
        if self.kind in KEY_KINDS:
            return KEY_KINDS[self.kind][0]
        if self.kind in ('evm_rpc', 'solana_rpc') and self.chain:
            return CHAIN_TO_RPC_PROVIDER.get(self.chain)
        if self.kind in ('evm_ws', 'solana_ws') and self.chain:
            return CHAIN_TO_WS_PROVIDER.get(self.chain)
        return None


def _merge_candidate(dst: Candidate, src: Candidate) -> None:
    dst.sources.extend(s for s in src.sources if s not in dst.sources)
    dst.account = dst.account or src.account
    dst.chain = dst.chain or src.chain
    dst.name = dst.name or src.name
    dst.provider_type = dst.provider_type or src.provider_type
    if src.db_id is not None:
        dst.db_id, dst.db_enabled = src.db_id, src.db_enabled
    dst.secret_slot = dst.secret_slot or src.secret_slot


class CandidateSet:
    def __init__(self):
        self._by_id: Dict[Tuple[str, str], Candidate] = {}

    def add(self, cand: Candidate) -> None:
        if not cand.url and not cand.key:
            return
        ident = cand.identity()
        if ident in self._by_id:
            _merge_candidate(self._by_id[ident], cand)
        else:
            self._by_id[ident] = cand

    def all(self) -> List[Candidate]:
        return list(self._by_id.values())


# =========================================================================
# Candidate collection — (a) spreadsheet
# =========================================================================

_LABEL_KEY_KINDS = [
    ('helius', 'helius_key'), ('etherscan', 'etherscan_key'),
    ('birdeye', 'birdeye_key'), ('goplus', 'goplus_key'),
]
# order matters: 'etherscan' must be tested before the bare 'eth' chain match
_LABEL_CHAINS = [
    ('solana', 'solana'), ('sol', 'solana'),
    ('ethereum', 'ethereum'), ('eth', 'ethereum'),
    ('bsc', 'bsc'), ('bnb', 'bsc'), ('binance', 'bsc'),
    ('base', 'base'), ('arb', 'arbitrum'),
    ('avax', 'avalanche'), ('avalanche', 'avalanche'),
    ('monad', 'monad'), ('polygon', 'polygon'), ('matic', 'polygon'),
    ('fantom', 'fantom'), ('cronos', 'cronos'), ('pulse', 'pulsechain'),
]
_URL_CHAIN_HINTS = [
    (r'solana|helius', 'solana'),
    (r'eth-mainnet|network=ethereum|/eth\b|/eth/|mainnet\.infura', 'ethereum'),
    (r'bsc|bnb|network=bsc', 'bsc'),
    (r'base-mainnet|network=base|/base\b|/base/', 'base'),
    (r'arb-mainnet|arbitrum|/arbitrum\b', 'arbitrum'),
    (r'avax|avalanche', 'avalanche'),
    (r'monad', 'monad'),
    (r'polygon|matic', 'polygon'),
]


def _label_to_kind(label: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (key_kind or None, chain or None) for a block label."""
    low = (label or '').strip().lower()
    if not low:
        return None, None
    for token, kind in _LABEL_KEY_KINDS:
        if token in low:
            return kind, KEY_KINDS[kind][1]
    for token, chain in _LABEL_CHAINS:
        if token in low:
            return None, chain
    return None, None


def _infer_chain_from_url(url: str) -> Optional[str]:
    low = (url or '').lower()
    for pattern, chain in _URL_CHAIN_HINTS:
        if re.search(pattern, low):
            return chain
    return None


def _cell(row: List[Any], idx: int) -> str:
    if idx >= len(row) or row[idx] is None:
        return ''
    return str(row[idx]).strip()


def _rows_from_file(path: Path) -> List[List[str]]:
    """Rows (list of cell strings) from an .xlsx sheet named 'RPCs' (or the
    first sheet) or from a CSV export. CSV is always supported."""
    if path.suffix.lower() in ('.xlsx', '.xlsm'):
        try:
            import openpyxl
        except ImportError:
            print("openpyxl is not installed here — export the sheet as CSV "
                  "(File > Download > CSV) and pass that file instead.")
            sys.exit(1)
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb['RPCs'] if 'RPCs' in wb.sheetnames else wb[wb.sheetnames[0]]
        return [[('' if c is None else str(c).strip()) for c in row]
                for row in ws.iter_rows(values_only=True)]
    # CSV (sniff delimiter , vs ;)
    text = path.read_text(encoding='utf-8-sig', errors='replace')
    delim = ';' if text.count(';') > text.count(',') else ','
    return [row for row in csv.reader(text.splitlines(), delimiter=delim)]


def collect_from_file(path: Path, out: CandidateSet) -> int:
    """Parse the operator spreadsheet layout:
    col A = chain/provider block label on the FIRST row of each block
            (persists down), col B = key, col C = base URL (final URL is
            C+B), col D = account email, col E = formula C&B (skipped),
    col F = occasional wss URL.
    """
    rows = _rows_from_file(path)
    current_label = ''
    added = 0
    for i, row in enumerate(rows, start=1):
        label_cell = _cell(row, 0)
        if label_cell:
            current_label = label_cell
        key_cell = _cell(row, 1)
        base_cell = _cell(row, 2)
        email_cell = _cell(row, 3)
        wss_cell = _cell(row, 5)
        # skip pasted formulas (CSV exports of col E never reach us; a stray
        # '=' cell anywhere is not data)
        if base_cell.startswith('='):
            base_cell = ''
        if key_cell.startswith('='):
            key_cell = ''
        if not (key_cell or base_cell or wss_cell):
            continue
        key_kind, chain = _label_to_kind(current_label)
        src = f'file:row{i}'
        devnet_label = bool(NON_MAINNET_RE.search(current_label))

        if key_kind:
            # bare API key block (Helius/Etherscan/Birdeye/GoPlus)
            if key_cell:
                out.add(Candidate(kind=key_kind, key=key_cell, chain=chain,
                                  name=current_label, account=email_cell, sources=[src]))
                added += 1
        else:
            url = ''
            if base_cell and key_cell:
                url = base_cell + key_cell
            elif base_cell.startswith(('http', 'ws')):
                url = base_cell
            elif key_cell.startswith(('http', 'ws')):
                url = key_cell
            if url:
                url_chain = chain or _infer_chain_from_url(url)
                name = current_label or url_chain or 'unlabeled'
                if devnet_label:
                    name = f'{name} (non-mainnet label)'
                if url.startswith(('ws://', 'wss://')):
                    kind = 'solana_ws' if url_chain == 'solana' else 'evm_ws'
                else:
                    kind = 'solana_rpc' if url_chain == 'solana' else 'evm_rpc'
                out.add(Candidate(kind=kind, url=url, chain=url_chain,
                                  name=name, account=email_cell, sources=[src]))
                added += 1
        if wss_cell.startswith(('ws://', 'wss://')):
            url_chain = chain or _infer_chain_from_url(wss_cell)
            kind = 'solana_ws' if url_chain == 'solana' else 'evm_ws'
            out.add(Candidate(kind=kind, url=wss_cell, chain=url_chain,
                              name=(current_label or 'unlabeled') + ' wss',
                              account=email_cell, sources=[src]))
            added += 1
    return added


# =========================================================================
# Candidate collection — (b) .env
# =========================================================================

def _clean(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip().strip('"').strip("'")
    if not value or value in ('null', 'None') or value.startswith('your_'):
        return None
    return value


def _numbered(env_var: str) -> List[Tuple[str, str]]:
    out = []
    for i in range(1, MAX_NUMBERED_KEYS + 1):
        name = env_var if i == 1 else f'{env_var}_{i}'
        v = _clean(os.getenv(name))
        if v:
            out.append((name, v))
    return out


# comma-list vars -> chain      (mirrors pool_engine._load_from_env)
_ENV_LIST_VARS = {
    'ETHEREUM_RPC_URLS': 'ethereum', 'BSC_RPC_URLS': 'bsc',
    'POLYGON_RPC_URLS': 'polygon', 'ARBITRUM_RPC_URLS': 'arbitrum',
    'BASE_RPC_URLS': 'base', 'MONAD_RPC_URLS': 'monad',
    'PULSECHAIN_RPC_URLS': 'pulsechain', 'FANTOM_RPC_URLS': 'fantom',
    'CRONOS_RPC_URLS': 'cronos', 'AVALANCHE_RPC_URLS': 'avalanche',
    'SOLANA_RPC_URLS': 'solana', 'SOLANA_BACKUP_RPCS': 'solana',
}
# single vars (numbered _2.._9 variants included) -> chain
_ENV_SINGLE_VARS = {
    'ETHEREUM_RPC_URL': 'ethereum', 'WEB3_PROVIDER_URL': 'ethereum',
    'WEB3_BACKUP_PROVIDER_1': 'ethereum', 'WEB3_BACKUP_PROVIDER_2': 'ethereum',
    'BSC_RPC_URL': 'bsc', 'POLYGON_RPC_URL': 'polygon',
    'ARBITRUM_RPC_URL': 'arbitrum', 'BASE_RPC_URL': 'base',
    'MONAD_RPC_URL': 'monad', 'PULSECHAIN_RPC_URL': 'pulsechain',
    'FANTOM_RPC_URL': 'fantom', 'CRONOS_RPC_URL': 'cronos',
    'AVALANCHE_RPC_URL': 'avalanche', 'SOLANA_RPC_URL': 'solana',
}
_ENV_WS_VARS = {
    'SOLANA_WS_URL': 'solana',
    'SNIPER_SOLANA_WSS_URL': 'solana',
    'SNIPER_EVM_WSS_URL': None,  # chain unknown; chainId is recorded as observed
}
_ENV_KEY_VARS = {
    'HELIUS_API_KEY': 'helius_key', 'ETHERSCAN_API_KEY': 'etherscan_key',
    'BIRDEYE_API_KEY': 'birdeye_key', 'GOPLUS_API_KEY': 'goplus_key',
}


def collect_from_env(out: CandidateSet) -> int:
    added = 0
    for var, chain in _ENV_LIST_VARS.items():
        raw = os.getenv(var, '')
        for url in [u.strip().strip('"').strip("'") for u in raw.split(',') if u.strip()]:
            if not _clean(url):
                continue
            kind = 'solana_rpc' if chain == 'solana' else 'evm_rpc'
            out.add(Candidate(kind=kind, url=url, chain=chain, name=var, sources=[f'env:{var}']))
            added += 1
    for var, chain in _ENV_SINGLE_VARS.items():
        for name, url in _numbered(var):
            kind = 'solana_rpc' if chain == 'solana' else 'evm_rpc'
            if url.startswith(('ws://', 'wss://')):
                kind = 'solana_ws' if chain == 'solana' else 'evm_ws'
            out.add(Candidate(kind=kind, url=url, chain=chain, name=name, sources=[f'env:{name}']))
            added += 1
    for var, chain in _ENV_WS_VARS.items():
        for name, url in _numbered(var):
            if not url.startswith(('ws://', 'wss://')):
                continue
            eff_chain = chain or _infer_chain_from_url(url)
            kind = 'solana_ws' if eff_chain == 'solana' else 'evm_ws'
            out.add(Candidate(kind=kind, url=url, chain=eff_chain, name=name, sources=[f'env:{name}']))
            added += 1
    for var, kind in _ENV_KEY_VARS.items():
        for name, key in _numbered(var):
            out.add(Candidate(kind=kind, key=key, chain=KEY_KINDS[kind][1],
                              name=name, sources=[f'env:{name}']))
            added += 1
    return added


# =========================================================================
# Candidate collection — (c) existing rpc_api_pool rows, (d) secrets
# =========================================================================

_PROVIDER_TO_KIND = {
    'HELIUS_API': 'helius_key', 'ETHERSCAN_API': 'etherscan_key',
    'BIRDEYE_API': 'birdeye_key', 'GOPLUS_API': 'goplus_key',
}


async def collect_from_db(db_pool, out: CandidateSet) -> int:
    """Re-verify ALL existing rpc_api_pool rows (including disabled ones)."""
    added = 0
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, provider_type, name, url, api_key, chain,
                   is_enabled, supports_ws, ws_url
            FROM rpc_api_pool ORDER BY provider_type, priority, id
        """)
    for row in rows:
        pt = row['provider_type']
        src = f"db:id={row['id']}"
        chain = row['chain']
        if pt in _PROVIDER_TO_KIND:
            kind = _PROVIDER_TO_KIND[pt]
            key = row['api_key']
            if not key and row['url'] and 'api-key=' in row['url']:
                try:
                    import urllib.parse as up
                    key = (up.parse_qs(up.urlparse(row['url']).query).get('api-key') or [None])[0]
                except Exception:
                    key = None
            if key:
                out.add(Candidate(kind=kind, key=key, chain=chain, name=row['name'],
                                  provider_type=pt, sources=[src],
                                  db_id=row['id'], db_enabled=row['is_enabled']))
                added += 1
            continue
        if pt.endswith('_WS'):
            kind = 'solana_ws' if chain == 'solana' else 'evm_ws'
            out.add(Candidate(kind=kind, url=row['url'], chain=chain, name=row['name'],
                              provider_type=pt, sources=[src],
                              db_id=row['id'], db_enabled=row['is_enabled']))
            added += 1
            continue
        if pt.endswith('_RPC'):
            kind = 'solana_rpc' if chain == 'solana' else 'evm_rpc'
            url = row['url']
            if row['api_key'] and 'api-key' not in (url or ''):
                sep = '&' if '?' in url else '?'
                url = f'{url}{sep}api-key={row["api_key"]}'
            out.add(Candidate(kind=kind, url=url, chain=chain, name=row['name'],
                              provider_type=pt, sources=[src],
                              db_id=row['id'], db_enabled=row['is_enabled']))
            added += 1
            continue
        # Other API types (JUPITER_API, 1INCH_API, ...) — no safe cheap
        # probe designed; report as UNTESTED, never touch on --apply.
        out.add(Candidate(kind='other_api', url=row['url'], chain=chain, name=row['name'],
                          provider_type=pt, sources=[src],
                          db_id=row['id'], db_enabled=row['is_enabled'],
                          note='no safe probe for this provider type'))
        added += 1
    return added


async def collect_from_secrets(secrets, out: CandidateSet) -> int:
    added = 0
    for var, kind in _ENV_KEY_VARS.items():
        for slot in range(1, MAX_NUMBERED_KEYS + 1):
            name = var if slot == 1 else f'{var}_{slot}'
            try:
                value = _clean(await secrets.get_async(name, log_access=False))
            except Exception:
                value = None
            if value:
                out.add(Candidate(kind=kind, key=value, chain=KEY_KINDS[kind][1],
                                  name=name, sources=[f'secrets:{name}'],
                                  secret_slot=name))
                added += 1
    return added


# =========================================================================
# Probes — one cheap, strictly read-only call chain per candidate
# =========================================================================

def _classify_body_error(message: str) -> Optional[str]:
    """Map an RPC error string to QUOTA/RATE/AUTH classes (None = no match)."""
    low = (message or '').lower()
    if '-32005' in low or any(w in low for w in (
            'quota', 'exceeded', 'credits', 'capacity', 'monthly limit',
            'usage limit', 'payment required', 'plan limit')):
        return QUOTA_EXHAUSTED
    if 'rate limit' in low or 'too many requests' in low or '429' in low:
        return RATE_LIMITED
    if any(w in low for w in ('unauthorized', 'forbidden', 'invalid api key',
                              'api key', 'authentication', 'not allowed', 'denied')):
        return AUTH_FAIL
    return None


def _status_to_class(http_status: int) -> Optional[str]:
    if http_status in (401, 403):
        return AUTH_FAIL
    if http_status == 429:
        return RATE_LIMITED
    if http_status == 402:
        return QUOTA_EXHAUSTED
    return None


class Prober:
    """All network probes. Subclassed by the --mock self-test."""

    def __init__(self, timeout_s: float = DEFAULT_TIMEOUT_S):
        self.timeout_s = timeout_s
        self._session = None

    async def __aenter__(self):
        import aiohttp
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout_s))
        return self

    async def __aexit__(self, *exc):
        if self._session:
            await self._session.close()

    async def _rpc(self, url: str, method: str, params: list) -> Tuple[Optional[Any], Optional[str], int]:
        """POST one JSON-RPC call. Returns (result, error_class_or_None, http_status).
        Raises asyncio.TimeoutError / connection errors upward."""
        payload = {'jsonrpc': '2.0', 'id': 1, 'method': method, 'params': params}
        async with self._session.post(url, json=payload) as resp:
            cls = _status_to_class(resp.status)
            if cls:
                return None, cls, resp.status
            if resp.status >= 400:
                return None, DEAD, resp.status
            try:
                body = await resp.json(content_type=None)
            except Exception:
                return None, DEAD, resp.status
            if isinstance(body, dict) and body.get('error'):
                err = body['error']
                msg = f"{err.get('code', '')} {err.get('message', '')}" if isinstance(err, dict) else str(err)
                return None, (_classify_body_error(msg) or DEAD), resp.status
            return (body.get('result') if isinstance(body, dict) else None), None, resp.status

    # --- EVM https --------------------------------------------------------
    async def probe_evm_http(self, cand: Candidate) -> None:
        t0 = time.monotonic()
        try:
            chain_id_hex, err, _ = await self._rpc(cand.url, 'eth_chainId', [])
            if err:
                cand.status = err
                return
            cand.latency_ms = int((time.monotonic() - t0) * 1000)
            try:
                observed_id = int(str(chain_id_hex), 16)
            except (TypeError, ValueError):
                cand.status = DEAD
                cand.note = 'eth_chainId returned garbage'
                return
            cand.observed = f'chainId={observed_id}'
            expected = EVM_CHAIN_IDS.get(cand.chain) if cand.chain else None
            if expected is not None and observed_id != expected:
                if cand.chain in SOFT_CHAIN_ID_CHAINS:
                    cand.note = (f'chainId {observed_id} != expected {expected} for '
                                 f'{cand.chain} — accepting with warning, verify manually')
                else:
                    cand.status = WRONG_NETWORK
                    cand.note = f'chainId {observed_id}, expected {expected} ({cand.chain})'
                    return
            elif expected is None:
                # unknown chain — record what we saw, try to back-fill chain
                for chain_name, cid in EVM_CHAIN_IDS.items():
                    if cid == observed_id:
                        cand.chain = cand.chain or chain_name
                        break
                if not cand.chain:
                    cand.note = f'unknown chain (chainId={observed_id}) — not importable'

            bn_hex, err, _ = await self._rpc(cand.url, 'eth_blockNumber', [])
            if err:
                cand.status = err
                cand.note = cand.note or 'eth_blockNumber failed'
                return
            # 1-block eth_getLogs capability probe (drives smart_money/copy EVM)
            try:
                bn = int(str(bn_hex), 16)
                blk = hex(max(0, bn - 1))
                _, logs_err, _ = await self._rpc(
                    cand.url, 'eth_getLogs', [{'fromBlock': blk, 'toBlock': blk}])
                cand.getlogs_ok = logs_err is None
                if logs_err:
                    cand.note = (cand.note + '; ' if cand.note else '') + 'eth_getLogs blocked'
            except (TypeError, ValueError):
                cand.getlogs_ok = False
            cand.status = OK
        except asyncio.TimeoutError:
            cand.status = TIMEOUT
        except Exception as e:
            cand.status = DEAD
            cand.note = type(e).__name__

    # --- EVM / Solana wss -------------------------------------------------
    async def _ws_roundtrip(self, url: str, request: dict) -> dict:
        import websockets
        async with websockets.connect(url, open_timeout=self.timeout_s,
                                      close_timeout=3, max_size=2 ** 20) as ws:
            await ws.send(json.dumps(request))
            raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout_s)
            return json.loads(raw)

    def _classify_ws_exception(self, e: Exception) -> Tuple[str, str]:
        import websockets
        if isinstance(e, asyncio.TimeoutError):
            return TIMEOUT, ''
        status = getattr(e, 'status_code', None)
        if isinstance(e, websockets.exceptions.InvalidStatusCode) or status:
            cls = _status_to_class(int(status or 0))
            return (cls or DEAD), f'HTTP {status}'
        return DEAD, type(e).__name__

    async def probe_evm_ws(self, cand: Candidate) -> None:
        t0 = time.monotonic()
        try:
            body = await self._ws_roundtrip(
                cand.url, {'jsonrpc': '2.0', 'id': 1, 'method': 'eth_chainId', 'params': []})
            cand.latency_ms = int((time.monotonic() - t0) * 1000)
            if body.get('error'):
                cand.status = _classify_body_error(str(body['error'])) or DEAD
                return
            observed_id = int(str(body.get('result')), 16)
            cand.observed = f'chainId={observed_id}'
            expected = EVM_CHAIN_IDS.get(cand.chain) if cand.chain else None
            if expected is not None and observed_id != expected and cand.chain not in SOFT_CHAIN_ID_CHAINS:
                cand.status = WRONG_NETWORK
                cand.note = f'chainId {observed_id}, expected {expected} ({cand.chain})'
                return
            if expected is None:
                for chain_name, cid in EVM_CHAIN_IDS.items():
                    if cid == observed_id:
                        cand.chain = cand.chain or chain_name
                        break
            cand.status = OK
        except asyncio.TimeoutError:
            cand.status = TIMEOUT
        except Exception as e:
            cand.status, cand.note = self._classify_ws_exception(e)

    async def probe_solana_ws(self, cand: Candidate) -> None:
        """connect + slotSubscribe (+ unsubscribe). Note: the Solana ws API is
        subscription-only, so the NETWORK cannot be verified over ws — pair
        it with the https genesis check; devnet-looking URLs are rejected."""
        import websockets
        t0 = time.monotonic()
        try:
            async with websockets.connect(cand.url, open_timeout=self.timeout_s,
                                          close_timeout=3, max_size=2 ** 20) as ws:
                await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 1,
                                          'method': 'slotSubscribe'}))
                raw = await asyncio.wait_for(ws.recv(), timeout=self.timeout_s)
                body = json.loads(raw)
                cand.latency_ms = int((time.monotonic() - t0) * 1000)
                if body.get('error'):
                    cand.status = _classify_body_error(str(body['error'])) or DEAD
                    return
                sub_id = body.get('result')
                if sub_id is not None:
                    try:
                        await ws.send(json.dumps({'jsonrpc': '2.0', 'id': 2,
                                                  'method': 'slotUnsubscribe',
                                                  'params': [sub_id]}))
                    except Exception:
                        pass
                if cand.looks_non_mainnet():
                    cand.status = WRONG_NETWORK
                    cand.note = 'devnet/testnet URL'
                    return
                cand.status = OK
                cand.note = cand.note or 'network not verifiable over ws (subscription API)'
        except asyncio.TimeoutError:
            cand.status = TIMEOUT
        except Exception as e:
            cand.status, cand.note = self._classify_ws_exception(e)

    # --- Solana https -----------------------------------------------------
    async def probe_solana_http(self, cand: Candidate, url: Optional[str] = None) -> None:
        url = url or cand.url
        t0 = time.monotonic()
        try:
            _, err, _ = await self._rpc(url, 'getSlot', [])
            if err:
                cand.status = err
                return
            cand.latency_ms = int((time.monotonic() - t0) * 1000)
            genesis, err, _ = await self._rpc(url, 'getGenesisHash', [])
            if err:
                # getSlot worked; genesis blocked is unusual but not fatal —
                # refuse to certify the network though.
                cand.status = WRONG_NETWORK if cand.looks_non_mainnet() else err
                cand.note = 'getGenesisHash failed'
                return
            cand.observed = f'genesis={str(genesis)[:8]}…'
            if genesis != SOLANA_MAINNET_GENESIS:
                cand.status = WRONG_NETWORK
                cand.note = 'genesis != mainnet-beta (devnet/testnet endpoint)'
                return
            cand.status = OK
        except asyncio.TimeoutError:
            cand.status = TIMEOUT
        except Exception as e:
            cand.status = DEAD
            cand.note = type(e).__name__

    # --- Bare API keys ----------------------------------------------------
    async def probe_helius_key(self, cand: Candidate) -> None:
        await self.probe_solana_http(
            cand, url=f'https://mainnet.helius-rpc.com/?api-key={cand.key}')

    async def probe_etherscan_key(self, cand: Candidate) -> None:
        url = ('https://api.etherscan.io/v2/api?chainid=1&module=proxy'
               f'&action=eth_blockNumber&apikey={cand.key}')
        t0 = time.monotonic()
        try:
            async with self._session.get(url) as resp:
                cls = _status_to_class(resp.status)
                if cls:
                    cand.status = cls
                    return
                body = await resp.json(content_type=None)
            cand.latency_ms = int((time.monotonic() - t0) * 1000)
            result = str(body.get('result', ''))
            if body.get('status') == '0' or body.get('message') == 'NOTOK':
                if 'rate limit' in result.lower():
                    cand.status = RATE_LIMITED
                elif 'invalid api key' in result.lower() or 'apikey' in result.lower():
                    cand.status = AUTH_FAIL
                else:
                    cand.status = _classify_body_error(result) or DEAD
                cand.note = result[:80]
                return
            if result.startswith('0x'):
                cand.status = OK
            else:
                cand.status = DEAD
                cand.note = f'unexpected body: {result[:60]}'
        except asyncio.TimeoutError:
            cand.status = TIMEOUT
        except Exception as e:
            cand.status = DEAD
            cand.note = type(e).__name__

    async def probe_birdeye_key(self, cand: Candidate) -> None:
        url = f'https://public-api.birdeye.so/defi/price?address={WSOL_MINT}'
        t0 = time.monotonic()
        try:
            async with self._session.get(
                    url, headers={'X-API-KEY': cand.key, 'x-chain': 'solana'}) as resp:
                cls = _status_to_class(resp.status)
                if cls:
                    cand.status = cls
                    return
                if resp.status >= 400:
                    cand.status = DEAD
                    cand.note = f'HTTP {resp.status}'
                    return
                body = await resp.json(content_type=None)
            cand.latency_ms = int((time.monotonic() - t0) * 1000)
            cand.status = OK if body.get('success') else AUTH_FAIL
            if not body.get('success'):
                cand.note = str(body.get('message', ''))[:80]
        except asyncio.TimeoutError:
            cand.status = TIMEOUT
        except Exception as e:
            cand.status = DEAD
            cand.note = type(e).__name__

    async def probe_goplus_key(self, cand: Candidate) -> None:
        """GoPlus real auth is an app_key+app_secret signed-token flow; a bare
        key cannot be POSITIVELY verified (the token_security route also
        serves unauthenticated traffic). Fail-soft per spec: reachability
        check only, classify UNTESTED unless the API hard-rejects."""
        url = f'https://api.gopluslabs.io/api/v1/token_security/1?contract_addresses={USDC_ETH}'
        try:
            async with self._session.get(
                    url, headers={'Authorization': f'Bearer {cand.key}'}) as resp:
                cls = _status_to_class(resp.status)
                if cls:
                    cand.status = cls
                    return
                cand.status = UNTESTED
                cand.note = ('endpoint reachable; GoPlus bare keys are not '
                             'verifiable without app_secret — not imported')
        except asyncio.TimeoutError:
            cand.status = TIMEOUT
        except Exception as e:
            cand.status = DEAD
            cand.note = type(e).__name__

    async def probe(self, cand: Candidate) -> None:
        try:
            if cand.kind == 'evm_rpc':
                await self.probe_evm_http(cand)
            elif cand.kind == 'evm_ws':
                await self.probe_evm_ws(cand)
            elif cand.kind == 'solana_rpc':
                await self.probe_solana_http(cand)
            elif cand.kind == 'solana_ws':
                await self.probe_solana_ws(cand)
            elif cand.kind == 'helius_key':
                await self.probe_helius_key(cand)
            elif cand.kind == 'etherscan_key':
                await self.probe_etherscan_key(cand)
            elif cand.kind == 'birdeye_key':
                await self.probe_birdeye_key(cand)
            elif cand.kind == 'goplus_key':
                await self.probe_goplus_key(cand)
            else:
                cand.status = UNTESTED
        except Exception as e:  # belt-and-braces: a probe bug must not kill the run
            cand.status = DEAD
            cand.note = f'probe crashed: {type(e).__name__}'
        # Never certify a non-mainnet-looking URL as OK, whatever the probe said
        if cand.status == OK and cand.looks_non_mainnet() and cand.kind != 'evm_rpc':
            # EVM candidates are already chain-id checked; Solana-side devnet
            # is caught by genesis, this guards ws/unknown kinds.
            cand.status = WRONG_NETWORK
            cand.note = 'devnet/testnet URL'


async def run_probes(cands: List[Candidate], prober: Prober,
                     concurrency: int = DEFAULT_CONCURRENCY) -> None:
    sem = asyncio.Semaphore(concurrency)

    async def one(c: Candidate):
        async with sem:
            await prober.probe(c)

    await asyncio.gather(*(one(c) for c in cands))


# =========================================================================
# Report
# =========================================================================

_STATUS_ORDER = {OK: 0, RATE_LIMITED: 1, QUOTA_EXHAUSTED: 2, UNTESTED: 3,
                 TIMEOUT: 4, WRONG_NETWORK: 5, AUTH_FAIL: 6, DEAD: 7}


def _group_key(c: Candidate) -> str:
    pt = c.effective_provider_type()
    if pt:
        return pt
    if c.kind in ('evm_ws', 'solana_ws'):
        return f'{(c.chain or "unknown").upper()}_WS (no provider_type)'
    return f'{(c.chain or "unknown").upper()}_{c.kind}'


def print_report(cands: List[Candidate]) -> None:
    groups: Dict[str, List[Candidate]] = {}
    for c in cands:
        groups.setdefault(_group_key(c), []).append(c)

    header = f'{"STATUS":<16}{"LAT":>6}  {"LOGS":<5}{"NAME":<28}{"ACCOUNT":<26}{"ENDPOINT / KEY":<52}NOTE'
    print()
    for group in sorted(groups):
        rows = sorted(groups[group], key=lambda c: (_STATUS_ORDER.get(c.status, 9), -(c.latency_ms or 10 ** 9)))
        print(f'=== {group} ({len(rows)}) ' + '=' * max(0, 70 - len(group)))
        print(header)
        for c in rows:
            lat = f'{c.latency_ms}' if c.latency_ms is not None else '-'
            logs = {True: 'yes', False: 'NO'}.get(c.getlogs_ok, '-')
            target = redact_url(c.url) if c.url else redact_secret(c.key)
            flags = ''
            if c.db_id is not None and c.db_enabled is False:
                flags = ' [db-disabled]'
            note = '; '.join(x for x in (c.observed, c.note) if x)
            print(f'{c.status + flags:<16}{lat:>6}  {logs:<5}{(c.name or "")[:27]:<28}'
                  f'{(c.account or "")[:25]:<26}{target[:51]:<52}{note}')
        print()

    totals: Dict[str, int] = {}
    for c in cands:
        totals[c.status] = totals.get(c.status, 0) + 1
    summary = ', '.join(f'{k}={v}' for k, v in sorted(totals.items(), key=lambda kv: _STATUS_ORDER.get(kv[0], 9)))
    print(f'TOTAL {len(cands)} candidates: {summary}')


def write_json_report(cands: List[Candidate], path: Path) -> None:
    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'totals': {},
        'note': ('URLs and keys are REDACTED in this report by design. '
                 'rpc_api_pool stores endpoint URLs (incl. embedded keys) '
                 'plaintext in Postgres; only bare API keys are Fernet-'
                 'encrypted via secure_credentials.'),
        'candidates': [],
    }
    for c in cands:
        payload['totals'][c.status] = payload['totals'].get(c.status, 0) + 1
        payload['candidates'].append({
            'kind': c.kind,
            'provider_type': c.effective_provider_type(),
            'chain': c.chain,
            'name': c.name,
            'account': c.account,
            'sources': c.sources,
            'db_id': c.db_id,
            'db_enabled': c.db_enabled,
            'url': redact_url(c.url) if c.url else None,
            'key': redact_secret(c.key) if c.key else None,
            'status': c.status,
            'latency_ms': c.latency_ms,
            'getlogs_ok': c.getlogs_ok,
            'observed': c.observed,
            'note': c.note,
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f'JSON report written: {path}')


# =========================================================================
# DB / secrets bootstrap (same pattern as scripts/fix_solana_bad_exits.py)
# =========================================================================

async def make_db_pool():
    import asyncpg
    db_host = os.getenv('DB_HOST', 'postgres')
    db_port = int(os.getenv('DB_PORT', 5432))
    db_name = os.getenv('DB_NAME', 'tradingbot')
    user_secret = Path('/run/secrets/db_user')
    db_user = user_secret.read_text().strip() if user_secret.exists() else os.getenv('DB_USER')
    pw_secret = Path('/run/secrets/db_password')
    db_password = pw_secret.read_text().strip() if pw_secret.exists() else os.getenv('DB_PASSWORD')
    if not db_user or not db_password:
        raise RuntimeError(
            'DB credentials unavailable. Run inside the trading-bot container '
            '(docker compose exec trading-bot ...) or export DB_USER/DB_PASSWORD.')
    return await asyncpg.create_pool(
        host=db_host, port=db_port, database=db_name,
        user=db_user, password=db_password, min_size=1, max_size=3)


# =========================================================================
# main
# =========================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--file', type=Path, help='operator spreadsheet (.xlsx or .csv)')
    p.add_argument('--apply', action='store_true',
                   help='import verified-OK endpoints/keys (default: dry-run report only)')
    p.add_argument('--mock', action='store_true',
                   help='offline self-test with stubbed probe responses')
    p.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT_S,
                   help=f'per-probe timeout seconds (default {DEFAULT_TIMEOUT_S:.0f})')
    p.add_argument('--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                   help=f'concurrent probes (default {DEFAULT_CONCURRENCY})')
    p.add_argument('--json-out', type=Path, default=Path(DEFAULT_JSON_OUT),
                   help=f'JSON report path (default {DEFAULT_JSON_OUT})')
    return p


async def main_async(args) -> int:
    if args.mock:
        return await run_mock_selftest()

    cset = CandidateSet()
    counts = {}
    if args.file:
        if not args.file.exists():
            print(f'File not found: {args.file}')
            return 1
        counts['file'] = collect_from_file(args.file, cset)
    counts['env'] = collect_from_env(cset)

    db_pool = None
    secrets = None
    try:
        db_pool = await make_db_pool()
    except Exception as e:
        print(f'WARNING: no database connection ({e}) — DB rows and encrypted '
              f'secrets will not be verified.')
        if args.apply:
            print('REFUSING --apply without a database connection (no partial writes).')
            return 2

    if db_pool:
        counts['db'] = await collect_from_db(db_pool, cset)
        try:
            from security.secrets_manager import secrets as _secrets
            _secrets.initialize(db_pool)
            secrets = _secrets
            counts['secrets'] = await collect_from_secrets(secrets, cset)
        except Exception as e:
            print(f'WARNING: secrets manager unavailable ({e})')
            if args.apply:
                print('REFUSING --apply: secrets manager failed to initialize '
                      '(no partial writes).')
                await db_pool.close()
                return 2
        if args.apply and (secrets is None or getattr(secrets, '_fernet', None) is None):
            print('REFUSING --apply: encryption key not loaded — verified API '
                  'keys could not be stored encrypted (no partial writes).')
            await db_pool.close()
            return 2

    cands = cset.all()
    if not cands:
        print('No candidates found (no --file, empty .env, empty DB).')
        if db_pool:
            await db_pool.close()
        return 1

    srcs = ', '.join(f'{k}={v}' for k, v in counts.items())
    print(f'Collected {len(cands)} unique candidates ({srcs}); probing with '
          f'timeout={args.timeout:.0f}s concurrency={args.concurrency} ...')

    async with Prober(timeout_s=args.timeout) as prober:
        await run_probes(cands, prober, concurrency=args.concurrency)

    print_report(cands)
    write_json_report(cands, args.json_out)

    if args.apply:
        await apply_results(cands, db_pool, secrets)
    else:
        ok = sum(1 for c in cands if c.status == OK)
        print(f'\nDRY-RUN ONLY — nothing written. Re-run with --apply to import '
              f'the {ok} verified-OK endpoint(s)/key(s).')

    if db_pool:
        await db_pool.close()
    return 0


# =========================================================================
# --apply: import the working set
# =========================================================================

_KEY_BASE_URLS = {  # mirrors pool_engine api_mappings base URLs
    'HELIUS_API': None,  # URL embeds the key
    'ETHERSCAN_API': 'https://api.etherscan.io',
    'BIRDEYE_API': 'https://public-api.birdeye.so',
    'GOPLUS_API': 'https://api.gopluslabs.io',
}


def _url_is_keyed(url: str) -> bool:
    """Heuristic: URL carries a provider key (path segment or query param)."""
    try:
        import urllib.parse as up
        parts = up.urlsplit(url)
        for k, v in up.parse_qsl(parts.query, keep_blank_values=True):
            if k.lower() in _KEY_QUERY_PARAMS and v:
                return True
        for seg in parts.path.split('/'):
            if (_KEYISH_SEGMENT.match(seg)
                    and (any(c.isdigit() for c in seg)
                         or (seg.lower() != seg and seg.upper() != seg))):
                return True
    except Exception:
        pass
    return False


def plan_key_slots(existing: Dict[int, Optional[str]], ok_keys: List[str],
                   failed_existing: set) -> Tuple[Dict[int, str], List[str]]:
    """Pure slot planner for numbered secrets (KEY, KEY_2 .. KEY_9).

    - a verified key already sitting in a slot KEEPS that slot (never reshuffle)
    - new keys fill EMPTY slots lowest-first
    - if no empty slot remains, slots whose current value verified hard-failed
      (AUTH_FAIL/DEAD) may be overwritten — loudly reported by the caller
    - slots holding any other value (OK / unverified / rate-limited) are
      never touched; keys with nowhere to go are returned as skipped
    """
    assignment: Dict[int, str] = {}
    remaining: List[str] = []
    for key in ok_keys:
        slot = next((s for s, v in existing.items() if v == key), None)
        if slot is not None and slot not in assignment:
            assignment[slot] = key
        else:
            remaining.append(key)
    empty = [s for s in range(1, MAX_NUMBERED_KEYS + 1)
             if not existing.get(s) and s not in assignment]
    replaceable = [s for s in sorted(existing)
                   if existing.get(s) and existing[s] in failed_existing
                   and s not in assignment]
    skipped: List[str] = []
    for key in remaining:
        if empty:
            assignment[empty.pop(0)] = key
        elif replaceable:
            assignment[replaceable.pop(0)] = key
        else:
            skipped.append(key)
    return assignment, skipped


async def _upsert_endpoint(conn, *, provider_type: str, name: str, url: str,
                           api_key: Optional[str], chain: Optional[str],
                           priority: int, notes: str,
                           supports_ws: bool = False) -> str:
    """INSERT a new pool row, or (on conflict) refresh status/last_checked
    ONLY — operator edits (priority, weight, name, is_enabled) are preserved.
    Returns 'inserted' or 'updated'."""
    endpoint_type = 'ws' if provider_type.endswith('_WS') else (
        'rpc' if provider_type.endswith('_RPC') else 'api')
    row = await conn.fetchrow("""
        INSERT INTO rpc_api_pool (
            endpoint_type, provider_type, name, url, api_key,
            status, is_enabled, priority, weight, chain, supports_ws, notes,
            last_health_check_at
        ) VALUES ($1, $2, $3, $4, $5, 'active', TRUE, $6, 100, $7, $8, $9, NOW())
        ON CONFLICT (provider_type, url) DO NOTHING
        RETURNING id
    """, endpoint_type, provider_type, name, url, api_key,
        priority, chain, supports_ws, notes)
    if row:
        return 'inserted'
    await conn.execute("""
        UPDATE rpc_api_pool
        SET status = 'active', last_health_check_at = NOW()
        WHERE provider_type = $1 AND url = $2
    """, provider_type, url)
    return 'updated'


async def apply_results(cands: List[Candidate], db_pool, secrets) -> None:
    """Write phase. Callers guarantee db_pool + secrets(+fernet) are live."""
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    inserted = updated = disabled = keys_stored = 0
    skipped: List[str] = []

    # ---- (a) verified RPC / WSS URLs -> rpc_api_pool ---------------------
    async with db_pool.acquire() as conn:
        for c in cands:
            if c.kind not in ('evm_rpc', 'solana_rpc', 'evm_ws', 'solana_ws'):
                continue
            if c.status != OK:
                if c.status in (RATE_LIMITED, QUOTA_EXHAUSTED) and c.db_id is None:
                    skipped.append(f'{c.name or redact_url(c.url)}: {c.status} '
                                   f'(auth looks fine — re-run --apply once the limit clears)')
                continue
            if c.looks_non_mainnet():
                skipped.append(f'{c.name or redact_url(c.url)}: devnet/testnet URL never imported')
                continue
            provider_type = c.effective_provider_type()
            if not provider_type:
                if c.kind in ('evm_ws', 'solana_ws'):
                    skipped.append(f'{c.name or redact_url(c.url)}: OK but no '
                                   f'{(c.chain or "?").upper()}_WS provider_type exists '
                                   f'(mig 011 taxonomy) — not inventing one, add manually if needed')
                else:
                    skipped.append(f'{c.name or redact_url(c.url)}: OK but chain unknown — not imported')
                continue
            if c.db_id is not None:
                # existing row: refresh status/last_checked only
                await conn.execute("""
                    UPDATE rpc_api_pool
                    SET status = 'active', last_health_check_at = NOW()
                    WHERE id = $1
                """, c.db_id)
                updated += 1
                if c.db_enabled is False:
                    skipped.append(f'{c.name}: verified OK but operator-disabled in DB '
                                   f'(id={c.db_id}) — left disabled, re-enable via /settings/rpc-api')
                continue
            priority = 50 if (_url_is_keyed(c.url) or c.account) else 100
            notes = f'[verify_and_import_rpcs {stamp}]'
            if c.account:
                notes += f' account={c.account}'
            if c.getlogs_ok is False:
                notes += ' getlogs=blocked'
            outcome = await _upsert_endpoint(
                conn, provider_type=provider_type,
                name=(c.name or provider_type)[:100], url=c.url, api_key=None,
                chain=c.chain, priority=priority, notes=notes,
                supports_ws=provider_type.endswith('_WS'))
            if outcome == 'inserted':
                inserted += 1
            else:
                updated += 1

        # ---- (c) hard-failing EXISTING rows -> disable (never delete) ----
        for c in cands:
            if c.db_id is None or c.status not in HARD_FAIL:
                continue
            result = await conn.execute("""
                UPDATE rpc_api_pool
                SET is_enabled = FALSE, status = 'disabled',
                    notes = COALESCE(notes, '') || $2,
                    last_health_check_at = NOW()
                WHERE id = $1 AND is_enabled = TRUE
            """, c.db_id, f' [verify: {c.status} {stamp}]')
            if result.endswith('1'):
                disabled += 1
                print(f'  disabled rpc_api_pool id={c.db_id} ({c.name}): {c.status}')
        for c in cands:
            if c.db_id is not None and c.status in (RATE_LIMITED, QUOTA_EXHAUSTED):
                print(f'  note: id={c.db_id} ({c.name}) is {c.status} — left ENABLED '
                      f'(transient; pool_engine cools it at runtime)')

    # ---- (b) verified bare API keys -> encrypted secrets + pool seed -----
    from config.pool_engine import PoolEngine  # reuse the exact URL convention
    for kind, (provider_type, chain, base_var) in KEY_KINDS.items():
        kind_cands = [c for c in cands if c.kind == kind]
        ok_keys = [c.key for c in kind_cands if c.status == OK]
        failed = {c.key for c in kind_cands if c.status in HARD_FAIL}
        if not ok_keys:
            for c in kind_cands:
                if c.status == UNTESTED:
                    skipped.append(f'{c.name or base_var}: UNTESTED — never imported unverified')
            continue
        # current slot occupancy (env + DB secrets, decrypted)
        existing: Dict[int, Optional[str]] = {}
        for slot in range(1, MAX_NUMBERED_KEYS + 1):
            name = base_var if slot == 1 else f'{base_var}_{slot}'
            try:
                existing[slot] = _clean(await secrets.get_async(name, log_access=False))
            except Exception:
                existing[slot] = None
        assignment, no_room = plan_key_slots(existing, ok_keys, failed)
        for key in no_room:
            skipped.append(f'{base_var}: key {redact_secret(key)} verified OK but all '
                           f'{MAX_NUMBERED_KEYS} slots are occupied — free a slot and re-run')
        for slot in sorted(assignment):
            key = assignment[slot]
            name = base_var if slot == 1 else f'{base_var}_{slot}'
            prev = existing.get(slot)
            if prev and prev != key:
                print(f'  OVERWRITING {name}: previous key {redact_secret(prev)} '
                      f'verified {AUTH_FAIL}/{DEAD}')
            ok = await secrets.set(name, key, category='api', is_sensitive=True)
            if not ok:
                skipped.append(f'{name}: secrets.set failed — key NOT stored')
                continue
            keys_stored += 1
            # seed the pool endpoint exactly like the Wave-F5 bootstrap
            url = PoolEngine._api_url_for_key(provider_type, key, slot,
                                              _KEY_BASE_URLS.get(provider_type))
            async with db_pool.acquire() as conn:
                outcome = await _upsert_endpoint(
                    conn, provider_type=provider_type, name=name, url=url,
                    api_key=key, chain=chain, priority=100,
                    notes=f'[verify_and_import_rpcs {stamp}]')
            if outcome == 'inserted':
                inserted += 1
        # slots holding hard-failed keys that were NOT overwritten
        for slot, value in existing.items():
            if value and value in failed and assignment.get(slot) != value \
                    and value not in assignment.values():
                name = base_var if slot == 1 else f'{base_var}_{slot}'
                print(f'  WARNING: secrets slot {name} holds a key that verified '
                      f'hard-failed ({redact_secret(value)}) — clear it via the '
                      f'/credentials page')

    # ---- (d) summary ------------------------------------------------------
    print(f'\nAPPLY SUMMARY: imported {inserted} new endpoint(s), refreshed '
          f'{updated} existing, disabled {disabled}, stored {keys_stored} '
          f'API key slot(s) encrypted, skipped {len(skipped)}.')
    for s in skipped:
        print(f'  skipped: {s}')
    print('\nReminder: run `docker compose restart trading-bot` so every module '
          'reloads the pool with the imported endpoints.')


async def run_mock_selftest() -> int:  # pragma: no cover - replaced in stage 3
    raise NotImplementedError('mock self-test lands in a later commit')


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print('\nInterrupted.')
        return 130


if __name__ == '__main__':
    sys.exit(main())
