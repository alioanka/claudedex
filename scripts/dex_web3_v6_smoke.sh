#!/usr/bin/env bash
# DEX web3 v6 API drift smoke (catalog: script_dex_web3_v6_imports;
# wave-2 commit 48d5f20). Asserts direct_dex + mev_protection import and
# that the v6 snake_case helpers + PoA middleware shim resolve.
set -euo pipefail
cd "${CLAUDEDEX_REPO_ROOT:-/app}"
python - <<'PY'
import sys
from trading.executors import direct_dex as _dd        # noqa: F401
from trading.executors import mev_protection as _mev   # noqa: F401
from web3 import Web3
missing = [n for n in ("to_checksum_address", "is_address", "is_connected")
           if not hasattr(Web3, n)]
poa_found = False
try:
    from web3.middleware import ExtraDataToPOAMiddleware  # noqa: F401
    poa_found = True
except Exception:
    try:
        from web3.middleware import geth_poa_middleware  # noqa: F401
        poa_found = True
    except Exception:
        pass
if missing:
    print("FAIL — v6 helpers missing:", missing); sys.exit(1)
if not poa_found:
    print("FAIL — no PoA middleware import path"); sys.exit(1)
print("PASS — direct_dex + mev_protection import + v6 helpers + PoA OK")
PY
