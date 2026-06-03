"""
scripts/download_kronos_weights.py — Download Kronos weights from HuggingFace.

Operator script: run ONCE before enabling Kronos in the advisor module.
This script only downloads; it does NOT start the advisor or modify any DB rows.

IMPORTANT — Kronos needs TWO downloads
--------------------------------------
Kronos is a two-stage model: a custom K-line TOKENIZER repo AND a MODEL repo.
They live in SEPARATE HuggingFace repos and BOTH must be present. The earlier
single-repo download was insufficient. Verified model<->tokenizer pairing from
the upstream README "Model Zoo" (https://github.com/shiyu-coder/Kronos):

    variant   model repo                 tokenizer repo                 max_context
    mini      NeoQuasar/Kronos-mini      NeoQuasar/Kronos-Tokenizer-2k   2048
    small     NeoQuasar/Kronos-small     NeoQuasar/Kronos-Tokenizer-base  512
    base      NeoQuasar/Kronos-base      NeoQuasar/Kronos-Tokenizer-base  512

The model code itself is VENDORED in the repo
(modules/advisor/core/kronos_vendor/, MIT) — no code download is needed, only
these pretrained weights.

Usage
-----
    python scripts/download_kronos_weights.py                  # mini (default)
    python scripts/download_kronos_weights.py --variant small  # Kronos-small
    python scripts/download_kronos_weights.py --variant base   # Kronos-base
    python scripts/download_kronos_weights.py --dir /app/data/kronos

After download
--------------
Set the following in your .env:
    ADVISOR_KRONOS_WEIGHTS_PATH=<the printed MODEL path>
The forecaster auto-discovers the sibling tokenizer dir; override only if needed:
    ADVISOR_KRONOS_TOKENIZER_PATH=<the printed TOKENIZER path>

Then in DB:
    UPDATE config_settings
    SET value='true'
    WHERE config_type='advisor_config' AND key='advisor_kronos_enabled';

Variant reference
-----------------
    mini  :  4.1 M params, ~50 MB   — CPU-feasible (a few s / call)
    small : 24.7 M params, ~100 MB  — CPU fine
    base  :102.3 M params, ~400 MB  — GPU recommended for production latency

License: MIT
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# variant -> (model repo id, tokenizer repo id). Both are downloaded.
_REPO_MAP: dict[str, tuple[str, str]] = {
    "mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-2k"),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"),
    "base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base"),
}

# Default to /app/data/kronos: /app/data is the mounted ./data volume, so
# weights downloaded here PERSIST across `docker compose up --build`. A bare
# /data path would be ephemeral container disk and wiped on every rebuild.
_DEFAULT_DIR = Path(os.getenv("KRONOS_WEIGHTS_BASE", "/app/data/kronos"))


def _check_huggingface_hub() -> None:
    """Ensure huggingface_hub is installed; print helpful message if not."""
    try:
        import huggingface_hub  # type: ignore[import]  # noqa: F401
    except ImportError:
        print(
            "ERROR: huggingface_hub is not installed.\n"
            "Install it with: pip install huggingface_hub\n"
            "Then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)


def _download_one(repo_id: str, local_dir: Path) -> Path:
    """Snapshot-download a single HF repo into local_dir."""
    from huggingface_hub import snapshot_download  # type: ignore[import]

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {local_dir}")
    print("  (do NOT interrupt — partial downloads can corrupt the directory)")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    print(f"  done: {local_dir}")
    return local_dir


def download(variant: str = "mini", base_dir: Path = _DEFAULT_DIR) -> tuple[Path, Path]:
    """
    Download BOTH the Kronos-{variant} model repo AND its matching tokenizer
    repo into base_dir.

    Returns
    -------
    (model_dir, tokenizer_dir)
    """
    _check_huggingface_hub()

    pair = _REPO_MAP.get(variant.lower())
    if pair is None:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {list(_REPO_MAP)}"
        )
    model_repo, tokenizer_repo = pair

    model_dir = base_dir / model_repo.split("/")[-1]          # Kronos-<variant>
    tokenizer_dir = base_dir / tokenizer_repo.split("/")[-1]  # Kronos-Tokenizer-*

    print("Kronos requires TWO downloads (model + tokenizer). Fetching both.")
    print()
    _download_one(model_repo, model_dir)
    print()
    _download_one(tokenizer_repo, tokenizer_dir)

    print()
    print("=" * 64)
    print("Download complete.")
    print(f"  MODEL     : {model_dir}")
    print(f"  TOKENIZER : {tokenizer_dir}")
    print()
    print("Next steps:")
    print("  1. Add to .env:")
    print(f"       ADVISOR_KRONOS_WEIGHTS_PATH={model_dir}")
    print("     (tokenizer is auto-discovered as the sibling dir above; to")
    print("      override explicitly add:)")
    print(f"       ADVISOR_KRONOS_TOKENIZER_PATH={tokenizer_dir}")
    print()
    print("  2. Enable Kronos in DB:")
    print("       UPDATE config_settings SET value='true'")
    print("       WHERE config_type='advisor_config'")
    print("         AND key='advisor_kronos_enabled';")
    print()
    print("  3. Set the variant in DB:")
    print(f"       UPDATE config_settings SET value='Kronos-{variant}'")
    print("       WHERE config_type='advisor_config'")
    print("         AND key='advisor_kronos_variant';")
    print()
    print("  4. Restart the advisor module.")

    return model_dir, tokenizer_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download Kronos model + matching tokenizer weights from HuggingFace "
            "Hub. BOTH repos are required before enabling Kronos in the advisor."
        )
    )
    parser.add_argument(
        "--variant",
        choices=list(_REPO_MAP),
        default="mini",
        help=(
            "Model size. 'mini' = 4.1M/~50MB CPU-feasible (default); "
            "'small' = 24.7M/~100MB; 'base' = 102.3M/~400MB GPU recommended."
        ),
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=_DEFAULT_DIR,
        help=f"Base directory for weights (default: {_DEFAULT_DIR}). "
             "'Kronos-{variant}' and 'Kronos-Tokenizer-*' subdirs are created.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    args = parser.parse_args()

    if args.dry_run:
        model_repo, tokenizer_repo = _REPO_MAP[args.variant]
        model_dir = args.dir / model_repo.split("/")[-1]
        tokenizer_dir = args.dir / tokenizer_repo.split("/")[-1]
        print(f"[dry-run] Would download MODEL     {model_repo} -> {model_dir}")
        print(f"[dry-run] Would download TOKENIZER {tokenizer_repo} -> {tokenizer_dir}")
        print(f"[dry-run] Set ADVISOR_KRONOS_WEIGHTS_PATH={model_dir} after download.")
        return

    download(variant=args.variant, base_dir=args.dir)


if __name__ == "__main__":
    main()
