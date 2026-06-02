"""
scripts/download_kronos_weights.py — Download Kronos model weights from HuggingFace.

Operator script: run ONCE before enabling Kronos in the advisor module.
This script only downloads; it does NOT start the advisor or modify any DB rows.

Usage
-----
    python scripts/download_kronos_weights.py                  # mini (default)
    python scripts/download_kronos_weights.py --variant small  # Kronos-small
    python scripts/download_kronos_weights.py --variant base   # Kronos-base
    python scripts/download_kronos_weights.py --dir /data/kronos

After download
--------------
Set the following in your .env (or Secure Credentials):
    ADVISOR_KRONOS_WEIGHTS_PATH=<the printed path>

Then in DB:
    UPDATE config_settings
    SET value='true'
    WHERE config_type='advisor_config' AND key='advisor_kronos_enabled';

Variant reference
-----------------
    mini  :  4.1 M params, ~50 MB   — CPU-feasible (<2 s / call)
    small : 24.7 M params, ~100 MB  — CPU fine (~3 s / call)
    base  :102.3 M params, ~400 MB  — GPU recommended for production latency

HuggingFace repo IDs: NeoQuasar/Kronos-{mini,small,base}
License: MIT
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


_REPO_MAP = {
    "mini":  "NeoQuasar/Kronos-mini",
    "small": "NeoQuasar/Kronos-small",
    "base":  "NeoQuasar/Kronos-base",
}

_DEFAULT_DIR = Path(os.getenv("KRONOS_WEIGHTS_BASE", "/data/kronos"))


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


def download(variant: str = "mini", base_dir: Path = _DEFAULT_DIR) -> Path:
    """
    Download Kronos-{variant} from HuggingFace Hub to base_dir/Kronos-{variant}.

    Parameters
    ----------
    variant  : "mini" | "small" | "base"
    base_dir : Local directory root; subdirectory is created automatically.

    Returns
    -------
    Path to the downloaded weights directory.
    """
    _check_huggingface_hub()

    from huggingface_hub import snapshot_download  # type: ignore[import]

    repo_id = _REPO_MAP.get(variant.lower())
    if repo_id is None:
        raise ValueError(
            f"Unknown variant '{variant}'. Choose from: {list(_REPO_MAP)}"
        )

    local_dir = base_dir / f"Kronos-{variant}"
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id} -> {local_dir}")
    print("This may take a minute depending on your connection speed.")
    print("Do NOT interrupt — partial downloads may leave the directory in a bad state.")
    print()

    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))

    print()
    print(f"Download complete: {local_dir}")
    print()
    print("Next steps:")
    print(f"  1. Add to .env (or Secure Credentials):")
    print(f"       ADVISOR_KRONOS_WEIGHTS_PATH={local_dir}")
    print()
    print("  2. Enable Kronos in DB:")
    print("       UPDATE config_settings")
    print("       SET value='true'")
    print("       WHERE config_type='advisor_config'")
    print("         AND key='advisor_kronos_enabled';")
    print()
    print("  3. Optionally set variant in DB:")
    print(f"       UPDATE config_settings SET value='Kronos-{variant}'")
    print("       WHERE config_type='advisor_config'")
    print("         AND key='advisor_kronos_variant';")
    print()
    print("  4. Restart the advisor module.")

    return local_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download Kronos foundation-model weights from HuggingFace Hub. "
            "Weights are required before enabling Kronos in the advisor module."
        )
    )
    parser.add_argument(
        "--variant",
        choices=list(_REPO_MAP),
        default="mini",
        help=(
            "Model size to download. "
            "'mini' = 4.1M/~50MB CPU-feasible (default); "
            "'small' = 24.7M/~100MB; "
            "'base' = 102.3M/~400MB GPU recommended."
        ),
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=_DEFAULT_DIR,
        help=f"Base directory for weights (default: {_DEFAULT_DIR}). "
             "A subdirectory 'Kronos-{variant}' is created inside it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded without actually downloading.",
    )
    args = parser.parse_args()

    if args.dry_run:
        repo_id = _REPO_MAP[args.variant]
        local_dir = args.dir / f"Kronos-{args.variant}"
        print(f"[dry-run] Would download {repo_id} to {local_dir}")
        print(f"[dry-run] Set ADVISOR_KRONOS_WEIGHTS_PATH={local_dir} after download.")
        return

    download(variant=args.variant, base_dir=args.dir)


if __name__ == "__main__":
    main()
