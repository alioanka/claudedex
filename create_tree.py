#!/usr/bin/env python3
"""
create_tree.py

Create a boilerplate file tree for the ClaudeDex project.

Usage:
    python create_tree.py                # creates ./ClaudeDex/...
    python create_tree.py --root .       # create files in current directory (no ClaudeDex root)
    python create_tree.py --root MyDir   # create under ./MyDir
    python create_tree.py --overwrite    # overwrite existing files
"""
import argparse
from pathlib import Path
from textwrap import dedent

STRUCTURE = {
    "": [
        "count_loc.py",
        "jup_text.py",
        "main.py",
        "scaffold.py",
        "setup.py",
        "setup_env_keys.py",
        "test_dexscreener_solana.py",
        "test_solana.py",
        "verify_references.py",
        "xref_symbol_db.py",
    ],
    "analysis": [
        "__init__.py",
        "dev_analyzer.py",
        "liquidity_monitor.py",
        "market_analyzer.py",
        "pump_predictor.py",
    ],
}


BASIC_TEMPLATE = """\
# {name}
\"\"\"Auto-generated placeholder for {name}.

Replace this content with your real implementation.
\"\"\"

def main():
    print("This is a placeholder for {name}.")

if __name__ == "__main__":
    main()
"""

INIT_TEMPLATE = """\
# Package initializer for the analysis package.
# Add package-level imports or metadata here.
"""

def create_tree(root: Path, overwrite: bool = False):
    created = []
    skipped = []
    root.mkdir(parents=True, exist_ok=True)

    for folder, files in STRUCTURE.items():
        target_dir = root / folder if folder else root
        target_dir.mkdir(parents=True, exist_ok=True)

        for filename in files:
            file_path = target_dir / filename
            if file_path.exists() and not overwrite:
                skipped.append(str(file_path))
                continue

            # Choose template
            if filename == "__init__.py":
                text = INIT_TEMPLATE
            elif filename.endswith(".py"):
                text = dedent(BASIC_TEMPLATE).format(name=filename)
            else:
                text = f"# {filename}\n"

            # Write file (overwrite or create)
            file_path.write_text(text, encoding="utf-8")
            created.append(str(file_path))

    return created, skipped

def main():
    p = argparse.ArgumentParser(description="Create ClaudeDex file tree.")
    p.add_argument("--root", "-r", default="ClaudeDex",
                   help="Root directory to create the tree under. Use '.' for current directory.")
    p.add_argument("--overwrite", "-o", action="store_true",
                   help="Overwrite files if they already exist.")
    args = p.parse_args()

    root = Path(args.root).resolve()
    created, skipped = create_tree(root, overwrite=args.overwrite)

    print(f"\nRoot: {root}\n")
    if created:
        print("Created files:")
        for f in created:
            print("  +", f)
    else:
        print("No files were created.")

    if skipped:
        print("\nSkipped (already existed):")
        for f in skipped:
            print("  -", f)

    print("\nDone.")

if __name__ == "__main__":
    main()
