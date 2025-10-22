#!/usr/bin/env python3
"""
generate_file_tree.py

Generate a tree-style text file describing the file/folder structure.

Usage examples:
    # create file_tree.txt in the current folder describing the current folder
    python generate_file_tree.py --root . 

    # create file_tree.txt describing a specific folder, write output to that folder
    python generate_file_tree.py --root "C:/Users/HP/Desktop/ClaudeDex"

    # write to a different output file
    python generate_file_tree.py --root . --output ./ClaudeDex_file_tree.txt

Options:
    --root   / -r   Root folder to read (default: current directory)
    --output / -o   Output file path (default: <root>/file_tree.txt)
    --hidden       Include hidden files and directories (default: False)
    --follow-symlinks Include following symlinks (default: False)
    --max-depth / -d  Max recursion depth (0 = only root; default: unlimited)
"""
import argparse
from pathlib import Path
from typing import List

TREE_BRANCH = "├── "
TREE_LAST = "└── "
TREE_PIPE = "│   "
TREE_SPACE = "    "

def list_tree_lines(root: Path, include_hidden: bool=False, follow_symlinks: bool=False, max_depth: int=-1) -> List[str]:
    """
    Return list of lines representing the tree starting at 'root'.
    Lists files first (alphabetical), then directories (alphabetical) to match the sample style.
    """
    root = root.resolve()
    lines: List[str] = []
    root_name = root.name or str(root)  # e.g., "ClaudeDex" or drive root

    lines.append(f"{root_name}/")
    
    def _walk(dir_path: Path, prefix: str, depth: int):
        if max_depth >= 0 and depth > max_depth:
            return

        try:
            entries = list(dir_path.iterdir())
        except PermissionError:
            lines.append(prefix + TREE_LAST + "[permission denied]")
            return

        # Filter hidden files if requested
        if not include_hidden:
            entries = [e for e in entries if not e.name.startswith(".")]

        # Sort: files first (alpha), then dirs (alpha)
        files = sorted([e for e in entries if e.is_file() or (e.is_symlink() and e.exists() and e.resolve().is_file())], key=lambda p: p.name.lower())
        dirs = sorted([e for e in entries if e.is_dir() or (e.is_symlink() and e.exists() and e.resolve().is_dir())], key=lambda p: p.name.lower())

        combined = files + dirs
        for idx, entry in enumerate(combined):
            is_last = (idx == len(combined) - 1)
            connector = TREE_LAST if is_last else TREE_BRANCH
            display_name = entry.name + ("/" if entry.is_dir() else "")
            # Show symlink arrow if it's a symlink
            if entry.is_symlink():
                try:
                    target = entry.resolve()
                    display_name = f"{entry.name} -> {target}" + ("/" if target.is_dir() else "")
                except Exception:
                    display_name = f"{entry.name} -> [broken symlink]"

            lines.append(prefix + connector + display_name)

            # Recurse into directories (and symlinked dirs if follow_symlinks)
            should_recurse = False
            if entry.is_dir():
                should_recurse = True
            elif entry.is_symlink():
                # symlink that points to dir
                try:
                    if follow_symlinks and entry.resolve().is_dir():
                        should_recurse = True
                except Exception:
                    should_recurse = False

            if should_recurse:
                new_prefix = prefix + (TREE_SPACE if is_last else TREE_PIPE)
                _walk(entry.resolve() if follow_symlinks else entry, new_prefix, depth + 1)

    _walk(root, "", 0)
    return lines

def write_tree_file(root: Path, out_path: Path, include_hidden: bool=False, follow_symlinks: bool=False, max_depth: int=-1) -> None:
    lines = list_tree_lines(root, include_hidden=include_hidden, follow_symlinks=follow_symlinks, max_depth=max_depth)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} lines to: {out_path}")

def main():
    p = argparse.ArgumentParser(description="Generate a file-tree text file for a folder.")
    p.add_argument("--root", "-r", default=".", help="Root directory to inspect (default: current directory).")
    p.add_argument("--output", "-o", default=None, help="Output file path (default: <root>/file_tree.txt).")
    p.add_argument("--hidden", action="store_true", help="Include hidden files / dotfiles.")
    p.add_argument("--follow-symlinks", action="store_true", help="Follow symlinked directories when recursing.")
    p.add_argument("--max-depth", "-d", type=int, default=-1, help="Max recursion depth (0=only root). Default: unlimited.")
    args = p.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR: root path does not exist: {root}")
        return

    default_out = root / "file_tree.txt"
    out_path = Path(args.output).resolve() if args.output else default_out

    write_tree_file(root, out_path, include_hidden=args.hidden, follow_symlinks=args.follow_symlinks, max_depth=args.max_depth)

if __name__ == "__main__":
    main()
