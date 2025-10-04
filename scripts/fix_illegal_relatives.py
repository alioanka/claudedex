#!/usr/bin/env python3
"""
Fix illegal / cross-top-level relative imports using real file tree.

- Rewrites imports like:   from ...utils.helpers import X   ->   from utils.helpers import X
- Only rewrites when:
  * the relative level would go beyond top-level (illegal), or
  * the target resolves to a different top-level package than the current file (cross-package; safer as absolute)
- Leaves valid local relatives (e.g. from .sibling import X) unchanged.

Dry-run by default. Use --write to actually apply changes.
Creates .bak backups unless --no-backup is passed.

Examples:
  python scripts/fix_illegal_relatives.py --root .
  python scripts/fix_illegal_relatives.py --root . --write --no-backup
"""

import argparse
import ast
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Tuple, Optional

EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", "node_modules",
    "build", "dist", ".mypy_cache", ".ruff_cache", ".pytest_cache",
    "logs", "output", "artefacts", "artifacts"
}

@dataclass
class Replacement:
    start: int       # lineno-1
    end: int         # end_lineno (exclusive)
    text: str

def iter_py_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn

def path_to_module(root: Path, file: Path) -> str:
    rel = file.relative_to(root)
    parts = list(rel.parts)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    return ".".join(parts)

def module_exists(root: Path, mod: str) -> bool:
    if not mod:
        return False
    p = root / Path(*mod.split("."))
    return (p.with_suffix(".py").exists() or (p / "__init__.py").exists())

def top_level_package(mod: str) -> Optional[str]:
    if not mod:
        return None
    return mod.split(".", 1)[0]

def compute_absolute_from_relative(curr_module: str, node: ast.ImportFrom) -> Optional[str]:
    """
    Given current *module* name (e.g., "data.collectors.social_data")
    and a relative ImportFrom node (level>0), compute the absolute target module string.

    Semantics:
    - The "current package" is the parent of curr_module.
    - level L climbs L packages up from the current package.
    - node.module may be None or a dotted tail to append.
    """
    # current package = parent of curr_module
    parts = curr_module.split(".")
    if len(parts) == 1:
        parent = []  # module at top-level
    else:
        parent = parts[:-1]  # drop basename

    L = node.level
    tail = (node.module or "").split(".") if node.module else []
    if L <= len(parent):
        base = parent[:len(parent) - L]
        abs_parts = base + tail
        return ".".join([p for p in abs_parts if p])
    else:
        # beyond top-level → treat tail as absolute top-level module
        return ".".join(tail) if tail else None  # None means "from ... import X" with no module

def rewrite_import_line(abs_module: str, node: ast.ImportFrom) -> str:
    """
    Build a canonical 'from abs_module import name [as alias], ...' line.
    """
    names = []
    for n in node.names:
        if n.asname:
            names.append(f"{n.name} as {n.asname}")
        else:
            names.append(n.name)
    return f"from {abs_module} import {', '.join(names)}"

def fix_file(root: Path, file: Path, write: bool, no_backup: bool) -> Tuple[int, List[str]]:
    """
    Return (num_changes, messages)
    """
    curr_mod = path_to_module(root, file)
    parent_mod = ".".join(curr_mod.split(".")[:-1]) if "." in curr_mod else ""
    msgs: List[str] = []

    try:
        src = file.read_text(encoding="utf-8")
    except Exception as e:
        return 0, [f"[skip] {file}: read error: {e}"]

    try:
        tree = ast.parse(src)
    except Exception as e:
        return 0, [f"[warn] {file}: syntax error: {e}"]

    lines = src.splitlines()
    replacements: List[Replacement] = []
    changes = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level <= 0:
            continue  # not relative
        # Compute absolute target from relative
        abs_mod = compute_absolute_from_relative(curr_mod, node)

        # Case: "from ... import X" (node.module None) and beyond top-level → cannot safely rewrite; warn.
        if not abs_mod:
            # We *could* try to turn into 'import X' if X is a module, but this is ambiguous; safer to warn.
            msgs.append(f"[warn] {file}:{node.lineno} -> relative import without module beyond top-level; manual check")
            continue

        # Check if target exists in tree
        if not module_exists(root, abs_mod):
            # If the relative was illegal (level too high), still warn and skip rewrite
            msgs.append(f"[warn] {file}:{node.lineno} -> target '{abs_mod}' not found; leaving as-is")
            continue

        # If it's a simple local relative within same top-level package and not beyond top-level, we can keep it
        cur_top = top_level_package(curr_mod)
        tgt_top = top_level_package(abs_mod)
        illegal = node.level > len(parent_mod.split("."))  # beyond top-level
        cross_pkg = (cur_top != tgt_top)  # crosses top-level boundary

        if not (illegal or cross_pkg):
            # Keep local relative imports as-is
            continue

        # Build replacement text
        new_line = rewrite_import_line(abs_mod, node)

        # Replace full span (supports multi-line imports) using lineno/end_lineno
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)   # inclusive
        new = Replacement(start=start, end=end, text=new_line)
        replacements.append(new)

    if not replacements:
        return 0, msgs

    # Apply replacements bottom-up so indexes stay valid
    replacements.sort(key=lambda r: r.start, reverse=True)
    new_lines = lines[:]
    for r in replacements:
        # splice: replace [start:end] with single new line
        new_lines[r.start:r.end] = [r.text]
        changes += 1

    if write and changes:
        if not no_backup:
            file.with_suffix(file.suffix + ".bak").write_text(src, encoding="utf-8")
        file.write_text("\n".join(new_lines) + ("\n" if src.endswith("\n") else ""), encoding="utf-8")
        msgs.append(f"[fix] {file} -> {changes} import(s) rewritten")
    else:
        msgs.append(f"[would-fix] {file} -> {changes} import(s)")

    return changes, msgs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--write", action="store_true", help="Apply changes (default is dry-run)")
    ap.add_argument("--no-backup", action="store_true", help="Do not create .bak backups")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    total_changes = 0
    all_msgs: List[str] = []

    for f in iter_py_files(root):
        ch, msgs = fix_file(root, f, write=args.write, no_backup=args.no_backup)
        total_changes += ch
        all_msgs.extend(msgs)

    for m in all_msgs:
        if "[fix]" in m or "[warn]" in m:
            print(m)
    print(f"[done] total files changed: {total_changes}")

if __name__ == "__main__":
    main()
