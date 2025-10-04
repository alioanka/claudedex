#!/usr/bin/env python3
"""
Dev-side guardrail:
- Finds type-hint names used in annotations (Tuple, List, Any, Optional, Union, Deque, Dict, Set, Literal, etc.)
- Inserts missing imports: "from typing import <Names>" and "from decimal import Decimal" when needed.
- Optionally adds: "from __future__ import annotations" to make hints lazy at runtime.
- Audits external imports vs requirements.txt and (optionally) appends missing packages.

Usage:
  python scripts/dev_autofix_imports.py --root . --write --add-future --update-reqs
  python scripts/dev_autofix_imports.py --root .          # dry-run
"""

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple

# --- CONFIG ---
DEFAULT_EXCLUDES = {".venv", "venv", "__pycache__", ".git", "build", "dist", "node_modules", ".mypy_cache", ".ruff_cache", ".pytest_cache", "logs"}
TYPING_NAMES: Set[str] = {
    "Any","Optional","Union","Tuple","List","Dict","Set","FrozenSet",
    "Callable","Iterable","Iterator","Mapping","MutableMapping","Sequence",
    "MutableSequence","Deque","DefaultDict","TypedDict","Literal","Protocol",
    "TypeVar","Generic","Awaitable","Coroutine","NoReturn"
}
# Common module->pip-name hints
PKG_HINTS = {
    "eth_account": "eth-account",
    "TA-Lib": "TA-Lib",
    "talib": "TA-Lib",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "ujson": "ujson",
    "orjson": "orjson",
    "textblob": "textblob",
    "web3": "web3",
    "dateutil": "python-dateutil",
    "hdwallet": "hdwallet",
    "dotenv": "python-dotenv",
    "jwt": "PyJWT",
    "psycopg2": "psycopg2-binary",
    "sklearn": "scikit-learn",
    "socketio": "python-socketio",
    "aiohttp_cors": "aiohttp-cors",
    # pick the one you actually use; adjust if different:
    "aiohttp_sse": "aiohttp-sse-client",
}
# Heuristic stdlib set: rely on py311 runtime when present
try:
    import sys as _sys
    STD_LIB = set(_sys.stdlib_module_names)  # py311+
except Exception:
    STD_LIB = set()

def find_py_files(root: Path) -> List[Path]:
    files = []
    for p, dnames, fnames in os.walk(root):
        # prune excluded dirs
        dnames[:] = [d for d in dnames if d not in DEFAULT_EXCLUDES]
        for f in fnames:
            if f.endswith(".py"):
                files.append(Path(p) / f)
    return files

def get_module_path(root: Path, file: Path) -> str:
    try:
        rel = file.relative_to(root).with_suffix("")
        return ".".join(rel.parts)
    except Exception:
        return file.stem

def parse_imports(src: str) -> Tuple[Set[str], Set[str], Set[str]]:
    """Return (imported_typing_names, imported_modules, from_decimal_names)"""
    imported_typing: Set[str] = set()
    imported_modules: Set[str] = set()
    from_decimal: Set[str] = set()
    try:
        tree = ast.parse(src)
    except Exception:
        return imported_typing, imported_modules, from_decimal
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                imported_modules.add(top)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            top = mod.split(".")[0] if mod else ""
            if top:
                imported_modules.add(top)
            if mod == "typing":
                for alias in node.names:
                    imported_typing.add(alias.name)
            if mod == "decimal":
                for alias in node.names:
                    from_decimal.add(alias.name)
    return imported_typing, imported_modules, from_decimal

def collect_annotation_names(tree: ast.AST) -> Set[str]:
    """Collect bare names used inside annotations."""
    used: Set[str] = set()
    def visit_ann(n):
        if isinstance(n, ast.Name):
            used.add(n.id)
        elif isinstance(n, ast.Attribute):
            # e.g., decimal.Decimal; we'll separately handle bare Decimal
            if isinstance(n.value, ast.Name) and n.attr == "Decimal":
                used.add("Decimal")
        elif isinstance(n, ast.Subscript):
            visit_ann(n.value)
            if hasattr(n, "slice"):
                for ch in ast.walk(n.slice):
                    if isinstance(ch, ast.Name):
                        used.add(ch.id)
        elif isinstance(n, (ast.Tuple, ast.List, ast.Set)):
            for el in n.elts:
                visit_ann(el)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns: visit_ann(node.returns)
            args = list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
            for a in args:
                if getattr(a, "annotation", None):
                    visit_ann(a.annotation)
        elif isinstance(node, ast.AnnAssign):
            if getattr(node, "annotation", None):
                visit_ann(node.annotation)
    return used

def build_local_name_index(root: Path) -> Set[str]:
    """
    Return a set of names that should be treated as *local* modules/packages:
    - top-level package dirs that contain __init__.py (e.g., core, data, trading, security, utils, config, monitoring, etc.)
    - every python file's basename (e.g., event_bus from core/event_bus.py)
    """
    names: Set[str] = set()
    for p, dnames, fnames in os.walk(root):
        # prune noise
        dnames[:] = [d for d in dnames if d not in DEFAULT_EXCLUDES]
        # package dirs
        for d in dnames:
            pkg_init = Path(p) / d / "__init__.py"
            if pkg_init.exists():
                names.add(d)
        # module basenames
        for f in fnames:
            if f.endswith(".py"):
                names.add(Path(f).stem)
    return names


def insertion_index(src: str) -> int:
    """Find the line index where new imports should be inserted (after docstring / __future__)."""
    lines = src.splitlines()
    i = 0
    # Skip shebang and encoding lines
    while i < len(lines) and (lines[i].startswith("#!") or lines[i].startswith("# -*-")):
        i += 1
    # Module docstring
    if i < len(lines) and re.match(r'^\s*[ruRU]?["\']{3}', lines[i]):
        # consume until closing triple-quote
        q = lines[i].lstrip().split(lines[i].lstrip()[0])[0] if lines[i].lstrip() else '"""'
        i += 1
        while i < len(lines) and '"""' not in lines[i] and "'''" not in lines[i]:
            i += 1
        i += 1
    # Future imports
    while i < len(lines) and lines[i].startswith("from __future__"):
        i += 1
    return i

def ensure_future_annotations(lines: List[str]) -> List[str]:
    """Insert `from __future__ import annotations` after docstring if not present."""
    if any(l.startswith("from __future__ import annotations") for l in lines[:5]):
        return lines
    idx = insertion_index("\n".join(lines))
    return lines[:idx] + ["from __future__ import annotations"] + lines[idx:]

def ensure_import_lines(src: str,
                        need_typing: Set[str],
                        need_decimal: bool,
                        add_future: bool) -> Tuple[str, bool]:
    """Return (new_src, changed)"""
    changed = False
    lines = src.splitlines()
    if add_future:
        new_lines = ensure_future_annotations(lines)
        if new_lines != lines:
            lines = new_lines
            changed = True

    # Build new import lines
    inserts: List[str] = []
    if need_typing:
        inserts.append(f"from typing import {', '.join(sorted(need_typing))}")
    if need_decimal:
        inserts.append("from decimal import Decimal")

    if not inserts:
        return src, changed

    idx = insertion_index("\n".join(lines))
    new_src = "\n".join(lines[:idx] + inserts + lines[idx:])
    return new_src, True

def find_external_modules(src: str) -> Set[str]:
    """Return top-level imported module names that look external (not stdlib)."""
    modules: Set[str] = set()
    try:
        tree = ast.parse(src)
    except Exception:
        return modules
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod and not mod.startswith("."):
                modules.add(mod.split(".")[0])
    return modules

def is_local_module(root: Path, name: str) -> bool:
    return (root / (name.replace(".", "/") + ".py")).exists() or (root / name / "__init__.py").exists()

def normalize_req_line(line: str) -> str:
    # get package token up to version spec or extras
    return re.split(r"[<>= \[]", line.strip(), maxsplit=1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--write", action="store_true", help="Apply changes (default is dry-run)")
    ap.add_argument("--add-future", action="store_true", help="Insert `from __future__ import annotations`")
    ap.add_argument("--update-reqs", action="store_true", help="Append missing packages to requirements.txt")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = find_py_files(root)
    print(f"[scan] Python files: {len(files)}")

    # Load requirements
    req_path = root / "requirements.txt"
    req_lines = req_path.read_text(encoding="utf-8").splitlines() if req_path.exists() else []
    req_pkgs = {normalize_req_line(l) for l in req_lines if l.strip() and not l.strip().startswith("#")}

    updated_files = 0
    missing_pkgs: Set[str] = set()

    for f in files:
        try:
            src = f.read_text(encoding="utf-8")
        except Exception as e:
            print(f"[skip] {f}: read error {e}")
            continue

        try:
            tree = ast.parse(src)
        except Exception as e:
            print(f"[warn] {f}: syntax error: {e}")
            continue

        used = collect_annotation_names(tree)
        imp_typing, imp_mods, dec_names = parse_imports(src)

        # Which typing names are missing?
        needed_typing = (used & TYPING_NAMES) - imp_typing
        need_decimal = ("Decimal" in used) and ("Decimal" not in dec_names)

        # before looping files:
        local_names = build_local_name_index(root)

        # External deps audit (module-level)
        exmods = find_external_modules(src)
        # inside the loop, when you compute exmods = find_external_modules(src):
        for m in exmods:
            top = m
            # skip stdlib
            if top in STD_LIB:
                continue
            # skip local: top-level package dirs OR any .py basename anywhere in repo
            if top in local_names:
                continue
            # also consider dots: e.g., 'trading' is local even if import was 'from trading.strategies...'
            if top in local_names:
                continue

            # map to pip-name
            pip_name = PKG_HINTS.get(top, top)
            if not any(normalize_req_line(line).lower() == pip_name.lower() for line in req_lines):
                missing_pkgs.add(pip_name)

        if not needed_typing and not need_decimal and not args.add_future:
            continue

        new_src, changed = ensure_import_lines(src, needed_typing, need_decimal, add_future=args.add_future)
        if changed and args.write:
            # backup
            f.with_suffix(f.suffix + ".bak").write_text(src, encoding="utf-8")
            f.write_text(new_src, encoding="utf-8")
            updated_files += 1
            print(f"[fix] {f} -> +typing:{sorted(needed_typing)} +decimal:{need_decimal}")
        elif changed:
            print(f"[would-fix] {f} -> +typing:{sorted(needed_typing)} +decimal:{need_decimal}")

    # Update requirements.txt if requested
    if args.update_reqs and missing_pkgs:
        print(f"[deps] appending to requirements.txt: {sorted(missing_pkgs)}")
        with req_path.open("a", encoding="utf-8") as rf:
            for p in sorted(missing_pkgs):
                # pin a few well-known packages if not present
                if p.lower() == "ta-lib":
                    rf.write("\nTA-Lib==0.4.32\n")
                elif p.lower() == "numpy":
                    rf.write("\nnumpy<2\n")
                else:
                    rf.write(f"\n{p}\n")

    print(f"[done] files updated: {updated_files}; missing packages detected: {sorted(missing_pkgs)}")

if __name__ == "__main__":
    main()
