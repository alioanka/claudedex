#!/usr/bin/env python3
"""
ClaudeDex Verifier (PLUS v3, with suggestions)

- Handles multi-line function signatures in docs
- Correctly parses nested type hints like List[Dict[str, str]]
- Parses class expectations from docs (e.g., `class Performance(Base)`)
- Validates: missing functions/classes, signature mismatches
- Enforces interface methods for subclasses (e.g., BaseExecutor requires cancel_order)
- Reports duplicates (in-file & per-class)
- NEW: Suggests closest names when a class/function is missing (e.g., "did you mean PerformanceMetrics?")
- Outputs Markdown + JSON reports
"""

import argparse, ast, json, re, glob, difflib
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# -----------------------------
# Data models
# -----------------------------

@dataclass
class ExpectedItem:
    name: str
    kind: str  # 'function' | 'class'
    is_async: Optional[bool] = None
    visibility: Optional[str] = None
    params: Optional[List[str]] = None
    has_varargs: Optional[bool] = None
    has_varkw: Optional[bool] = None
    returns: Optional[str] = None
    wrapper_for: Optional[str] = None
    decorators: Optional[List[str]] = None

@dataclass
class FoundMethod:
    name: str
    qualified_name: str
    is_async: bool
    lineno: int
    scope: str                # "module" or "class:ClassName" or "func:outer"
    params: List[str]
    has_varargs: bool
    has_varkw: bool
    returns: Optional[str]
    visibility: str
    decorators: List[str]
    calls: List[str]

@dataclass
class FileReport:
    path: str
    exists: bool
    expected: List[ExpectedItem]
    found_methods: List[FoundMethod]
    found_classes: List[str]
    class_bases: Dict[str, List[str]]
    missing: List[str]
    async_mismatches: List[str]
    signature_mismatches: List[str]
    visibility_mismatches: List[str]
    wrapper_mismatches: List[str]
    duplicates_in_file: List[str]
    extras: List[str]

# -----------------------------
# Helpers
# -----------------------------

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def _norm(s: str) -> str:
    return s.strip().strip("`").strip()

def _ann_to_str(node: Optional[ast.AST]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return getattr(node, 'id', None) or getattr(node, 'attr', None) or str(type(node).__name__)

def _params_from_args(args: ast.arguments) -> Tuple[List[str], bool, bool]:
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    has_varargs = args.vararg is not None
    has_varkw = args.kwarg is not None
    return names, has_varargs, has_varkw

def _visibility_from_name(name: str) -> str:
    return "private" if name.startswith("_") else "public"

def _resolve_filename_under_root(root: Path, rel_path: str) -> Optional[Path]:
    """
    If rel_path exists relative to root, return it.
    If it's a bare filename, search uniquely under root/** and return a unique match.
    """
    candidate = (root / rel_path)
    if candidate.exists():
        return candidate
    if '/' in rel_path or '\\' in rel_path:
        return None
    matches = glob.glob(str(root / f"**/{rel_path}"), recursive=True)
    if len(matches) == 1:
        return Path(matches[0])
    return None

# -----------------------------
# Docs parser (multi-line & nested types safe) + CLASS parsing
# -----------------------------

DOC_FILE_HEADER_RE = re.compile(r"^\s*(?:\d+\.|\-)\s+([A-Za-z0-9_\-\/\.]+)\s*(?:\-|\:)?")

def _split_params_safely(raw: str) -> Tuple[List[str], bool, bool]:
    """
    Split parameters at top-level commas only, ignoring commas inside [], (), <>.
    Strip type hints/defaults, preserve varargs/varkw flags.
    """
    params = []
    buf = []
    depth_sq = depth_par = depth_ang = 0
    has_varargs = False
    has_varkw = False
    for ch in raw:
        if ch == '[': depth_sq += 1
        elif ch == ']': depth_sq = max(0, depth_sq-1)
        elif ch == '(': depth_par += 1
        elif ch == ')': depth_par = max(0, depth_par-1)
        elif ch == '<': depth_ang += 1
        elif ch == '>': depth_ang = max(0, depth_ang-1)
        if ch == ',' and depth_sq == depth_par == depth_ang == 0:
            token = ''.join(buf).strip()
            if token:
                if token.startswith('**'): has_varkw = True
                elif token.startswith('*'): has_varargs = True
                name = token.split(':', 1)[0].split('=', 1)[0].strip().lstrip('*')
                if name: params.append(name)
            buf = []
        else:
            buf.append(ch)
    token = ''.join(buf).strip()
    if token:
        if token.startswith('**'): has_varkw = True
        elif token.startswith('*'): has_varargs = True
        name = token.split(':', 1)[0].split('=', 1)[0].strip().lstrip('*')
        if name: params.append(name)
    return params, has_varargs, has_varkw

def parse_expectations_from_md(md_path: Path) -> Dict[str, List[ExpectedItem]]:
    """
    Tolerant parser for key_methods-style docs:
    - File bullet/header like: "- data/storage/models.py - SQLAlchemy Models"
    - Followed by class lines: "class Trade(Base)" (any text after name is ignored)
    - And def lines (possibly multi-line)
    """
    text = _read_text(md_path)
    lines = text.splitlines()
    file_to_expected: Dict[str, List[ExpectedItem]] = defaultdict(list)
    current_file: Optional[str] = None

    i = 0
    while i < len(lines):
        line = _norm(lines[i])

        # File header / bullet
        m = DOC_FILE_HEADER_RE.match(line)
        if m:
            current_file = m.group(1)
            i += 1
            continue

        if current_file is None:
            i += 1
            continue

        # CLASS expectations (e.g., "class Performance(Base)")
        if "class " in line and not line.startswith("#"):
            cm = re.search(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b", line)
            if cm:
                cname = cm.group(1)
                file_to_expected[current_file].append(ExpectedItem(
                    name=cname, kind="class"
                ))
            i += 1
            continue

        # FUNCTION expectations (multi-line signature support)
        if "def " in line and "(" in line and not line.startswith("#"):
            sig_lines = [line]
            open_parens = line.count("(") - line.count(")")
            j = i + 1
            while open_parens > 0 and j < len(lines):
                nxt = _norm(lines[j])
                sig_lines.append(nxt)
                open_parens += nxt.count("(") - nxt.count(")")
                j += 1
            full_sig = " ".join(sig_lines)
            is_async = "async def" in full_sig
            fm = re.search(r"(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*(?:->\s*([^\s\[]+))?", full_sig)
            if fm:
                name = fm.group(1)
                raw_params = fm.group(2)
                ret = fm.group(3)
                params, has_varargs, has_varkw = _split_params_safely(raw_params)
                file_to_expected[current_file].append(ExpectedItem(
                    name=name, kind="function", is_async=is_async,
                    params=params, has_varargs=has_varargs, has_varkw=has_varkw, returns=ret
                ))
            i = j
            continue

        i += 1

    return dict(file_to_expected)

# -----------------------------
# Structure parser (file tree)
# -----------------------------

def parse_structure_md(md_path: Path) -> List[str]:
    """
    Parse corrected_file_structure.md into a list of Python file paths.
    Supports ASCII tree diagrams and simple regex fallback.
    """
    text = _read_text(md_path)

    def _parse_tree_block_lines(text: str) -> List[Tuple[int, str, bool]]:
        out = []
        for raw in text.splitlines():
            line = raw.rstrip()
            if '── ' not in line:
                continue
            try:
                prefix, name = line.split('── ', 1)
            except ValueError:
                continue
            level = 0; i = 0
            while i < len(prefix):
                if prefix[i:i+4] in ('│   ', '    '):
                    level += 1; i += 4
                else:
                    i += 1
            is_dir = name.endswith('/')
            out.append((level, name.strip(), is_dir))
        return out

    entries = _parse_tree_block_lines(text)
    paths: List[str] = []
    if entries:
        stack: List[str] = []
        for level, name, is_dir in entries:
            stack = stack[:level]
            if is_dir:
                stack.append(name[:-1])
            else:
                full = "/".join(stack + [name])
                paths.append(full)
    else:
        paths = re.findall(r"([A-Za-z0-9_\-\/\.]+\.py)\b", text)

    seen, ordered = set(), []
    for c in paths:
        if c not in seen and c.endswith(".py"):
            ordered.append(c)
            seen.add(c)
    return ordered

# -----------------------------
# AST collector
# -----------------------------

class _CallCollector(ast.NodeVisitor):
    def __init__(self): self.calls = []
    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name): self.calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute): self.calls.append(node.func.attr)
        self.generic_visit(node)

class _AstCollector(ast.NodeVisitor):
    def __init__(self):
        self.methods: List[FoundMethod] = []
        self.classes: List[str] = []
        self.class_bases: Dict[str, List[str]] = {}
        self._class_stack: List[str] = []
        self._func_stack: List[str] = []
    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        bases = []
        for b in node.bases:
            try: s = ast.unparse(b)
            except Exception: s = getattr(b, 'id', None) or getattr(b, 'attr', None) or ""
            tail = s.split('.')[-1] if s else ""
            if tail: bases.append(tail)
        self.class_bases[node.name] = bases
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()
    def _visit_func(self, node, is_async):
        name = node.name
        if self._class_stack:
            scope = f"class:{self._class_stack[-1]}"; qname = f"{self._class_stack[-1]}.{name}"
        elif self._func_stack:
            scope = f"func:{self._func_stack[-1]}"; qname = f"{self._func_stack[-1]}.{name}"
        else:
            scope = "module"; qname = name
        args = node.args
        params, has_varargs, has_varkw = _params_from_args(args)
        returns = _ann_to_str(node.returns)
        visibility = _visibility_from_name(name)
        cc = _CallCollector()
        for stmt in node.body:
            cc.visit(stmt)
        self.methods.append(FoundMethod(name, qname, is_async, node.lineno, scope, params, has_varargs, has_varkw, returns, visibility, [], cc.calls))
        self._func_stack.append(name); self.generic_visit(node); self._func_stack.pop()
    def visit_FunctionDef(self, node): return self._visit_func(node, False)
    def visit_AsyncFunctionDef(self, node): return self._visit_func(node, True)

def collect_file_symbols(py_path: Path):
    code = _read_text(py_path)
    if not code:
        return [], [], {}, [f"Empty: {py_path}"]
    try:
        tree = ast.parse(code)
    except Exception as e:
        return [], [], {}, [f"AST parse error in {py_path}: {e}"]
    c = _AstCollector()
    c.visit(tree)
    return c.methods, c.classes, c.class_bases, []

# -----------------------------
# Verification
# -----------------------------

def _compare_signature(expected: ExpectedItem, found_variants: List[FoundMethod], file_path: str) -> List[str]:
    issues: List[str] = []
    for fm in found_variants:
        exp_params = list(expected.params or [])
        f_params = list(fm.params)
        # ignore self/cls leading for instance/class methods
        if f_params and f_params[0] in ("self", "cls"):
            f_params = f_params[1:]
        if exp_params and exp_params != f_params:
            issues.append(f"{file_path}:{fm.lineno} {fm.qualified_name} params mismatch expected={exp_params} found={f_params}")
    return issues

INTERFACES = {
    "BaseExecutor": {"execute_trade", "cancel_order", "modify_order", "validate_order", "get_order_status"},
}

def _enforce_interfaces(class_bases: Dict[str, List[str]], methods: List[FoundMethod]) -> List[str]:
    required = set()
    for bases in class_bases.values():
        for b in bases:
            if b in INTERFACES:
                required |= INTERFACES[b]
    present = {m.name for m in methods}
    return [f"def {n}()" for n in sorted(required - present)]

def _closest(names: List[str], target: str, n: int = 3, cutoff: float = 0.6) -> List[str]:
    """Suggest up to n names close to target (case-sensitive first, fallback to case-insensitive)."""
    if not names:
        return []
    hits = difflib.get_close_matches(target, names, n=n, cutoff=cutoff)
    if hits:
        return hits
    # try case-insensitive if no hits
    lower_map = {name.lower(): name for name in names}
    hits_lower = difflib.get_close_matches(target.lower(), list(lower_map.keys()), n=n, cutoff=cutoff)
    return [lower_map[h] for h in hits_lower]

def generate_reports(root: Path, expected_map: Dict[str, List[ExpectedItem]], all_files_to_check: Optional[List[str]] = None):
    reports: Dict[str, FileReport] = {}
    if not all_files_to_check:
        all_files_to_check = sorted(expected_map.keys())

    for rel_path in all_files_to_check:
        resolved = _resolve_filename_under_root(root, rel_path)
        fpath = resolved if resolved else (root / rel_path)
        exists = fpath.exists()

        expected = expected_map.get(rel_path, [])

        found_methods: List[FoundMethod] = []
        found_classes: List[str] = []
        class_bases: Dict[str, List[str]] = {}
        parse_errors: List[str] = []

        if exists and fpath.suffix == ".py":
            found_methods, found_classes, class_bases, parse_errors = collect_file_symbols(fpath)

        missing: List[str] = []
        sig_mism: List[str] = []
        async_mismatches: List[str] = []
        visibility_mismatches: List[str] = []
        wrapper_mismatches: List[str] = []
        extras: List[str] = []

        # Expected checks: classes + functions (with suggestions when missing)
        for exp in expected:
            if exp.kind == "class":
                if exp.name not in found_classes:
                    suggestions = _closest(found_classes, exp.name)
                    hint = f" (closest: {', '.join(suggestions)})" if suggestions else ""
                    missing.append(f"class {exp.name}{hint}")
                continue

            variants = [m for m in found_methods if m.name == exp.name]
            if not variants:
                suggestions = _closest([m.name for m in found_methods], exp.name)
                hint = f" (closest: {', '.join(suggestions)})" if suggestions else ""
                missing.append(f"def {exp.name}(){hint}")
                continue

            sig_mism.extend(_compare_signature(exp, variants, str(fpath.relative_to(root))))
            # (hooks for async/visibility/wrapper available if you later want them strict)

        # Interface enforcement (e.g., BaseExecutor)
        missing += _enforce_interfaces(class_bases, found_methods)

        # Duplicate detection (per-file/per-class)
        scope_name_counts = Counter((m.scope, m.name) for m in found_methods)
        duplicates_in_file = [
            f"{scope}::{name} x{cnt}"
            for (scope, name), cnt in scope_name_counts.items()
            if cnt > 1
        ]

        rep = FileReport(
            path=str(fpath.relative_to(root)) if exists else rel_path,
            exists=exists,
            expected=expected,
            found_methods=found_methods,
            found_classes=found_classes,
            class_bases=class_bases,
            missing=missing,
            async_mismatches=async_mismatches,
            signature_mismatches=sig_mism,
            visibility_mismatches=visibility_mismatches,
            wrapper_mismatches=wrapper_mismatches,
            duplicates_in_file=duplicates_in_file,
            extras=extras,
        )

        for err in parse_errors:
            rep.missing.append(f"[ParseError] {err}")

        reports[rel_path] = rep

    return reports, {}

# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="ClaudeDex verifier (PLUS v3 + suggestions)")
    p.add_argument("--root", default=".", help="Project root")
    p.add_argument("--docs", nargs="+", default=["docs/*.md"], help="Docs; glob patterns allowed (PowerShell-safe)")
    p.add_argument("--structure", default=None, help="corrected_file_structure.md path")
    p.add_argument("--markdown-out", default="verifier_report.md")
    p.add_argument("--json-out", default="verifier_report.json")
    p.add_argument("--fail-on-missing", action="store_true", help="Exit with code 2 if any missing/signature errors exist")
    args = p.parse_args()

    root = Path(args.root).resolve()

    # Expand doc globs so PowerShell literal '*.md' works
    doc_paths: List[Path] = []
    for pat in args.docs:
        expanded = glob.glob(str(pat)) or glob.glob(str(root / pat))
        if expanded:
            doc_paths += [Path(p) for p in expanded]
        else:
            print(f"[WARN] Doc not found: {pat}")

    # Build expectations from docs
    expected_map: Dict[str, List[ExpectedItem]] = defaultdict(list)
    for dp in doc_paths:
        part = parse_expectations_from_md(dp)
        for k, v in part.items():
            expected_map[k].extend(v)

    # Files to check from structure (if provided)
    files_to_check: Optional[List[str]] = None
    if args.structure:
        sp = Path(args.structure)
        if not sp.exists():
            sp2 = (root / args.structure)
            if sp2.exists(): sp = sp2
        if sp.exists():
            files_to_check = parse_structure_md(sp)

    reports, _ = generate_reports(root, expected_map, files_to_check)

    # JSON report
    Path(args.json_out).write_text(
        json.dumps({k: asdict(v) for k, v in reports.items()}, indent=2),
        encoding="utf-8"
    )

    # Markdown report (simple but useful)
    md_lines = []
    for k, r in reports.items():
        md_lines.append(f"### {k}")
        md_lines.append(f"- Missing: {r.missing}")
        md_lines.append(f"- Sig mismatches: {r.signature_mismatches}")
        md_lines.append(f"- Duplicates: {r.duplicates_in_file}")
        md_lines.append("")
    Path(args.markdown_out).write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[OK] Wrote reports to {args.markdown_out} and {args.json_out}")

    if args.fail_on_missing:
        # Fail if any missing class/method or signature issues exist
        total_issues = sum(len(r.missing) + len(r.signature_mismatches) for r in reports.values())
        if total_issues > 0:
            raise SystemExit(2)

if __name__ == "__main__":
    main()
