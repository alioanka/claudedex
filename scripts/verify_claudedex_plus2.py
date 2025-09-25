
#!/usr/bin/env python3
# (content continues from previous attempt)

import argparse
import ast
import json
import re
import glob
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set

@dataclass
class ExpectedItem:
    name: str
    kind: str
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
    scope: str
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
    candidate = (root / rel_path)
    if candidate.exists():
        return candidate
    if '/' in rel_path or '\\' in rel_path:
        return None
    matches = glob.glob(str(root / f"**/{rel_path}"), recursive=True)
    if len(matches) == 1:
        return Path(matches[0])
    return None

DOC_FILE_HEADER_RE = re.compile(r"^\s*(?:\d+\.|\-)\s+([A-Za-z0-9_\-\/\.]+)\s*(?:\-|\:)?")

def parse_expectations_from_md(md_path: Path) -> Dict[str, List[ExpectedItem]]:
    text = _read_text(md_path)
    lines = text.splitlines()
    file_to_expected: Dict[str, List[ExpectedItem]] = defaultdict(list)
    current_file: Optional[str] = None

    for raw in lines:
        line = _norm(raw)
        if not line:
            continue

        m = DOC_FILE_HEADER_RE.match(line)
        if m:
            current_file = m.group(1)
            continue
        if current_file is None:
            continue

        if "class " in line:
            cm = re.search(r"class\s+([A-Za-z_][A-Za-z0-9_]*)", line)
            if cm:
                name = cm.group(1)
                flags = re.findall(r"\[([^\]]+)\]", line)
                visibility = None
                for f in flags:
                    if f.lower() == "public":
                        visibility = "public"
                    elif f.lower() == "private":
                        visibility = "private"
                file_to_expected[current_file].append(ExpectedItem(name=name, kind="class", visibility=visibility))
            continue

        if "def " in line:
            is_async = "async def" in line
            fm = re.search(r"def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*?)\)\s*(?:->\s*([^\s\[]]+))?", line)
            if fm:
                name = fm.group(1)
                raw_params = fm.group(2)
                ret = fm.group(3)
                params = []
                has_varargs = False
                has_varkw = False
                for p in [p.strip() for p in raw_params.split(",")] if raw_params.strip() else []:
                    if not p: continue
                    if p.startswith("**"): has_varkw = True; continue
                    if p.startswith("*"): has_varargs = True; continue
                    pname = p.split(":")[0].split("=")[0].strip()
                    if pname: params.append(pname)

                flags = re.findall(r"\[([^\]]+)\]", line)
                visibility = None
                wrapper_for = None
                for f in flags:
                    fl = f.lower()
                    if fl == "public": visibility = "public"
                    elif fl == "private": visibility = "private"
                    elif fl == "async": is_async = True
                    elif fl.startswith("wrapper="): wrapper_for = f.split("=", 1)[1].strip()

                file_to_expected[current_file].append(ExpectedItem(
                    name=name, kind="function", is_async=is_async, visibility=visibility,
                    params=params, has_varargs=has_varargs, has_varkw=has_varkw, returns=ret,
                    wrapper_for=wrapper_for, decorators=[]
                ))
            continue

    return dict(file_to_expected)

def _parse_tree_block_lines(text: str) -> List[Tuple[int, str, bool]]:
    out: List[Tuple[int, str, bool]] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if '── ' not in line: continue
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

def parse_structure_md(md_path: Path) -> List[str]:
    text = _read_text(md_path)
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
    seen = set(); ordered = []
    for c in paths:
        if c not in seen and c.endswith(".py"):
            ordered.append(c); seen.add(c)
    return ordered

class _CallCollector(ast.NodeVisitor):
    def __init__(self):
        self.calls: List[str] = []
    def visit_Call(self, node: ast.Call):
        name = None
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            name = node.func.attr
        if name: self.calls.append(name)
        self.generic_visit(node)

class _AstCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.methods: List[FoundMethod] = []
        self.classes: List[str] = []
        self.class_bases: Dict[str, List[str]] = {}
        self._class_stack: List[str] = []
        self._func_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.classes.append(node.name)
        bases = []
        for b in node.bases:
            try: s = ast.unparse(b)
            except Exception:
                s = getattr(b, 'id', None) or getattr(b, 'attr', None) or ""
            tail = s.split('.')[-1] if s else ""
            if tail: bases.append(tail)
        self.class_bases[node.name] = bases
        self._class_stack.append(node.name)
        self.generic_visit(node)
        self._class_stack.pop()

    def _visit_func(self, node: ast.AST, is_async: bool) -> Any:
        name = getattr(node, 'name', '?')
        if self._class_stack:
            scope = f"class:{self._class_stack[-1]}"; qname = f"{self._class_stack[-1]}.{name}"
        elif self._func_stack:
            scope = f"func:{self._func_stack[-1]}"; qname = f"{self._func_stack[-1]}.{name}"
        else:
            scope = "module"; qname = name

        args: ast.arguments = getattr(node, 'args', ast.arguments())
        params, has_varargs, has_varkw = _params_from_args(args)
        returns = _ann_to_str(getattr(node, 'returns', None))
        visibility = _visibility_from_name(name)

        decos = []
        for d in getattr(node, 'decorator_list', []):
            try: decos.append(ast.unparse(d))
            except Exception:
                if isinstance(d, ast.Name): decos.append(d.id)
                elif isinstance(d, ast.Attribute): decos.append(d.attr)

        cc = _CallCollector()
        for stmt in getattr(node, 'body', []):
            cc.visit(stmt)

        self.methods.append(FoundMethod(
            name=name, qualified_name=qname, is_async=is_async, lineno=getattr(node,'lineno',-1),
            scope=scope, params=params, has_varargs=has_varargs, has_varkw=has_varkw,
            returns=returns, visibility=visibility, decorators=decos, calls=cc.calls
        ))

        self._func_stack.append(name); self.generic_visit(node); self._func_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        return self._visit_func(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        return self._visit_func(node, is_async=True)

def collect_file_symbols(py_path: Path):
    code = _read_text(py_path)
    if not code:
        return [], [], {}, [f"Empty or unreadable: {py_path}"]
    try:
        tree = ast.parse(code)
    except Exception as e:
        return [], [], {}, [f"AST parse error in {py_path}: {e}"]
    collector = _AstCollector()
    collector.visit(tree)
    return collector.methods, collector.classes, collector.class_bases, []

def _compare_signature(expected: ExpectedItem, found_variants: List[FoundMethod], file_path: str) -> List[str]:
    issues: List[str] = []
    for fm in found_variants:
        exp_params = list(expected.params or [])
        f_params = list(fm.params)
        if f_params and f_params[0] in ("self", "cls") and (not exp_params or exp_params[0] != f_params[0]):
            f_params = f_params[1:]
        if exp_params and exp_params != f_params:
            issues.append(f"{file_path}:{fm.lineno} {fm.qualified_name} params mismatch expected={exp_params} found={f_params}")
        if expected.has_varargs is not None and expected.has_varargs != fm.has_varargs:
            issues.append(f"{file_path}:{fm.lineno} {fm.qualified_name} varargs (*) expected={expected.has_varargs} found={fm.has_varargs}")
        if expected.has_varkw is not None and expected.has_varkw != fm.has_varkw:
            issues.append(f"{file_path}:{fm.lineno} {fm.qualified_name} varkw (**) expected={expected.has_varkw} found={fm.has_varkw}")
        if expected.returns is not None:
            fr = fm.returns or "None"; er = expected.returns or "None"
            if fr != er:
                issues.append(f"{file_path}:{fm.lineno} {fm.qualified_name} return mismatch expected={er} found={fr}")
    return issues

def _check_visibility(expected: ExpectedItem, found_variants: List[FoundMethod], file_path: str) -> List[str]:
    issues: List[str] = []
    if expected.visibility is None: return issues
    req = expected.visibility; ok = False
    for fm in found_variants:
        if fm.visibility == req: ok = True
        else: issues.append(f"{file_path}:{fm.lineno} {fm.qualified_name} visibility mismatch expected={req} found={fm.visibility}")
    if ok: return []
    return issues

def _check_wrapper(expected: ExpectedItem, found_variants: List[FoundMethod], file_path: str) -> List[str]:
    issues: List[str] = []
    if not expected.wrapper_for: return issues
    target_tail = expected.wrapper_for.split(".")[-1]
    any_calls = False
    for fm in found_variants:
        calls_target = (target_tail in fm.calls) or any(d.endswith("wraps") for d in fm.decorators)
        if calls_target: any_calls = True
        else: issues.append(f"{file_path}:{fm.lineno} {fm.qualified_name} documented wrapper for '{expected.wrapper_for}' but no call to '{target_tail}' detected")
    if any_calls: return []
    return issues

INTERFACES = {
    "BaseExecutor": {"execute_trade", "cancel_order", "modify_order", "validate_order", "get_order_status"},
}

def _enforce_interfaces(class_bases: Dict[str, List[str]], methods: List[FoundMethod]) -> List[str]:
    required = set()
    for cls, bases in class_bases.items():
        for base in bases:
            if base in INTERFACES:
                required |= INTERFACES[base]
    if not required: return []
    present = {m.name for m in methods}
    return [f"def {n}()" for n in sorted(required - present)]

def generate_reports(root: Path, expected_map: Dict[str, List[ExpectedItem]], all_files_to_check: Optional[List[str]] = None, include_cross_file_duplicates: bool = True):
    reports: Dict[str, FileReport] = {}
    cross_counter: Dict[str, List[str]] = defaultdict(list)

    if not all_files_to_check:
        all_files_to_check = sorted(expected_map.keys())

    for rel_path in all_files_to_check:
        resolved = _resolve_filename_under_root(root, rel_path)
        fpath = resolved if resolved is not None else (root / rel_path)
        exists = fpath.exists()
        expected = expected_map.get(rel_path, [])

        found_methods, found_classes, class_bases, parse_errors = ([], [], {}, [])
        if exists and fpath.suffix == ".py":
            found_methods, found_classes, class_bases, parse_errors = collect_file_symbols(fpath)

        found_names = {m.name for m in found_methods} | set(found_classes)

        missing = []
        async_mismatches = []
        signature_mismatches = []
        visibility_mismatches = []
        wrapper_mismatches = []

        for exp in expected:
            if exp.kind == "class":
                if exp.name not in found_classes:
                    missing.append(f"class {exp.name}")
                continue
            variants = [m for m in found_methods if m.name == exp.name]
            if not variants:
                missing.append(f"def {exp.name}()"); continue
            if exp.is_async is not None and not any(m.is_async == exp.is_async for m in variants):
                async_mismatches.append(f"{str(fpath.relative_to(root))}::{exp.name} expected async={exp.is_async} but found {[m.is_async for m in variants]}")
            signature_mismatches.extend(_compare_signature(exp, variants, str(fpath.relative_to(root))))
            visibility_mismatches.extend(_check_visibility(exp, variants, str(fpath.relative_to(root))))
            wrapper_mismatches.extend(_check_wrapper(exp, variants, str(fpath.relative_to(root))))

        # Interface enforcement
        missing += _enforce_interfaces(class_bases, found_methods)

        scope_name_counts = Counter((m.scope, m.name) for m in found_methods)
        duplicates_in_file = [f"{scope}::{name} x{cnt}" for (scope, name), cnt in scope_name_counts.items() if cnt > 1]

        extras = []
        if expected:
            expected_names = {e.name for e in expected if e.kind == "function"} | {e.name for e in expected if e.kind == "class"}
            extras = sorted([n for n in found_names if n not in expected_names])

        for m in found_methods:
            if m.scope == "module":
                cross_counter[m.name].append(str(fpath.relative_to(root)))

        rep = FileReport(
            path=str(fpath.relative_to(root)) if exists else rel_path,
            exists=exists, expected=expected, found_methods=found_methods, found_classes=found_classes,
            class_bases=class_bases, missing=missing, async_mismatches=async_mismatches,
            signature_mismatches=signature_mismatches, visibility_mismatches=visibility_mismatches,
            wrapper_mismatches=wrapper_mismatches, duplicates_in_file=duplicates_in_file, extras=extras
        )
        for err in parse_errors:
            rep.missing.append(f"[ParseError] {err}")
        reports[rel_path] = rep

    cross_dups = {}
    if include_cross_file_duplicates:
        for name, files in cross_counter.items():
            uniq = sorted(set(files))
            if len(uniq) >= 2: cross_dups[name] = uniq

    return reports, cross_dups

def render_markdown(reports: Dict[str, FileReport], cross_dups: Dict[str, List[str]]) -> str:
    total_files = len(reports)
    missing_total = sum(len(r.missing) for r in reports.values())
    dup_files = sum(1 for r in reports.values() if r.duplicates_in_file)
    not_found = [r.path for r in reports.values() if not r.exists]

    md = []
    md.append("# ClaudeDex Verifier Report (PLUS v2)")
    md.append("")
    md.append("## Summary")
    md.append(f"- Files checked: **{total_files}**")
    md.append(f"- Total missing items: **{missing_total}**")
    md.append(f"- Files with in-file duplicates: **{dup_files}**")
    if not_found:
        md.append(f"- Files not found: **{len(not_found)}**")
        md.append("  - " + ", ".join(not_found))
    md.append("")
    if cross_dups:
        md.append("## Cross-file duplicate top-level functions")
        for name, files in sorted(cross_dups.items()):
            md.append(f"- `{name}`: " + ", ".join(files))
        md.append("")

    md.append("## Per-file details")
    for key in sorted(reports.keys()):
        r = reports[key]
        status = "✅ OK" if (not r.missing and not r.signature_mismatches and not r.visibility_mismatches and not r.wrapper_mismatches) and r.exists else "⚠️ Issues"
        if not r.exists: status = "❌ Missing file"
        md.append(f"### `{r.path}` — {status}")
        md.append(f"- Exists: **{r.exists}**")
        if r.expected:
            parts = []
            for e in r.expected:
                if e.kind == "class":
                    parts.append(f"class {e.name}" + (f" [{e.visibility}]" if e.visibility else ""))
                else:
                    sig = f"def {e.name}({', '.join(e.params or [])})"
                    if e.returns: sig += f" -> {e.returns}"
                    flags = []
                    if e.is_async: flags.append("async")
                    if e.visibility: flags.append(e.visibility)
                    if e.wrapper_for: flags.append(f"wrapper={e.wrapper_for}")
                    if flags: sig += " [" + ", ".join(flags) + "]"
                    parts.append(sig)
            md.append("- Expected items: " + "; ".join(parts))
        if r.class_bases:
            md.append(f"- Class bases: " + "; ".join(f"{k} -> {','.join(v)}" for k,v in r.class_bases.items()))
        if r.missing: md.append(f"- **Missing** ({len(r.missing)}): " + ", ".join(r.missing))
        if r.async_mismatches: md.append(f"- **Async mismatches** ({len(r.async_mismatches)}): " + "; ".join(r.async_mismatches))
        if r.signature_mismatches:
            md.append(f"- **Signature mismatches** ({len(r.signature_mismatches)}):")
            for s in r.signature_mismatches[:30]: md.append(f"  - {s}")
            if len(r.signature_mismatches) > 30: md.append("  - ...")
        if r.visibility_mismatches:
            md.append(f"- **Visibility mismatches** ({len(r.visibility_mismatches)}):")
            for s in r.visibility_mismatches[:30]: md.append(f"  - {s}")
            if len(r.visibility_mismatches) > 30: md.append("  - ...")
        if r.wrapper_mismatches:
            md.append(f"- **Wrapper issues** ({len(r.wrapper_mismatches)}):")
            for s in r.wrapper_mismatches[:30]: md.append(f"  - {s}")
            if len(r.wrapper_mismatches) > 30: md.append("  - ...")
        if r.duplicates_in_file: md.append(f"- **Duplicates** ({len(r.duplicates_in_file)}): " + "; ".join(r.duplicates_in_file))
        if r.extras: md.append(f"- Extras (not in expected list): " + ", ".join(r.extras[:30]) + (" ..." if len(r.extras) > 30 else ""))
        if r.found_methods or r.found_classes:
            sample_methods = ", ".join(sorted({m.qualified_name for m in r.found_methods})[:15])
            sample_classes = ", ".join(sorted(set(r.found_classes))[:10])
            if sample_classes: md.append(f"- Classes: {sample_classes}")
            if sample_methods: md.append(f"- Methods: {sample_methods}")
        md.append("")
    return "\n".join(md)

def main():
    parser = argparse.ArgumentParser(description="Verify ClaudeDex project against docs with deep checks.")
    parser.add_argument("--root", default=".", help="Project root (default: current dir)")
    parser.add_argument("--docs", nargs="+", default=["docs/*.md"], help="Docs; glob patterns allowed (PowerShell-safe)")
    parser.add_argument("--structure", default=None, help="corrected_file_structure.md path")
    parser.add_argument("--markdown-out", default="verifier_report.md", help="Markdown report path")
    parser.add_argument("--json-out", default="verifier_report.json", help="JSON report path")
    parser.add_argument("--fail-on-missing", action="store_true", help="Exit non-zero if any expected item is missing")
    args = parser.parse_args()

    root = Path(args.root).resolve()

    # Expand doc globs so PowerShell literal '*.md' works
    doc_paths: List[Path] = []
    for pat in args.docs:
        expanded = glob.glob(str(pat)) or glob.glob(str(root / pat))
        if not expanded:
            print(f"[WARN] Doc not found: {pat}")
            continue
        doc_paths += [Path(p) for p in expanded]

    expected_map: Dict[str, List[ExpectedItem]] = defaultdict(list)
    for p in doc_paths:
        part = parse_expectations_from_md(p)
        for k, v in part.items():
            expected_map[k].extend(v)

    files_to_check: Optional[List[str]] = None
    if args.structure:
        sp = Path(args.structure)
        if not sp.exists():
            sp2 = (root / args.structure)
            if sp2.exists(): sp = sp2
        if sp.exists():
            files_to_check = parse_structure_md(sp)

    reports, cross_dups = generate_reports(root, expected_map, files_to_check)

    payload = {
        "summary": {
            "files_checked": len(reports),
            "missing_total": sum(len(r.missing) for r in reports.values()),
            "signature_mismatches_total": sum(len(r.signature_mismatches) for r in reports.values()),
            "visibility_mismatches_total": sum(len(r.visibility_mismatches) for r in reports.values()),
            "wrapper_issues_total": sum(len(r.wrapper_mismatches) for r in reports.values()),
            "files_with_duplicates": sum(1 for r in reports.values() if r.duplicates_in_file),
            "files_not_found": [r.path for r in reports.values() if not r.exists],
        },
        "files": {
            path: {
                "exists": r.exists,
                "expected": [asdict(e) for e in r.expected],
                "found_classes": r.found_classes,
                "class_bases": r.class_bases,
                "found_methods": [asdict(m) for m in r.found_methods],
                "missing": r.missing,
                "async_mismatches": r.async_mismatches,
                "signature_mismatches": r.signature_mismatches,
                "visibility_mismatches": r.visibility_mismatches,
                "wrapper_mismatches": r.wrapper_mismatches,
                "duplicates_in_file": r.duplicates_in_file,
                "extras": r.extras,
            }
            for path, r in reports.items()
        },
        "cross_file_duplicates_top_level_functions": cross_dups,
    }

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md = render_markdown(reports, cross_dups)
    with open(args.markdown_out, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"[OK] Wrote Markdown report to: {args.markdown_out}")
    print(f"[OK] Wrote JSON report to: {args.json_out}")
    if args.fail_on_missing and payload["summary"]["missing_total"] > 0:
        raise SystemExit(2)

if __name__ == "__main__":
    main()
