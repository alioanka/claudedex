#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xref_symbol_db.py — Two-phase cross-reference checker with a persisted mini-DB.

Phase A: --build
  - Scans project, writes:
      file_tree.md
      symbol_db.json  (modules → classes/methods/functions with signatures + duplicates)
      symbol_db.md    (human-readable index; public + private)

Phase B: --verify
  - Loads symbol_db.json and verifies calls WITHOUT re-parsing referenced modules:
      missing_definition, missing_method, signature_mismatch, import_failure
  - Also reports duplicates (intra-module) and collisions (cross-module, informational)

Notes:
  - Cross-module collisions only consider PUBLIC names (no leading underscore) to avoid noise.
"""

import argparse, os, ast, json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any
from fnmatch import fnmatch

DEFAULT_EXCLUDE_DIR = {".venv","__pycache__","build","dist","logs","node_modules",".tox",".mypy_cache",".pytest_cache",".git"}
DEFAULT_EXCLUDE_GLOB = {"* copy.py"}

def is_py(path:str)->bool: return path.endswith(".py")

def to_module(root:str, file_path:str)->str:
    rel = os.path.relpath(file_path, root).replace(os.sep,"/")
    if rel.endswith("__init__.py"):
        rel = rel[:-len("__init__.py")]
    elif rel.endswith(".py"):
        rel = rel[:-3]
    rel = rel.strip("/")
    return rel.replace("/",".") if rel else ""

def read_file(p:str)->str:
    for enc in ("utf-8","latin-1"):
        try:
            with open(p,"r",encoding=enc) as f: return f.read()
        except Exception: pass
    return ""

def dotted_attr(node: ast.AST)->Optional[str]:
    parts=[]; cur=node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr); cur=cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id); parts.reverse()
        return ".".join(parts)
    return None

def arg_summary(node: ast.Call):
    pos = len(node.args)
    kwn = {k.arg for k in node.keywords if k.arg is not None}
    return {"positional":pos, "keyword":len(kwn)}, kwn

# ---------- Phase A: Build DB ----------

def scan_files(root:str, exclude_dirs:Set[str], exclude_glob:Set[str])->List[str]:
    files=[]
    for dp, dn, fns in os.walk(root):
        dn[:] = [d for d in dn if d not in exclude_dirs]
        for fn in fns:
            if any(fnmatch(fn, pat) for pat in exclude_glob): continue
            fp=os.path.join(dp,fn)
            if is_py(fp): files.append(fp)
    return files

def build_file_tree_md(root:str, files:List[str])->str:
    from collections import defaultdict
    root_abs=os.path.abspath(root)
    paths=[os.path.relpath(f, root_abs).replace("\\","/") for f in files]
    tree=defaultdict(list)
    for p in paths:
        tree[os.path.dirname(p)].append(os.path.basename(p))
    lines=["# File Tree\n","```\n", os.path.basename(root_abs)+"/\n"]
    def walk(d, prefix=""):
        subs = sorted({k for k in tree if os.path.dirname(k)==d and os.path.basename(k)!=""})
        files_here = sorted(tree.get(d, []))
        for f in files_here:
            lines.append(f"{prefix}├── {f}\n")
        for i, sub in enumerate(subs):
            lines.append(f"{prefix}└── {os.path.basename(sub)}/\n")
            walk(sub, prefix + "    ")
    walk("")
    lines.append("```\n")
    return "".join(lines)

def extract_sig(fn: ast.FunctionDef | ast.AsyncFunctionDef, is_method: bool) -> Dict[str, Any]:
    a=fn.args
    def names(L): return [x.arg for x in L]
    return dict(
        name=fn.name,
        params=names(a.args),
        kwonly=names(a.kwonlyargs),
        vararg=bool(a.vararg),
        kwarg=bool(a.kwarg),
        defaults=len(a.defaults),
        posonly=names(a.posonlyargs),
        is_method=is_method
    )

def build_symbol_db(root:str, files:List[str])->Dict[str,Any]:
    db = {}  # module -> { filepath, classes, functions, dups }
    # for cross-module collisions
    class_to_defs: Dict[str, List[Tuple[str,str,int]]] = {}
    func_to_defs: Dict[str, List[Tuple[str,str,int]]] = {}

    for fp in files:
        mod = to_module(root, fp)
        src = read_file(fp)
        if not src.strip(): continue
        try:
            tree=ast.parse(src, filename=fp)
        except Exception:
            continue
        classes={}
        functions={}
        dups={"functions":{}, "classes":{}, "methods":{}}  # name->[lines] / (cls,meth)->[lines]
        function_first: Dict[str,int]={}
        class_first: Dict[str,int]={}

        # pass 1: collect top-level defs + duplicates in module
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                if node.name in function_first:
                    dups["functions"].setdefault(node.name, sorted({function_first[node.name]})).append(node.lineno)
                else:
                    function_first[node.name]=node.lineno
                functions[node.name]=extract_sig(node, is_method=False)
            elif isinstance(node, ast.ClassDef):
                if node.name in class_first:
                    dups["classes"].setdefault(node.name, sorted({class_first[node.name]})).append(node.lineno)
                else:
                    class_first[node.name]=node.lineno
                # pass 2: methods + duplicate methods
                methods={}
                method_first: Dict[str,int]={}
                for b in node.body:
                    if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if b.name in method_first:
                            key=(node.name, b.name)
                            dups["methods"].setdefault(f"{key[0]}.{key[1]}", sorted({method_first[b.name]})).append(b.lineno)
                        else:
                            method_first[b.name]=b.lineno
                        methods[b.name]=extract_sig(b, is_method=True)
                classes[node.name]={"methods":methods,"has_init":("__init__" in methods)}

        db[mod]={
            "filepath": fp,
            "classes": classes,
            "functions": functions,
            "dups": dups
        }

        # collect public for collisions
        for cls, meta in classes.items():
            if not cls.startswith("_"):
                class_to_defs.setdefault(cls, []).append((mod, fp, class_first.get(cls, 1)))
        for fn, sig in functions.items():
            if not fn.startswith("_"):
                func_to_defs.setdefault(fn, []).append((mod, fp, function_first.get(fn, sig.get("lineno", 1))))

    # attach collisions to db root
    db["_collisions"] = {
        "classes": {k:v for k,v in class_to_defs.items() if len({m for (m,_,_) in v})>1},
        "functions": {k:v for k,v in func_to_defs.items() if len({m for (m,_,_) in v})>1},
    }
    return db

def write_symbol_md(root:str, db:Dict[str,Any], out_md:str):
    lines=["# Symbol Index (Mini DB)\n"]
    for module in sorted(k for k in db.keys() if k != "_collisions"):
        entry=db[module]
        lines.append(f"\n## {module}\n")
        lines.append(f"- File: `{os.path.relpath(entry['filepath'], root)}`\n")
        if entry["functions"]:
            lines.append("\n### Functions\n")
            for fn, sig in sorted(entry["functions"].items()):
                params = ", ".join(sig["params"])
                lines.append(f"- `{fn}({params})`\n")
        if entry["classes"]:
            lines.append("\n### Classes & Methods\n")
            for cls, meta in sorted(entry["classes"].items()):
                lines.append(f"- **{cls}**\n")
                for m, sig in sorted(meta["methods"].items()):
                    params = list(sig["params"])
                    if sig["is_method"] and params and params[0]=="self":
                        params=params[1:]
                    lines.append(f"  - `{m}({', '.join(params)})`\n")
    # collisions summary
    col = db.get("_collisions", {})
    lines.append("\n---\n## Collisions (public names)\n")
    lines.append(f"- Class name collisions: {len(col.get('classes',{}))}\n")
    lines.append(f"- Function name collisions: {len(col.get('functions',{}))}\n")
    with open(out_md,"w",encoding="utf-8") as f: f.write("".join(lines))

# ---------- Phase B: Verify using DB ----------

@dataclass
class Problem:
    kind: str
    message: str
    filepath: str
    module: str
    lineno: int
    context: Dict[str, Any]=field(default_factory=dict)

def import_table(tree: ast.AST, cur_module:str) -> Tuple[Dict[str,str], Dict[str,Tuple[str,str]], List[str]]:
    def resolve_relative(level:int, imported:str|None)->Optional[str]:
        if level<=0: return imported
        base = cur_module.split(".")
        if len(base)<level: return imported
        up = base[:-level]
        if imported: up = up + imported.split(".")
        return ".".join([p for p in up if p])

    imports={}
    imported_objects={}
    star=[]
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                imports[a.asname or a.name.split(".")[-1]] = a.name
        elif isinstance(n, ast.ImportFrom):
            abs_mod = resolve_relative(getattr(n,"level",0) or 0, n.module)
            if len(n.names)==1 and n.names[0].name=="*":
                star.append(abs_mod or "")
            else:
                for a in n.names:
                    imported_objects[a.asname or a.name]=(abs_mod or "", a.name)
    return imports, imported_objects, star

def ensure_in_db(db, module)->bool:
    return module in db and module != "_collisions"

def class_has_method(db, module, cls, method)->bool:
    return ensure_in_db(db,module) and cls in db[module]["classes"] and method in db[module]["classes"][cls]["methods"]

def synth_init_sig()->Dict[str,Any]:
    return dict(name="__init__", params=["self"], kwonly=[], vararg=False, kwarg=False, defaults=0, posonly=[], is_method=True)

def get_sig(db, module, fq)->Optional[Dict[str,Any]]:
    parts=fq.split(".")
    if len(parts)<2: return None
    mod=".".join(parts[:-1]); leaf=parts[-1]
    if not ensure_in_db(db, mod): return None
    entry=db[mod]
    if leaf in entry["functions"]:
        return entry["functions"][leaf]
    if len(parts)>=3:
        cls=parts[-2]; meth=parts[-1]
        if class_has_method(db, mod, cls, meth):
            return entry["classes"][cls]["methods"][meth]
    # constructor
    if leaf in entry["classes"]:
        if "__init__" in entry["classes"][leaf]["methods"]:
            return entry["classes"][leaf]["methods"]["__init__"]
        return synth_init_sig()
    return None

def check_sig(sig:Dict[str,Any], given_pos:int, given_kw:Set[str])->Optional[str]:
    params=list(sig["posonly"])+list(sig["params"])
    if sig["is_method"] and params and params[0] in ("self","cls"):
        params=params[1:]
    required = len(params) - sig["defaults"]
    allowed_pos = len(params)
    if not sig["vararg"] and given_pos>allowed_pos:
        return f"Too many positional args: {given_pos}>{allowed_pos}"
    if given_pos < min(required, allowed_pos):
        return f"Missing required positional args: required {required}, given {given_pos}"
    if not sig["kwarg"]:
        known=set(params)  # (kwonly can be included if desired)
        unknown = set(given_kw) - known
        if unknown:
            return f"Unknown keyword(s): {', '.join(sorted(unknown))}"
    return None

def verify_with_db(root:str, files:List[str], db:Dict[str,Any])->Tuple[List[Problem], Dict[str,int]]:
    problems: List[Problem]=[]
    skipped_external=0

    # report stored duplicates immediately (from DB)
    for module, entry in db.items():
        if module == "_collisions": continue
        fpath = entry["filepath"]
        for name, lines in entry.get("dups",{}).get("classes",{}).items():
            problems.append(Problem("duplicate_class_in_module", f"Duplicate class in module: {module}.{name}", fpath, module, lines[-1], {"lines": sorted(set(lines))}))
        for name, lines in entry.get("dups",{}).get("functions",{}).items():
            problems.append(Problem("duplicate_function_in_module", f"Duplicate function in module: {module}.{name}", fpath, module, lines[-1], {"lines": sorted(set(lines))}))
        for key, lines in entry.get("dups",{}).get("methods",{}).items():
            problems.append(Problem("duplicate_method_in_class", f"Duplicate method in class: {module}.{key}", fpath, module, lines[-1], {"lines": sorted(set(lines))}))

    # cross-module collisions (public)
    cols = db.get("_collisions", {})
    for name, defs in cols.get("classes", {}).items():
        problems.append(Problem("name_collision_class_across_modules", f"Class name '{name}' appears in multiple modules", "", "", 1, {"definitions": defs}))
    for name, defs in cols.get("functions", {}).items():
        problems.append(Problem("name_collision_function_across_modules", f"Function name '{name}' appears in multiple modules", "", "", 1, {"definitions": defs}))

    for fp in files:
        mod = to_module(root, fp)
        src=read_file(fp)
        if not src.strip(): continue
        try:
            tree=ast.parse(src, filename=fp)
        except Exception as e:
            problems.append(Problem("import_failure", f"AST parse error: {e}", fp, mod, 1))
            continue

        imports, imported_objects, star_imports = import_table(tree, mod)

        # import name shadowing
        entry = db.get(mod, {})
        if entry:
            local_funcs = set(entry.get("functions", {}).keys())
            local_classes = set(entry.get("classes", {}).keys())
            for alias in list(imports.keys()) + list(imported_objects.keys()):
                if alias in local_funcs or alias in local_classes:
                    problems.append(Problem("import_name_shadowing", f"Imported name shadows local definition: {alias}", fp, mod, 1, {"alias": alias}))

        class_stack: List[str]=[]

        def map_calls(body: ast.AST, local_types:Dict[str,str], class_attr_types:Dict[str,str]):
            for node in ast.walk(body):
                if isinstance(node, ast.Assign):
                    # var = ClassName(...)
                    if isinstance(node.value, ast.Call):
                        ctor=node.value.func
                        fq=None
                        if isinstance(ctor, ast.Name) and ctor.id in imported_objects:
                            m, name = imported_objects[ctor.id]
                            if m: fq=f"{m}.{name}"
                        elif isinstance(ctor, ast.Attribute):
                            dot=dotted_attr(ctor)
                            if dot:
                                head=dot.split(".")[0]
                                if head in imports:
                                    fq=dot.replace(head, imports[head], 1)
                        if fq:
                            for t in node.targets:
                                if isinstance(t, ast.Name):
                                    local_types[t.id]=fq
                    # self.attr = ClassName(...)
                    if any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id=="self" for t in node.targets):
                        if isinstance(node.value, ast.Call):
                            ctor=node.value.func
                            fq=None
                            if isinstance(ctor, ast.Name) and ctor.id in imported_objects:
                                m,name=imported_objects[ctor.id]
                                if m: fq=f"{m}.{name}"
                            elif isinstance(ctor, ast.Attribute):
                                dot=dotted_attr(ctor)
                                if dot:
                                    head=dot.split(".")[0]
                                    if head in imports:
                                        fq=dot.replace(head, imports[head], 1)
                            if fq:
                                for t in node.targets:
                                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id=="self":
                                        class_attr_types[t.attr]=fq

                if isinstance(node, ast.Call):
                    expr=None; kind="unknown"
                    if isinstance(node.func, ast.Attribute):
                        expr=dotted_attr(node.func) or "<attr>"
                        rootnode=node.func
                        while isinstance(rootnode, ast.Attribute):
                            rootnode=rootnode.value
                        if isinstance(rootnode, ast.Name) and rootnode.id=="self":
                            kind="self_attr_or_method"
                        else:
                            base=node.func.value
                            if isinstance(base, ast.Name):
                                if base.id in imports: kind="module"
                                elif base.id in local_types: kind="instance"
                    elif isinstance(node.func, ast.Name):
                        expr=node.func.id; kind="name"
                    else:
                        expr="<callable>"

                    argsum, kwnames = arg_summary(node)

                    target=None
                    if kind=="self_attr_or_method":
                        toks=expr.split(".")
                        if len(toks)==2 and toks[0]=="self":
                            if ensure_in_db(db, mod):
                                cls_name = class_stack[-1] if class_stack else None
                                if cls_name and class_has_method(db, mod, cls_name, toks[1]):
                                    target=f"{mod}.{cls_name}.{toks[1]}"
                        elif len(toks)>=3 and toks[0]=="self":
                            attr=toks[1]; meth=toks[-1]
                            fq = class_attr_types.get(attr)
                            if fq:
                                m=".".join(fq.split(".")[:-1]); c=fq.split(".")[-1]
                                target=f"{m}.{c}.{meth}"
                    elif kind=="module":
                        head=expr.split(".")[0]
                        m=imports.get(head)
                        if m:
                            tail=".".join(expr.split(".")[1:])
                            target=f"{m}.{tail}"
                    elif kind=="instance":
                        head=expr.split(".")[0]
                        fq=local_types.get(head)
                        if fq:
                            m=".".join(fq.split(".")[:-1]); c=fq.split(".")[-1]; meth=expr.split(".")[-1]
                            target=f"{m}.{c}.{meth}"
                    elif kind=="name":
                        if ensure_in_db(db, mod) and expr in db[mod]["functions"]:
                            target=f"{mod}.{expr}"
                        else:
                            ref=imported_objects.get(expr)
                            if ref:
                                m, name = ref
                                if m:
                                    target=f"{m}.{name}"

                    if not target:
                        continue
                    tgt_mod=".".join(target.split(".")[:-1])
                    if not ensure_in_db(db, tgt_mod):
                        skipped_external+=1
                        continue

                    sig=get_sig(db, tgt_mod, target)
                    if not sig:
                        parts=target.split(".")
                        if len(parts)>=3 and parts[-2] in db[tgt_mod]["classes"]:
                            problems.append(Problem("missing_method", f"Method not found: {target}", fp, mod, getattr(node,"lineno",1), {"expr":expr, "target":target}))
                        else:
                            problems.append(Problem("missing_definition", f"Missing definition: {target}", fp, mod, getattr(node,"lineno",1), {"expr":expr, "target":target}))
                        continue

                    err = check_sig(sig, argsum["positional"], set(kwnames))
                    if err:
                        problems.append(Problem("signature_mismatch", f"{target}: {err}", fp, mod, getattr(node,"lineno",1), {
                            "called_expr": expr, "args": argsum, "keywords": sorted(list(kwnames))
                        }))

        # walk classes/methods
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                class_stack.append(node.name)
                class_attr_types={}
                for b in node.body:
                    if isinstance(b,(ast.FunctionDef, ast.AsyncFunctionDef)):
                        local_types={}
                        map_calls(b, local_types, class_attr_types)
                class_stack.pop()
            elif isinstance(node,(ast.FunctionDef, ast.AsyncFunctionDef)):
                local_types={}
                class_attr_types={}
                map_calls(node, local_types, class_attr_types)

    stats={"skipped_external": skipped_external}
    return problems, stats

# ---------- CLI ----------

def parse_args():
    ap=argparse.ArgumentParser(description="Two-phase cross-reference checker (Mini-DB)")
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--build", action="store_true", help="Build file_tree.md, symbol_db.json, symbol_db.md")
    ap.add_argument("--verify", action="store_true", help="Verify using symbol_db.json")
    ap.add_argument("--md-out", default="xref_report.md", help="Markdown report (verify)")
    ap.add_argument("--json-out", default="xref_report.json", help="JSON report (verify)")
    ap.add_argument("--exclude", default="", help="Comma-separated dir excludes")
    ap.add_argument("--exclude-glob", default="", help="Comma-separated file globs to exclude")
    return ap.parse_args()

def main():
    args=parse_args()
    exclude=set([e.strip() for e in args.exclude.split(",") if e.strip()]) | DEFAULT_EXCLUDE_DIR
    exclude_glob=set([e.strip() for e in args.exclude_glob.split(",") if e.strip()]) | DEFAULT_EXCLUDE_GLOB
    root=os.path.abspath(args.root)

    if args.build:
        files = scan_files(root, exclude, exclude_glob)
        with open("file_tree.md","w",encoding="utf-8") as f:
            f.write(build_file_tree_md(root, files))
        db = build_symbol_db(root, files)
        with open("symbol_db.json","w",encoding="utf-8") as f: json.dump(db,f,indent=2,ensure_ascii=False)
        write_symbol_md(root, db, "symbol_db.md")
        print(f"[build] Files: {len(files)} | Modules: {len(db)-1} | Collisions(classes:{len(db.get('_collisions',{}).get('classes',{}))}, functions:{len(db.get('_collisions',{}).get('functions',{}))})")
        return

    if args.verify:
        with open("symbol_db.json","r",encoding="utf-8") as f:
            db=json.load(f)
        files = scan_files(root, exclude, exclude_glob)
        problems, stats = verify_with_db(root, files, db)

        by_kind={}
        for p in problems:
            by_kind.setdefault(p.kind, []).append(dict(
                message=p.message, filepath=p.filepath, module=p.module, lineno=p.lineno, context=p.context
            ))
        with open(args.json_out,"w",encoding="utf-8") as f:
            json.dump({"problems":by_kind, "stats":stats}, f, indent=2, ensure_ascii=False)

        lines=["# XRef Verification Report (Mini-DB)\n",
               f"- Root: `{root}`\n",
               f"- Files scanned: `{len(files)}`\n",
               f"- Skipped external: `{stats['skipped_external']}`\n\n"]
        sections = [
            ("Missing Definitions","missing_definition"),
            ("Missing Methods","missing_method"),
            ("Signature Mismatches","signature_mismatch"),
            ("Import Failures (Parse)","import_failure"),
            ("Import Name Shadowing","import_name_shadowing"),
            ("Duplicate Classes (in module)","duplicate_class_in_module"),
            ("Duplicate Functions (in module)","duplicate_function_in_module"),
            ("Duplicate Methods (in class)","duplicate_method_in_class"),
            ("Class Name Collisions (across modules)","name_collision_class_across_modules"),
            ("Function Name Collisions (across modules)","name_collision_function_across_modules"),
        ]
        total=0
        for title, key in sections:
            items=by_kind.get(key,[])
            total+=len(items)
            lines.append(f"\n## {title} ({len(items)})\n")
            if not items:
                lines.append("- None\n"); continue
            for it in items:
                lines.append(f"- **{it['message']}**  \n  File: `{it['filepath']}`:{it['lineno']}\n")
                ctx=it.get("context") or {}
                for k,v in ctx.items():
                    lines.append(f"  - {k}: `{v}`\n")
        lines.append(f"\n---\n**Total findings:** {total}\n")
        with open(args.md_out,"w",encoding="utf-8") as f: f.write("".join(lines))
        print(f"[verify] Problems: {total} (see {args.md_out} / {args.json_out})")
        return

    print("Nothing to do. Use --build or --verify.")

if __name__=="__main__":
    main()
