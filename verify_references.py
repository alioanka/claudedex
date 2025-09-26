#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_references.py — Cross-file reference & signature verifier (Patched + Duplicates/Collisions)

- Project-only checks by default (stdlib/3rd-party → skipped)
- Resolves: imports/aliases (incl. relative), self.attr via __init__ param hints AND direct constructors
- Treats `module.Class(...)` as a constructor and validates against Class.__init__
- Reports:
    missing_definition, missing_method, signature_mismatch,
    import_object_missing, import_failure, config_unresolved,
    duplicate_class_in_module, duplicate_function_in_module, duplicate_method_in_class,
    name_collision_class_across_modules, name_collision_function_across_modules,
    import_name_shadowing
- No "Did you mean" hints
- Skips duck-typed/unknown receivers (logger/queue/etc.)
- Optional strict config dotted-ref checks
"""

import argparse, ast, json, os, re
from fnmatch import fnmatch
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set, Any

SigInfo = namedtuple("SigInfo", ["name","params","kwonly","vararg","kwarg","defaults","posonly","is_method"])

@dataclass
class DefInfo:
    kind: str
    module: str
    qualname: str
    filepath: str
    lineno: int
    signature: Optional[SigInfo]=None

@dataclass
class ClassInfo:
    methods: Dict[str, DefInfo] = field(default_factory=dict)
    attr_types: Dict[str, str] = field(default_factory=dict)  # self.attr -> fq class

@dataclass
class ModuleInfo:
    module: str
    filepath: str
    functions: Dict[str, DefInfo] = field(default_factory=dict)
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    imports: Dict[str, str] = field(default_factory=dict)           # alias -> module
    imported_objects: Dict[str, str] = field(default_factory=dict)  # alias -> "module:Name"
    star_imports: List[str] = field(default_factory=list)
    var_types: Dict[str, str] = field(default_factory=dict)         # local var -> fq class
    # Duplicate bookkeeping (intra-module)
    function_def_linenos: Dict[str, int] = field(default_factory=dict)
    class_def_linenos: Dict[str, int] = field(default_factory=dict)
    dup_functions: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    dup_classes: Dict[str, List[int]] = field(default_factory=lambda: defaultdict(list))
    dup_methods: Dict[Tuple[str,str], List[int]] = field(default_factory=lambda: defaultdict(list))
    errors: List[str] = field(default_factory=list)

@dataclass
class CallSite:
    filepath: str
    module: str
    lineno: int
    col: int
    expr: str
    call_kind: str
    resolved_target: Optional[str]
    arg_summary: Dict[str,int]
    keywords: Set[str]
    enclosing_class: Optional[str]
    enclosing_func: Optional[str]

@dataclass
class Problem:
    kind: str
    message: str
    filepath: str
    module: str
    lineno: int
    context: Dict[str, Any]=field(default_factory=dict)

CONFIG_EXTS = (".json",".env",".ini")
DEFAULT_DIR_EXCLUDES = {".venv","__pycache__","build","dist","logs","node_modules",".tox",".mypy_cache",".pytest_cache",".git"}
DEFAULT_GLOB_EXCLUDES = {"* copy.py"}
LIKELY_DUCK_NAMES = {"logger","log","queue","producer","consumer","publisher","subscriber","handler","client"}

def is_py(path:str)->bool: return path.endswith(".py")
def is_cfg(path:str)->bool: return path.endswith(CONFIG_EXTS)

def to_module(root:str, file_path:str)->str:
    rel = os.path.relpath(file_path, root).replace(os.sep,"/")
    if rel.endswith("__init__.py"):
        rel = rel[:-len("__init__.py")]
    elif rel.endswith(".py"):
        rel = rel[:-3]
    rel = rel.strip("/")
    return rel.replace("/",".") if rel else ""

def file_for_module(root:str, module:str)->Optional[str]:
    base = os.path.join(root, *module.split("."))
    for cand in (base+".py", os.path.join(base,"__init__.py")):
        if os.path.isfile(cand):
            return cand
    return None

def extract_sig(fn: ast.FunctionDef | ast.AsyncFunctionDef, is_method: bool) -> SigInfo:
    a = fn.args
    def names(L): return [x.arg for x in L]
    return SigInfo(
        name=fn.name, params=names(a.args), kwonly=names(a.kwonlyargs),
        vararg=a.vararg.arg if a.vararg else None,
        kwarg=a.kwarg.arg if a.kwarg else None,
        defaults=len(a.defaults), posonly=names(a.posonlyargs),
        is_method=is_method
    )

def arg_summary(node: ast.Call)->Tuple[Dict[str,int], Set[str]]:
    pos = len(node.args)
    kw = {k.arg for k in node.keywords if k.arg is not None}
    return {"positional":pos, "keyword":len(kw)}, kw

def dotted_attr(node: ast.AST)->Optional[str]:
    parts=[]; cur=node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr); cur=cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id); parts.reverse()
        return ".".join(parts)
    return None

def read_file(p:str)->str:
    for enc in ("utf-8","latin-1"):
        try:
            with open(p,"r",encoding=enc) as f: return f.read()
        except Exception: pass
    return ""

def resolve_relative(module:str, level:int, imported:str|None)->Optional[str]:
    if level<=0: return imported
    base = module.split(".")
    if len(base)<level: return imported
    up = base[:-level]
    if imported: up = up + imported.split(".")
    return ".".join([p for p in up if p])

class PyIndexer(ast.NodeVisitor):
    def __init__(self, root:str, filepath:str, module:str):
        self.root=root; self.filepath=filepath; self.module=module
        self.info=ModuleInfo(module=module, filepath=filepath)
        self.class_stack: List[str]=[]

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            mod = alias.name
            asname = alias.asname or mod.split(".")[-1]
            self.info.imports[asname] = mod

    def visit_ImportFrom(self, node: ast.ImportFrom):
        abs_mod = resolve_relative(self.module, getattr(node,"level",0) or 0, node.module)
        if abs_mod is None: abs_mod = ""
        if len(node.names)==1 and node.names[0].name=="*":
            self.info.star_imports.append(abs_mod); return
        for alias in node.names:
            name = alias.name
            asname = alias.asname or name
            self.info.imported_objects[asname] = f"{abs_mod}:{name}"

    def visit_ClassDef(self, node: ast.ClassDef):
        # duplicate class in module?
        if node.name in self.info.class_def_linenos:
            self.info.dup_classes[node.name].extend([self.info.class_def_linenos[node.name], node.lineno])
        else:
            self.info.class_def_linenos[node.name] = node.lineno

        self.class_stack.append(node.name)
        ci = self.info.classes.setdefault(node.name, ClassInfo())

        for body in node.body:
            if isinstance(body,(ast.FunctionDef, ast.AsyncFunctionDef)):
                # duplicate method in class?
                if body.name in ci.methods:
                    self.info.dup_methods[(node.name, body.name)].extend([ci.methods[body.name].lineno, body.lineno])
                sig = extract_sig(body, is_method=True)
                d = DefInfo(kind="method", module=self.module,
                            qualname=f"{self.module}.{node.name}.{body.name}",
                            filepath=self.filepath, lineno=body.lineno, signature=sig)
                ci.methods[body.name]=d

        # __init__ param annotations + self.attr = param
        for body in node.body:
            if isinstance(body,(ast.FunctionDef, ast.AsyncFunctionDef)) and body.name=="__init__":
                ann: Dict[str,str]={}
                for arg in body.args.args:
                    if arg.arg=="self": continue
                    if arg.annotation:
                        if isinstance(arg.annotation, ast.Name):
                            ann[arg.arg]=arg.annotation.id
                        elif isinstance(arg.annotation, ast.Attribute):
                            s = dotted_attr(arg.annotation)
                            if s: ann[arg.arg]=s.split(".")[-1]
                for stmt in body.body:
                    if isinstance(stmt, ast.Assign):
                        for tgt in stmt.targets:
                            if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id=="self":
                                if isinstance(stmt.value, ast.Name):
                                    param = stmt.value.id
                                    if param in ann:
                                        type_name = ann[param]
                                        fq = None
                                        if type_name in self.info.imported_objects:
                                            fq = self.info.imported_objects[type_name].replace(":",".")
                                        elif type_name in self.info.classes:
                                            fq = f"{self.module}.{type_name}"
                                        if fq:
                                            ci.attr_types[tgt.attr]=fq
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # duplicate function in module?
        if node.name in self.info.function_def_linenos:
            self.info.dup_functions[node.name].extend([self.info.function_def_linenos[node.name], node.lineno])
        else:
            self.info.function_def_linenos[node.name] = node.lineno

        sig = extract_sig(node, is_method=False)
        self.info.functions[node.name]=DefInfo("function", self.module, f"{self.module}.{node.name}", self.filepath, node.lineno, sig)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if node.name in self.info.function_def_linenos:
            self.info.dup_functions[node.name].extend([self.info.function_def_linenos[node.name], node.lineno])
        else:
            self.info.function_def_linenos[node.name] = node.lineno
        sig = extract_sig(node, is_method=False)
        self.info.functions[node.name]=DefInfo("function", self.module, f"{self.module}.{node.name}", self.filepath, node.lineno, sig)

    def visit_Assign(self, node: ast.Assign):
        # local: var = ClassName(...) or var = mod.ClassName(...)
        try:
            if isinstance(node.value, ast.Call):
                ctor=node.value.func
                dotted=None
                if isinstance(ctor, ast.Name):
                    name=ctor.id
                    if name in self.info.imported_objects:
                        dotted=self.info.imported_objects[name].replace(":",".")
                elif isinstance(ctor, ast.Attribute):
                    dotted=dotted_attr(ctor)
                    if dotted:
                        first=dotted.split(".")[0]
                        if first in self.info.imports:
                            dotted=dotted.replace(first, self.info.imports[first], 1)
                if dotted:
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            self.info.var_types[t.id]=dotted
        except Exception:
            pass

        # self.attr = ClassName(...) direct constructor inference
        try:
            if any(isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id=="self"
                   for t in getattr(node,"targets", [])) and isinstance(node.value, ast.Call):
                ctor = node.value.func
                dotted=None
                if isinstance(ctor, ast.Name):
                    if ctor.id in self.info.imported_objects:
                        dotted=self.info.imported_objects[ctor.id].replace(":", ".")
                elif isinstance(ctor, ast.Attribute):
                    dotted=dotted_attr(ctor)
                    if dotted:
                        first=dotted.split(".")[0]
                        if first in self.info.imports:
                            dotted=dotted.replace(first, self.info.imports[first], 1)
                if dotted and self.class_stack:
                    clsname=self.class_stack[-1]
                    ci=self.info.classes.setdefault(clsname, ClassInfo())
                    for t in node.targets:
                        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id=="self":
                            ci.attr_types[t.attr]=dotted
        except Exception:
            pass

class CallCollector(ast.NodeVisitor):
    def __init__(self, mi: ModuleInfo):
        self.mi=mi
        self.class_stack: List[str]=[]
        self.func_stack: List[str]=[]
        self.calls: List[CallSite]=[]

    def visit_ClassDef(self, node: ast.ClassDef):
        self.class_stack.append(node.name); self.generic_visit(node); self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.func_stack.append(node.name); self.generic_visit(node); self.func_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.func_stack.append(node.name); self.generic_visit(node); self.func_stack.pop()

    def visit_Call(self, node: ast.Call):
        expr=None; kind="unknown"
        if isinstance(node.func, ast.Attribute):
            expr=dotted_attr(node.func) or "<attr>"
            root=node.func
            while isinstance(root, ast.Attribute):
                root=root.value
            if isinstance(root, ast.Name) and root.id=="self":
                kind="self_attr_or_method"
            else:
                base=node.func.value
                if isinstance(base, ast.Name):
                    if base.id in self.mi.imports: kind="module"
                    elif base.id in self.mi.var_types: kind="instance"
        elif isinstance(node.func, ast.Name):
            expr=node.func.id; kind="name"
        else:
            expr="<callable>"
        argsum, kwn = arg_summary(node)
        self.calls.append(CallSite(
            filepath=self.mi.filepath, module=self.mi.module, lineno=node.lineno, col=node.col_offset,
            expr=expr, call_kind=kind, resolved_target=None, arg_summary=argsum, keywords=kwn,
            enclosing_class=self.class_stack[-1] if self.class_stack else None,
            enclosing_func=self.func_stack[-1] if self.func_stack else None
        ))
        self.generic_visit(node)

@dataclass
class CollMap:
    # cross-module collisions (public names only)
    class_to_defs: Dict[str, List[Tuple[str,str,int]]] = field(default_factory=lambda: defaultdict(list))  # name -> [(module, file, line)]
    func_to_defs: Dict[str, List[Tuple[str,str,int]]] = field(default_factory=lambda: defaultdict(list))

class Analyzer:
    def __init__(self, root:str, exclude:Set[str], exclude_glob:Set[str], strict_config:bool, project_only:bool=True):
        self.root=os.path.abspath(root)
        self.exclude=exclude | DEFAULT_DIR_EXCLUDES
        self.exclude_glob=exclude_glob | DEFAULT_GLOB_EXCLUDES
        self.strict_config=strict_config
        self.project_only=project_only
        self.modules: Dict[str, ModuleInfo]={}
        self.calls: List[CallSite]=[]
        self.problems: List[Problem]=[]
        self.skipped_external=0
        self.skipped_unknown=0
        self.coll = CollMap()

    def _is_glob_excluded(self, path:str)->bool:
        name=os.path.basename(path)
        return any(fnmatch(name, pat) for pat in self.exclude_glob)

    def walk(self)->Tuple[List[str],List[str]]:
        py,cfg=[],[]
        for dp, dn, fns in os.walk(self.root):
            dn[:] = [d for d in dn if d not in self.exclude]
            for fn in fns:
                fp=os.path.join(dp,fn)
                if self._is_glob_excluded(fp): continue
                if is_py(fp): py.append(fp)
                elif is_cfg(fp): cfg.append(fp)
        return py,cfg

    def index(self, files: List[str]):
        # parse & index
        for fp in files:
            mod = to_module(self.root, fp)
            src = read_file(fp)
            if not src.strip(): continue
            try:
                tree = ast.parse(src, filename=fp)
            except Exception as e:
                self.problems.append(Problem("import_failure", f"AST parse error: {e}", fp, mod, 1)); continue
            idx = PyIndexer(self.root, fp, mod); idx.visit(tree)
            self.modules[mod]=idx.info

            # collect for collisions (public names only)
            for cls, _ci in idx.info.classes.items():
                if not cls.startswith("_"):
                    self.coll.class_to_defs[cls].append((mod, fp, idx.info.class_def_linenos.get(cls, 1)))
            for fn, di in idx.info.functions.items():
                if not fn.startswith("_"):
                    self.coll.func_to_defs[fn].append((mod, fp, di.lineno))

        # callsites
        for mod, mi in self.modules.items():
            src = read_file(mi.filepath)
            if not src.strip(): continue
            try:
                tree = ast.parse(src, filename=mi.filepath)
            except Exception: continue
            cc=CallCollector(mi); cc.visit(tree); self.calls.extend(cc.calls)

        # Validate "from x import Y" when x is project module
        for mod, mi in self.modules.items():
            # import name shadowing (alias clashes with local)
            for alias in list(mi.imports.keys()) + list(mi.imported_objects.keys()):
                if alias in mi.functions or alias in mi.classes:
                    self.problems.append(Problem(
                        "import_name_shadowing",
                        f"Imported name shadows local definition: {alias}",
                        mi.filepath, mi.module, 1, {"alias": alias}
                    ))
            for alias, ref in mi.imported_objects.items():
                src_mod, obj = ref.split(":",1)
                if not src_mod:  # relative (e.g., from . import Y) — defer
                    continue
                proj_fp = file_for_module(self.root, src_mod)
                if not proj_fp:   # external → ignore
                    continue
                if src_mod not in self.modules:
                    s = read_file(proj_fp)
                    try:
                        tree = ast.parse(s, filename=proj_fp)
                        idx = PyIndexer(self.root, proj_fp, src_mod); idx.visit(tree)
                        self.modules[src_mod]=idx.info
                    except Exception:
                        continue
                mi_src = self.modules.get(src_mod)
                if not mi_src: continue
                if obj not in mi_src.functions and obj not in mi_src.classes:
                    self.problems.append(Problem(
                        "import_object_missing",
                        f"from {src_mod} import {obj} -> {obj} not found in project module",
                        mi.filepath, mi.module, 1, {"imported_as": alias}
                    ))

        # Record duplicates (intra-module)
        for mod, mi in self.modules.items():
            for name, lines in mi.dup_classes.items():
                if lines:
                    self.problems.append(Problem(
                        "duplicate_class_in_module",
                        f"Duplicate class in module: {mod}.{name}",
                        mi.filepath, mod, lines[-1], {"lines": sorted(set(lines))}
                    ))
            for name, lines in mi.dup_functions.items():
                if lines:
                    self.problems.append(Problem(
                        "duplicate_function_in_module",
                        f"Duplicate function in module: {mod}.{name}",
                        mi.filepath, mod, lines[-1], {"lines": sorted(set(lines))}
                    ))
            for (cls, meth), lines in mi.dup_methods.items():
                if lines:
                    self.problems.append(Problem(
                        "duplicate_method_in_class",
                        f"Duplicate method in class: {mod}.{cls}.{meth}",
                        mi.filepath, mod, lines[-1], {"lines": sorted(set(lines))}
                    ))

        # Record collisions (cross-module, public only; informational)
        for name, defs in self.coll.class_to_defs.items():
            mods = {m for (m,_,_) in defs}
            if len(mods) > 1:
                self.problems.append(Problem(
                    "name_collision_class_across_modules",
                    f"Class name '{name}' appears in multiple modules",
                    "", "", 1, {"definitions": defs}
                ))
        for name, defs in self.coll.func_to_defs.items():
            mods = {m for (m,_,_) in defs}
            if len(mods) > 1:
                self.problems.append(Problem(
                    "name_collision_function_across_modules",
                    f"Function name '{name}' appears in multiple modules",
                    "", "", 1, {"definitions": defs}
                ))

    def ensure_indexed(self, module:str)->bool:
        if module in self.modules: return True
        f=file_for_module(self.root, module)
        if not f: return False
        src=read_file(f)
        try:
            tree=ast.parse(src, filename=f)
            idx=PyIndexer(self.root, f, module); idx.visit(tree)
            self.modules[module]=idx.info
            return True
        except Exception:
            return False

    def locate(self, fq: str) -> Optional[DefInfo]:
        parts = fq.split(".")
        if len(parts) < 2: return None
        module = ".".join(parts[:-1]); leaf = parts[-1]
        mi = self.modules.get(module)
        if not mi: return None
        if leaf in mi.functions:
            return mi.functions[leaf]
        if len(parts) >= 3:
            cls = parts[-2]; meth = parts[-1]
            ci = mi.classes.get(cls)
            if ci and meth in ci.methods:
                return ci.methods[meth]
        ci = mi.classes.get(leaf)
        if ci:
            if "__init__" in ci.methods:
                return ci.methods["__init__"]
            return DefInfo("method", module, f"{module}.{leaf}.__init__", mi.filepath, 1,
                           SigInfo("__init__", ["self"], [], None, None, 0, [], True))
        return None

    def check_sig(self, defsig: SigInfo, call: CallSite)->Optional[str]:
        params=list(defsig.posonly)+list(defsig.params)
        if defsig.is_method and params and params[0] in ("self","cls"):
            params=params[1:]
        required = len(params) - defsig.defaults
        allowed_pos = len(params)
        given_pos = call.arg_summary["positional"]
        if defsig.vararg is None and given_pos>allowed_pos:
            return f"Too many positional args: given {given_pos}, allowed {allowed_pos}"
        if given_pos < min(required, allowed_pos):
            return f"Missing required positional args: required {required}, given {given_pos}"
        if defsig.kwarg is None:
            known=set(params+list(defsig.kwonly))
            unknown=call.keywords - known
            if unknown:
                return f"Unknown keyword(s): {', '.join(sorted(unknown))}"
        return None

    def resolve_self_chain(self, call: CallSite)->Optional[str]:
        mi=self.modules.get(call.module)
        if not mi or not call.enclosing_class: return None
        ci=mi.classes.get(call.enclosing_class)
        if not ci: return None
        if call.expr.count(".")==1 and call.expr.startswith("self."):
            meth=call.expr.split(".")[-1]
            if meth in ci.methods:
                return f"{call.module}.{call.enclosing_class}.{meth}"
            return None
        tokens=call.expr.split(".")
        if len(tokens)>=3 and tokens[0]=="self":
            attr=tokens[1]; meth=tokens[-1]
            fq_class = ci.attr_types.get(attr)
            if fq_class:
                mod=".".join(fq_class.split(".")[:-1]); cls=fq_class.split(".")[-1]
                return f"{mod}.{cls}.{meth}"
        return None

    def locate_def(self, fq:str)->Optional[DefInfo]:
        mod = ".".join(fq.split(".")[:-1])
        if mod and mod not in self.modules:
            if not self.ensure_indexed(mod):
                return None
        return self.locate(fq)

    def resolve_call(self, c: CallSite):
        mi=self.modules.get(c.module)
        if not mi: return
        target=None

        if c.call_kind in {"instance","name"}:
            head = c.expr.split(".")[0]
            if head in LIKELY_DUCK_NAMES:
                self.skipped_unknown += 1
                return

        if c.call_kind=="self_attr_or_method":
            target=self.resolve_self_chain(c)
            if not target:
                self.skipped_unknown += 1
                return
        elif c.call_kind=="module":
            first=c.expr.split(".")[0]
            modpath=mi.imports.get(first)
            if not modpath:
                self.skipped_unknown += 1
                return
            tail=".".join(c.expr.split(".")[1:])
            target=f"{modpath}.{tail}"
        elif c.call_kind=="instance":
            first=c.expr.split(".")[0]
            cls_fq=mi.var_types.get(first)
            if not cls_fq:
                self.skipped_unknown += 1
                return
            mod=".".join(cls_fq.split(".")[:-1]); cls=cls_fq.split(".")[-1]; meth=c.expr.split(".")[-1]
            target=f"{mod}.{cls}.{meth}"
        elif c.call_kind=="name":
            name=c.expr
            if name in mi.functions:
                target=f"{mi.module}.{name}"
            else:
                imp=mi.imported_objects.get(name)
                if imp:
                    target=imp.replace(":",".")
                else:
                    for star in mi.star_imports:
                        cand=f"{star}.{name}"
                        if self.locate_def(cand):
                            target=cand; break
                    if not target:
                        self.skipped_unknown += 1
                        return
        else:
            self.skipped_unknown += 1
            return

        c.resolved_target=target
        if not target: return
        tgt_mod=".".join(target.split(".")[:-1])

        if self.project_only and not (tgt_mod in self.modules or file_for_module(self.root, tgt_mod)):
            self.skipped_external += 1
            return

        di=self.locate_def(target)
        if not di:
            mi_tgt = self.modules.get(tgt_mod)
            if mi_tgt:
                parts = target.split(".")
                if len(parts) >= 3:
                    cls = parts[-2]; meth = parts[-1]
                    ci = mi_tgt.classes.get(cls)
                    if ci and meth not in ci.methods:
                        self.problems.append(Problem("missing_method",
                                                     f"Method not found: {tgt_mod}.{cls}.{meth}",
                                                     c.filepath, c.module, c.lineno,
                                                     {"expr": c.expr, "target": target}))
                        return
            self.problems.append(Problem("missing_definition",
                                         f"Missing definition: {target}",
                                         c.filepath, c.module, c.lineno,
                                         {"expr": c.expr, "target": target}))
            return

        err=self.check_sig(di.signature, c)
        if err:
            self.problems.append(Problem("signature_mismatch",
                                         f"{di.qualname}: {err}",
                                         c.filepath, c.module, c.lineno,
                                         {"called_expr":c.expr,
                                          "declared":str(di.signature),
                                          "args":{"positional":c.arg_summary["positional"],
                                                  "keywords":sorted(list(c.keywords))}}))

    def scan_configs(self, cfg_files: List[str], strict:bool):
        if not strict: return
        pat=re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)\b")
        for fp in cfg_files:
            text=read_file(fp)
            for m in pat.finditer(text):
                token=m.group(1)
                parts=token.split(".")
                resolved=False
                for k in range(len(parts)-1,0,-1):
                    mod=".".join(parts[:k]); sym=".".join(parts[k:])
                    if not self.ensure_indexed(mod): 
                        continue
                    mi=self.modules.get(mod)
                    if not mi: continue
                    if sym in mi.functions or sym in mi.classes:
                        resolved=True; break
                    if "." in sym:
                        cls,meth=sym.split(".",1)
                        if cls in mi.classes and meth in mi.classes[cls].methods:
                            resolved=True; break
                if not resolved:
                    self.problems.append(Problem("config_unresolved",
                                                 f"Unresolved dotted reference in config: {token}",
                                                 fp, "", 1, {"token":token}))

    def analyze(self):
        for c in self.calls:
            self.resolve_call(c)

    def report_data(self)->Dict[str,List[Dict[str,Any]]]:
        out=defaultdict(list)
        for p in self.problems:
            out[p.kind].append({
                "message":p.message,"filepath":p.filepath,"module":p.module,
                "lineno":p.lineno,"context":p.context
            })
        out["_stats"]=[{
            "skipped_external": self.skipped_external,
            "skipped_unknown": self.skipped_unknown,
            "modules_indexed": len(self.modules),
            "calls_analyzed": len(self.calls),
        }]
        return out

    def report_md(self, res:Dict[str,List[Dict]] )->str:
        def sec(t): return f"\n## {t}\n"
        lines=[]
        lines.append("# Cross-file Reference & Signature Verification Report\n")
        lines.append(f"- Root: `{self.root}`")
        lines.append(f"- Python modules indexed: `{len(self.modules)}`")
        lines.append(f"- Calls analyzed: `{len(self.calls)}`")
        lines.append(f"- Skipped (external): `{self.skipped_external}`")
        lines.append(f"- Skipped (unknown/duck-typed): `{self.skipped_unknown}`\n")
        sections=[
            ("Missing Definitions","missing_definition"),
            ("Missing Methods","missing_method"),
            ("Signature Mismatches","signature_mismatch"),
            ("Import Failures (Parse)","import_failure"),
            ("Import Object Missing","import_object_missing"),
            ("Import Name Shadowing","import_name_shadowing"),
            ("Duplicate Classes (in module)","duplicate_class_in_module"),
            ("Duplicate Functions (in module)","duplicate_function_in_module"),
            ("Duplicate Methods (in class)","duplicate_method_in_class"),
            ("Class Name Collisions (across modules)","name_collision_class_across_modules"),
            ("Function Name Collisions (across modules)","name_collision_function_across_modules"),
            ("Unresolved Config References","config_unresolved"),
        ]
        total=0
        for title,key in sections:
            items=res.get(key,[]); total+=len(items)
            lines.append(sec(f"{title} ({len(items)})"))
            if not items: lines.append("- None\n"); continue
            for it in items:
                lines.append(f"- **{it['message']}**  \n  File: `{it['filepath']}`:{it['lineno']}")
                ctx=it.get("context") or {}
                for k,v in ctx.items():
                    lines.append(f"  - {k}: `{v}`")
            lines.append("")
        lines.append(f"\n---\n**Total findings:** {total}\n")
        lines.append("> Notes:\n> - External/stdlib/3rd-party calls are skipped.\n> - Duck-typed/unknown receivers are skipped to avoid noise.\n> - Cross-module name collisions are informational.\n")
        return "\n".join(lines)

def parse_args():
    ap=argparse.ArgumentParser(description="Cross-file reference & signature verifier")
    ap.add_argument("--root", default=".", help="Project root")
    ap.add_argument("--exclude", default="", help="Comma-separated directories to exclude")
    ap.add_argument("--exclude-glob", default="", help="Comma-separated filename globs to exclude")
    ap.add_argument("--md-out", default="ref_report.md", help="Markdown output")
    ap.add_argument("--json-out", default="ref_report.json", help="JSON output")
    ap.add_argument("--strict-config", action="store_true", help="Flag unresolved dotted refs in configs")
    ap.add_argument("--allow-external", action="store_true", help="If set, DO check external modules (not recommended)")
    return ap.parse_args()

def main():
    args=parse_args()
    exclude=set([e.strip() for e in args.exclude.split(",") if e.strip()])
    exclude_glob=set([e.strip() for e in args.exclude_glob.split(",") if e.strip()])
    az=Analyzer(args.root, exclude, exclude_glob, strict_config=args.strict_config, project_only=(not args.allow_external))
    py,cfg=az.walk()
    az.index(py)
    az.scan_configs(cfg, args.strict_config)
    az.analyze()
    res=az.report_data()
    with open(args.json_out,"w",encoding="utf-8") as f: json.dump(res,f,indent=2,ensure_ascii=False)
    with open(args.md_out,"w",encoding="utf-8") as f: f.write(az.report_md(res))
    total=sum(len(v) for k,v in res.items() if not k.startswith("_"))
    print(f"[verify_references] Analyzed {len(py)} Python files and {len(cfg)} config files.")
    print(f"[verify_references] Modules indexed: {len(az.modules)} | Calls analyzed: {len(az.calls)}")
    print(f"[verify_references] Skipped external: {az.skipped_external} | Skipped unknown: {az.skipped_unknown}")
    print(f"[verify_references] Findings: {total} (see {args.md_out} / {args.json_out})")

if __name__=="__main__":
    main()
