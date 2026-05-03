"""Deterministic structural extraction from source code using tree-sitter. Outputs nodes+edges dicts."""
from __future__ import annotations
import importlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any
from .cache import load_cached, save_cached


def _make_id(*parts: str) -> str:
    """Build a stable node ID from one or more name parts."""
    combined = "_".join(p.strip("_.") for p in parts if p)
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", combined)
    return cleaned.strip("_").lower()


def _file_stem(path: Path) -> str:
    """Return a stem qualified with the parent directory name to avoid ID collisions
    when multiple files share the same filename in different directories (#550)."""
    parent = path.parent.name
    if parent and parent not in (".", ""):
        return f"{parent}.{path.stem}"
    return path.stem


_TSCONFIG_ALIAS_CACHE: dict[str, dict[str, str]] = {}


def _load_tsconfig_aliases(start_dir: Path) -> dict[str, str]:
    """Walk up from start_dir to find tsconfig.json and return compilerOptions.paths aliases.

    Returns a dict mapping alias prefix (e.g. "@/") to resolved base dir (e.g. "src/").
    Result is cached by tsconfig path string.
    """
    current = start_dir.resolve()
    for candidate in [current, *current.parents]:
        tsconfig = candidate / "tsconfig.json"
        if tsconfig.exists():
            key = str(tsconfig)
            if key not in _TSCONFIG_ALIAS_CACHE:
                try:
                    data = json.loads(tsconfig.read_text(encoding="utf-8"))
                    paths = data.get("compilerOptions", {}).get("paths", {})
                    aliases: dict[str, str] = {}
                    for alias, targets in paths.items():
                        if not targets:
                            continue
                        # Strip trailing /* from alias and target
                        alias_prefix = alias.rstrip("/*")
                        target_base = targets[0].rstrip("/*")
                        aliases[alias_prefix] = str(candidate / target_base)
                    _TSCONFIG_ALIAS_CACHE[key] = aliases
                except Exception:
                    _TSCONFIG_ALIAS_CACHE[key] = {}
            return _TSCONFIG_ALIAS_CACHE[key]
    return {}


# ── LanguageConfig dataclass ─────────────────────────────────────────────────

@dataclass
class LanguageConfig:
    ts_module: str                                   # e.g. "tree_sitter_python"
    ts_language_fn: str = "language"                 # attr to call: e.g. tslang.language()

    class_types: frozenset = frozenset()
    function_types: frozenset = frozenset()
    import_types: frozenset = frozenset()
    call_types: frozenset = frozenset()
    static_prop_types: frozenset = frozenset()
    helper_fn_names: frozenset = frozenset()
    container_bind_methods: frozenset = frozenset()
    event_listener_properties: frozenset = frozenset()

    # Name extraction
    name_field: str = "name"
    name_fallback_child_types: tuple = ()

    # Body detection
    body_field: str = "body"
    body_fallback_child_types: tuple = ()   # e.g. ("declaration_list", "compound_statement")

    # Call name extraction
    call_function_field: str = "function"           # field on call node for callee
    call_accessor_node_types: frozenset = frozenset()  # member/attribute nodes
    call_accessor_field: str = "attribute"          # field on accessor for method name

    # Stop recursion at these types in walk_calls
    function_boundary_types: frozenset = frozenset()

    # Import handler: called for import nodes instead of generic handling
    import_handler: Callable | None = None

    # Optional custom name resolver for functions (C, C++ declarator unwrapping)
    resolve_function_name_fn: Callable | None = None

    # Extra label formatting for functions: if True, functions get "name()" label
    function_label_parens: bool = True

    # Extra walk hook called after generic dispatch (for JS arrow functions, C# namespaces, etc.)
    extra_walk_fn: Callable | None = None


# ── Generic helpers ───────────────────────────────────────────────────────────

def _read_text(node, source: bytes) -> str:
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _resolve_name(node, source: bytes, config: LanguageConfig) -> str | None:
    """Get the name from a node using config.name_field, falling back to child types."""
    if config.resolve_function_name_fn is not None:
        # For C/C++ where the name is inside a declarator
        return None  # caller handles this separately
    n = node.child_by_field_name(config.name_field)
    if n:
        return _read_text(n, source)
    for child in node.children:
        if child.type in config.name_fallback_child_types:
            return _read_text(child, source)
    return None


def _find_body(node, config: LanguageConfig):
    """Find the body node using config.body_field, falling back to child types."""
    b = node.child_by_field_name(config.body_field)
    if b:
        return b
    for child in node.children:
        if child.type in config.body_fallback_child_types:
            return child
    return None


# ── Import handlers ───────────────────────────────────────────────────────────

def _import_python(node, source: bytes, file_nid: str, stem: str, edges: list, str_path: str) -> None:
    t = node.type
    if t == "import_statement":
        for child in node.children:
            if child.type in ("dotted_name", "aliased_import"):
                raw = _read_text(child, source)
                module_name = raw.split(" as ")[0].strip().lstrip(".")
                tgt_nid = _make_id(module_name)
                edges.append({
                    "source": file_nid,
                    "target": tgt_nid,
                    "relation": "imports",
                    "confidence": "EXTRACTED",
                    "source_file": str_path,
                    "source_location": f"L{node.start_point[0] + 1}",
                    "weight": 1.0,
                })
    elif t == "import_from_statement":
        module_node = node.child_by_field_name("module_name")
        if module_node:
            raw = _read_text(module_node, source)
            if raw.startswith("."):
                # Relative import - resolve to full path so IDs match file node IDs
                dots = len(raw) - len(raw.lstrip("."))
                module_name = raw.lstrip(".")
                base = Path(str_path).parent
                for _ in range(dots - 1):
                    base = base.parent
                rel = (module_name.replace(".", "/") + ".py") if module_name else "__init__.py"
                tgt_nid = _make_id(str(base / rel))
            else:
                tgt_nid = _make_id(raw)
            edges.append({
                "source": file_nid,
                "target": tgt_nid,
                "relation": "imports_from",
                "confidence": "EXTRACTED",
                "source_file": str_path,
                "source_location": f"L{node.start_point[0] + 1}",
                "weight": 1.0,
            })


def _import_js(node, source: bytes, file_nid: str, stem: str, edges: list, str_path: str) -> None:
    for child in node.children:
        if child.type == "string":
            raw = _read_text(child, source).strip("'\"` ")
            if not raw:
                break
            if raw.startswith("."):
                # Relative import - resolve to full path so IDs match file node IDs
                # normpath removes ".." segments so the ID matches the target file's own node ID
                resolved = Path(os.path.normpath(Path(str_path).parent / raw))
                # TypeScript ESM: imports written as .js but actual file is .ts/.tsx
                if resolved.suffix == ".js":
                    resolved = resolved.with_suffix(".ts")
                elif resolved.suffix == ".jsx":
                    resolved = resolved.with_suffix(".tsx")
                tgt_nid = _make_id(str(resolved))
            else:
                # Check tsconfig.json path aliases (e.g. "@/" → "src/") before treating as external (#575)
                aliases = _load_tsconfig_aliases(Path(str_path).parent)
                resolved_alias = None
                for alias_prefix, alias_base in aliases.items():
                    if raw == alias_prefix or raw.startswith(alias_prefix + "/"):
                        rest = raw[len(alias_prefix):].lstrip("/")
                        resolved_alias = Path(os.path.normpath(Path(alias_base) / rest))
                        break
                if resolved_alias is not None:
                    tgt_nid = _make_id(str(resolved_alias))
                else:
                    # Bare/scoped import (node_modules) - use last segment; dropped as external
                    module_name = raw.split("/")[-1]
                    if not module_name:
                        break
                    tgt_nid = _make_id(module_name)
            edges.append({
                "source": file_nid,
                "target": tgt_nid,
                "relation": "imports_from",
                "confidence": "EXTRACTED",
                "source_file": str_path,
                "source_location": f"L{node.start_point[0] + 1}",
                "weight": 1.0,
            })
            break


def _import_java(node, source: bytes, file_nid: str, stem: str, edges: list, str_path: str) -> None:
    def _walk_scoped(n) -> str:
        parts: list[str] = []
        cur = n
        while cur:
            if cur.type == "scoped_identifier":
                name_node = cur.child_by_field_name("name")
                if name_node:
                    parts.append(_read_text(name_node, source))
                cur = cur.child_by_field_name("scope")
            elif cur.type == "identifier":
                parts.append(_read_text(cur, source))
                break
            else:
                break
        parts.reverse()
        return ".".join(parts)

    for child in node.children:
        if child.type in ("scoped_identifier", "identifier"):
            path_str = _walk_scoped(child)
            module_name = path_str.split(".")[-1].strip("*").strip(".") or (
                path_str.split(".")[-2] if len(path_str.split(".")) > 1 else path_str
            )
            if module_name:
                tgt_nid = _make_id(module_name)
                edges.append({
                    "source": file_nid,
                    "target": tgt_nid,
                    "relation": "imports",
                    "confidence": "EXTRACTED",
                    "source_file": str_path,
                    "source_location": f"L{node.start_point[0] + 1}",
                    "weight": 1.0,
                })
            break


def _import_c(node, source: bytes, file_nid: str, stem: str, edges: list, str_path: str) -> None:
    for child in node.children:
        if child.type in ("string_literal", "system_lib_string", "string"):
            raw = _read_text(child, source).strip('"<> ')
            module_name = raw.split("/")[-1].split(".")[0]
            if module_name:
                tgt_nid = _make_id(module_name)
                edges.append({
                    "source": file_nid,
                    "target": tgt_nid,
                    "relation": "imports",
                    "confidence": "EXTRACTED",
                    "source_file": str_path,
                    "source_location": f"L{node.start_point[0] + 1}",
                    "weight": 1.0,
                })
            break


def _import_csharp(node, source: bytes, file_nid: str, stem: str, edges: list, str_path: str) -> None:
    for child in node.children:
        if child.type in ("qualified_name", "identifier", "name_equals"):
            raw = _read_text(child, source)
            module_name = raw.split(".")[-1].strip()
            if module_name:
                tgt_nid = _make_id(module_name)
                edges.append({
                    "source": file_nid,
                    "target": tgt_nid,
                    "relation": "imports",
                    "confidence": "EXTRACTED",
                    "source_file": str_path,
                    "source_location": f"L{node.start_point[0] + 1}",
                    "weight": 1.0,
                })
            break


def _import_kotlin(node, source: bytes, file_nid: str, stem: str, edges: list, str_path: str) -> None:
    path_node = node.child_by_field_name("path")
    if path_node:
        raw = _read_text(path_node, source)
        module_name = raw.split(".")[-1].strip()
        if module_name:
            tgt_nid = _make_id(module_name)
            edges.append({
                "source": file_nid,
                "target": tgt_nid,
                "relation": "imports",
                "confidence": "EXTRACTED",
                "source_file": str_path,
                "source_location": f"L{node.start_point[0] + 1}",
                "weight": 1.0,
            })
        return
    # Fallback: find identifier child
    for child in node.children:
        if child.type == "identifier":
            raw = _read_text(child, source)
            tgt_nid = _make_id(raw)
            edges.append({
                "source": file_nid,
                "target": tgt_nid,
                "relation": "imports",
                "confidence": "EXTRACTED",
                "source_file": str_path,
                "source_location": f"L{node.start_point[0] + 1}",
                "weight": 1.0,
            })
            break


def _import_scala(node, source: bytes, file_nid: str, stem: str, edges: list, str_path: str) -> None:
    for child in node.children:
        if child.type in ("stable_id", "identifier"):
            raw = _read_text(child, source)
            module_name = raw.split(".")[-1].strip("{} ")
            if module_name and module_name != "_":
                tgt_nid = _make_id(module_name)
                edges.append({
                    "source": file_nid,
                    "target": tgt_nid,
                    "relation": "imports",
                    "confidence": "EXTRACTED",
                    "source_file": str_path,
                    "source_location": f"L{node.start_point[0] + 1}",
                    "weight": 1.0,
                })
            break


def _import_php(node, source: bytes, file_nid: str, stem: str, edges: list, str_path: str) -> None:
    for child in node.children:
        if child.type in ("qualified_name", "name", "identifier"):
            raw = _read_text(child, source)
            module_name = raw.split("\\")[-1].strip()
            if module_name:
                tgt_nid = _make_id(module_name)
                edges.append({
                    "source": file_nid,
                    "target": tgt_nid,
                    "relation": "imports",
                    "confidence": "EXTRACTED",
                    "source_file": str_path,
                    "source_location": f"L{node.start_point[0] + 1}",
                    "weight": 1.0,
                })
            break


# ── C/C++ function name helpers ───────────────────────────────────────────────

def _get_c_func_name(node, source: bytes) -> str | None:
    """Recursively unwrap declarator to find the innermost identifier (C)."""
    if node.type == "identifier":
        return _read_text(node, source)
    decl = node.child_by_field_name("declarator")
    if decl:
        return _get_c_func_name(decl, source)
    for child in node.children:
        if child.type == "identifier":
            return _read_text(child, source)
    return None


def _get_cpp_func_name(node, source: bytes) -> str | None:
    """Recursively unwrap declarator to find the innermost identifier (C++)."""
    if node.type == "identifier":
        return _read_text(node, source)
    if node.type == "qualified_identifier":
        name_node = node.child_by_field_name("name")
        if name_node:
            return _read_text(name_node, source)
    decl = node.child_by_field_name("declarator")
    if decl:
        return _get_cpp_func_name(decl, source)
    for child in node.children:
        if child.type == "identifier":
            return _read_text(child, source)
    return None


# ── JS/TS extra walk for arrow functions ──────────────────────────────────────

def _js_extra_walk(node, source: bytes, file_nid: str, stem: str, str_path: str,
                   nodes: list, edges: list, seen_ids: set, function_bodies: list,
                   parent_class_nid: str | None, add_node_fn, add_edge_fn) -> bool:
    """Handle lexical_declaration (arrow functions) for JS/TS. Returns True if handled."""
    if node.type == "lexical_declaration":
        for child in node.children:
            if child.type == "variable_declarator":
                value = child.child_by_field_name("value")
                if value and value.type == "arrow_function":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        func_name = _read_text(name_node, source)
                        line = child.start_point[0] + 1
                        func_nid = _make_id(stem, func_name)
                        add_node_fn(func_nid, f"{func_name}()", line)
                        add_edge_fn(file_nid, func_nid, "contains", line)
                        body = value.child_by_field_name("body")
                        if body:
                            function_bodies.append((func_nid, body))
        return True
    return False


# ── C# extra walk for namespace declarations ──────────────────────────────────

def _csharp_extra_walk(node, source: bytes, file_nid: str, stem: str, str_path: str,
                       nodes: list, edges: list, seen_ids: set, function_bodies: list,
                       parent_class_nid: str | None, add_node_fn, add_edge_fn,
                       walk_fn) -> bool:
    """Handle namespace_declaration for C#. Returns True if handled."""
    if node.type == "namespace_declaration":
        name_node = node.child_by_field_name("name")
        if name_node:
            ns_name = _read_text(name_node, source)
            ns_nid = _make_id(stem, ns_name)
            line = node.start_point[0] + 1
            add_node_fn(ns_nid, ns_name, line)
            add_edge_fn(file_nid, ns_nid, "contains", line)
        body = node.child_by_field_name("body")
        if body:
            for child in body.children:
                walk_fn(child, parent_class_nid)
        return True
    return False


# ── Swift extra walk for enum cases ──────────────────────────────────────────

def _swift_extra_walk(node, source: bytes, file_nid: str, stem: str, str_path: str,
                      nodes: list, edges: list, seen_ids: set, function_bodies: list,
                      parent_class_nid: str | None, add_node_fn, add_edge_fn) -> bool:
    """Handle enum_entry for Swift. Returns True if handled."""
    if node.type == "enum_entry" and parent_class_nid:
        for child in node.children:
            if child.type == "simple_identifier":
                case_name = _read_text(child, source)
                case_nid = _make_id(parent_class_nid, case_name)
                line = node.start_point[0] + 1
                add_node_fn(case_nid, case_name, line)
                add_edge_fn(parent_class_nid, case_nid, "case_of", line)
        return True
    return False


# ── Language configs ──────────────────────────────────────────────────────────

_PYTHON_CONFIG = LanguageConfig(
    ts_module="tree_sitter_python",
    class_types=frozenset({"class_definition"}),
    function_types=frozenset({"function_definition"}),
    import_types=frozenset({"import_statement", "import_from_statement"}),
    call_types=frozenset({"call"}),
    call_function_field="function",
    call_accessor_node_types=frozenset({"attribute"}),
    call_accessor_field="attribute",
    function_boundary_types=frozenset({"function_definition"}),
    import_handler=_import_python,
)

_JS_CONFIG = LanguageConfig(
    ts_module="tree_sitter_javascript",
    class_types=frozenset({"class_declaration"}),
    function_types=frozenset({"function_declaration", "method_definition"}),
    import_types=frozenset({"import_statement"}),
    call_types=frozenset({"call_expression"}),
    call_function_field="function",
    call_accessor_node_types=frozenset({"member_expression"}),
    call_accessor_field="property",
    function_boundary_types=frozenset({"function_declaration", "arrow_function", "method_definition"}),
    import_handler=_import_js,
)

_TS_CONFIG = LanguageConfig(
    ts_module="tree_sitter_typescript",
    ts_language_fn="language_typescript",
    class_types=frozenset({"class_declaration"}),
    function_types=frozenset({"function_declaration", "method_definition"}),
    import_types=frozenset({"import_statement"}),
    call_types=frozenset({"call_expression"}),
    call_function_field="function",
    call_accessor_node_types=frozenset({"member_expression"}),
    call_accessor_field="property",
    function_boundary_types=frozenset({"function_declaration", "arrow_function", "method_definition"}),
    import_handler=_import_js,
)

_JAVA_CONFIG = LanguageConfig(
    ts_module="tree_sitter_java",
    class_types=frozenset({"class_declaration", "interface_declaration"}),
    function_types=frozenset({"method_declaration", "constructor_declaration"}),
    import_types=frozenset({"import_declaration"}),
    call_types=frozenset({"method_invocation"}),
    call_function_field="name",
    call_accessor_node_types=frozenset(),
    function_boundary_types=frozenset({"method_declaration", "constructor_declaration"}),
    import_handler=_import_java,
)

_C_CONFIG = LanguageConfig(
    ts_module="tree_sitter_c",
    class_types=frozenset(),
    function_types=frozenset({"function_definition"}),
    import_types=frozenset({"preproc_include"}),
    call_types=frozenset({"call_expression"}),
    call_function_field="function",
    call_accessor_node_types=frozenset({"field_expression"}),
    call_accessor_field="field",
    function_boundary_types=frozenset({"function_definition"}),
    import_handler=_import_c,
    resolve_function_name_fn=_get_c_func_name,
)

_CPP_CONFIG = LanguageConfig(
    ts_module="tree_sitter_cpp",
    class_types=frozenset({"class_specifier"}),
    function_types=frozenset({"function_definition"}),
    import_types=frozenset({"preproc_include"}),
    call_types=frozenset({"call_expression"}),
    call_function_field="function",
    call_accessor_node_types=frozenset({"field_expression", "qualified_identifier"}),
    call_accessor_field="field",
    function_boundary_types=frozenset({"function_definition"}),
    import_handler=_import_c,
    resolve_function_name_fn=_get_cpp_func_name,
)

_RUBY_CONFIG = LanguageConfig(
    ts_module="tree_sitter_ruby",
    class_types=frozenset({"class"}),
    function_types=frozenset({"method", "singleton_method"}),
    import_types=frozenset(),
    call_types=frozenset({"call"}),
    call_function_field="method",
    call_accessor_node_types=frozenset(),
    name_fallback_child_types=("constant", "scope_resolution", "identifier"),
    body_fallback_child_types=("body_statement",),
    function_boundary_types=frozenset({"method", "singleton_method"}),
)

_CSHARP_CONFIG = LanguageConfig(
    ts_module="tree_sitter_c_sharp",
    class_types=frozenset({"class_declaration", "interface_declaration"}),
    function_types=frozenset({"method_declaration"}),
    import_types=frozenset({"using_directive"}),
    call_types=frozenset({"invocation_expression"}),
    call_function_field="function",
    call_accessor_node_types=frozenset({"member_access_expression"}),
    call_accessor_field="name",
    body_fallback_child_types=("declaration_list",),
    function_boundary_types=frozenset({"method_declaration"}),
    import_handler=_import_csharp,
)

_KOTLIN_CONFIG = LanguageConfig(
    ts_module="tree_sitter_kotlin",
    class_types=frozenset({"class_declaration", "object_declaration"}),
    function_types=frozenset({"function_declaration"}),
    import_types=frozenset({"import_header"}),
    call_types=frozenset({"call_expression"}),
    call_function_field="",
    call_accessor_node_types=frozenset({"navigation_expression"}),
    call_accessor_field="",
    name_fallback_child_types=("simple_identifier",),
    body_fallback_child_types=("function_body", "class_body"),
    function_boundary_types=frozenset({"function_declaration"}),
    import_handler=_import_kotlin,
)

_SCALA_CONFIG = LanguageConfig(
    ts_module="tree_sitter_scala",
    class_types=frozenset({"class_definition", "object_definition"}),
    function_types=frozenset({"function_definition"}),
    import_types=frozenset({"import_declaration"}),
    call_types=frozenset({"call_expression"}),
    call_function_field="",
    call_accessor_node_types=frozenset({"field_expression"}),
    call_accessor_field="field",
    name_fallback_child_types=("identifier",),
    body_fallback_child_types=("template_body",),
    function_boundary_types=frozenset({"function_definition"}),
    import_handler=_import_scala,
)

_PHP_CONFIG = LanguageConfig(
    ts_module="tree_sitter_php",
    ts_language_fn="language_php",
    class_types=frozenset({"class_declaration"}),
    function_types=frozenset({"function_definition", "method_declaration"}),
    import_types=frozenset({"namespace_use_clause"}),
    call_types=frozenset({"function_call_expression", "member_call_expression", "scoped_call_expression", "class_constant_access_expression"}),
    static_prop_types=frozenset({"scoped_property_access_expression"}),
    helper_fn_names=frozenset({"config"}),
    container_bind_methods=frozenset({"bind", "singleton", "scoped", "instance"}),
    event_listener_properties=frozenset({"listen", "subscribe"}),
    call_function_field="function",
    call_accessor_node_types=frozenset({"member_call_expression"}),
    call_accessor_field="name",
    name_fallback_child_types=("name",),
    body_fallback_child_types=("declaration_list", "compound_statement"),
    function_boundary_types=frozenset({"function_definition", "method_declaration"}),
    import_handler=_import_php,
)


def _import_lua(node, source: bytes, file_nid: str, stem: str, edges: list, str_path: str) -> None:
    """Extract require('module') from Lua variable_declaration nodes."""
    text = _read_text(node, source)
    import re
    m = re.search(r"""require\s*[\('"]\s*['"]?([^'")\s]+)""", text)
    if m:
        module_name = m.group(1).split(".")[-1]
        if module_name:
            edges.append({
                "source": file_nid,
                "target": module_name,
                "relation": "imports",
                "confidence": "EXTRACTED",
                "confidence_score": 1.0,
                "source_file": str_path,
                "source_location": str(node.start_point[0] + 1),
                "weight": 1.0,
            })


_LUA_CONFIG = LanguageConfig(
    ts_module="tree_sitter_lua",
    ts_language_fn="language",
    class_types=frozenset(),
    function_types=frozenset({"function_declaration"}),
    import_types=frozenset({"variable_declaration"}),
    call_types=frozenset({"function_call"}),
    call_function_field="name",
    call_accessor_node_types=frozenset({"method_index_expression"}),
    call_accessor_field="name",
    name_fallback_child_types=("identifier", "method_index_expression"),
    body_fallback_child_types=("block",),
    function_boundary_types=frozenset({"function_declaration"}),
    import_handler=_import_lua,
)


def _import_swift(node, source: bytes, file_nid: str, stem: str, edges: list, str_path: str) -> None:
    for child in node.children:
        if child.type == "identifier":
            raw = _read_text(child, source)
            tgt_nid = _make_id(raw)
            edges.append({
                "source": file_nid,
                "target": tgt_nid,
                "relation": "imports",
                "confidence": "EXTRACTED",
                "source_file": str_path,
                "source_location": f"L{node.start_point[0] + 1}",
                "weight": 1.0,
            })
            break


_SWIFT_CONFIG = LanguageConfig(
    ts_module="tree_sitter_swift",
    class_types=frozenset({"class_declaration", "protocol_declaration"}),
    function_types=frozenset({"function_declaration", "init_declaration", "deinit_declaration", "subscript_declaration"}),
    import_types=frozenset({"import_declaration"}),
    call_types=frozenset({"call_expression"}),
    call_function_field="",
    call_accessor_node_types=frozenset({"navigation_expression"}),
    call_accessor_field="",
    name_fallback_child_types=("simple_identifier", "type_identifier", "user_type"),
    body_fallback_child_types=("class_body", "protocol_body", "function_body", "enum_class_body"),
    function_boundary_types=frozenset({"function_declaration", "init_declaration", "deinit_declaration", "subscript_declaration"}),
    import_handler=_import_swift,
)


# ── Generic extractor ─────────────────────────────────────────────────────────

def _extract_generic(path: Path, config: LanguageConfig) -> dict:
    """Generic AST extractor driven by LanguageConfig."""
    try:
        mod = importlib.import_module(config.ts_module)
        from tree_sitter import Language, Parser
        lang_fn = getattr(mod, config.ts_language_fn, None)
        if lang_fn is None:
            # Fallback for PHP: try "language_php" then "language"
            lang_fn = getattr(mod, "language", None)
        if lang_fn is None:
            return {"nodes": [], "edges": [], "error": f"No language function in {config.ts_module}"}
        language = Language(lang_fn())
    except ImportError:
        return {"nodes": [], "edges": [], "error": f"{config.ts_module} not installed"}
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    try:
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    function_bodies: list[tuple[str, object]] = []
    pending_listen_edges: list[tuple[str, str, int]] = []

    def add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({
                "id": nid,
                "label": label,
                "file_type": "code",
                "source_file": str_path,
                "source_location": f"L{line}",
            })

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", weight: float = 1.0) -> None:
        edges.append({
            "source": src,
            "target": tgt,
            "relation": relation,
            "confidence": confidence,
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": weight,
        })

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    def walk(node, parent_class_nid: str | None = None) -> None:
        t = node.type

        # Import types
        if t in config.import_types:
            if config.import_handler:
                config.import_handler(node, source, file_nid, stem, edges, str_path)
            return

        # Class types
        if t in config.class_types:
            # Resolve class name
            name_node = node.child_by_field_name(config.name_field)
            if name_node is None:
                for child in node.children:
                    if child.type in config.name_fallback_child_types:
                        name_node = child
                        break
            if not name_node:
                return
            class_name = _read_text(name_node, source)
            class_nid = _make_id(stem, class_name)
            line = node.start_point[0] + 1
            add_node(class_nid, class_name, line)
            add_edge(file_nid, class_nid, "contains", line)

            # Python-specific: inheritance
            if config.ts_module == "tree_sitter_python":
                args = node.child_by_field_name("superclasses")
                if args:
                    for arg in args.children:
                        if arg.type == "identifier":
                            base = _read_text(arg, source)
                            base_nid = _make_id(stem, base)
                            if base_nid not in seen_ids:
                                base_nid = _make_id(base)
                                if base_nid not in seen_ids:
                                    nodes.append({
                                        "id": base_nid,
                                        "label": base,
                                        "file_type": "code",
                                        "source_file": "",
                                        "source_location": "",
                                    })
                                    seen_ids.add(base_nid)
                            add_edge(class_nid, base_nid, "inherits", line)

            # Swift-specific: conformance / inheritance
            if config.ts_module == "tree_sitter_swift":
                for child in node.children:
                    if child.type == "inheritance_specifier":
                        for sub in child.children:
                            if sub.type in ("user_type", "type_identifier"):
                                base = _read_text(sub, source)
                                base_nid = _make_id(stem, base)
                                if base_nid not in seen_ids:
                                    base_nid = _make_id(base)
                                    if base_nid not in seen_ids:
                                        nodes.append({
                                            "id": base_nid,
                                            "label": base,
                                            "file_type": "code",
                                            "source_file": "",
                                            "source_location": "",
                                        })
                                        seen_ids.add(base_nid)
                                add_edge(class_nid, base_nid, "inherits", line)

            # C#-specific: inheritance / interface implementation via base_list
            if config.ts_module == "tree_sitter_c_sharp":
                for child in node.children:
                    if child.type == "base_list":
                        for sub in child.children:
                            if sub.type in ("identifier", "generic_name"):
                                if sub.type == "generic_name":
                                    name_child = sub.child_by_field_name("name")
                                    base = _read_text(name_child, source) if name_child else _read_text(sub.children[0], source)
                                else:
                                    base = _read_text(sub, source)
                                base_nid = _make_id(stem, base)
                                if base_nid not in seen_ids:
                                    base_nid = _make_id(base)
                                    if base_nid not in seen_ids:
                                        nodes.append({
                                            "id": base_nid,
                                            "label": base,
                                            "file_type": "code",
                                            "source_file": "",
                                            "source_location": "",
                                        })
                                        seen_ids.add(base_nid)
                                add_edge(class_nid, base_nid, "inherits", line)

            # Java-specific: extends (superclass) / implements (interfaces) / interface-extends
            if config.ts_module == "tree_sitter_java":
                def _emit_java_parent(base_name: str, rel: str, at_line: int) -> None:
                    if not base_name:
                        return
                    base_nid = _make_id(stem, base_name)
                    if base_nid not in seen_ids:
                        base_nid = _make_id(base_name)
                        if base_nid not in seen_ids:
                            nodes.append({
                                "id": base_nid,
                                "label": base_name,
                                "file_type": "code",
                                "source_file": "",
                                "source_location": "",
                            })
                            seen_ids.add(base_nid)
                    add_edge(class_nid, base_nid, rel, at_line)

                sup = node.child_by_field_name("superclass")
                if sup is not None:
                    for sub in sup.children:
                        if sub.type == "type_identifier":
                            _emit_java_parent(_read_text(sub, source), "extends", line)
                            break

                ifs = node.child_by_field_name("interfaces")
                if ifs is not None:
                    for sub in ifs.children:
                        if sub.type == "type_list":
                            for tid in sub.children:
                                if tid.type == "type_identifier":
                                    _emit_java_parent(_read_text(tid, source), "implements", line)

                if t == "interface_declaration":
                    for child in node.children:
                        if child.type == "extends_interfaces":
                            for sub in child.children:
                                if sub.type == "type_list":
                                    for tid in sub.children:
                                        if tid.type == "type_identifier":
                                            _emit_java_parent(_read_text(tid, source), "extends", line)

            # Find body and recurse
            body = _find_body(node, config)
            if body:
                for child in body.children:
                    walk(child, parent_class_nid=class_nid)
            return

        # Event listener property arrays: $listen = [Event::class => [Listener::class]]
        if (t == "property_declaration"
                and parent_class_nid
                and config.event_listener_properties):
            for element in node.children:
                if element.type != "property_element":
                    continue
                prop_name: str | None = None
                array_node = None
                for c in element.children:
                    if c.type == "variable_name":
                        for sc in c.children:
                            if sc.type == "name":
                                prop_name = _read_text(sc, source)
                                break
                    elif c.type == "array_creation_expression":
                        array_node = c
                if (prop_name is None
                        or prop_name not in config.event_listener_properties
                        or array_node is None):
                    continue
                for entry in array_node.children:
                    if entry.type != "array_element_initializer":
                        continue
                    event_cls: str | None = None
                    listener_arr = None
                    for sub in entry.children:
                        if sub.type == "class_constant_access_expression" and event_cls is None:
                            for sc in sub.children:
                                if sc.is_named and sc.type in ("name", "qualified_name"):
                                    event_cls = _read_text(sc, source)
                                    break
                        elif sub.type == "array_creation_expression":
                            listener_arr = sub
                    if not event_cls or listener_arr is None:
                        continue
                    for listener_entry in listener_arr.children:
                        if listener_entry.type != "array_element_initializer":
                            continue
                        for item in listener_entry.children:
                            if item.type != "class_constant_access_expression":
                                continue
                            for sc in item.children:
                                if sc.is_named and sc.type in ("name", "qualified_name"):
                                    listener_cls = _read_text(sc, source)
                                    line_no = item.start_point[0] + 1
                                    pending_listen_edges.append((event_cls, listener_cls, line_no))
                                    break
                            break
            return

        # Function types
        if t in config.function_types:
            # Swift deinit/subscript have no name field — resolve before generic fallback
            if t == "deinit_declaration":
                func_name: str | None = "deinit"
            elif t == "subscript_declaration":
                func_name = "subscript"
            elif config.resolve_function_name_fn is not None:
                # C/C++ style: use declarator
                declarator = node.child_by_field_name("declarator")
                func_name = None
                if declarator:
                    func_name = config.resolve_function_name_fn(declarator, source)
            else:
                name_node = node.child_by_field_name(config.name_field)
                if name_node is None:
                    for child in node.children:
                        if child.type in config.name_fallback_child_types:
                            name_node = child
                            break
                func_name = _read_text(name_node, source) if name_node else None

            if not func_name:
                return

            line = node.start_point[0] + 1
            if parent_class_nid:
                func_nid = _make_id(parent_class_nid, func_name)
                add_node(func_nid, f".{func_name}()", line)
                add_edge(parent_class_nid, func_nid, "method", line)
            else:
                func_nid = _make_id(stem, func_name)
                add_node(func_nid, f"{func_name}()", line)
                add_edge(file_nid, func_nid, "contains", line)

            body = _find_body(node, config)
            if body:
                function_bodies.append((func_nid, body))
            return

        # JS/TS arrow functions and C# namespaces — language-specific extra handling
        if config.ts_module in ("tree_sitter_javascript", "tree_sitter_typescript"):
            if _js_extra_walk(node, source, file_nid, stem, str_path,
                              nodes, edges, seen_ids, function_bodies,
                              parent_class_nid, add_node, add_edge):
                return

        if config.ts_module == "tree_sitter_c_sharp":
            if _csharp_extra_walk(node, source, file_nid, stem, str_path,
                                   nodes, edges, seen_ids, function_bodies,
                                   parent_class_nid, add_node, add_edge, walk):
                return

        if config.ts_module == "tree_sitter_swift":
            if _swift_extra_walk(node, source, file_nid, stem, str_path,
                                  nodes, edges, seen_ids, function_bodies,
                                  parent_class_nid, add_node, add_edge):
                return

        # Default: recurse
        for child in node.children:
            walk(child, parent_class_nid=None)

    walk(root)

    # ── Call-graph pass ───────────────────────────────────────────────────────
    label_to_nid: dict[str, str] = {}
    for n in nodes:
        raw = n["label"]
        normalised = raw.strip("()").lstrip(".")
        label_to_nid[normalised.lower()] = n["id"]

    seen_call_pairs: set[tuple[str, str]] = set()
    seen_static_ref_pairs: set[tuple[str, str, str]] = set()
    seen_helper_ref_pairs: set[tuple[str, str, str]] = set()
    seen_bind_pairs: set[tuple[str, str, str]] = set()
    raw_calls: list[dict] = []  # unresolved calls for cross-file resolution in extract()

    def _php_class_const_scope(n) -> str | None:
        scope = n.child_by_field_name("scope")
        if scope is None:
            for c in n.children:
                if c.is_named and c.type in ("name", "qualified_name", "identifier"):
                    scope = c
                    break
        if scope is None:
            return None
        return _read_text(scope, source)

    def walk_calls(node, caller_nid: str) -> None:
        if node.type in config.function_boundary_types:
            return

        if node.type in config.call_types:
            callee_name: str | None = None
            is_member_call: bool = False

            # Special handling per language
            if config.ts_module == "tree_sitter_swift":
                # Swift: first child may be simple_identifier or navigation_expression
                first = node.children[0] if node.children else None
                if first:
                    if first.type == "simple_identifier":
                        callee_name = _read_text(first, source)
                    elif first.type == "navigation_expression":
                        is_member_call = True
                        for child in first.children:
                            if child.type == "navigation_suffix":
                                for sc in child.children:
                                    if sc.type == "simple_identifier":
                                        callee_name = _read_text(sc, source)
            elif config.ts_module == "tree_sitter_kotlin":
                # Kotlin: first child may be simple_identifier or navigation_expression
                first = node.children[0] if node.children else None
                if first:
                    if first.type == "simple_identifier":
                        callee_name = _read_text(first, source)
                    elif first.type == "navigation_expression":
                        is_member_call = True
                        for child in reversed(first.children):
                            if child.type == "simple_identifier":
                                callee_name = _read_text(child, source)
                                break
            elif config.ts_module == "tree_sitter_scala":
                # Scala: first child
                first = node.children[0] if node.children else None
                if first:
                    if first.type == "identifier":
                        callee_name = _read_text(first, source)
                    elif first.type == "field_expression":
                        is_member_call = True
                        field = first.child_by_field_name("field")
                        if field:
                            callee_name = _read_text(field, source)
                        else:
                            for child in reversed(first.children):
                                if child.type == "identifier":
                                    callee_name = _read_text(child, source)
                                    break
            elif config.ts_module == "tree_sitter_c_sharp" and node.type == "invocation_expression":
                # C#: try name field, then first named child
                name_node = node.child_by_field_name("name")
                if name_node:
                    callee_name = _read_text(name_node, source)
                else:
                    for child in node.children:
                        if child.is_named:
                            raw = _read_text(child, source)
                            if "." in raw:
                                callee_name = raw.split(".")[-1]
                                is_member_call = True
                            else:
                                callee_name = raw
                            break
            elif config.ts_module == "tree_sitter_php":
                # PHP: distinguish call expression subtypes
                if node.type == "function_call_expression":
                    func_node = node.child_by_field_name("function")
                    if func_node:
                        callee_name = _read_text(func_node, source)
                elif node.type == "scoped_call_expression":
                    # Static method call: Helper::format() → callee = "Helper"
                    scope_node = node.child_by_field_name("scope")
                    if scope_node:
                        callee_name = _read_text(scope_node, source)
                else:
                    # member_call_expression: $obj->method()
                    is_member_call = True
                    name_node = node.child_by_field_name("name")
                    if name_node:
                        callee_name = _read_text(name_node, source)
            elif config.ts_module == "tree_sitter_cpp":
                # C++: function field, then field_expression/qualified_identifier
                func_node = node.child_by_field_name(config.call_function_field) if config.call_function_field else None
                if func_node:
                    if func_node.type == "identifier":
                        callee_name = _read_text(func_node, source)
                    elif func_node.type in ("field_expression", "qualified_identifier"):
                        is_member_call = True
                        name = func_node.child_by_field_name("field") or func_node.child_by_field_name("name")
                        if name:
                            callee_name = _read_text(name, source)
            else:
                # Generic: get callee from call_function_field
                func_node = node.child_by_field_name(config.call_function_field) if config.call_function_field else None
                if func_node:
                    if func_node.type == "identifier":
                        callee_name = _read_text(func_node, source)
                    elif func_node.type in config.call_accessor_node_types:
                        is_member_call = True
                        if config.call_accessor_field:
                            attr = func_node.child_by_field_name(config.call_accessor_field)
                            if attr:
                                callee_name = _read_text(attr, source)
                    else:
                        # Try reading the node directly (e.g. Java name field is the callee)
                        callee_name = _read_text(func_node, source)

            if callee_name:
                tgt_nid = label_to_nid.get(callee_name.lower())
                if tgt_nid and tgt_nid != caller_nid:
                    pair = (caller_nid, tgt_nid)
                    if pair not in seen_call_pairs:
                        seen_call_pairs.add(pair)
                        line = node.start_point[0] + 1
                        edges.append({
                            "source": caller_nid,
                            "target": tgt_nid,
                            "relation": "calls",
                            "confidence": "EXTRACTED",
                            "source_file": str_path,
                            "source_location": f"L{line}",
                            "weight": 1.0,
                        })
                elif callee_name and not tgt_nid:
                    # Callee not in this file — save for cross-file resolution in extract()
                    raw_calls.append({
                        "caller_nid": caller_nid,
                        "callee": callee_name,
                        "is_member_call": is_member_call,
                        "source_file": str_path,
                        "source_location": f"L{node.start_point[0] + 1}",
                    })

            # Helper function calls: config('foo.bar') → uses_config edge to "foo"
            if (callee_name and callee_name in config.helper_fn_names):
                args_node = node.child_by_field_name("arguments")
                first_key: str | None = None
                if args_node:
                    for arg in args_node.children:
                        if arg.type != "argument":
                            continue
                        for inner in arg.children:
                            if inner.type == "string":
                                for sc in inner.children:
                                    if sc.type == "string_content":
                                        first_key = _read_text(sc, source)
                                        break
                                break
                        if first_key:
                            break
                if first_key:
                    segment = first_key.split(".")[0]
                    tgt_nid = (label_to_nid.get(segment.lower())
                               or label_to_nid.get(f"{segment}.php".lower()))
                    if tgt_nid and tgt_nid != caller_nid:
                        relation = f"uses_{callee_name}"
                        pair3 = (caller_nid, tgt_nid, relation)
                        if pair3 not in seen_helper_ref_pairs:
                            seen_helper_ref_pairs.add(pair3)
                            line = node.start_point[0] + 1
                            edges.append({
                                "source": caller_nid,
                                "target": tgt_nid,
                                "relation": relation,
                                "confidence": "EXTRACTED",
                                "confidence_score": 1.0,
                                "source_file": str_path,
                                "source_location": f"L{line}",
                                "weight": 1.0,
                            })

            # Service container bindings: $this->app->bind(Foo::class, Bar::class)
            if (node.type == "member_call_expression"
                    and callee_name
                    and callee_name in config.container_bind_methods):
                args_node = node.child_by_field_name("arguments")
                class_args: list[str] = []
                if args_node:
                    for arg in args_node.children:
                        if arg.type != "argument":
                            continue
                        for inner in arg.children:
                            if inner.type == "class_constant_access_expression":
                                cls = _php_class_const_scope(inner)
                                if cls:
                                    class_args.append(cls)
                                break
                        if len(class_args) >= 2:
                            break
                if len(class_args) == 2:
                    contract_name, impl_name = class_args
                    contract_nid = label_to_nid.get(contract_name.lower())
                    impl_nid = label_to_nid.get(impl_name.lower())
                    if contract_nid and impl_nid and contract_nid != impl_nid:
                        pair3 = (contract_nid, impl_nid, "bound_to")
                        if pair3 not in seen_bind_pairs:
                            seen_bind_pairs.add(pair3)
                            line = node.start_point[0] + 1
                            edges.append({
                                "source": contract_nid,
                                "target": impl_nid,
                                "relation": "bound_to",
                                "confidence": "EXTRACTED",
                                "confidence_score": 1.0,
                                "source_file": str_path,
                                "source_location": f"L{line}",
                                "weight": 1.0,
                            })

        # Static property access: Foo::$bar → uses_static_prop edge
        if node.type in config.static_prop_types:
            scope_node = node.child_by_field_name("scope")
            if scope_node is None:
                for child in node.children:
                    if child.is_named and child.type in ("name", "qualified_name", "identifier"):
                        scope_node = child
                        break
            if scope_node is not None:
                class_name = _read_text(scope_node, source)
                tgt_nid = label_to_nid.get(class_name.lower())
                if tgt_nid and tgt_nid != caller_nid:
                    pair3 = (caller_nid, tgt_nid, "uses_static_prop")
                    if pair3 not in seen_static_ref_pairs:
                        seen_static_ref_pairs.add(pair3)
                        line = node.start_point[0] + 1
                        edges.append({
                            "source": caller_nid,
                            "target": tgt_nid,
                            "relation": "uses_static_prop",
                            "confidence": "EXTRACTED",
                            "confidence_score": 1.0,
                            "source_file": str_path,
                            "source_location": f"L{line}",
                            "weight": 1.0,
                        })

        # PHP class constant access: Foo::BAR → references_constant edge
        if config.ts_module == "tree_sitter_php" and node.type == "class_constant_access_expression":
            class_name = _php_class_const_scope(node)
            if class_name:
                tgt_nid = label_to_nid.get(class_name.lower())
                if tgt_nid and tgt_nid != caller_nid:
                    pair3 = (caller_nid, tgt_nid, "references_constant")
                    if pair3 not in seen_static_ref_pairs:
                        seen_static_ref_pairs.add(pair3)
                        line = node.start_point[0] + 1
                        edges.append({
                            "source": caller_nid,
                            "target": tgt_nid,
                            "relation": "references_constant",
                            "confidence": "EXTRACTED",
                            "confidence_score": 1.0,
                            "source_file": str_path,
                            "source_location": f"L{line}",
                            "weight": 1.0,
                        })

        for child in node.children:
            walk_calls(child, caller_nid)

    for caller_nid, body_node in function_bodies:
        walk_calls(body_node, caller_nid)

    # ── Event listener pass ───────────────────────────────────────────────────
    seen_listen_pairs: set[tuple[str, str]] = set()
    for event_name, listener_name, line in pending_listen_edges:
        event_nid = label_to_nid.get(event_name.lower())
        listener_nid = label_to_nid.get(listener_name.lower())
        if not event_nid or not listener_nid or event_nid == listener_nid:
            continue
        pair2 = (event_nid, listener_nid)
        if pair2 in seen_listen_pairs:
            continue
        seen_listen_pairs.add(pair2)
        edges.append({
            "source": event_nid,
            "target": listener_nid,
            "relation": "listened_by",
            "confidence": "EXTRACTED",
            "confidence_score": 1.0,
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": 1.0,
        })

    # ── Clean edges ───────────────────────────────────────────────────────────
    valid_ids = seen_ids
    clean_edges = []
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in valid_ids and (tgt in valid_ids or edge["relation"] in ("imports", "imports_from")):
            clean_edges.append(edge)

    return {"nodes": nodes, "edges": clean_edges, "raw_calls": raw_calls}


# ── Python rationale extraction ───────────────────────────────────────────────

_RATIONALE_PREFIXES = ("# NOTE:", "# IMPORTANT:", "# HACK:", "# WHY:", "# RATIONALE:", "# TODO:", "# FIXME:")


def _extract_python_rationale(path: Path, result: dict) -> None:
    """Post-pass: extract docstrings and rationale comments from Python source.
    Mutates result in-place by appending to result['nodes'] and result['edges'].
    """
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
        language = Language(tspython.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception:
        return

    stem = _file_stem(path)
    str_path = str(path)
    nodes = result["nodes"]
    edges = result["edges"]
    seen_ids = {n["id"] for n in nodes}
    file_nid = _make_id(str(path))

    def _get_docstring(body_node) -> tuple[str, int] | None:
        if not body_node:
            return None
        for child in body_node.children:
            if child.type == "expression_statement":
                for sub in child.children:
                    if sub.type in ("string", "concatenated_string"):
                        text = source[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                        text = text.strip("\"'").strip('"""').strip("'''").strip()
                        if len(text) > 20:
                            return text, child.start_point[0] + 1
            break
        return None

    def _add_rationale(text: str, line: int, parent_nid: str) -> None:
        label = text[:80].replace("\r\n", " ").replace("\r", " ").replace("\n", " ").strip()
        rid = _make_id(stem, "rationale", str(line))
        if rid not in seen_ids:
            seen_ids.add(rid)
            nodes.append({
                "id": rid,
                "label": label,
                "file_type": "rationale",
                "source_file": str_path,
                "source_location": f"L{line}",
            })
        edges.append({
            "source": rid,
            "target": parent_nid,
            "relation": "rationale_for",
            "confidence": "EXTRACTED",
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": 1.0,
        })

    # Module-level docstring
    ds = _get_docstring(root)
    if ds:
        _add_rationale(ds[0], ds[1], file_nid)

    # Class and function docstrings
    def walk_docstrings(node, parent_nid: str) -> None:
        t = node.type
        if t == "class_definition":
            name_node = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            if name_node and body:
                class_name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                nid = _make_id(stem, class_name)
                ds = _get_docstring(body)
                if ds:
                    _add_rationale(ds[0], ds[1], nid)
                for child in body.children:
                    walk_docstrings(child, nid)
            return
        if t == "function_definition":
            name_node = node.child_by_field_name("name")
            body = node.child_by_field_name("body")
            if name_node and body:
                func_name = source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                nid = _make_id(parent_nid, func_name) if parent_nid != file_nid else _make_id(stem, func_name)
                ds = _get_docstring(body)
                if ds:
                    _add_rationale(ds[0], ds[1], nid)
            return
        for child in node.children:
            walk_docstrings(child, parent_nid)

    walk_docstrings(root, file_nid)

    # Rationale comments (# NOTE:, # IMPORTANT:, etc.)
    source_text = source.decode("utf-8", errors="replace")
    for lineno, line_text in enumerate(source_text.splitlines(), start=1):
        stripped = line_text.strip()
        if any(stripped.startswith(p) for p in _RATIONALE_PREFIXES):
            _add_rationale(stripped, lineno, file_nid)


# ── Public API ────────────────────────────────────────────────────────────────

def extract_python(path: Path) -> dict:
    """Extract classes, functions, and imports from a .py file via tree-sitter AST."""
    result = _extract_generic(path, _PYTHON_CONFIG)
    if "error" not in result:
        _extract_python_rationale(path, result)
    return result


def extract_js(path: Path) -> dict:
    """Extract classes, functions, arrow functions, and imports from a .js/.ts/.tsx file."""
    config = _TS_CONFIG if path.suffix in (".ts", ".tsx") else _JS_CONFIG
    return _extract_generic(path, config)


def extract_java(path: Path) -> dict:
    """Extract classes, interfaces, methods, constructors, and imports from a .java file."""
    return _extract_generic(path, _JAVA_CONFIG)


def extract_c(path: Path) -> dict:
    """Extract functions and includes from a .c/.h file."""
    return _extract_generic(path, _C_CONFIG)


def extract_cpp(path: Path) -> dict:
    """Extract functions, classes, and includes from a .cpp/.cc/.cxx/.hpp file."""
    return _extract_generic(path, _CPP_CONFIG)


def extract_ruby(path: Path) -> dict:
    """Extract classes, methods, singleton methods, and calls from a .rb file."""
    return _extract_generic(path, _RUBY_CONFIG)


def extract_csharp(path: Path) -> dict:
    """Extract classes, interfaces, methods, namespaces, and usings from a .cs file."""
    return _extract_generic(path, _CSHARP_CONFIG)


def extract_kotlin(path: Path) -> dict:
    """Extract classes, objects, functions, and imports from a .kt/.kts file."""
    return _extract_generic(path, _KOTLIN_CONFIG)


def extract_scala(path: Path) -> dict:
    """Extract classes, objects, functions, and imports from a .scala file."""
    return _extract_generic(path, _SCALA_CONFIG)


def extract_php(path: Path) -> dict:
    """Extract classes, functions, methods, namespace uses, and calls from a .php file."""
    return _extract_generic(path, _PHP_CONFIG)


def extract_blade(path: Path) -> dict:
    """Extract @include, <livewire:> components, and wire:click bindings from Blade templates."""
    import re
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"error": f"cannot read {path}"}

    file_nid = _make_id(str(path))
    nodes = [{"id": file_nid, "label": path.name, "file_type": "code",
              "source_file": str(path), "source_location": None}]
    edges = []

    # @include('path.to.partial') or @include("path.to.partial")
    for m in re.finditer(r"@include\(['\"]([^'\"]+)['\"]", src):
        tgt = m.group(1).replace(".", "/")
        tgt_nid = _make_id(tgt)
        if tgt_nid not in {n["id"] for n in nodes}:
            nodes.append({"id": tgt_nid, "label": m.group(1), "file_type": "code",
                          "source_file": str(path), "source_location": None})
        edges.append({"source": file_nid, "target": tgt_nid, "relation": "includes",
                      "confidence": "EXTRACTED", "confidence_score": 1.0,
                      "source_file": str(path), "source_location": None, "weight": 1.0})

    # <livewire:component.name /> or <livewire:component.name>
    for m in re.finditer(r"<livewire:([\w.\-]+)", src):
        tgt_nid = _make_id(m.group(1))
        if tgt_nid not in {n["id"] for n in nodes}:
            nodes.append({"id": tgt_nid, "label": m.group(1), "file_type": "code",
                          "source_file": str(path), "source_location": None})
        edges.append({"source": file_nid, "target": tgt_nid, "relation": "uses_component",
                      "confidence": "EXTRACTED", "confidence_score": 1.0,
                      "source_file": str(path), "source_location": None, "weight": 1.0})

    # wire:click="methodName"
    for m in re.finditer(r'wire:click=["\']([^"\']+)["\']', src):
        tgt_nid = _make_id(m.group(1))
        if tgt_nid not in {n["id"] for n in nodes}:
            nodes.append({"id": tgt_nid, "label": m.group(1), "file_type": "code",
                          "source_file": str(path), "source_location": None})
        edges.append({"source": file_nid, "target": tgt_nid, "relation": "binds_method",
                      "confidence": "EXTRACTED", "confidence_score": 1.0,
                      "source_file": str(path), "source_location": None, "weight": 1.0})

    return {"nodes": nodes, "edges": edges}


def extract_dart(path: Path) -> dict:
    """Extract classes, mixins, functions, imports, and calls from a .dart file using regex."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"error": f"cannot read {path}"}

    file_nid = _make_id(str(path))
    nodes = [{"id": file_nid, "label": path.name, "file_type": "code",
              "source_file": str(path), "source_location": None}]
    edges = []
    defined: set[str] = set()

    # Classes and mixins
    for m in re.finditer(r"^\s*(?:abstract\s+)?(?:class|mixin)\s+(\w+)", src, re.MULTILINE):
        nid = _make_id(str(path), m.group(1))
        if nid not in defined:
            nodes.append({"id": nid, "label": m.group(1), "file_type": "code",
                          "source_file": str(path), "source_location": None})
            edges.append({"source": file_nid, "target": nid, "relation": "defines",
                          "confidence": "EXTRACTED", "confidence_score": 1.0,
                          "source_file": str(path), "source_location": None, "weight": 1.0})
            defined.add(nid)

    # Top-level and member functions/methods
    for m in re.finditer(r"^\s*(?:static\s+|async\s+)?(?:\w+\s+)+(\w+)\s*\(", src, re.MULTILINE):
        name = m.group(1)
        if name in {"if", "for", "while", "switch", "catch", "return"}:
            continue
        nid = _make_id(str(path), name)
        if nid not in defined:
            nodes.append({"id": nid, "label": name, "file_type": "code",
                          "source_file": str(path), "source_location": None})
            edges.append({"source": file_nid, "target": nid, "relation": "defines",
                          "confidence": "EXTRACTED", "confidence_score": 1.0,
                          "source_file": str(path), "source_location": None, "weight": 1.0})
            defined.add(nid)

    # import 'package:...' or import '...'
    for m in re.finditer(r"""^import\s+['"]([^'"]+)['"]""", src, re.MULTILINE):
        pkg = m.group(1)
        tgt_nid = _make_id(pkg)
        if tgt_nid not in defined:
            nodes.append({"id": tgt_nid, "label": pkg, "file_type": "code",
                          "source_file": str(path), "source_location": None})
            defined.add(tgt_nid)
        edges.append({"source": file_nid, "target": tgt_nid, "relation": "imports",
                      "confidence": "EXTRACTED", "confidence_score": 1.0,
                      "source_file": str(path), "source_location": None, "weight": 1.0})

    return {"nodes": nodes, "edges": edges}


def extract_verilog(path: Path) -> dict:
    """Extract modules, functions, tasks, package imports, and instantiations from .v/.sv files."""
    try:
        import tree_sitter_verilog as tsverilog
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_verilog not installed"}

    try:
        language = Language(tsverilog.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()

    def add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                          "source_file": str_path, "source_location": f"L{line}",
                          "confidence_score": 1.0})

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", score: float = 1.0) -> None:
        edges.append({"source": src, "target": tgt, "relation": relation,
                      "confidence": confidence, "confidence_score": score,
                      "source_file": str_path, "source_location": f"L{line}", "weight": 1.0})

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    def walk(node, module_nid: str | None = None) -> None:
        t = node.type

        if t == "module_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                mod_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                nid = _make_id(stem, mod_name)
                add_node(nid, mod_name, line)
                add_edge(file_nid, nid, "defines", line)
                for child in node.children:
                    walk(child, nid)
                return

        elif t in ("function_declaration", "function_prototype"):
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                parent = module_nid or file_nid
                nid = _make_id(parent, func_name)
                add_node(nid, f"{func_name}()", line)
                add_edge(parent, nid, "contains", line)

        elif t == "task_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                task_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                parent = module_nid or file_nid
                nid = _make_id(parent, task_name)
                add_node(nid, task_name, line)
                add_edge(parent, nid, "contains", line)

        elif t == "package_import_declaration":
            for child in node.children:
                if child.type == "package_import_item":
                    pkg_text = _read_text(child, source)
                    pkg_name = pkg_text.split("::")[0].strip()
                    if pkg_name:
                        line = node.start_point[0] + 1
                        tgt_nid = _make_id(pkg_name)
                        add_node(tgt_nid, pkg_name, line)
                        src = module_nid or file_nid
                        add_edge(src, tgt_nid, "imports_from", line)

        elif t == "module_instantiation":
            # module_type instantiates another module
            type_node = node.child_by_field_name("module_type")
            if type_node and module_nid:
                inst_type = _read_text(type_node, source).strip()
                if inst_type:
                    line = node.start_point[0] + 1
                    tgt_nid = _make_id(inst_type)
                    add_node(tgt_nid, inst_type, line)
                    add_edge(module_nid, tgt_nid, "instantiates", line)

        for child in node.children:
            walk(child, module_nid)

    walk(root)
    return {"nodes": nodes, "edges": edges}


_DBT_PROJECT_CACHE: dict[str, bool] = {}


def _is_dbt_project_file(path: Path) -> bool:
    """Walk up parents looking for a `dbt_project.yml` to decide if `path` belongs
    to a dbt project. Result cached per-directory to avoid repeated stat() calls
    when many .sql files share ancestors."""
    try:
        parent = path.parent.resolve()
    except OSError:
        return False
    key = str(parent)
    if key in _DBT_PROJECT_CACHE:
        return _DBT_PROJECT_CACHE[key]
    cur = parent
    seen: list[str] = []
    found = False
    while True:
        seen.append(str(cur))
        if (cur / "dbt_project.yml").exists():
            found = True
            break
        if cur.parent == cur:
            break
        cur = cur.parent
    for s in seen:
        _DBT_PROJECT_CACHE[s] = found
    return found


_DBT_FNS = {"ref", "source", "var", "config"}


def extract_dbt_sql(path: Path) -> dict:
    """Extract dbt model lineage from a .sql file by parsing it as Jinja2.
    Emits edges for `ref('x')` (relation `references`),
    `source('s','t')` (`references_source`), `var('v')` (`uses_var`).
    `config(...)` kwargs (alias, materialized) become attributes on the file node.

    Used by `extract_sql` when the file lives inside a dbt project (parent dir
    chain contains `dbt_project.yml`) or when Jinja markers (`{{` or `{%`) are
    detected in the source. Falls back gracefully when `tree-sitter-jinja` is not
    installed — caller should retry with `extract_sql`'s plain-SQL path."""
    try:
        import tree_sitter_jinja as tsj
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_jinja not installed. Run: pip install tree-sitter-jinja"}

    try:
        language = Language(tsj.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    str_path = str(path)
    file_nid = _make_id(path.stem)
    file_node: dict = {"id": file_nid, "label": path.stem, "file_type": "code",
                       "source_file": str_path, "source_location": "L1"}
    nodes: list[dict] = [file_node]
    edges: list[dict] = []
    seen_targets: set[str] = set()

    def _read(n) -> str:
        return source[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    def _strip_quotes(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            return s[1:-1]
        return s

    def _add_target_node_and_edge(target_id: str, label: str, relation: str, line: int) -> None:
        if target_id in seen_targets:
            return
        seen_targets.add(target_id)
        nodes.append({"id": target_id, "label": label, "file_type": "code",
                      "source_file": "", "source_location": ""})
        edges.append({"source": file_nid, "target": target_id, "relation": relation,
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "source_location": f"L{line}", "weight": 1.0})

    def walk(node) -> None:
        if node.type in ("call", "fn_call"):
            named = [c for c in node.children if c.is_named]
            if named:
                fn_name = _read(named[0]).strip()
                if fn_name in _DBT_FNS:
                    args: list[str] = []
                    stack = [node]
                    while stack:
                        n2 = stack.pop()
                        if n2.type in ("lit_string", "string", "string_literal"):
                            args.append(_strip_quotes(_read(n2)))
                        else:
                            stack.extend(n2.children)
                    line = node.start_point[0] + 1
                    if fn_name == "ref" and args:
                        target = args[0]
                        _add_target_node_and_edge(_make_id(target), target, "references", line)
                    elif fn_name == "source" and len(args) >= 2:
                        schema, table = args[0], args[1]
                        _add_target_node_and_edge(
                            _make_id("source", schema, table),
                            f"{schema}.{table}",
                            "references_source",
                            line,
                        )
                    elif fn_name == "var" and args:
                        _add_target_node_and_edge(
                            _make_id("var", args[0]),
                            f"var.{args[0]}",
                            "uses_var",
                            line,
                        )
                    elif fn_name == "config":
                        cfg_text = _read(node)
                        m = re.search(r"alias\s*=\s*['\"]([^'\"]+)['\"]", cfg_text)
                        if m:
                            file_node["dbt_alias"] = m.group(1)
                        m = re.search(r"materialized\s*=\s*['\"]([^'\"]+)['\"]", cfg_text)
                        if m:
                            file_node["dbt_materialized"] = m.group(1)
                        # Tags can be tags=['a', 'b'] or tags=["a"] — capture all bare strings
                        # inside the tags=[ ... ] list. Used by the bridges resolver to match
                        # Airflow `selects_dbt: tag:X` selectors to actual dbt models.
                        tags_match = re.search(r"tags\s*=\s*\[([^\]]*)\]", cfg_text)
                        if tags_match:
                            inner = tags_match.group(1)
                            tags = re.findall(r"['\"]([^'\"]+)['\"]", inner)
                            if tags:
                                file_node["dbt_tags"] = tags
        for c in node.children:
            walk(c)

    walk(root)
    return {"nodes": nodes, "edges": edges}


_DBT_YAML_TOP_SECTIONS = {"models", "sources", "seeds", "snapshots", "exposures",
                          "metrics", "semantic_models", "analyses"}
# Keys whose values describe per-entity structure (column lists, tests, …) —
# we skip recursing into them so individual column "name:" pairs don't get
# misread as model definitions.
_DBT_YAML_LEAF_KEYS = {"columns", "tests", "data_tests", "config", "meta",
                       "constraints", "freshness", "loaded_at_field",
                       "description", "external", "quoting",
                       "loaded_at_query", "tags", "identifier",
                       "database", "schema"}


def extract_dbt_yaml(path: Path) -> dict:
    """Extract dbt schema-style YAML structure (`models:`, `sources:`, `seeds:`,
    `snapshots:`, `exposures:`, `metrics:`) into nodes/edges.

    Each entity becomes a node whose ID matches what `extract_dbt_sql` would
    produce for the corresponding `.sql` file (e.g. a `name: silver_x` under
    `models:` produces id `silver_x`, the same id as the `silver_x.sql` file
    node). So loading both extractions into the same graph naturally merges
    documentation onto the model node.

    Designed to be **opt-in**: `.yml` / `.yaml` files are NOT auto-classified
    as code by `detect.py` (they remain in `DOC_EXTENSIONS` so non-dbt YAML
    corpora — Kubernetes, Kustomize, Helm — keep going through the semantic
    pipeline). Callers that want dbt YAML structurally extracted invoke this
    function explicitly (e.g. `graphify ast-only` does so for dbt projects).
    """
    try:
        import tree_sitter_yaml as tsy
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_yaml not installed. Run: pip install tree-sitter-yaml"}

    try:
        language = Language(tsy.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    str_path = str(path)
    file_nid = _make_id("yaml", path.stem)
    nodes: list[dict] = [{"id": file_nid, "label": path.name, "file_type": "code",
                          "source_file": str_path, "source_location": "L1"}]
    edges: list[dict] = []
    seen_ids: set[str] = {file_nid}

    def _read(n) -> str:
        return source[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    def _ensure_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                          "source_file": str_path, "source_location": f"L{line}"})

    def _add_named_node(section: str, name_val: str, line: int) -> None:
        if section == "sources":
            nid = _make_id("source", name_val)
            label, rel = f"source.{name_val}", "defines_source"
        elif section == "seeds":
            nid = _make_id("seed", name_val)
            label, rel = f"seed.{name_val}", "defines_seed"
        elif section == "snapshots":
            nid = _make_id("snapshot", name_val)
            label, rel = name_val, "documents_snapshot"
        elif section == "models":
            # Same id `extract_dbt_sql` produces for the corresponding .sql
            nid = _make_id(name_val)
            label, rel = name_val, "documents"
        else:
            nid = _make_id(section, name_val)
            label, rel = f"{section}.{name_val}", f"defines_{section}"
        _ensure_node(nid, label, line)
        edges.append({"source": file_nid, "target": nid, "relation": rel,
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "source_location": f"L{line}", "weight": 1.0})

    def walk_pairs(node, current_section: str | None = None) -> None:
        for child in node.children:
            if child.type in ("block_mapping_pair", "flow_pair"):
                key_node = child.child_by_field_name("key")
                value_node = child.child_by_field_name("value")
                if key_node is None or value_node is None:
                    nk = [c for c in child.children if c.is_named]
                    if len(nk) >= 2:
                        key_node, value_node = nk[0], nk[1]
                if key_node is None:
                    continue
                key_text = _read(key_node).strip()
                line = child.start_point[0] + 1

                # Top-level section — descend with section context
                if current_section is None and key_text in _DBT_YAML_TOP_SECTIONS:
                    if value_node:
                        walk_pairs(value_node, current_section=key_text)
                    continue
                # Inside a section, "name: foo" defines an entity
                if current_section and key_text == "name" and value_node:
                    name_val = _read(value_node).strip().strip("'\"")
                    if name_val and "{{" not in name_val:
                        _add_named_node(current_section, name_val, line)
                    continue
                # Skip per-entity sub-sections (columns/tests/etc.)
                if current_section and key_text in _DBT_YAML_LEAF_KEYS:
                    continue
                # `tables:` inside sources keeps section=sources
                if current_section == "sources" and key_text == "tables" and value_node:
                    walk_pairs(value_node, current_section="sources")
                    continue
                # Recurse with same section
                if value_node:
                    walk_pairs(value_node, current_section=current_section)
            else:
                walk_pairs(child, current_section=current_section)

    walk_pairs(root)
    return {"nodes": nodes, "edges": edges}


def extract_sql(path: Path) -> dict:
    """Extract tables, views, functions, and relationships from .sql files via tree-sitter.

    When the file lives inside a dbt project (a `dbt_project.yml` sits in any parent
    directory) or contains Jinja markers (`{{` or `{%`), routes to `extract_dbt_sql`
    which understands ref()/source()/var()/config(). Falls back to the plain
    tree-sitter-sql path when `tree-sitter-jinja` is unavailable."""
    try:
        source_peek = path.read_bytes()
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    looks_dbt = _is_dbt_project_file(path) or b"{{" in source_peek or b"{%" in source_peek
    if looks_dbt:
        result = extract_dbt_sql(path)
        if "error" not in result or "tree_sitter_jinja not installed" not in result.get("error", ""):
            return result
        # Fall through to plain SQL extractor when jinja parser is missing

    try:
        import tree_sitter_sql as tssql
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_sql not installed. Run: pip install tree-sitter-sql"}

    try:
        language = Language(tssql.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = re.sub(r"[^a-z0-9]", "_", path.stem.lower())
    str_path = str(path)
    file_nid = _make_id(str_path)
    nodes: list[dict] = [{"id": file_nid, "label": path.name, "file_type": "code",
                           "source_file": str_path, "source_location": None}]
    edges: list[dict] = []
    seen_ids: set[str] = {file_nid}
    table_nids: dict[str, str] = {}  # name → nid for reference resolution

    def _read(n) -> str:
        return source[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    def _obj_name(n) -> str | None:
        for c in n.children:
            if c.type == "object_reference":
                for cc in c.children:
                    if cc.type == "identifier":
                        return _read(cc)
        return None

    def _add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                           "source_file": str_path, "source_location": f"L{line}"})
            edges.append({"source": file_nid, "target": nid, "relation": "contains",
                           "confidence": "EXTRACTED", "source_file": str_path,
                           "source_location": f"L{line}", "weight": 1.0})

    def _add_edge(src: str, tgt: str, relation: str, line: int) -> None:
        edges.append({"source": src, "target": tgt, "relation": relation,
                       "confidence": "EXTRACTED", "source_file": str_path,
                       "source_location": f"L{line}", "weight": 1.0})

    def walk(node) -> None:
        t = node.type
        line = node.start_point[0] + 1

        if t == "create_table":
            name = _obj_name(node)
            if name:
                nid = _make_id(stem, name)
                _add_node(nid, name, line)
                table_nids[name.lower()] = nid
                # Foreign key REFERENCES
                for col in node.children:
                    if col.type == "column_definitions":
                        for cd in col.children:
                            if cd.type != "column_definition":
                                continue
                            ref_name: str | None = None
                            found_ref = False
                            for cc in cd.children:
                                if cc.type == "keyword_references":
                                    found_ref = True
                                elif found_ref and cc.type == "object_reference":
                                    for ccc in cc.children:
                                        if ccc.type == "identifier":
                                            ref_name = _read(ccc)
                                    break
                            if ref_name:
                                ref_nid = _make_id(stem, ref_name)
                                _add_edge(nid, ref_nid, "references", line)

        elif t == "create_view":
            name = _obj_name(node)
            if name:
                nid = _make_id(stem, name)
                _add_node(nid, name, line)
                table_nids[name.lower()] = nid
                # FROM/JOIN table references inside view body
                _walk_from_refs(node, nid, line)

        elif t == "create_function":
            name = _obj_name(node)
            if name:
                nid = _make_id(stem, name)
                _add_node(nid, f"{name}()", line)
                _walk_from_refs(node, nid, line)

        elif t == "create_procedure":
            name = _obj_name(node)
            if name:
                nid = _make_id(stem, name)
                _add_node(nid, f"{name}()", line)
                _walk_from_refs(node, nid, line)

        for child in node.children:
            walk(child)

    def _walk_from_refs(node, caller_nid: str, line: int) -> None:
        """Recursively find FROM/JOIN table references inside a node."""
        if node.type in ("from", "join"):
            for c in node.children:
                if c.type == "relation":
                    for cc in c.children:
                        if cc.type == "object_reference":
                            for ccc in cc.children:
                                if ccc.type == "identifier":
                                    tbl = _read(ccc)
                                    tbl_nid = _make_id(stem, tbl)
                                    _add_edge(caller_nid, tbl_nid, "reads_from",
                                              c.start_point[0] + 1)
        for child in node.children:
            _walk_from_refs(child, caller_nid, line)

    for stmt in root.children:
        if stmt.type == "statement":
            for child in stmt.children:
                walk(child)

    return {"nodes": nodes, "edges": edges}


def extract_terraform(path: Path) -> dict:
    """Extract resources, modules, data sources, variables, outputs and their cross-references
    from .tf / .hcl files via tree-sitter-hcl. Supports Terraform-specific reference patterns:
    `module.x`, `var.y`, `data.x.y`, `local.z`, and `<resource_type>.<name>`."""
    try:
        import tree_sitter_hcl as tshcl
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_hcl not installed. Run: pip install tree-sitter-hcl"}

    try:
        language = Language(tshcl.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    str_path = str(path)
    file_nid = _make_id(str_path)
    nodes: list[dict] = [{"id": file_nid, "label": path.name, "file_type": "code",
                          "source_file": str_path, "source_location": None}]
    edges: list[dict] = []
    seen_ids: set[str] = {file_nid}

    def _read(n) -> str:
        return source[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    def _strip_quotes(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            return s[1:-1]
        return s

    def _ensure_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                          "source_file": str_path, "source_location": f"L{line}" if line else None})

    def _add_edge(src: str, tgt: str, relation: str, line: int) -> None:
        edges.append({"source": src, "target": tgt, "relation": relation,
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "source_location": f"L{line}", "weight": 1.0})

    _BLOCK_TYPES = {"resource", "module", "data", "variable", "output", "locals",
                    "provider", "terraform", "check"}
    _REF_PATTERNS = [
        (re.compile(r"\bmodule\.([A-Za-z_][\w-]*)\b"), "uses_module", "module"),
        (re.compile(r"\bvar\.([A-Za-z_][\w-]*)\b"), "uses_var", "var"),
        (re.compile(r"\bdata\.([A-Za-z_][\w-]*)\.([A-Za-z_][\w-]*)\b"), "uses_data", "data"),
        (re.compile(r"\blocal\.([A-Za-z_][\w-]*)\b"), "uses_local", "local"),
    ]
    _RESOURCE_REF = re.compile(r"\b([a-z][a-z0-9_]+\.[A-Za-z_][\w-]+)\.[A-Za-z_]")
    _NON_RESOURCE_PREFIXES = {"module", "var", "data", "local", "self", "each", "count", "path", "terraform"}

    def walk(node) -> None:
        if node.type == "block":
            named = [c for c in node.children if c.is_named]
            if not named or named[0].type != "identifier":
                for c in node.children:
                    walk(c)
                return
            block_type_node = named[0]
            block_type = _read(block_type_node)
            if block_type not in _BLOCK_TYPES:
                for c in node.children:
                    walk(c)
                return

            labels: list[str] = []
            for c in node.children:
                if c.type == "string_lit":
                    labels.append(_strip_quotes(_read(c)).strip("\""))

            line = node.start_point[0] + 1
            if block_type == "resource" and len(labels) >= 2:
                block_id = _make_id("tf", labels[0], labels[1])
                block_label = f"{labels[0]}.{labels[1]}"
            elif block_type == "module" and labels:
                block_id = _make_id("tf_module", labels[0])
                block_label = f"module.{labels[0]}"
            elif block_type == "data" and len(labels) >= 2:
                block_id = _make_id("tf_data", labels[0], labels[1])
                block_label = f"data.{labels[0]}.{labels[1]}"
            elif block_type == "variable" and labels:
                block_id = _make_id("tf_var", labels[0])
                block_label = f"var.{labels[0]}"
            elif block_type == "output" and labels:
                block_id = _make_id("tf_output", labels[0])
                block_label = f"output.{labels[0]}"
            else:
                block_id = _make_id("tf", block_type, *labels)
                block_label = block_type + ("." + ".".join(labels) if labels else "")

            _ensure_node(block_id, block_label, line)
            _add_edge(file_nid, block_id, "contains", line)

            body_text = _read(node)
            for pattern, rel, kind in _REF_PATTERNS:
                for m in pattern.finditer(body_text):
                    if kind == "module":
                        tid = _make_id("tf_module", m.group(1))
                        tlabel = f"module.{m.group(1)}"
                    elif kind == "var":
                        tid = _make_id("tf_var", m.group(1))
                        tlabel = f"var.{m.group(1)}"
                    elif kind == "data":
                        tid = _make_id("tf_data", m.group(1), m.group(2))
                        tlabel = f"data.{m.group(1)}.{m.group(2)}"
                    elif kind == "local":
                        tid = _make_id("tf_local", m.group(1))
                        tlabel = f"local.{m.group(1)}"
                    else:
                        continue
                    if tid != block_id:
                        _ensure_node(tid, tlabel, 0)
                        _add_edge(block_id, tid, rel, line)

            for m in _RESOURCE_REF.finditer(body_text):
                ref_target = m.group(1)
                parts = ref_target.split(".")
                if len(parts) != 2:
                    continue
                type_part, name_part = parts
                if type_part in _NON_RESOURCE_PREFIXES:
                    continue
                tid = _make_id("tf", type_part, name_part)
                if tid != block_id:
                    _ensure_node(tid, ref_target, 0)
                    _add_edge(block_id, tid, "references_resource", line)
            return

        for c in node.children:
            walk(c)

    walk(root)
    return {"nodes": nodes, "edges": edges}


# ── Airflow / Astronomer Cosmos ───────────────────────────────────────────────

_AIRFLOW_IMPORT_MARKERS = (
    b"from airflow", b"import airflow",
    b"from cosmos", b"import cosmos",
    b"from astronomer", b"import astronomer",
)
_AIRFLOW_OPERATOR_SUFFIXES = ("Operator", "Sensor")
_AIRFLOW_TASKGROUP_NAMES = {"TaskGroup", "DbtTaskGroup"}
_AIRFLOW_DAG_NAMES = {"DAG", "DbtDag"}


def _is_airflow_file(source: bytes) -> bool:
    """Cheap test: does this Python file mention an airflow / cosmos import in its
    head?  Only DAG files will."""
    head = source[:5000]
    return any(marker in head for marker in _AIRFLOW_IMPORT_MARKERS)


def extract_airflow_dag(path: Path) -> dict:
    """Extract Airflow / Astronomer Cosmos DAG structure from a .py file:
    DAGs, tasks, task dependencies (`>>`/`<<`/`chain()`), Cosmos `DbtDag`
    bindings (dbt project + select/exclude selectors), `BashOperator` dbt
    selectors, `Variable.get`/`BaseHook.get_connection` references, and
    `Dataset` declarations.

    Returns an EMPTY result for non-Airflow Python files (detected by a head
    scan for `from airflow` / `from cosmos` imports), so it's safe to dispatch
    on every `.py` file.
    """
    try:
        import tree_sitter_python as tspy
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_python not installed"}

    try:
        source = path.read_bytes()
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    if not _is_airflow_file(source):
        return {"nodes": [], "edges": []}

    try:
        parser = Parser(Language(tspy.language()))
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    str_path = str(path)
    file_nid = _make_id("airflow", path.stem)
    nodes: list[dict] = [{"id": file_nid, "label": path.name, "file_type": "code",
                          "source_file": str_path, "source_location": "L1"}]
    edges: list[dict] = []
    seen_ids: set[str] = {file_nid}
    # Map Python variable name → node id, so `t1 >> t2` can resolve to graph nodes.
    var_to_node: dict[str, str] = {}
    # Module-level `NAME = "string"` constants, so we can resolve
    # `BaseHook.get_connection(STANNP_CONN_ID)` to the actual literal.
    var_to_str: dict[str, str] = {}
    # Module-level `NAME = SomeCall(...)` assignments, so DbtDag(render_config=name)
    # can be resolved back to the RenderConfig(...) call AST.
    var_to_call_node: dict[str, object] = {}

    def _read(n) -> str:
        return source[n.start_byte:n.end_byte].decode("utf-8", errors="replace")

    def _strip_quotes(s: str) -> str:
        s = s.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            return s[1:-1]
        return s

    def _resolve_string(node) -> str | None:
        """Extract a string literal from various AST shapes:
        plain `string`, `concatenated_string` (`"a" "b"`),
        `parenthesized_expression` wrapping any of the above, f-string
        (literal parts are kept, interpolations are stripped), and bare
        `identifier` resolved against `var_to_str`."""
        if node is None:
            return None
        t = node.type
        if t == "string":
            # f-string handling: walk children, keep `string_content`, drop `interpolation`
            parts = []
            has_interp = False
            for c in node.children:
                if c.type in ("string_content", "string_literal"):
                    parts.append(_read(c))
                elif c.type == "interpolation":
                    has_interp = True
                elif c.type in ("string_start", "string_end", "escape_sequence"):
                    if c.type == "escape_sequence":
                        # Decode common escapes naively
                        raw = _read(c)
                        parts.append(raw.encode().decode("unicode_escape", errors="replace"))
            if parts:
                return "".join(parts)
            # Fall back to text minus quotes
            return _strip_quotes(_read(node))
        if t == "concatenated_string":
            return "".join((_resolve_string(c) or "") for c in node.children if c.is_named)
        if t == "parenthesized_expression":
            for c in node.children:
                if c.is_named:
                    return _resolve_string(c)
            return None
        if t == "identifier":
            return var_to_str.get(_read(node))
        return None

    def _ensure_node(nid: str, label: str, line: int = 1, **extras) -> None:
        if nid in seen_ids:
            return
        seen_ids.add(nid)
        n = {"id": nid, "label": label, "file_type": "code",
             "source_file": str_path, "source_location": f"L{line}"}
        for k, v in extras.items():
            if v is not None:
                n[k] = v
        nodes.append(n)

    def _add_edge(src: str, tgt: str, relation: str, line: int = 0) -> None:
        edges.append({"source": src, "target": tgt, "relation": relation,
                      "confidence": "EXTRACTED", "source_file": str_path,
                      "source_location": f"L{line}" if line else "", "weight": 1.0})

    def _set_node_attr(nid: str, key: str, value) -> None:
        for n in nodes:
            if n["id"] == nid:
                n[key] = value
                return

    def _call_function_name(call_node) -> str:
        fn = call_node.child_by_field_name("function")
        if fn is None:
            return ""
        return _read(fn)

    def _kwarg_node(call_node, key: str):
        args = call_node.child_by_field_name("arguments")
        if args is None:
            return None
        for c in args.children:
            if c.type == "keyword_argument":
                k = c.child_by_field_name("name")
                v = c.child_by_field_name("value")
                if k and v and _read(k) == key:
                    return v
        return None

    def _kwarg_string(call_node, key: str) -> str | None:
        v = _kwarg_node(call_node, key)
        return _resolve_string(v) if v is not None else None

    def _resolve_call(node):
        """If node is a `call`, return it.  If it's an `identifier` we've seen
        bound to a call (e.g. `render_config = RenderConfig(...)`), return that
        underlying call AST.  Otherwise None."""
        if node is None:
            return None
        if node.type == "call":
            return node
        if node.type == "identifier":
            return var_to_call_node.get(_read(node))
        return None

    def _string_list(list_node) -> list[str]:
        if list_node is None or list_node.type != "list":
            return []
        return [_strip_quotes(_read(c)) for c in list_node.children if c.type == "string"]

    # ─── Pass 0: collect module-level variable bindings ────────────────────
    # `STANNP_CONN_ID = "stannp_api"` so Variable/Connection lookups resolve;
    # `render_config = RenderConfig(...)` so DbtDag(render_config=name) resolves.

    def _walk_p0(node) -> None:
        if node.type == "assignment":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is not None and left.type == "identifier" and right is not None:
                name = _read(left)
                resolved = _resolve_string(right)
                if resolved is not None:
                    var_to_str[name] = resolved
                elif right.type == "call":
                    var_to_call_node[name] = right
        for c in node.children:
            _walk_p0(c)

    _walk_p0(root)

    # ─── Pass 1: DAG / Operator / Cosmos / Dataset declarations ─────────────

    def _parse_dbt_selectors_from_bash(cmd: str, owner_nid: str, line: int) -> None:
        try:
            import shlex
            tokens = shlex.split(cmd, posix=True)
        except Exception:
            return
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in ("--select", "-s") and i + 1 < len(tokens):
                sel = tokens[i + 1]
                sel_nid = _make_id("dbt_selector", sel)
                _ensure_node(sel_nid, f"dbt: {sel}", line=line, dbt_selector=sel)
                _add_edge(owner_nid, sel_nid, "selects_dbt", line)
                i += 2
            elif t == "--exclude" and i + 1 < len(tokens):
                exc = tokens[i + 1]
                exc_nid = _make_id("dbt_selector", exc)
                _ensure_node(exc_nid, f"dbt: {exc}", line=line, dbt_selector=exc)
                _add_edge(owner_nid, exc_nid, "excludes_dbt", line)
                i += 2
            else:
                i += 1

    def _handle_call(call_node, var_name: str | None, line: int) -> str | None:
        fn_full = _call_function_name(call_node)
        bare = fn_full.split(".")[-1] if fn_full else ""

        # ── DAG ──
        if bare in _AIRFLOW_DAG_NAMES:
            dag_id = _kwarg_string(call_node, "dag_id") or var_name or "anonymous_dag"
            schedule = (_kwarg_string(call_node, "schedule")
                        or _kwarg_string(call_node, "schedule_interval"))
            nid = _make_id("airflow_dag", dag_id)
            _ensure_node(nid, f"DAG: {dag_id}", line=line,
                         dag_id=dag_id, dag_schedule=schedule, cosmos=(bare == "DbtDag"))
            _add_edge(file_nid, nid, "defines_dag", line)

            # Cosmos-specific: extract project_config + render_config selectors.
            # Both can be either inline `=ProjectConfig(...)` or assigned via a
            # variable upstream (`pc = ProjectConfig(...); DbtDag(project_config=pc)`).
            if bare == "DbtDag":
                proj_cfg = _resolve_call(_kwarg_node(call_node, "project_config"))
                if proj_cfg is not None:
                    project_path = None
                    args = proj_cfg.child_by_field_name("arguments")
                    if args:
                        # First positional string OR `dbt_project_path=` kwarg
                        for c in args.children:
                            if c.type == "string":
                                project_path = _resolve_string(c)
                                break
                            if c.type == "keyword_argument":
                                kk = c.child_by_field_name("name")
                                vv = c.child_by_field_name("value")
                                if kk and vv and _read(kk) in ("dbt_project_path", "project_path"):
                                    project_path = _resolve_string(vv)
                                    if project_path:
                                        break
                    if project_path:
                        proj_nid = _make_id("dbt_project", project_path)
                        _ensure_node(proj_nid, f"dbt project: {project_path}",
                                     line=line, dbt_project_path=project_path)
                        _add_edge(nid, proj_nid, "uses_dbt_project", line)
                render_cfg = _resolve_call(_kwarg_node(call_node, "render_config"))
                if render_cfg is not None:
                    select_node = _kwarg_node(render_cfg, "select")
                    exclude_node = _kwarg_node(render_cfg, "exclude")
                    for sel in _string_list(select_node):
                        sel_nid = _make_id("dbt_selector", sel)
                        _ensure_node(sel_nid, f"dbt: {sel}", line=line, dbt_selector=sel)
                        _add_edge(nid, sel_nid, "selects_dbt", line)
                    for exc in _string_list(exclude_node):
                        exc_nid = _make_id("dbt_selector", exc)
                        _ensure_node(exc_nid, f"dbt: {exc}", line=line, dbt_selector=exc)
                        _add_edge(nid, exc_nid, "excludes_dbt", line)

            if var_name:
                var_to_node[var_name] = nid
            return nid

        # ── Cosmos task group ──
        if bare in _AIRFLOW_TASKGROUP_NAMES:
            group_id = _kwarg_string(call_node, "group_id") or var_name or "task_group"
            nid = _make_id("airflow_taskgroup", group_id)
            _ensure_node(nid, f"task_group: {group_id}", line=line, group_id=group_id)
            _add_edge(file_nid, nid, "defines_taskgroup", line)
            if var_name:
                var_to_node[var_name] = nid
            return nid

        # ── Operator / Sensor ──
        if bare and any(bare.endswith(suf) for suf in _AIRFLOW_OPERATOR_SUFFIXES):
            task_id = _kwarg_string(call_node, "task_id") or var_name or "anonymous_task"
            nid = _make_id("airflow_task", task_id)
            _ensure_node(nid, f"task: {task_id}", line=line,
                         task_id=task_id, airflow_operator=bare)
            _add_edge(file_nid, nid, "defines_task", line)

            if bare == "BashOperator":
                cmd = _kwarg_string(call_node, "bash_command")
                if cmd:
                    _set_node_attr(nid, "bash_command", cmd[:300])
                    _parse_dbt_selectors_from_bash(cmd, nid, line)
            if bare == "PythonOperator":
                pc_node = _kwarg_node(call_node, "python_callable")
                if pc_node is not None and pc_node.type == "identifier":
                    callable_name = _read(pc_node)
                    callable_nid = _make_id("python_callable", callable_name)
                    _ensure_node(callable_nid, f"{callable_name}()",
                                 line=line, python_callable=callable_name)
                    _add_edge(nid, callable_nid, "calls_python", line)

            if var_name:
                var_to_node[var_name] = nid
            return nid

        # ── Dataset ──
        if bare == "Dataset":
            args = call_node.child_by_field_name("arguments")
            if args:
                for c in args.children:
                    if c.type == "string":
                        ds_uri = _strip_quotes(_read(c))
                        ds_nid = _make_id("airflow_dataset", ds_uri)
                        _ensure_node(ds_nid, f"dataset: {ds_uri}",
                                     line=line, dataset_uri=ds_uri)
                        if var_name:
                            var_to_node[var_name] = ds_nid
                        return ds_nid
        return None

    def _walk_p1(node) -> None:
        # `with DAG(...) as dag:` style
        if node.type == "with_statement":
            for c in node.children:
                if c.type != "with_clause":
                    continue
                for item in c.children:
                    if item.type != "with_item":
                        continue
                    for sub in item.children:
                        if sub.type == "as_pattern":
                            value_node = sub.child_by_field_name("value")
                            alias_node = sub.child_by_field_name("alias")
                            if value_node is None and sub.children:
                                value_node = sub.children[0]
                            if alias_node is None:
                                for x in sub.children:
                                    if x.type == "as_pattern_target":
                                        alias_node = x
                                        break
                            var_name = None
                            if alias_node is not None:
                                # alias_node may be `as_pattern_target` containing identifier
                                if alias_node.type == "identifier":
                                    var_name = _read(alias_node)
                                else:
                                    for x in alias_node.children:
                                        if x.type == "identifier":
                                            var_name = _read(x)
                                            break
                            if value_node is not None and value_node.type == "call":
                                _handle_call(value_node, var_name, node.start_point[0] + 1)
                        elif sub.type == "call":
                            _handle_call(sub, None, node.start_point[0] + 1)
        # `name = Operator(...)` style
        if node.type == "assignment":
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            if left is not None and right is not None and left.type == "identifier" and right.type == "call":
                _handle_call(right, _read(left), node.start_point[0] + 1)
        for c in node.children:
            _walk_p1(c)

    _walk_p1(root)

    # ─── Pass 2: task dependencies (>> << chain) ───────────────────────────

    def _resolve_dep_operand(node) -> list[str]:
        if node is None:
            return []
        if node.type == "identifier":
            nid = var_to_node.get(_read(node))
            return [nid] if nid else []
        if node.type == "list":
            out = []
            for c in node.children:
                if c.type == "identifier":
                    nid = var_to_node.get(_read(c))
                    if nid:
                        out.append(nid)
            return out
        if node.type == "binary_operator":
            # chained `t1 >> t2 >> t3` — resolve recursively, emit edges, return rightmost
            op = node.child_by_field_name("operator")
            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            left_nids = _resolve_dep_operand(left)
            right_nids = _resolve_dep_operand(right)
            if op is not None and _read(op) in (">>", "<<"):
                line = node.start_point[0] + 1
                if _read(op) == ">>":
                    for ln in left_nids:
                        for rn in right_nids:
                            _add_edge(ln, rn, "depends_on", line)
                    return right_nids
                else:
                    for ln in left_nids:
                        for rn in right_nids:
                            _add_edge(rn, ln, "depends_on", line)
                    return left_nids
            return right_nids or left_nids
        return []

    def _walk_p2(node) -> None:
        if node.type == "binary_operator":
            op = node.child_by_field_name("operator")
            if op is not None and _read(op) in (">>", "<<"):
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                left_nids = _resolve_dep_operand(left)
                right_nids = _resolve_dep_operand(right)
                line = node.start_point[0] + 1
                if _read(op) == ">>":
                    for ln in left_nids:
                        for rn in right_nids:
                            _add_edge(ln, rn, "depends_on", line)
                else:
                    for ln in left_nids:
                        for rn in right_nids:
                            _add_edge(rn, ln, "depends_on", line)
                return  # avoid walking into chained children twice
        if node.type == "call":
            fn_name = _call_function_name(node)
            bare = fn_name.split(".")[-1] if fn_name else ""
            if bare in ("chain", "chain_linear"):
                args = node.child_by_field_name("arguments")
                if args:
                    seq = []
                    for c in args.children:
                        if c.type == "identifier":
                            nid = var_to_node.get(_read(c))
                            if nid:
                                seq.append(nid)
                    line = node.start_point[0] + 1
                    for i in range(len(seq) - 1):
                        _add_edge(seq[i], seq[i + 1], "depends_on", line)
        for c in node.children:
            _walk_p2(c)

    _walk_p2(root)

    # ─── Pass 3: runtime dependencies (Variable, Connection) ────────────────

    def _walk_p3(node) -> None:
        if node.type == "call":
            fn_name = _call_function_name(node)
            args = node.child_by_field_name("arguments")
            line = node.start_point[0] + 1
            if fn_name == "Variable.get" and args is not None:
                first = next((c for c in args.children if c.is_named), None)
                var_name = _resolve_string(first)
                if var_name:
                    nid = _make_id("airflow_variable", var_name)
                    _ensure_node(nid, f"variable: {var_name}",
                                 line=line, airflow_variable=var_name)
                    _add_edge(file_nid, nid, "uses_variable", line)
            if fn_name == "BaseHook.get_connection" and args is not None:
                first = next((c for c in args.children if c.is_named), None)
                conn_name = _resolve_string(first)
                if conn_name:
                    nid = _make_id("airflow_connection", conn_name)
                    _ensure_node(nid, f"connection: {conn_name}",
                                 line=line, airflow_connection=conn_name)
                    _add_edge(file_nid, nid, "uses_connection", line)
        for c in node.children:
            _walk_p3(c)

    _walk_p3(root)

    return {"nodes": nodes, "edges": edges}


def extract_python_with_airflow(path: Path) -> dict:
    """Wrapper: runs the standard Python extractor and additionally augments
    the result with Airflow / Cosmos DAG structure when the file is an Airflow
    DAG (detected by import marker scan).  Non-Airflow .py files get the plain
    `extract_python` output, no overhead beyond a 5KB head read."""
    py_result = extract_python(path)
    af_result = extract_airflow_dag(path)
    if af_result.get("nodes"):
        # Skip the file-level airflow node — keep only its real children
        # (the python extractor already created a file-level node)
        af_nodes = af_result["nodes"]
        if af_nodes and af_nodes[0]["label"] == path.name:
            af_file_nid = af_nodes[0]["id"]
            # Remap edges from airflow file_nid → python file_nid for cleaner merge
            py_file_label = path.name
            py_file_id = next((n["id"] for n in py_result["nodes"]
                               if n.get("label") == py_file_label), None)
            if py_file_id:
                for e in af_result["edges"]:
                    if e["source"] == af_file_nid:
                        e["source"] = py_file_id
                    if e["target"] == af_file_nid:
                        e["target"] = py_file_id
                af_nodes = af_nodes[1:]  # drop airflow's file-level duplicate
        py_result["nodes"].extend(af_nodes)
        py_result["edges"].extend(af_result["edges"])
    return py_result


def extract_lua(path: Path) -> dict:
    """Extract functions, methods, require() imports, and calls from a .lua file."""
    return _extract_generic(path, _LUA_CONFIG)


def extract_swift(path: Path) -> dict:
    """Extract classes, structs, protocols, functions, imports, and calls from a .swift file."""
    return _extract_generic(path, _SWIFT_CONFIG)


# ── Julia extractor (custom walk) ────────────────────────────────────────────

def extract_julia(path: Path) -> dict:
    """Extract modules, structs, functions, imports, and calls from a .jl file."""
    try:
        import tree_sitter_julia as tsjulia
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree-sitter-julia not installed"}

    try:
        language = Language(tsjulia.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    function_bodies: list[tuple[str, object]] = []

    def add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({
                "id": nid,
                "label": label,
                "file_type": "code",
                "source_file": str_path,
                "source_location": f"L{line}",
            })

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", weight: float = 1.0) -> None:
        edges.append({
            "source": src,
            "target": tgt,
            "relation": relation,
            "confidence": confidence,
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": weight,
        })

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    def _func_name_from_signature(sig_node) -> str | None:
        """Extract function name from a Julia signature node (call_expression > identifier)."""
        for child in sig_node.children:
            if child.type == "call_expression":
                callee = child.children[0] if child.children else None
                if callee and callee.type == "identifier":
                    return _read_text(callee, source)
        return None

    def walk_calls(body_node, func_nid: str) -> None:
        if body_node is None:
            return
        t = body_node.type
        if t in ("function_definition", "short_function_definition"):
            return
        if t == "call_expression" and body_node.children:
            callee = body_node.children[0]
            # Direct call: foo(...)
            if callee.type == "identifier":
                callee_name = _read_text(callee, source)
                target_nid = _make_id(stem, callee_name)
                add_edge(func_nid, target_nid, "calls", body_node.start_point[0] + 1,
                         confidence="EXTRACTED")
            # Method call: obj.method(...)
            elif callee.type == "field_expression" and len(callee.children) >= 3:
                method_node = callee.children[-1]
                method_name = _read_text(method_node, source)
                target_nid = _make_id(stem, method_name)
                add_edge(func_nid, target_nid, "calls", body_node.start_point[0] + 1,
                         confidence="EXTRACTED")
        for child in body_node.children:
            walk_calls(child, func_nid)

    def walk(node, scope_nid: str) -> None:
        t = node.type

        # Module
        if t == "module_definition":
            name_node = next((c for c in node.children if c.type == "identifier"), None)
            if name_node:
                mod_name = _read_text(name_node, source)
                mod_nid = _make_id(stem, mod_name)
                line = node.start_point[0] + 1
                add_node(mod_nid, mod_name, line)
                add_edge(file_nid, mod_nid, "defines", line)
                for child in node.children:
                    walk(child, mod_nid)
            return

        # Struct (struct / mutable struct — both map to struct_definition in tree-sitter-julia)
        if t == "struct_definition":
            # type_head may contain: identifier (simple) or binary_expression (Foo <: Bar)
            type_head = next((c for c in node.children if c.type == "type_head"), None)
            if type_head:
                bin_expr = next((c for c in type_head.children if c.type == "binary_expression"), None)
                if bin_expr:
                    # First identifier is the struct name, last is the supertype
                    identifiers = [c for c in bin_expr.children if c.type == "identifier"]
                    if identifiers:
                        struct_name = _read_text(identifiers[0], source)
                        struct_nid = _make_id(stem, struct_name)
                        line = node.start_point[0] + 1
                        add_node(struct_nid, struct_name, line)
                        add_edge(scope_nid, struct_nid, "defines", line)
                        if len(identifiers) >= 2:
                            super_name = _read_text(identifiers[-1], source)
                            add_edge(struct_nid, _make_id(stem, super_name), "inherits",
                                     line, confidence="EXTRACTED")
                else:
                    name_node = next((c for c in type_head.children if c.type == "identifier"), None)
                    if name_node:
                        struct_name = _read_text(name_node, source)
                        struct_nid = _make_id(stem, struct_name)
                        line = node.start_point[0] + 1
                        add_node(struct_nid, struct_name, line)
                        add_edge(scope_nid, struct_nid, "defines", line)
            return

        # Abstract type
        if t == "abstract_definition":
            # type_head > identifier
            type_head = next((c for c in node.children if c.type == "type_head"), None)
            if type_head:
                name_node = next((c for c in type_head.children if c.type == "identifier"), None)
                if name_node:
                    abs_name = _read_text(name_node, source)
                    abs_nid = _make_id(stem, abs_name)
                    line = node.start_point[0] + 1
                    add_node(abs_nid, abs_name, line)
                    add_edge(scope_nid, abs_nid, "defines", line)
            return

        # Function: function foo(...) ... end
        if t == "function_definition":
            sig_node = next((c for c in node.children if c.type == "signature"), None)
            if sig_node:
                func_name = _func_name_from_signature(sig_node)
                if func_name:
                    func_nid = _make_id(stem, func_name)
                    line = node.start_point[0] + 1
                    add_node(func_nid, f"{func_name}()", line)
                    add_edge(scope_nid, func_nid, "defines", line)
                    function_bodies.append((func_nid, node))
            return

        # Short function: foo(x) = expr
        if t == "assignment":
            lhs = node.children[0] if node.children else None
            if lhs and lhs.type == "call_expression" and lhs.children:
                callee = lhs.children[0]
                if callee.type == "identifier":
                    func_name = _read_text(callee, source)
                    func_nid = _make_id(stem, func_name)
                    line = node.start_point[0] + 1
                    add_node(func_nid, f"{func_name}()", line)
                    add_edge(scope_nid, func_nid, "defines", line)
                    # Only walk the RHS (index 2 after lhs and operator) to avoid self-loops
                    rhs = node.children[-1] if len(node.children) >= 3 else None
                    if rhs:
                        function_bodies.append((func_nid, rhs))
            return

        # Using / Import
        if t in ("using_statement", "import_statement"):
            line = node.start_point[0] + 1
            for child in node.children:
                if child.type == "identifier":
                    mod_name = _read_text(child, source)
                    imp_nid = _make_id(mod_name)
                    add_node(imp_nid, mod_name, line)
                    add_edge(scope_nid, imp_nid, "imports", line)
                elif child.type == "selected_import":
                    identifiers = [c for c in child.children if c.type == "identifier"]
                    if identifiers:
                        pkg_name = _read_text(identifiers[0], source)
                        pkg_nid = _make_id(pkg_name)
                        add_node(pkg_nid, pkg_name, line)
                        add_edge(scope_nid, pkg_nid, "imports", line)
            return

        for child in node.children:
            walk(child, scope_nid)

    walk(root, file_nid)

    for func_nid, body_node in function_bodies:
        # For function_definition nodes, walk children directly to avoid
        # the boundary check returning early on the top-level node itself.
        # Skip the "signature" child — it contains the function's own call_expression
        # which would create a self-loop.
        if body_node.type == "function_definition":
            for child in body_node.children:
                if child.type != "signature":
                    walk_calls(child, func_nid)
        else:
            walk_calls(body_node, func_nid)

    return {"nodes": nodes, "edges": edges}


# ── Go extractor (custom walk) ────────────────────────────────────────────────

def extract_go(path: Path) -> dict:
    """Extract functions, methods, type declarations, and imports from a .go file."""
    try:
        import tree_sitter_go as tsgo
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree-sitter-go not installed"}

    try:
        language = Language(tsgo.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    # Use directory name as package scope so methods on the same type across
    # multiple files in a package share one canonical type node.
    pkg_scope = path.parent.name or stem
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    function_bodies: list[tuple[str, object]] = []
    go_imported_pkgs: set[str] = set()  # local names of imported packages

    def add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({
                "id": nid,
                "label": label,
                "file_type": "code",
                "source_file": str_path,
                "source_location": f"L{line}",
            })

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", weight: float = 1.0) -> None:
        edges.append({
            "source": src,
            "target": tgt,
            "relation": relation,
            "confidence": confidence,
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": weight,
        })

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    def walk(node) -> None:
        t = node.type

        if t == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                func_nid = _make_id(stem, func_name)
                add_node(func_nid, f"{func_name}()", line)
                add_edge(file_nid, func_nid, "contains", line)
                body = node.child_by_field_name("body")
                if body:
                    function_bodies.append((func_nid, body))
            return

        if t == "method_declaration":
            receiver = node.child_by_field_name("receiver")
            receiver_type: str | None = None
            if receiver:
                for param in receiver.children:
                    if param.type == "parameter_declaration":
                        type_node = param.child_by_field_name("type")
                        if type_node:
                            raw = _read_text(type_node, source).lstrip("*").strip()
                            receiver_type = raw
                        break
            name_node = node.child_by_field_name("name")
            if name_node:
                method_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                if receiver_type:
                    parent_nid = _make_id(pkg_scope, receiver_type)
                    add_node(parent_nid, receiver_type, line)
                    method_nid = _make_id(parent_nid, method_name)
                    add_node(method_nid, f".{method_name}()", line)
                    add_edge(parent_nid, method_nid, "method", line)
                else:
                    method_nid = _make_id(stem, method_name)
                    add_node(method_nid, f"{method_name}()", line)
                    add_edge(file_nid, method_nid, "contains", line)
                body = node.child_by_field_name("body")
                if body:
                    function_bodies.append((method_nid, body))
            return

        if t == "type_declaration":
            for child in node.children:
                if child.type == "type_spec":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        type_name = _read_text(name_node, source)
                        line = child.start_point[0] + 1
                        type_nid = _make_id(pkg_scope, type_name)
                        add_node(type_nid, type_name, line)
                        add_edge(file_nid, type_nid, "contains", line)
            return

        if t == "import_declaration":
            for child in node.children:
                if child.type == "import_spec_list":
                    for spec in child.children:
                        if spec.type == "import_spec":
                            path_node = spec.child_by_field_name("path")
                            if path_node:
                                raw = _read_text(path_node, source).strip('"')
                                # Prefix with go_pkg_ so stdlib names (e.g. "context")
                                # don't collide with local files of the same basename.
                                tgt_nid = _make_id("go", "pkg", raw)
                                add_edge(file_nid, tgt_nid, "imports_from", spec.start_point[0] + 1)
                                # Track local name (alias or last path segment)
                                alias = spec.child_by_field_name("name")
                                local_name = _read_text(alias, source) if alias else raw.split("/")[-1]
                                if local_name and local_name != "_" and local_name != ".":
                                    go_imported_pkgs.add(local_name)
                elif child.type == "import_spec":
                    path_node = child.child_by_field_name("path")
                    if path_node:
                        raw = _read_text(path_node, source).strip('"')
                        tgt_nid = _make_id("go", "pkg", raw)
                        add_edge(file_nid, tgt_nid, "imports_from", child.start_point[0] + 1)
                        alias = child.child_by_field_name("name")
                        local_name = _read_text(alias, source) if alias else raw.split("/")[-1]
                        if local_name and local_name != "_" and local_name != ".":
                            go_imported_pkgs.add(local_name)
            return

        for child in node.children:
            walk(child)

    walk(root)

    label_to_nid: dict[str, str] = {}
    for n in nodes:
        raw = n["label"]
        normalised = raw.strip("()").lstrip(".")
        label_to_nid[normalised.lower()] = n["id"]

    seen_call_pairs: set[tuple[str, str]] = set()
    raw_calls: list[dict] = []

    def walk_calls(node, caller_nid: str) -> None:
        if node.type in ("function_declaration", "method_declaration"):
            return
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            callee_name: str | None = None
            is_member_call: bool = False
            if func_node:
                if func_node.type == "identifier":
                    callee_name = _read_text(func_node, source)
                elif func_node.type == "selector_expression":
                    field = func_node.child_by_field_name("field")
                    operand = func_node.child_by_field_name("operand")
                    receiver_name = _read_text(operand, source) if operand else ""
                    # Package-qualified call (e.g. fmt.Println) → allow cross-file resolution.
                    # Receiver method call (e.g. s.logger.Log) → skip, no import evidence.
                    is_member_call = receiver_name not in go_imported_pkgs
                    if field:
                        callee_name = _read_text(field, source)
            if callee_name:
                tgt_nid = label_to_nid.get(callee_name.lower())
                if tgt_nid and tgt_nid != caller_nid:
                    pair = (caller_nid, tgt_nid)
                    if pair not in seen_call_pairs:
                        seen_call_pairs.add(pair)
                        line = node.start_point[0] + 1
                        edges.append({
                            "source": caller_nid,
                            "target": tgt_nid,
                            "relation": "calls",
                            "confidence": "EXTRACTED",
                            "source_file": str_path,
                            "source_location": f"L{line}",
                            "weight": 1.0,
                        })
                elif callee_name:
                    raw_calls.append({
                        "caller_nid": caller_nid,
                        "callee": callee_name,
                        "is_member_call": is_member_call,
                        "source_file": str_path,
                        "source_location": f"L{node.start_point[0] + 1}",
                    })
        for child in node.children:
            walk_calls(child, caller_nid)

    for caller_nid, body_node in function_bodies:
        walk_calls(body_node, caller_nid)

    valid_ids = seen_ids
    clean_edges = []
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in valid_ids and (tgt in valid_ids or edge["relation"] in ("imports", "imports_from")):
            clean_edges.append(edge)

    return {"nodes": nodes, "edges": clean_edges, "raw_calls": raw_calls}


# ── Rust extractor (custom walk) ──────────────────────────────────────────────

def extract_rust(path: Path) -> dict:
    """Extract functions, structs, enums, traits, impl methods, and use declarations from a .rs file."""
    try:
        import tree_sitter_rust as tsrust
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree-sitter-rust not installed"}

    try:
        language = Language(tsrust.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    function_bodies: list[tuple[str, object]] = []

    def add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({
                "id": nid,
                "label": label,
                "file_type": "code",
                "source_file": str_path,
                "source_location": f"L{line}",
            })

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", weight: float = 1.0) -> None:
        edges.append({
            "source": src,
            "target": tgt,
            "relation": relation,
            "confidence": confidence,
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": weight,
        })

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    def walk(node, parent_impl_nid: str | None = None) -> None:
        t = node.type

        if t == "function_item":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                if parent_impl_nid:
                    func_nid = _make_id(parent_impl_nid, func_name)
                    add_node(func_nid, f".{func_name}()", line)
                    add_edge(parent_impl_nid, func_nid, "method", line)
                else:
                    func_nid = _make_id(stem, func_name)
                    add_node(func_nid, f"{func_name}()", line)
                    add_edge(file_nid, func_nid, "contains", line)
                body = node.child_by_field_name("body")
                if body:
                    function_bodies.append((func_nid, body))
            return

        if t in ("struct_item", "enum_item", "trait_item"):
            name_node = node.child_by_field_name("name")
            if name_node:
                item_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                item_nid = _make_id(stem, item_name)
                add_node(item_nid, item_name, line)
                add_edge(file_nid, item_nid, "contains", line)
            return

        if t == "impl_item":
            type_node = node.child_by_field_name("type")
            impl_nid: str | None = None
            if type_node:
                type_name = _read_text(type_node, source).strip()
                impl_nid = _make_id(stem, type_name)
                add_node(impl_nid, type_name, node.start_point[0] + 1)
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    walk(child, parent_impl_nid=impl_nid)
            return

        if t == "use_declaration":
            arg = node.child_by_field_name("argument")
            if arg:
                raw = _read_text(arg, source)
                clean = raw.split("{")[0].rstrip(":").rstrip("*").rstrip(":")
                module_name = clean.split("::")[-1].strip()
                if module_name:
                    tgt_nid = _make_id(module_name)
                    add_edge(file_nid, tgt_nid, "imports_from", node.start_point[0] + 1)
            return

        for child in node.children:
            walk(child, parent_impl_nid=None)

    walk(root)

    label_to_nid: dict[str, str] = {}
    for n in nodes:
        raw = n["label"]
        normalised = raw.strip("()").lstrip(".")
        label_to_nid[normalised.lower()] = n["id"]

    seen_call_pairs: set[tuple[str, str]] = set()
    raw_calls: list[dict] = []

    def walk_calls(node, caller_nid: str) -> None:
        if node.type == "function_item":
            return
        if node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            callee_name: str | None = None
            is_member_call: bool = False
            if func_node:
                if func_node.type == "identifier":
                    callee_name = _read_text(func_node, source)
                elif func_node.type == "field_expression":
                    is_member_call = True
                    field = func_node.child_by_field_name("field")
                    if field:
                        callee_name = _read_text(field, source)
                elif func_node.type == "scoped_identifier":
                    name = func_node.child_by_field_name("name")
                    if name:
                        callee_name = _read_text(name, source)
            if callee_name:
                tgt_nid = label_to_nid.get(callee_name.lower())
                if tgt_nid and tgt_nid != caller_nid:
                    pair = (caller_nid, tgt_nid)
                    if pair not in seen_call_pairs:
                        seen_call_pairs.add(pair)
                        line = node.start_point[0] + 1
                        edges.append({
                            "source": caller_nid,
                            "target": tgt_nid,
                            "relation": "calls",
                            "confidence": "EXTRACTED",
                            "source_file": str_path,
                            "source_location": f"L{line}",
                            "weight": 1.0,
                        })
                else:
                    raw_calls.append({
                        "caller_nid": caller_nid,
                        "callee": callee_name,
                        "is_member_call": is_member_call,
                        "source_file": str_path,
                        "source_location": f"L{node.start_point[0] + 1}",
                    })
        for child in node.children:
            walk_calls(child, caller_nid)

    for caller_nid, body_node in function_bodies:
        walk_calls(body_node, caller_nid)

    valid_ids = seen_ids
    clean_edges = []
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src in valid_ids and (tgt in valid_ids or edge["relation"] in ("imports", "imports_from")):
            clean_edges.append(edge)

    return {"nodes": nodes, "edges": clean_edges, "raw_calls": raw_calls}


# ── Zig ───────────────────────────────────────────────────────────────────────

def extract_zig(path: Path) -> dict:
    """Extract functions, structs, enums, unions, and imports from a .zig file."""
    try:
        import tree_sitter_zig as tszig
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_zig not installed"}

    try:
        language = Language(tszig.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    function_bodies: list[tuple[str, Any]] = []

    def add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                          "source_file": str_path, "source_location": f"L{line}"})

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", weight: float = 1.0) -> None:
        edges.append({"source": src, "target": tgt, "relation": relation,
                      "confidence": confidence, "source_file": str_path,
                      "source_location": f"L{line}", "weight": weight})

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    def _extract_import(node) -> None:
        for child in node.children:
            if child.type == "builtin_function":
                bi = None
                args = None
                for c in child.children:
                    if c.type == "builtin_identifier":
                        bi = _read_text(c, source)
                    elif c.type == "arguments":
                        args = c
                if bi in ("@import", "@cImport") and args:
                    for arg in args.children:
                        if arg.type in ("string_literal", "string"):
                            raw = _read_text(arg, source).strip('"')
                            module_name = raw.split("/")[-1].split(".")[0]
                            if module_name:
                                tgt_nid = _make_id(module_name)
                                add_edge(file_nid, tgt_nid, "imports_from",
                                         node.start_point[0] + 1)
                            return
            elif child.type == "field_expression":
                _extract_import(child)
                return

    def walk(node, parent_struct_nid: str | None = None) -> None:
        t = node.type

        if t == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                func_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                if parent_struct_nid:
                    func_nid = _make_id(parent_struct_nid, func_name)
                    add_node(func_nid, f".{func_name}()", line)
                    add_edge(parent_struct_nid, func_nid, "method", line)
                else:
                    func_nid = _make_id(stem, func_name)
                    add_node(func_nid, f"{func_name}()", line)
                    add_edge(file_nid, func_nid, "contains", line)
                body = node.child_by_field_name("body")
                if body:
                    function_bodies.append((func_nid, body))
            return

        if t == "variable_declaration":
            name_node = None
            value_node = None
            for child in node.children:
                if child.type == "identifier":
                    name_node = child
                elif child.type in ("struct_declaration", "enum_declaration",
                                    "union_declaration", "builtin_function",
                                    "field_expression"):
                    value_node = child

            if value_node and value_node.type == "struct_declaration":
                if name_node:
                    struct_name = _read_text(name_node, source)
                    line = node.start_point[0] + 1
                    struct_nid = _make_id(stem, struct_name)
                    add_node(struct_nid, struct_name, line)
                    add_edge(file_nid, struct_nid, "contains", line)
                    for child in value_node.children:
                        walk(child, parent_struct_nid=struct_nid)
                return

            if value_node and value_node.type in ("enum_declaration", "union_declaration"):
                if name_node:
                    type_name = _read_text(name_node, source)
                    line = node.start_point[0] + 1
                    type_nid = _make_id(stem, type_name)
                    add_node(type_nid, type_name, line)
                    add_edge(file_nid, type_nid, "contains", line)
                return

            if value_node and value_node.type in ("builtin_function", "field_expression"):
                _extract_import(node)
            return

        for child in node.children:
            walk(child, parent_struct_nid)

    walk(root)

    seen_call_pairs: set[tuple[str, str]] = set()
    raw_calls: list[dict] = []

    def walk_calls(node, caller_nid: str) -> None:
        if node.type == "function_declaration":
            return
        if node.type == "call_expression":
            fn = node.child_by_field_name("function")
            if fn:
                fn_text = _read_text(fn, source)
                callee = fn_text.split(".")[-1]
                is_member_call = "." in fn_text
                tgt_nid = next((n["id"] for n in nodes if n["label"] in
                                (f"{callee}()", f".{callee}()")), None)
                if tgt_nid and tgt_nid != caller_nid:
                    pair = (caller_nid, tgt_nid)
                    if pair not in seen_call_pairs:
                        seen_call_pairs.add(pair)
                        add_edge(caller_nid, tgt_nid, "calls",
                                 node.start_point[0] + 1,
                                 confidence="EXTRACTED", weight=1.0)
                elif callee:
                    raw_calls.append({
                        "caller_nid": caller_nid,
                        "callee": callee,
                        "is_member_call": is_member_call,
                        "source_file": str_path,
                        "source_location": f"L{node.start_point[0] + 1}",
                    })
        for child in node.children:
            walk_calls(child, caller_nid)

    for caller_nid, body_node in function_bodies:
        walk_calls(body_node, caller_nid)

    clean_edges = [e for e in edges if e["source"] in seen_ids and
                   (e["target"] in seen_ids or e["relation"] == "imports_from")]
    return {"nodes": nodes, "edges": clean_edges, "raw_calls": raw_calls}


# ── PowerShell ────────────────────────────────────────────────────────────────

def extract_powershell(path: Path) -> dict:
    """Extract functions, classes, methods, and using statements from a .ps1 file."""
    try:
        import tree_sitter_powershell as tsps
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_powershell not installed"}

    try:
        language = Language(tsps.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    function_bodies: list[tuple[str, Any]] = []

    def add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                          "source_file": str_path, "source_location": f"L{line}"})

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", weight: float = 1.0) -> None:
        edges.append({"source": src, "target": tgt, "relation": relation,
                      "confidence": confidence, "source_file": str_path,
                      "source_location": f"L{line}", "weight": weight})

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    _PS_SKIP = frozenset({
        "using", "return", "if", "else", "elseif", "foreach", "for",
        "while", "do", "switch", "try", "catch", "finally", "throw",
        "break", "continue", "exit", "param", "begin", "process", "end",
    })

    def _find_script_block_body(node):
        for child in node.children:
            if child.type == "script_block":
                for sc in child.children:
                    if sc.type == "script_block_body":
                        return sc
                return child
        return None

    def walk(node, parent_class_nid: str | None = None) -> None:
        t = node.type

        if t == "function_statement":
            name_node = next((c for c in node.children if c.type == "function_name"), None)
            if name_node:
                func_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                func_nid = _make_id(stem, func_name)
                add_node(func_nid, f"{func_name}()", line)
                add_edge(file_nid, func_nid, "contains", line)
                body = _find_script_block_body(node)
                if body:
                    function_bodies.append((func_nid, body))
            return

        if t == "class_statement":
            name_node = next((c for c in node.children if c.type == "simple_name"), None)
            if name_node:
                class_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                class_nid = _make_id(stem, class_name)
                add_node(class_nid, class_name, line)
                add_edge(file_nid, class_nid, "contains", line)
                for child in node.children:
                    walk(child, parent_class_nid=class_nid)
            return

        if t == "class_method_definition":
            name_node = next((c for c in node.children if c.type == "simple_name"), None)
            if name_node:
                method_name = _read_text(name_node, source)
                line = node.start_point[0] + 1
                if parent_class_nid:
                    method_nid = _make_id(parent_class_nid, method_name)
                    add_node(method_nid, f".{method_name}()", line)
                    add_edge(parent_class_nid, method_nid, "method", line)
                else:
                    method_nid = _make_id(stem, method_name)
                    add_node(method_nid, f"{method_name}()", line)
                    add_edge(file_nid, method_nid, "contains", line)
                body = _find_script_block_body(node)
                if body:
                    function_bodies.append((method_nid, body))
            return

        if t == "command":
            cmd_name_node = next((c for c in node.children if c.type == "command_name"), None)
            if cmd_name_node:
                cmd_text = _read_text(cmd_name_node, source).lower()
                if cmd_text == "using":
                    tokens = []
                    for child in node.children:
                        if child.type == "command_elements":
                            for el in child.children:
                                if el.type == "generic_token":
                                    tokens.append(_read_text(el, source))
                    module_tokens = [t for t in tokens
                                     if t.lower() not in ("namespace", "module", "assembly")]
                    if module_tokens:
                        module_name = module_tokens[-1].split(".")[-1]
                        add_edge(file_nid, _make_id(module_name), "imports_from",
                                 node.start_point[0] + 1)
            return

        for child in node.children:
            walk(child, parent_class_nid)

    walk(root)

    label_to_nid = {n["label"].strip("()").lstrip(".").lower(): n["id"] for n in nodes}
    seen_call_pairs: set[tuple[str, str]] = set()
    raw_calls: list[dict] = []

    def walk_calls(node, caller_nid: str) -> None:
        if node.type in ("function_statement", "class_statement"):
            return
        if node.type == "command":
            cmd_name_node = next((c for c in node.children if c.type == "command_name"), None)
            if cmd_name_node:
                cmd_text = _read_text(cmd_name_node, source)
                if cmd_text.lower() not in _PS_SKIP:
                    tgt_nid = label_to_nid.get(cmd_text.lower())
                    if tgt_nid and tgt_nid != caller_nid:
                        pair = (caller_nid, tgt_nid)
                        if pair not in seen_call_pairs:
                            seen_call_pairs.add(pair)
                            add_edge(caller_nid, tgt_nid, "calls",
                                     node.start_point[0] + 1,
                                     confidence="EXTRACTED", weight=1.0)
                    elif cmd_text:
                        raw_calls.append({
                            "caller_nid": caller_nid,
                            "callee": cmd_text,
                            "is_member_call": False,
                            "source_file": str_path,
                            "source_location": f"L{node.start_point[0] + 1}",
                        })
        for child in node.children:
            walk_calls(child, caller_nid)

    for caller_nid, body_node in function_bodies:
        walk_calls(body_node, caller_nid)

    clean_edges = [e for e in edges if e["source"] in seen_ids and
                   (e["target"] in seen_ids or e["relation"] == "imports_from")]
    return {"nodes": nodes, "edges": clean_edges, "raw_calls": raw_calls}


# ── Cross-file import resolution ──────────────────────────────────────────────

def _resolve_cross_file_imports(
    per_file: list[dict],
    paths: list[Path],
) -> list[dict]:
    """
    Two-pass import resolution: turn file-level imports into class-level edges.

    Pass 1 - build a global map: class/function name → node_id, per stem.
    Pass 2 - for each `from .module import Name`, look up Name in the global
              map and add a direct INFERRED edge from each class in the
              importing file to the imported entity.

    This turns:
        auth.py --imports_from--> models.py          (obvious, filtered out)
    Into:
        DigestAuth --uses--> Response  [INFERRED]    (cross-file, interesting!)
        BasicAuth  --uses--> Request   [INFERRED]
    """
    try:
        import tree_sitter_python as tspython
        from tree_sitter import Language, Parser
    except ImportError:
        return []

    language = Language(tspython.language())
    parser = Parser(language)

    # Pass 1: name → node_id across all files
    # Map: stem → {ClassName: node_id}
    stem_to_entities: dict[str, dict[str, str]] = {}
    for file_result in per_file:
        for node in file_result.get("nodes", []):
            src = node.get("source_file", "")
            if not src:
                continue
            stem = Path(src).stem
            label = node.get("label", "")
            nid = node.get("id", "")
            # Only index real classes/functions (not file nodes, not method stubs)
            if label and not label.endswith((")", ".py")) and "_" not in label[:1]:
                stem_to_entities.setdefault(stem, {})[label] = nid

    # Pass 2: for each file, find `from .X import A, B, C` and resolve
    new_edges: list[dict] = []
    stem_to_path: dict[str, Path] = {p.stem: p for p in paths}

    for file_result, path in zip(per_file, paths):
        stem = _file_stem(path)
        str_path = str(path)

        # Find all classes defined in this file (the importers)
        local_classes = [
            n["id"] for n in file_result.get("nodes", [])
            if n.get("source_file") == str_path
            and not n["label"].endswith((")", ".py"))
            and n["id"] != _make_id(stem)  # exclude file-level node
        ]
        if not local_classes:
            continue

        # Parse imports from this file
        try:
            source = path.read_bytes()
            tree = parser.parse(source)
        except Exception:
            continue

        def walk_imports(node) -> None:
            if node.type == "import_from_statement":
                # Find the module name - handles both absolute and relative imports.
                # Relative: `from .models import X` → relative_import → dotted_name
                # Absolute: `from models import X`  → module_name field
                target_stem: str | None = None
                for child in node.children:
                    if child.type == "relative_import":
                        # Dig into relative_import → dotted_name → identifier
                        for sub in child.children:
                            if sub.type == "dotted_name":
                                raw = source[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                                target_stem = raw.split(".")[-1]
                                break
                        break
                    if child.type == "dotted_name" and target_stem is None:
                        raw = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                        target_stem = raw.split(".")[-1]

                if not target_stem or target_stem not in stem_to_entities:
                    return

                # Collect imported names: dotted_name children of import_from_statement
                # that come AFTER the 'import' keyword token.
                imported_names: list[str] = []
                past_import_kw = False
                for child in node.children:
                    if child.type == "import":
                        past_import_kw = True
                        continue
                    if not past_import_kw:
                        continue
                    if child.type == "dotted_name":
                        imported_names.append(
                            source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                        )
                    elif child.type == "aliased_import":
                        # `import X as Y` - take the original name
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            imported_names.append(
                                source[name_node.start_byte:name_node.end_byte].decode("utf-8", errors="replace")
                            )

                line = node.start_point[0] + 1
                for name in imported_names:
                    tgt_nid = stem_to_entities[target_stem].get(name)
                    if tgt_nid:
                        for src_class_nid in local_classes:
                            new_edges.append({
                                "source": src_class_nid,
                                "target": tgt_nid,
                                "relation": "uses",
                                "confidence": "INFERRED",
                                "source_file": str_path,
                                "source_location": f"L{line}",
                                "weight": 0.8,
                            })
            for child in node.children:
                walk_imports(child)

        walk_imports(tree.root_node)

    return new_edges


def _resolve_cross_file_java_imports(
    per_file: list[dict],
    paths: list[Path],
) -> list[dict]:
    """Two-pass Java import resolution.

    Pass 1: build a global index {ClassName: [node_id, ...]} across all Java nodes.
    Pass 2: re-parse each Java file; for every `import a.b.C;`, resolve C against
    the index. Wildcard and stdlib imports produce no edge.
    """
    try:
        import tree_sitter_java as tsjava
        from tree_sitter import Language, Parser
    except ImportError:
        return []

    language = Language(tsjava.language())
    parser = Parser(language)

    # Pass 1: class-name → node_id index (only internal, uppercase-starting names)
    name_to_ids: dict[str, list[str]] = {}
    for file_result in per_file:
        for node in file_result.get("nodes", []):
            label = node.get("label", "")
            nid = node.get("id", "")
            src = node.get("source_file", "")
            if not label or not nid or not src:
                continue
            if label.endswith(")") or label.endswith(".java"):
                continue
            if not label[0].isalpha() or not label[0].isupper():
                continue
            name_to_ids.setdefault(label, []).append(nid)

    # Pass 2: resolve imports to real node IDs
    new_edges: list[dict] = []
    seen_pairs: set[tuple[str, str]] = set()
    for path in paths:
        file_nid = _make_id(str(path))
        try:
            source = path.read_bytes()
            tree = parser.parse(source)
        except Exception:
            continue

        def walk(n) -> None:
            if n.type == "import_declaration":
                raw = _read_text(n, source).strip()
                body = raw[len("import"):].strip().rstrip(";").strip()
                if body.startswith("static "):
                    body = body[len("static "):].strip()
                if body.endswith(".*"):
                    return
                parts = body.split(".")
                if not parts:
                    return
                last = parts[-1]
                if last and last[0].islower() and len(parts) >= 2:
                    last = parts[-2]
                at_line = n.start_point[0] + 1
                for tgt_nid in name_to_ids.get(last, []):
                    if tgt_nid == file_nid:
                        continue
                    key = (file_nid, tgt_nid)
                    if key in seen_pairs:
                        continue
                    seen_pairs.add(key)
                    new_edges.append({
                        "source": file_nid,
                        "target": tgt_nid,
                        "relation": "imports",
                        "confidence": "EXTRACTED",
                        "confidence_score": 1.0,
                        "source_file": str(path),
                        "source_location": f"L{at_line}",
                        "weight": 1.0,
                    })
            for child in n.children:
                walk(child)

        walk(tree.root_node)

    return new_edges


def extract_objc(path: Path) -> dict:
    """Extract interfaces, implementations, protocols, methods, and imports from .m/.mm/.h files."""
    try:
        import tree_sitter_objc as tsobjc
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_objc not installed"}

    try:
        language = Language(tsobjc.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    method_bodies: list[tuple[str, Any]] = []

    def add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                          "source_file": str_path, "source_location": f"L{line}"})

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", weight: float = 1.0) -> None:
        edges.append({"source": src, "target": tgt, "relation": relation,
                      "confidence": confidence, "source_file": str_path,
                      "source_location": f"L{line}", "weight": weight})

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    def _read(node) -> str:
        return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    def _get_name(node, field: str) -> str | None:
        n = node.child_by_field_name(field)
        return _read(n) if n else None

    def walk(node, parent_nid: str | None = None) -> None:
        t = node.type
        line = node.start_point[0] + 1

        if t == "preproc_include":
            # #import <Foundation/Foundation.h> or #import "MyClass.h"
            for child in node.children:
                if child.type == "system_lib_string":
                    raw = _read(child).strip("<>")
                    module = raw.split("/")[-1].replace(".h", "")
                    if module:
                        tgt_nid = _make_id(module)
                        add_edge(file_nid, tgt_nid, "imports", line)
                elif child.type == "string_literal":
                    # recurse into string_literal to find string_content
                    for sub in child.children:
                        if sub.type == "string_content":
                            raw = _read(sub)
                            module = raw.split("/")[-1].replace(".h", "")
                            if module:
                                tgt_nid = _make_id(module)
                                add_edge(file_nid, tgt_nid, "imports", line)
            return

        if t == "class_interface":
            # @interface ClassName : SuperClass <Protocols>
            # children: @interface, identifier(name), ':', identifier(super), parameterized_arguments, ...
            identifiers = [c for c in node.children if c.type == "identifier"]
            if not identifiers:
                for child in node.children:
                    walk(child, parent_nid)
                return
            name = _read(identifiers[0])
            cls_nid = _make_id(stem, name)
            add_node(cls_nid, name, line)
            add_edge(file_nid, cls_nid, "contains", line)
            # superclass is second identifier after ':'
            colon_seen = False
            for child in node.children:
                if child.type == ":":
                    colon_seen = True
                elif colon_seen and child.type == "identifier":
                    super_nid = _make_id(_read(child))
                    add_edge(cls_nid, super_nid, "inherits", line)
                    colon_seen = False
                elif child.type == "parameterized_arguments":
                    # protocols adopted
                    for sub in child.children:
                        if sub.type == "type_name":
                            for s in sub.children:
                                if s.type == "type_identifier":
                                    proto_nid = _make_id(_read(s))
                                    add_edge(cls_nid, proto_nid, "imports", line)
                elif child.type == "method_declaration":
                    walk(child, cls_nid)
            return

        if t == "class_implementation":
            # @implementation ClassName
            name = None
            for child in node.children:
                if child.type == "identifier":
                    name = _read(child)
                    break
            if not name:
                for child in node.children:
                    walk(child, parent_nid)
                return
            impl_nid = _make_id(stem, name)
            if impl_nid not in seen_ids:
                add_node(impl_nid, name, line)
                add_edge(file_nid, impl_nid, "contains", line)
            for child in node.children:
                if child.type == "implementation_definition":
                    for sub in child.children:
                        walk(sub, impl_nid)
            return

        if t == "protocol_declaration":
            name = None
            for child in node.children:
                if child.type == "identifier":
                    name = _read(child)
                    break
            if name:
                proto_nid = _make_id(stem, name)
                add_node(proto_nid, f"<{name}>", line)
                add_edge(file_nid, proto_nid, "contains", line)
                for child in node.children:
                    walk(child, proto_nid)
            return

        if t in ("method_declaration", "method_definition"):
            container = parent_nid or file_nid
            # method name is the first identifier child (simple selector)
            # for compound selectors: identifier + method_parameter pairs
            parts = []
            for child in node.children:
                if child.type == "identifier":
                    parts.append(_read(child))
                elif child.type == "method_parameter":
                    for sub in child.children:
                        if sub.type == "identifier":
                            # selector keyword before ':'
                            pass
            method_name = "".join(parts) if parts else None
            if method_name:
                method_nid = _make_id(container, method_name)
                add_node(method_nid, f"-{method_name}", line)
                add_edge(container, method_nid, "method", line)
                if t == "method_definition":
                    method_bodies.append((method_nid, node))
            return

        for child in node.children:
            walk(child, parent_nid)

    walk(root)

    # Second pass: resolve calls inside method bodies
    all_method_nids = {n["id"] for n in nodes if n["id"] != file_nid}
    seen_calls: set[tuple[str, str]] = set()
    for caller_nid, body_node in method_bodies:
        def walk_calls(n) -> None:
            if n.type == "message_expression":
                # [receiver selector]
                for child in n.children:
                    if child.type in ("selector", "keyword_argument_list"):
                        sel = []
                        if child.type == "selector":
                            sel.append(_read(child))
                        else:
                            for sub in child.children:
                                if sub.type == "keyword_argument":
                                    for s in sub.children:
                                        if s.type == "selector":
                                            sel.append(_read(s))
                        method_name = "".join(sel)
                        for candidate in all_method_nids:
                            if candidate.endswith(_make_id("", method_name).lstrip("_")):
                                pair = (caller_nid, candidate)
                                if pair not in seen_calls and caller_nid != candidate:
                                    seen_calls.add(pair)
                                    add_edge(caller_nid, candidate, "calls", body_node.start_point[0] + 1,
                                             confidence="EXTRACTED", weight=1.0)
            for child in n.children:
                walk_calls(child)
        walk_calls(body_node)

    return {"nodes": nodes, "edges": edges, "input_tokens": 0, "output_tokens": 0}


def extract_elixir(path: Path) -> dict:
    """Extract modules, functions, imports, and calls from a .ex/.exs file."""
    try:
        import tree_sitter_elixir as tselixir
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree_sitter_elixir not installed"}

    try:
        language = Language(tselixir.language())
        parser = Parser(language)
        source = path.read_bytes()
        tree = parser.parse(source)
        root = tree.root_node
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}

    stem = _file_stem(path)
    str_path = str(path)
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_ids: set[str] = set()
    function_bodies: list[tuple[str, Any]] = []

    def add_node(nid: str, label: str, line: int) -> None:
        if nid not in seen_ids:
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": label, "file_type": "code",
                          "source_file": str_path, "source_location": f"L{line}"})

    def add_edge(src: str, tgt: str, relation: str, line: int,
                 confidence: str = "EXTRACTED", weight: float = 1.0) -> None:
        edges.append({"source": src, "target": tgt, "relation": relation,
                      "confidence": confidence, "source_file": str_path,
                      "source_location": f"L{line}", "weight": weight})

    file_nid = _make_id(str(path))
    add_node(file_nid, path.name, 1)

    _IMPORT_KEYWORDS = frozenset({"alias", "import", "require", "use"})

    def _get_alias_text(node) -> str | None:
        for child in node.children:
            if child.type == "alias":
                return source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
        return None

    def walk(node, parent_module_nid: str | None = None) -> None:
        if node.type != "call":
            for child in node.children:
                walk(child, parent_module_nid)
            return

        identifier_node = None
        arguments_node = None
        do_block_node = None
        for child in node.children:
            if child.type == "identifier":
                identifier_node = child
            elif child.type == "arguments":
                arguments_node = child
            elif child.type == "do_block":
                do_block_node = child

        if identifier_node is None:
            for child in node.children:
                walk(child, parent_module_nid)
            return

        keyword = source[identifier_node.start_byte:identifier_node.end_byte].decode("utf-8", errors="replace")
        line = node.start_point[0] + 1

        if keyword == "defmodule":
            module_name = _get_alias_text(arguments_node) if arguments_node else None
            if not module_name:
                return
            module_nid = _make_id(stem, module_name)
            add_node(module_nid, module_name, line)
            add_edge(file_nid, module_nid, "contains", line)
            if do_block_node:
                for child in do_block_node.children:
                    walk(child, parent_module_nid=module_nid)
            return

        if keyword in ("def", "defp"):
            func_name = None
            if arguments_node:
                for child in arguments_node.children:
                    if child.type == "call":
                        for sub in child.children:
                            if sub.type == "identifier":
                                func_name = source[sub.start_byte:sub.end_byte].decode("utf-8", errors="replace")
                                break
                    elif child.type == "identifier":
                        func_name = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                        break
            if not func_name:
                return
            container = parent_module_nid or file_nid
            func_nid = _make_id(container, func_name)
            add_node(func_nid, f"{func_name}()", line)
            if parent_module_nid:
                add_edge(parent_module_nid, func_nid, "method", line)
            else:
                add_edge(file_nid, func_nid, "contains", line)
            if do_block_node:
                function_bodies.append((func_nid, do_block_node))
            return

        if keyword in _IMPORT_KEYWORDS and arguments_node:
            module_name = _get_alias_text(arguments_node)
            if module_name:
                tgt_nid = _make_id(module_name)
                add_edge(file_nid, tgt_nid, "imports", line)
            return

        for child in node.children:
            walk(child, parent_module_nid)

    walk(root)

    label_to_nid: dict[str, str] = {}
    for n in nodes:
        normalised = n["label"].strip("()").lstrip(".")
        label_to_nid[normalised.lower()] = n["id"]

    seen_call_pairs: set[tuple[str, str]] = set()
    raw_calls: list[dict] = []
    _SKIP_KEYWORDS = frozenset({
        "def", "defp", "defmodule", "defmacro", "defmacrop",
        "defstruct", "defprotocol", "defimpl", "defguard",
        "alias", "import", "require", "use",
        "if", "unless", "case", "cond", "with", "for",
    })

    def walk_calls(node, caller_nid: str) -> None:
        if node.type != "call":
            for child in node.children:
                walk_calls(child, caller_nid)
            return
        for child in node.children:
            if child.type == "identifier":
                kw = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                if kw in _SKIP_KEYWORDS:
                    for c in node.children:
                        walk_calls(c, caller_nid)
                    return
                break
        callee_name: str | None = None
        is_member_call: bool = False
        for child in node.children:
            if child.type == "dot":
                is_member_call = True
                dot_text = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                parts = dot_text.rstrip(".").split(".")
                if parts:
                    callee_name = parts[-1]
                break
            if child.type == "identifier":
                callee_name = source[child.start_byte:child.end_byte].decode("utf-8", errors="replace")
                break
        if callee_name:
            tgt_nid = label_to_nid.get(callee_name.lower())
            if tgt_nid and tgt_nid != caller_nid:
                pair = (caller_nid, tgt_nid)
                if pair not in seen_call_pairs:
                    seen_call_pairs.add(pair)
                    add_edge(caller_nid, tgt_nid, "calls",
                             node.start_point[0] + 1, confidence="EXTRACTED", weight=1.0)
            else:
                raw_calls.append({
                    "caller_nid": caller_nid,
                    "callee": callee_name,
                    "is_member_call": is_member_call,
                    "source_file": str_path,
                    "source_location": f"L{node.start_point[0] + 1}",
                })
        for child in node.children:
            walk_calls(child, caller_nid)

    for caller_nid, body in function_bodies:
        walk_calls(body, caller_nid)

    clean_edges = [e for e in edges if e["source"] in seen_ids and
                   (e["target"] in seen_ids or e["relation"] == "imports")]
    return {"nodes": nodes, "edges": clean_edges, "raw_calls": raw_calls, "input_tokens": 0, "output_tokens": 0}


# ── Main extract and collect_files ────────────────────────────────────────────


def _check_tree_sitter_version() -> None:
    """Raise a clear error if tree-sitter is too old for the new Language API."""
    try:
        from tree_sitter import LANGUAGE_VERSION
    except ImportError:
        raise ImportError(
            "tree-sitter is not installed. Run: pip install 'tree-sitter>=0.23.0'"
        )
    # Language API v2 starts at LANGUAGE_VERSION 14
    if LANGUAGE_VERSION < 14:
        import tree_sitter as _ts
        raise RuntimeError(
            f"tree-sitter {getattr(_ts, '__version__', 'unknown')} is too old. "
            f"graphify requires tree-sitter >= 0.23.0 (Language API v2). "
            f"Run: pip install --upgrade tree-sitter"
        )


def extract(paths: list[Path], cache_root: Path | None = None) -> dict:
    """Extract AST nodes and edges from a list of code files.

    Two-pass process:
    1. Per-file structural extraction (classes, functions, imports)
    2. Cross-file import resolution: turns file-level imports into
       class-level INFERRED edges (DigestAuth --uses--> Response)

    Args:
        paths: files to extract from
        cache_root: explicit root for graphify-out/cache/ (overrides the
            inferred common path prefix). Pass Path('.') when running on a
            subdirectory so the cache stays at ./graphify-out/cache/.
    """
    _check_tree_sitter_version()
    per_file: list[dict] = []

    # Infer a common root for cache keys (use first diverging segment, not sum of all matches)
    try:
        if not paths:
            root = Path(".")
        elif len(paths) == 1:
            root = paths[0].parent
        else:
            min_parts = min(len(p.parts) for p in paths)
            common_len = 0
            for i in range(min_parts):
                if len({p.parts[i] for p in paths}) == 1:
                    common_len += 1
                else:
                    break
            root = Path(*paths[0].parts[:common_len]) if common_len else Path(".")
    except Exception:
        root = Path(".")
    root = root.resolve()

    _DISPATCH: dict[str, Any] = {
        ".py": extract_python_with_airflow,
        ".js": extract_js,
        ".jsx": extract_js,
        ".mjs": extract_js,
        ".ts": extract_js,
        ".tsx": extract_js,
        ".go": extract_go,
        ".rs": extract_rust,
        ".java": extract_java,
        ".c": extract_c,
        ".h": extract_c,
        ".cpp": extract_cpp,
        ".cc": extract_cpp,
        ".cxx": extract_cpp,
        ".hpp": extract_cpp,
        ".rb": extract_ruby,
        ".cs": extract_csharp,
        ".kt": extract_kotlin,
        ".kts": extract_kotlin,
        ".scala": extract_scala,
        ".php": extract_php,
        ".swift": extract_swift,
        ".lua": extract_lua,
        ".toc": extract_lua,
        ".zig": extract_zig,
        ".ps1": extract_powershell,
        ".ex": extract_elixir,
        ".exs": extract_elixir,
        ".m": extract_objc,
        ".mm": extract_objc,
        ".jl": extract_julia,
        ".vue": extract_js,
        ".svelte": extract_js,
        ".dart": extract_dart,
        ".v": extract_verilog,
        ".sv": extract_verilog,
        ".sql": extract_sql,
        ".tf": extract_terraform,
        ".hcl": extract_terraform,
        # .yml / .yaml are NOT auto-classified as code (still in DOC_EXTENSIONS
        # so non-dbt YAML stays in the semantic pipeline), but if a caller
        # explicitly hands us one — e.g. `graphify ast-only` discovering schema
        # YAML in a dbt project — route through the dbt YAML extractor.
        # In a non-dbt context, extract_dbt_yaml emits an empty/single-node
        # result, which is the right no-op behaviour (no false structural edges).
        ".yml": extract_dbt_yaml,
        ".yaml": extract_dbt_yaml,
    }

    total = len(paths)
    _PROGRESS_INTERVAL = 100
    for i, path in enumerate(paths):
        if total >= _PROGRESS_INTERVAL and i % _PROGRESS_INTERVAL == 0 and i > 0:
            # Progress goes to stderr so machine-readable consumers
            # (graphify ast-only, query, etc.) get clean stdout.
            print(f"  AST extraction: {i}/{total} files ({i * 100 // total}%)", flush=True, file=sys.stderr)
        # .blade.php must be checked before suffix lookup since Path.suffix returns .php
        if path.name.endswith(".blade.php"):
            extractor = extract_blade
        else:
            extractor = _DISPATCH.get(path.suffix)
        if extractor is None:
            continue
        cached = load_cached(path, cache_root or root)
        if cached is not None:
            per_file.append(cached)
            continue
        result = extractor(path)
        if "error" not in result:
            save_cached(path, result, cache_root or root)
        per_file.append(result)
    if total >= _PROGRESS_INTERVAL:
        print(f"  AST extraction: {total}/{total} files (100%)", flush=True, file=sys.stderr)

    all_nodes: list[dict] = []
    all_edges: list[dict] = []
    for result in per_file:
        all_nodes.extend(result.get("nodes", []))
        all_edges.extend(result.get("edges", []))

    # Remap file node IDs from absolute-path-derived to project-relative so
    # graph.json edge endpoints are stable across machines (#502)
    id_remap: dict[str, str] = {}
    for path in paths:
        old_id = _make_id(str(path))
        try:
            new_id = _make_id(str(path.relative_to(root)))
        except ValueError:
            continue
        if old_id != new_id:
            id_remap[old_id] = new_id
    if id_remap:
        for n in all_nodes:
            if n.get("id") in id_remap:
                n["id"] = id_remap[n["id"]]
        for e in all_edges:
            if e.get("source") in id_remap:
                e["source"] = id_remap[e["source"]]
            if e.get("target") in id_remap:
                e["target"] = id_remap[e["target"]]

    # Add cross-file class-level edges (Python only - uses Python parser internally)
    py_paths = [p for p in paths if p.suffix == ".py"]
    if py_paths:
        py_results = [r for r, p in zip(per_file, paths) if p.suffix == ".py"]
        try:
            cross_file_edges = _resolve_cross_file_imports(py_results, py_paths)
            all_edges.extend(cross_file_edges)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Cross-file import resolution failed, skipping: %s", exc)

    # Cross-file Java import resolution
    java_paths = [p for p in paths if p.suffix == ".java"]
    if java_paths:
        java_results = [r for r, p in zip(per_file, paths) if p.suffix == ".java"]
        try:
            all_edges.extend(_resolve_cross_file_java_imports(java_results, java_paths))
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Java cross-file import resolution failed, skipping: %s", exc)

    # Cross-file call resolution for all languages
    # Each extractor saved unresolved calls in raw_calls. Now that we have all
    # nodes from all files, resolve any callee that exists in another file.
    # Build name → ALL matching node IDs so we can skip ambiguous common names
    # (e.g. "log", "execute", "find") that appear in multiple files — resolving
    # those inflates god_nodes ranking with spurious cross-file edges.
    global_label_to_nids: dict[str, list[str]] = {}
    for n in all_nodes:
        raw = n.get("label", "")
        normalised = raw.strip("()").lstrip(".")
        if normalised:
            key = normalised.lower()
            global_label_to_nids.setdefault(key, []).append(n["id"])

    existing_pairs = {(e["source"], e["target"]) for e in all_edges}
    for result in per_file:
        for rc in result.get("raw_calls", []):
            callee = rc.get("callee", "")
            if not callee:
                continue
            # Skip member-call callees: obj.log() → "log" has no import evidence
            # and collides with any top-level function named "log" in the corpus.
            if rc.get("is_member_call"):
                continue
            candidates = global_label_to_nids.get(callee.lower(), [])
            # Skip ambiguous names that resolve to multiple nodes — these are
            # common short names (log, execute, find) with no import evidence
            # to pick the right target; emitting all edges inflates god_nodes.
            if len(candidates) != 1:
                continue
            tgt = candidates[0]
            caller = rc["caller_nid"]
            if tgt != caller and (caller, tgt) not in existing_pairs:
                existing_pairs.add((caller, tgt))
                all_edges.append({
                    "source": caller,
                    "target": tgt,
                    "relation": "calls",
                    "confidence": "INFERRED",
                    "confidence_score": 0.8,
                    "source_file": rc.get("source_file", ""),
                    "source_location": rc.get("source_location"),
                    "weight": 1.0,
                })

    # Relativize source_file fields so paths are portable across machines (#555)
    for item in all_nodes + all_edges:
        sf = item.get("source_file")
        if not sf:
            continue
        sf_path = Path(sf)
        if not sf_path.is_absolute():
            continue
        try:
            item["source_file"] = str(sf_path.relative_to(root))
        except ValueError:
            pass

    # Cross-system bridges — runs after all per-file extractors finish so it
    # can spot entities that exist in multiple systems (e.g. an Airflow Cosmos
    # selector `tag:X` and a dbt model that carries tag X). Runs deterministically
    # off node attributes already present; no LLM, no extra parsing.
    try:
        from graphify.bridges import compute_bridges
        bridge_nodes, bridge_edges = compute_bridges(all_nodes, all_edges)
        all_nodes.extend(bridge_nodes)
        all_edges.extend(bridge_edges)
    except Exception:
        # Bridge resolution is opportunistic — never fail the whole extraction
        # because a heuristic crashed. Real failures should be debuggable via
        # the bridges_to edge tags but never block the pipeline.
        pass

    return {
        "nodes": all_nodes,
        "edges": all_edges,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def collect_files(target: Path, *, follow_symlinks: bool = False, root: Path | None = None) -> list[Path]:
    if target.is_file():
        return [target]
    from graphify.detect import _load_graphifyignore, _is_ignored, CODE_EXTENSIONS
    # Derive from CODE_EXTENSIONS so this set never drifts behind the canonical
    # list in detect.py (historically it had to be updated by hand and silently
    # missed `.dart`, `.ejs`, `.ex`, `.exs`, `.jl`, `.jsx`, `.mjs`, `.sv`,
    # `.svelte`, `.v`, `.vue`).
    _EXTENSIONS = set(CODE_EXTENSIONS)
    ignore_root = root if root is not None else target
    patterns = _load_graphifyignore(ignore_root)

    def _ignored(p: Path) -> bool:
        return bool(patterns and _is_ignored(p, ignore_root, patterns))

    if not follow_symlinks:
        results: list[Path] = []
        for ext in sorted(_EXTENSIONS):
            results.extend(
                p for p in target.rglob(f"*{ext}")
                if not any(part.startswith(".") for part in p.parts)
                and not _ignored(p)
            )
        return sorted(results)
    # Walk with symlink following + cycle detection
    results = []
    for dirpath, dirnames, filenames in os.walk(target, followlinks=True):
        if os.path.islink(dirpath):
            real = os.path.realpath(dirpath)
            parent_real = os.path.realpath(os.path.dirname(dirpath))
            if parent_real == real or parent_real.startswith(real + os.sep):
                dirnames.clear()
                continue
        dp = Path(dirpath)
        if any(part.startswith(".") for part in dp.parts):
            dirnames.clear()
            continue
        for fname in filenames:
            p = dp / fname
            if p.suffix in _EXTENSIONS and not fname.startswith(".") and not _ignored(p):
                results.append(p)
    return sorted(results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m graphify.extract <file_or_dir> ...", file=sys.stderr)
        sys.exit(1)

    paths: list[Path] = []
    for arg in sys.argv[1:]:
        paths.extend(collect_files(Path(arg)))

    result = extract(paths)
    print(json.dumps(result, indent=2))
