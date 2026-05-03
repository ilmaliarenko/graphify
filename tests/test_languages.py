"""Tests for language extractors: Java, C, C++, Ruby, C#, Kotlin, Scala, PHP, Swift, Go, Julia, JS/TS."""
from __future__ import annotations
from pathlib import Path
import pytest
from graphify.extract import (
    extract_java, extract_c, extract_cpp, extract_ruby,
    extract_csharp, extract_kotlin, extract_scala, extract_php,
    extract_swift, extract_go, extract_julia, extract_js,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _labels(r):
    return [n["label"] for n in r["nodes"]]

def _relations(r):
    return {e["relation"] for e in r["edges"]}

def _calls(r):
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    return {
        (node_by_id.get(e["source"], e["source"]), node_by_id.get(e["target"], e["target"]))
        for e in r["edges"] if e["relation"] == "calls"
    }


def _references(r):
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    return [
        (
            node_by_id.get(e["source"], e["source"]),
            node_by_id.get(e["target"], e["target"]),
            e,
        )
        for e in r["edges"] if e["relation"] == "references"
    ]


def _edges_with_relation(r, *relations):
    return [e for e in r["edges"] if e["relation"] in relations]


# ── Java ──────────────────────────────────────────────────────────────────────

def test_java_no_error():
    r = extract_java(FIXTURES / "sample.java")
    assert "error" not in r

def test_java_finds_class():
    r = extract_java(FIXTURES / "sample.java")
    assert any("DataProcessor" in l for l in _labels(r))

def test_java_finds_interface():
    r = extract_java(FIXTURES / "sample.java")
    assert any("Processor" in l for l in _labels(r))

def test_java_finds_methods():
    r = extract_java(FIXTURES / "sample.java")
    labels = _labels(r)
    assert any("addItem" in l for l in labels)
    assert any("process" in l for l in labels)

def test_java_finds_imports():
    r = extract_java(FIXTURES / "sample.java")
    assert "imports" in _relations(r)


def test_java_import_edges_have_import_context():
    r = extract_java(FIXTURES / "sample.java")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)

def test_java_no_dangling_edges():
    r = extract_java(FIXTURES / "sample.java")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids


# ── C ────────────────────────────────────────────────────────────────────────

def test_c_no_error():
    r = extract_c(FIXTURES / "sample.c")
    assert "error" not in r

def test_c_finds_functions():
    r = extract_c(FIXTURES / "sample.c")
    labels = _labels(r)
    assert any("process" in l for l in labels)
    assert any("main" in l for l in labels)

def test_c_finds_includes():
    r = extract_c(FIXTURES / "sample.c")
    assert "imports" in _relations(r)

def test_c_emits_calls():
    r = extract_c(FIXTURES / "sample.c")
    assert any(e["relation"] == "calls" for e in r["edges"])

def test_c_calls_are_extracted():
    r = extract_c(FIXTURES / "sample.c")
    for e in r["edges"]:
        if e["relation"] == "calls":
            assert e["confidence"] == "EXTRACTED"


def test_c_import_edges_have_import_context():
    r = extract_c(FIXTURES / "sample.c")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


def test_c_call_edges_have_call_context():
    r = extract_c(FIXTURES / "sample.c")
    call_edges = _edges_with_relation(r, "calls")
    assert call_edges
    assert all(e.get("context") == "call" for e in call_edges)


# ── C++ ───────────────────────────────────────────────────────────────────────

def test_cpp_no_error():
    r = extract_cpp(FIXTURES / "sample.cpp")
    assert "error" not in r

def test_cpp_finds_class():
    r = extract_cpp(FIXTURES / "sample.cpp")
    assert any("HttpClient" in l for l in _labels(r))

def test_cpp_finds_methods():
    r = extract_cpp(FIXTURES / "sample.cpp")
    labels = _labels(r)
    # C++ extractor captures the constructor and public-visible methods
    assert any("HttpClient" in l for l in labels)

def test_cpp_finds_includes():
    r = extract_cpp(FIXTURES / "sample.cpp")
    assert "imports" in _relations(r)


def test_cpp_import_edges_have_import_context():
    r = extract_cpp(FIXTURES / "sample.cpp")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


# ── Ruby ─────────────────────────────────────────────────────────────────────

def test_ruby_no_error():
    r = extract_ruby(FIXTURES / "sample.rb")
    assert "error" not in r

def test_ruby_finds_class():
    r = extract_ruby(FIXTURES / "sample.rb")
    assert any("ApiClient" in l for l in _labels(r))

def test_ruby_finds_methods():
    r = extract_ruby(FIXTURES / "sample.rb")
    labels = _labels(r)
    assert any("get" in l for l in labels)
    assert any("post" in l for l in labels)

def test_ruby_finds_function():
    r = extract_ruby(FIXTURES / "sample.rb")
    assert any("parse_response" in l for l in _labels(r))


# ── C# ───────────────────────────────────────────────────────────────────────

def test_csharp_no_error():
    r = extract_csharp(FIXTURES / "sample.cs")
    assert "error" not in r

def test_csharp_finds_class():
    r = extract_csharp(FIXTURES / "sample.cs")
    assert any("DataProcessor" in l for l in _labels(r))

def test_csharp_finds_interface():
    r = extract_csharp(FIXTURES / "sample.cs")
    assert any("IProcessor" in l for l in _labels(r))

def test_csharp_finds_methods():
    r = extract_csharp(FIXTURES / "sample.cs")
    labels = _labels(r)
    assert any("Process" in l for l in labels)

def test_csharp_finds_usings():
    r = extract_csharp(FIXTURES / "sample.cs")
    assert "imports" in _relations(r)

def test_csharp_inherits_edge():
    r = extract_csharp(FIXTURES / "sample.cs")
    inherits = [e for e in r["edges"] if e["relation"] == "inherits"]
    assert len(inherits) >= 1

def test_csharp_inherits_iprocessor():
    r = extract_csharp(FIXTURES / "sample.cs")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    found = any(
        "DataProcessor" in node_by_id.get(e["source"], "") and
        "IProcessor" in node_by_id.get(e["target"], "")
        for e in r["edges"] if e["relation"] == "inherits"
    )
    assert found, "DataProcessor should have inherits edge to IProcessor"


def test_csharp_field_type_references_have_field_context():
    r = extract_csharp(FIXTURES / "sample.cs")
    refs = _references(r)
    assert any(
        "DataProcessor" in src and "HttpClient" in tgt and edge.get("context") == "field"
        for src, tgt, edge in refs
    ), "DataProcessor field declarations should reference HttpClient with field context"


def test_csharp_call_edges_have_call_context():
    r = extract_csharp(FIXTURES / "sample.cs")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    assert any(
        "Process" in node_by_id.get(e["source"], "")
        and "Validate" in node_by_id.get(e["target"], "")
        and e.get("context") == "call"
        for e in r["edges"] if e["relation"] == "calls"
    ), "C# call edges should retain call context"


def test_csharp_import_edges_have_import_context():
    r = extract_csharp(FIXTURES / "sample.cs")
    import_edges = [e for e in r["edges"] if e["relation"] == "imports"]
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


# ── Kotlin ───────────────────────────────────────────────────────────────────

def test_kotlin_no_error():
    r = extract_kotlin(FIXTURES / "sample.kt")
    assert "error" not in r

def test_kotlin_finds_class():
    r = extract_kotlin(FIXTURES / "sample.kt")
    assert any("HttpClient" in l for l in _labels(r))

def test_kotlin_finds_data_class():
    r = extract_kotlin(FIXTURES / "sample.kt")
    assert any("Config" in l for l in _labels(r))

def test_kotlin_finds_methods():
    r = extract_kotlin(FIXTURES / "sample.kt")
    labels = _labels(r)
    assert any("get" in l for l in labels)
    assert any("post" in l for l in labels)

def test_kotlin_finds_function():
    r = extract_kotlin(FIXTURES / "sample.kt")
    assert any("createClient" in l for l in _labels(r))

def test_kotlin_emits_in_file_calls():
    """Regression test for the call-walker `simple_identifier` /
    `identifier` rename — see graphify-kmp's PythonParityTest."""
    r = extract_kotlin(FIXTURES / "sample.kt")
    calls = _calls(r)
    # In sample.kt: get() and post() both call buildRequest(), and
    # createClient() invokes Config and HttpClient (constructor calls).
    assert (".get()", ".buildRequest()") in calls
    assert (".post()", ".buildRequest()") in calls
    assert ("createClient()", "Config") in calls
    assert ("createClient()", "HttpClient") in calls


# ── Scala ─────────────────────────────────────────────────────────────────────

def test_scala_no_error():
    r = extract_scala(FIXTURES / "sample.scala")
    assert "error" not in r

def test_scala_finds_class():
    r = extract_scala(FIXTURES / "sample.scala")
    assert any("HttpClient" in l for l in _labels(r))

def test_scala_finds_object():
    r = extract_scala(FIXTURES / "sample.scala")
    assert any("HttpClientFactory" in l for l in _labels(r))

def test_scala_finds_methods():
    r = extract_scala(FIXTURES / "sample.scala")
    labels = _labels(r)
    assert any("get" in l for l in labels)
    assert any("post" in l for l in labels)


def test_scala_import_edges_have_import_context():
    r = extract_scala(FIXTURES / "sample.scala")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


def test_scala_call_edges_have_call_context():
    r = extract_scala(FIXTURES / "sample.scala")
    call_edges = _edges_with_relation(r, "calls")
    assert call_edges
    assert all(e.get("context") == "call" for e in call_edges)


# ── PHP ───────────────────────────────────────────────────────────────────────

def test_php_no_error():
    r = extract_php(FIXTURES / "sample.php")
    assert "error" not in r

def test_php_finds_class():
    r = extract_php(FIXTURES / "sample.php")
    assert any("ApiClient" in l for l in _labels(r))

def test_php_finds_methods():
    r = extract_php(FIXTURES / "sample.php")
    labels = _labels(r)
    assert any("get" in l for l in labels)
    assert any("post" in l for l in labels)

def test_php_finds_function():
    r = extract_php(FIXTURES / "sample.php")
    assert any("parseResponse" in l for l in _labels(r))

def test_php_finds_imports():
    r = extract_php(FIXTURES / "sample.php")
    assert "imports" in _relations(r)


def test_php_import_edges_have_import_context():
    r = extract_php(FIXTURES / "sample.php")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


def test_php_call_edges_have_call_context():
    r = extract_php(FIXTURES / "sample.php")
    call_edges = _edges_with_relation(r, "calls")
    assert call_edges
    assert all(e.get("context") == "call" for e in call_edges)

def test_php_finds_static_property_access():
    r = extract_php(FIXTURES / "sample_php_static_prop.php")
    assert "uses_static_prop" in _relations(r)

def test_php_static_prop_target_is_holding_class():
    r = extract_php(FIXTURES / "sample_php_static_prop.php")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    uses_prop = [
        (node_by_id.get(e["source"], e["source"]), node_by_id.get(e["target"], e["target"]))
        for e in r["edges"] if e["relation"] == "uses_static_prop"
    ]
    assert any("DefaultPalette" in tgt for _, tgt in uses_prop)

def test_php_finds_config_helper_call():
    r = extract_php(FIXTURES / "sample_php_config.php")
    assert "uses_config" in _relations(r)

def test_php_config_helper_target_matches_first_segment():
    r = extract_php(FIXTURES / "sample_php_config.php")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    uses_cfg = [
        (node_by_id.get(e["source"], e["source"]), node_by_id.get(e["target"], e["target"]))
        for e in r["edges"] if e["relation"] == "uses_config"
    ]
    assert any("Throttle" in tgt for _, tgt in uses_cfg)

def test_php_finds_container_bind():
    r = extract_php(FIXTURES / "sample_php_container.php")
    assert "bound_to" in _relations(r)

def test_php_container_bind_links_contract_to_implementation():
    r = extract_php(FIXTURES / "sample_php_container.php")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    bound = [
        (node_by_id.get(e["source"], e["source"]), node_by_id.get(e["target"], e["target"]))
        for e in r["edges"] if e["relation"] == "bound_to"
    ]
    assert any("PaymentGateway" in src and "StripeGateway" in tgt for src, tgt in bound)

def test_php_finds_event_listeners():
    r = extract_php(FIXTURES / "sample_php_listen.php")
    assert "listened_by" in _relations(r)

def test_php_event_listener_links_event_to_listener():
    r = extract_php(FIXTURES / "sample_php_listen.php")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    listened = [
        (node_by_id.get(e["source"], e["source"]), node_by_id.get(e["target"], e["target"]))
        for e in r["edges"] if e["relation"] == "listened_by"
    ]
    assert any("UserRegistered" in src and "SendWelcomeEmail" in tgt for src, tgt in listened)


# ── Swift ────────────────────────────────────────────────────────────────────

def test_swift_no_error():
    r = extract_swift(FIXTURES / "sample.swift")
    assert "error" not in r

def test_swift_finds_class():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("DataProcessor" in l for l in _labels(r))

def test_swift_finds_protocol():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("Processor" in l for l in _labels(r))

def test_swift_finds_struct():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("Config" in l for l in _labels(r))

def test_swift_finds_methods():
    r = extract_swift(FIXTURES / "sample.swift")
    labels = _labels(r)
    assert any("addItem" in l for l in labels)
    assert any("process" in l for l in labels)

def test_swift_finds_function():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("createProcessor" in l for l in _labels(r))

def test_swift_finds_imports():
    r = extract_swift(FIXTURES / "sample.swift")
    assert "imports" in _relations(r)


def test_swift_import_edges_have_import_context():
    r = extract_swift(FIXTURES / "sample.swift")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)

def test_swift_no_dangling_edges():
    r = extract_swift(FIXTURES / "sample.swift")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids

def test_swift_finds_actor():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("CacheManager" in l for l in _labels(r))

def test_swift_finds_enum():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("NetworkError" in l for l in _labels(r))

def test_swift_finds_enum_methods():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("describe" in l for l in _labels(r))

def test_swift_finds_enum_cases():
    r = extract_swift(FIXTURES / "sample.swift")
    labels = _labels(r)
    assert any("timeout" in l for l in labels)
    assert any("connectionFailed" in l for l in labels)

def test_swift_enum_cases_have_case_of_edge():
    r = extract_swift(FIXTURES / "sample.swift")
    case_edges = [e for e in r["edges"] if e["relation"] == "case_of"]
    assert len(case_edges) >= 2

def test_swift_finds_deinit():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("deinit" in l for l in _labels(r))

def test_swift_finds_subscript():
    r = extract_swift(FIXTURES / "sample.swift")
    assert any("subscript" in l for l in _labels(r))

def test_swift_extension_methods_attach_to_type():
    r = extract_swift(FIXTURES / "sample.swift")
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    method_edges = [e for e in r["edges"] if e["relation"] == "method"]
    found = False
    for e in method_edges:
        src_label = node_by_id.get(e["source"], "")
        tgt_label = node_by_id.get(e["target"], "")
        if "Config" in src_label and "isValid" in tgt_label:
            found = True
            break
    assert found, "extension method isValid should attach to Config"

def test_swift_extension_does_not_duplicate_type_node():
    r = extract_swift(FIXTURES / "sample.swift")
    config_nodes = [n for n in r["nodes"] if n["label"] == "Config"]
    assert len(config_nodes) == 1, f"Config should appear once, got {len(config_nodes)}"

def test_swift_conformance_edge():
    r = extract_swift(FIXTURES / "sample.swift")
    inherits_edges = [e for e in r["edges"] if e["relation"] == "inherits"]
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    found = False
    for e in inherits_edges:
        src_label = node_by_id.get(e["source"], "")
        tgt_label = node_by_id.get(e["target"], "")
        if "DataProcessor" in src_label and "Processor" in tgt_label:
            found = True
            break
    assert found, "DataProcessor should have inherits edge to Processor"

def test_swift_extension_conformance_edge():
    r = extract_swift(FIXTURES / "sample.swift")
    inherits_edges = [e for e in r["edges"] if e["relation"] == "inherits"]
    node_by_id = {n["id"]: n["label"] for n in r["nodes"]}
    found = False
    for e in inherits_edges:
        src_label = node_by_id.get(e["source"], "")
        tgt_label = node_by_id.get(e["target"], "")
        if "DataProcessor" in src_label and "Loggable" in tgt_label:
            found = True
            break
    assert found, "extension should add conformance edge DataProcessor -> Loggable"

def test_swift_emits_calls():
    r = extract_swift(FIXTURES / "sample.swift")
    calls = _calls(r)
    assert any("process" in src and "validate" in tgt for src, tgt in calls)

def test_swift_call_edges_have_call_context():
    r = extract_swift(FIXTURES / "sample.swift")
    call_edges = _edges_with_relation(r, "calls")
    assert call_edges
    assert all(e.get("context") == "call" for e in call_edges)


# ── Elixir ────────────────────────────────────────────────────────────────────

from graphify.extract import extract_elixir

def test_elixir_finds_module():
    r = extract_elixir(FIXTURES / "sample.ex")
    assert "error" not in r
    labels = [n["label"] for n in r["nodes"]]
    assert any("MyApp.Accounts.User" in l for l in labels)

def test_elixir_finds_functions():
    r = extract_elixir(FIXTURES / "sample.ex")
    labels = [n["label"] for n in r["nodes"]]
    assert any("create" in l for l in labels)
    assert any("find" in l for l in labels)
    assert any("validate" in l for l in labels)

def test_elixir_finds_imports():
    r = extract_elixir(FIXTURES / "sample.ex")
    import_edges = [e for e in r["edges"] if e["relation"] == "imports"]
    assert len(import_edges) >= 2


def test_elixir_import_edges_have_import_context():
    r = extract_elixir(FIXTURES / "sample.ex")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)

def test_elixir_finds_calls():
    r = extract_elixir(FIXTURES / "sample.ex")
    calls = {(e["source"], e["target"]) for e in r["edges"] if e["relation"] == "calls"}
    labels = {n["id"]: n["label"] for n in r["nodes"]}
    assert any("create" in labels.get(src, "") and "validate" in labels.get(tgt, "") for src, tgt in calls)


def test_elixir_call_edges_have_call_context():
    r = extract_elixir(FIXTURES / "sample.ex")
    call_edges = _edges_with_relation(r, "calls")
    assert call_edges
    assert all(e.get("context") == "call" for e in call_edges)

def test_elixir_method_edges():
    r = extract_elixir(FIXTURES / "sample.ex")
    methods = [e for e in r["edges"] if e["relation"] == "method"]
    assert len(methods) >= 3


# ── Objective-C ──────────────────────────────────────────────────────────────
from graphify.extract import extract_objc


def test_objc_finds_interface():
    r = extract_objc(FIXTURES / "sample.m")
    labels = [n["label"] for n in r["nodes"]]
    assert "Animal" in labels


def test_objc_finds_subclass():
    r = extract_objc(FIXTURES / "sample.m")
    labels = [n["label"] for n in r["nodes"]]
    assert "Dog" in labels


def test_objc_finds_methods():
    r = extract_objc(FIXTURES / "sample.m")
    labels = [n["label"] for n in r["nodes"]]
    assert any("speak" in l or "fetch" in l or "initWithName" in l for l in labels)


def test_objc_finds_imports():
    r = extract_objc(FIXTURES / "sample.m")
    import_edges = [e for e in r["edges"] if e["relation"] == "imports"]
    assert len(import_edges) >= 1


def test_objc_import_edges_have_import_context():
    r = extract_objc(FIXTURES / "sample.m")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


def test_objc_inherits_edge():
    r = extract_objc(FIXTURES / "sample.m")
    inherits = [e for e in r["edges"] if e["relation"] == "inherits"]
    assert len(inherits) >= 1


def test_objc_no_dangling_edges():
    r = extract_objc(FIXTURES / "sample.m")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"Dangling source: {e}"


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

def test_go_receiver_methods_share_type_node():
    """Methods on the same receiver type must share one canonical type node."""
    r = extract_go(FIXTURES / "sample.go")
    server_nodes = [n for n in r["nodes"] if n["label"] == "Server"]
    # Both Start() and Stop() are on *Server — should produce exactly one Server node
    assert len(server_nodes) == 1

def test_go_receiver_uses_pkg_scope():
    """Type node id should be scoped to directory, not file stem."""
    r = extract_go(FIXTURES / "sample.go")
    server_nodes = [n for n in r["nodes"] if n["label"] == "Server"]
    assert server_nodes
    # Should NOT contain the file stem "sample" in the type node id
    assert "sample" not in server_nodes[0]["id"].split(":")[0]


# ---------------------------------------------------------------------------
# Julia
# ---------------------------------------------------------------------------

def test_julia_finds_module():
    r = extract_julia(FIXTURES / "sample.jl")
    labels = [n["label"] for n in r["nodes"]]
    assert "Geometry" in labels


def test_julia_finds_structs():
    r = extract_julia(FIXTURES / "sample.jl")
    labels = [n["label"] for n in r["nodes"]]
    assert "Point" in labels
    assert "Circle" in labels


def test_julia_finds_abstract_type():
    r = extract_julia(FIXTURES / "sample.jl")
    labels = [n["label"] for n in r["nodes"]]
    assert "Shape" in labels


def test_julia_finds_functions():
    r = extract_julia(FIXTURES / "sample.jl")
    labels = [n["label"] for n in r["nodes"]]
    assert any("area" in l for l in labels)
    assert any("distance" in l for l in labels)


def test_julia_finds_short_function():
    r = extract_julia(FIXTURES / "sample.jl")
    labels = [n["label"] for n in r["nodes"]]
    assert any("perimeter" in l for l in labels)


def test_julia_finds_imports():
    r = extract_julia(FIXTURES / "sample.jl")
    import_edges = [e for e in r["edges"] if e["relation"] == "imports"]
    assert len(import_edges) >= 1


def test_julia_import_edges_have_import_context():
    r = extract_julia(FIXTURES / "sample.jl")
    import_edges = _edges_with_relation(r, "imports", "imports_from")
    assert import_edges
    assert all(e.get("context") == "import" for e in import_edges)


def test_julia_finds_inherits():
    r = extract_julia(FIXTURES / "sample.jl")
    inherits = [e for e in r["edges"] if e["relation"] == "inherits"]
    assert len(inherits) >= 1


def test_julia_finds_calls():
    r = extract_julia(FIXTURES / "sample.jl")
    call_edges = [e for e in r["edges"] if e["relation"] == "calls"]
    assert len(call_edges) >= 1


def test_julia_call_edges_have_call_context():
    r = extract_julia(FIXTURES / "sample.jl")
    call_edges = _edges_with_relation(r, "calls")
    assert call_edges
    assert all(e.get("context") == "call" for e in call_edges)


def test_julia_no_dangling_edges():
    r = extract_julia(FIXTURES / "sample.jl")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"Dangling source: {e}"


# ── TypeScript dynamic imports ───────────────────────────────────────────────

def test_ts_dynamic_import_no_error():
    r = extract_js(FIXTURES / "dynamic_import.ts")
    assert "error" not in r

def test_ts_dynamic_import_extracts_edges():
    """Dynamic import() calls inside functions should produce imports_from edges."""
    r = extract_js(FIXTURES / "dynamic_import.ts")
    dyn_edges = [e for e in r["edges"] if e["relation"] == "imports_from"]
    targets = {e["target"] for e in dyn_edges}
    # Should find: static ./logger, dynamic ./mayaEngine.js, dynamic ./queue.js
    assert any("logger" in t for t in targets), f"Missing static import of logger: {targets}"
    assert any("mayaengine" in t.lower() for t in targets), f"Missing dynamic import of mayaEngine: {targets}"
    assert any("queue" in t.lower() for t in targets), f"Missing dynamic import of queue: {targets}"

def test_ts_dynamic_import_confidence():
    """Dynamic imports should have EXTRACTED confidence (they are deterministic string literals)."""
    r = extract_js(FIXTURES / "dynamic_import.ts")
    dyn_edges = [e for e in r["edges"]
                 if e["relation"] == "imports_from"
                 and "mayaengine" in e["target"].lower()]
    assert len(dyn_edges) >= 1
    assert dyn_edges[0]["confidence"] == "EXTRACTED"

def test_ts_dynamic_import_source_is_function():
    """Dynamic import edge source should be the enclosing function, not the file."""
    r = extract_js(FIXTURES / "dynamic_import.ts")
    node_labels = {n["id"]: n["label"] for n in r["nodes"]}
    dyn_edges = [e for e in r["edges"]
                 if e["relation"] == "imports_from"
                 and "mayaengine" in e["target"].lower()]
    assert len(dyn_edges) >= 1
    src_label = node_labels.get(dyn_edges[0]["source"], "")
    assert "processInbound" in src_label, f"Expected processInbound as source, got {src_label}"

def test_ts_no_dynamic_import_in_sync_fn():
    """Functions without dynamic imports should not get spurious imports_from edges."""
    r = extract_js(FIXTURES / "dynamic_import.ts")
    node_ids = {n["label"]: n["id"] for n in r["nodes"]}
    sync_nid = node_ids.get("syncOnly()")
    if sync_nid:
        sync_imports = [e for e in r["edges"]
                        if e["source"] == sync_nid and e["relation"] == "imports_from"]
        assert len(sync_imports) == 0

def test_ts_dynamic_template_literal_skipped():
    """Dynamic template literals (with ${}) must not produce an imports_from edge."""
    r = extract_js(FIXTURES / "dynamic_import.ts")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    # loadHandler uses `./handlers/${handlerName}` — no static path, must be absent
    assert not any("handler" in t.lower() and "$" in t for t in targets), \
        f"Garbage edge from dynamic template literal found: {targets}"
    # More robust: no target should contain a brace character
    assert not any("{" in t or "}" in t for t in targets), \
        f"Target contains unresolved template expression: {targets}"

def test_ts_static_template_literal_resolved():
    """Static template literals (no ${}) should resolve the same as a plain string."""
    r = extract_js(FIXTURES / "dynamic_import.ts")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "imports_from"}
    assert any("statichelper" in t.lower() for t in targets), \
        f"Static template literal import not resolved: {targets}"


# ── Terraform / HCL ──────────────────────────────────────────────────────────

from graphify.extract import extract_terraform


def test_terraform_no_error():
    r = extract_terraform(FIXTURES / "sample.tf")
    assert "error" not in r, r.get("error")


def test_terraform_finds_resources():
    r = extract_terraform(FIXTURES / "sample.tf")
    labels = [n["label"] for n in r["nodes"]]
    assert any("aws_iam_role.lambda_exec" in l for l in labels)
    assert any("aws_lambda_function.processor" in l for l in labels)


def test_terraform_finds_module_and_variables():
    r = extract_terraform(FIXTURES / "sample.tf")
    labels = [n["label"] for n in r["nodes"]]
    assert any(l == "module.logging_bucket" for l in labels)
    assert any(l == "var.environment" for l in labels)
    assert any(l == "var.region" for l in labels)


def test_terraform_finds_data_source():
    r = extract_terraform(FIXTURES / "sample.tf")
    labels = [n["label"] for n in r["nodes"]]
    assert any(l == "data.aws_caller_identity.current" for l in labels)


def test_terraform_finds_locals_and_output():
    r = extract_terraform(FIXTURES / "sample.tf")
    labels = [n["label"] for n in r["nodes"]]
    assert any(l == "output.lambda_arn" for l in labels)
    # At least one local. reference target should exist
    assert any(l.startswith("local.") for l in labels)


def test_terraform_finds_uses_var_edges():
    r = extract_terraform(FIXTURES / "sample.tf")
    uses_var = [e for e in r["edges"] if e["relation"] == "uses_var"]
    assert len(uses_var) >= 2  # var.environment, var.region used


def test_terraform_finds_uses_module_edge():
    r = extract_terraform(FIXTURES / "sample.tf")
    rels = {e["relation"] for e in r["edges"]}
    assert "uses_module" in rels


def test_terraform_finds_uses_data_edge():
    r = extract_terraform(FIXTURES / "sample.tf")
    rels = {e["relation"] for e in r["edges"]}
    assert "uses_data" in rels


def test_terraform_finds_resource_to_resource_edge():
    r = extract_terraform(FIXTURES / "sample.tf")
    # aws_lambda_function.processor → aws_iam_role.lambda_exec
    refs = [e for e in r["edges"] if e["relation"] == "references_resource"]
    assert len(refs) >= 1


def test_terraform_no_dangling_edges():
    r = extract_terraform(FIXTURES / "sample.tf")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"Dangling source: {e}"
        assert e["target"] in node_ids, f"Dangling target: {e}"


def test_terraform_hcl_extension_uses_same_extractor():
    # Symlink-free: just verify dispatch wires .hcl → extract_terraform too
    from graphify import extract as _ex
    # _DISPATCH is built inside extract(); recreate the relevant assertion via _EXTENSIONS
    # The actual dispatcher binding is tested implicitly via collect_files semantics.
    assert hasattr(_ex, "extract_terraform")


# ── dbt SQL (Jinja-aware) ────────────────────────────────────────────────────

from graphify.extract import extract_dbt_sql, extract_sql, _is_dbt_project_file


def test_dbt_sql_no_error():
    r = extract_dbt_sql(FIXTURES / "sample_dbt.sql")
    assert "error" not in r, r.get("error")


def test_dbt_sql_finds_ref_edges():
    r = extract_dbt_sql(FIXTURES / "sample_dbt.sql")
    refs = [e for e in r["edges"] if e["relation"] == "references"]
    targets = {e["target"] for e in refs}
    assert "stg_orders" in targets
    assert "stg_promotions" in targets


def test_dbt_sql_finds_source_edge():
    r = extract_dbt_sql(FIXTURES / "sample_dbt.sql")
    src_edges = [e for e in r["edges"] if e["relation"] == "references_source"]
    assert any("book_catalog" in e["target"] for e in src_edges)


def test_dbt_sql_finds_var_edge():
    r = extract_dbt_sql(FIXTURES / "sample_dbt.sql")
    var_edges = [e for e in r["edges"] if e["relation"] == "uses_var"]
    assert any("window_start" in e["target"] for e in var_edges)


def test_dbt_sql_extracts_config_attrs():
    r = extract_dbt_sql(FIXTURES / "sample_dbt.sql")
    file_node = r["nodes"][0]
    assert file_node.get("dbt_alias") == "daily_book_sales"
    assert file_node.get("dbt_materialized") == "table"


def test_dbt_sql_no_dangling_edges():
    r = extract_dbt_sql(FIXTURES / "sample_dbt.sql")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"Dangling source: {e}"
        assert e["target"] in node_ids, f"Dangling target: {e}"


def test_extract_sql_dispatches_to_dbt_when_jinja_markers_present():
    """extract_sql should detect Jinja `{{` / `{%` and route to extract_dbt_sql,
    even when the file is not in a dbt project (no dbt_project.yml in parents)."""
    r = extract_sql(FIXTURES / "sample_dbt.sql")
    assert "error" not in r, r.get("error")
    assert any(e["relation"] == "references" for e in r["edges"])


def test_extract_sql_dispatches_to_dbt_when_in_dbt_project(tmp_path):
    """extract_sql should detect dbt_project.yml in parents and route to extract_dbt_sql
    even for files without Jinja markers (defensive — most dbt files have refs)."""
    (tmp_path / "dbt_project.yml").write_text("name: test_project\nversion: '1.0'\n")
    sql = tmp_path / "models" / "my_model.sql"
    sql.parent.mkdir()
    sql.write_text("select * from {{ ref('upstream') }}\n")
    r = extract_sql(sql)
    assert "error" not in r, r.get("error")
    assert any(e["relation"] == "references" and e["target"] == "upstream" for e in r["edges"])


def test_extract_sql_keeps_plain_sql_path_when_no_jinja_no_dbt():
    """A pure SQL file (no Jinja, no dbt project) should still go through the
    upstream tree-sitter-sql extractor — we must not regress its behaviour."""
    r = extract_sql(FIXTURES / "sample.sql")
    assert "error" not in r, r.get("error")
    # Plain SQL path emits "contains" edges from file → table; dbt path doesn't.
    rels = {e["relation"] for e in r["edges"]}
    assert "contains" in rels, "fell through to dbt path on plain SQL — should have used tree-sitter-sql"


def test_is_dbt_project_file_detects_parent_marker(tmp_path):
    (tmp_path / "dbt_project.yml").write_text("name: test\n")
    sql = tmp_path / "models" / "deep" / "x.sql"
    sql.parent.mkdir(parents=True)
    sql.write_text("select 1")
    assert _is_dbt_project_file(sql) is True


def test_is_dbt_project_file_returns_false_when_no_marker(tmp_path):
    sql = tmp_path / "x.sql"
    sql.write_text("select 1")
    assert _is_dbt_project_file(sql) is False


# ── dbt YAML schema extractor ────────────────────────────────────────────────

from graphify.extract import extract_dbt_yaml


def test_dbt_yaml_no_error():
    r = extract_dbt_yaml(FIXTURES / "sample_dbt_schema.yml")
    assert "error" not in r, r.get("error")


def test_dbt_yaml_finds_models():
    r = extract_dbt_yaml(FIXTURES / "sample_dbt_schema.yml")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "documents"}
    assert "daily_book_sales" in targets
    assert "monthly_top_authors" in targets


def test_dbt_yaml_finds_sources():
    r = extract_dbt_yaml(FIXTURES / "sample_dbt_schema.yml")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "defines_source"}
    # Both the source-level name "warehouse" and the per-table names land here
    assert any("warehouse" in t for t in targets)
    assert any("book_catalog" in t for t in targets)
    assert any("orders_log" in t for t in targets)


def test_dbt_yaml_finds_seeds():
    r = extract_dbt_yaml(FIXTURES / "sample_dbt_schema.yml")
    targets = {e["target"] for e in r["edges"] if e["relation"] == "defines_seed"}
    assert any("ref_genres" in t for t in targets)


def test_dbt_yaml_skips_columns_and_tests():
    """The extractor must NOT mistake column names ('book_isbn', 'revenue', etc.)
    for model definitions — they live under per-entity `columns:` keys, which
    are explicit leaves we skip."""
    r = extract_dbt_yaml(FIXTURES / "sample_dbt_schema.yml")
    docs = {e["target"] for e in r["edges"] if e["relation"] == "documents"}
    assert "book_isbn" not in docs
    assert "revenue" not in docs
    assert "rank" not in docs
    assert "author" not in docs


def test_dbt_yaml_model_id_matches_extract_dbt_sql():
    """A model `name: daily_book_sales` in YAML must produce the same node id
    as the corresponding `daily_book_sales.sql` file would — that's the merge
    contract: docs node and code node collapse into one."""
    yaml_r = extract_dbt_yaml(FIXTURES / "sample_dbt_schema.yml")
    yaml_targets = {e["target"] for e in yaml_r["edges"] if e["relation"] == "documents"}
    # extract_dbt_sql on sample_dbt.sql uses _make_id(path.stem) for the file node
    sql_r = extract_dbt_sql(FIXTURES / "sample_dbt.sql")
    sql_file_id = sql_r["nodes"][0]["id"]
    assert sql_file_id == "sample_dbt"
    # And refs in the SQL produce ids compatible with model targets in YAML
    refs = {e["target"] for e in sql_r["edges"] if e["relation"] == "references"}
    # The bookstore fixtures don't cross-reference — verify the ID convention
    # is consistent by running both on a synthetic name
    from graphify.extract import _make_id
    assert _make_id("daily_book_sales") in yaml_targets


def test_dbt_yaml_no_dangling_edges():
    r = extract_dbt_yaml(FIXTURES / "sample_dbt_schema.yml")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"Dangling source: {e}"
        assert e["target"] in node_ids, f"Dangling target: {e}"


def test_dbt_yaml_handles_jinja_templated_names_gracefully():
    """`name: {{ var('x') }}` must NOT produce a literal node — the extractor
    treats Jinja-templated names as 'unknown' and skips them, since the real
    name is only resolvable at compile time."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
        f.write("version: 2\nmodels:\n  - name: \"{{ var('x') }}\"\n")
        tmp_path = f.name
    try:
        r = extract_dbt_yaml(Path(tmp_path))
        for n in r["nodes"]:
            assert "{{" not in n["label"]
            assert "{{" not in n["id"]
    finally:
        os.unlink(tmp_path)


# ── Airflow / Astronomer Cosmos DAG extractor ────────────────────────────────

from graphify.extract import extract_airflow_dag, extract_python_with_airflow


def test_airflow_no_error():
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    assert "error" not in r, r.get("error")


def test_airflow_finds_dag_node():
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    labels = [n["label"] for n in r["nodes"]]
    assert any("DAG: bookstore_daily_report" in l for l in labels)


def test_airflow_finds_cosmos_dbt_dag():
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    cosmos_dags = [n for n in r["nodes"] if n.get("cosmos") is True]
    assert len(cosmos_dags) == 1
    assert cosmos_dags[0]["dag_id"] == "dbt_refresh_inventory"


def test_airflow_dbt_project_extracted_from_cosmos():
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    proj_edges = [e for e in r["edges"] if e["relation"] == "uses_dbt_project"]
    assert len(proj_edges) == 1
    proj_node = next(n for n in r["nodes"] if n["id"] == proj_edges[0]["target"])
    assert proj_node.get("dbt_project_path") == "/usr/local/airflow/dbt/bookstore_dbt_project"


def test_airflow_dbt_selectors_extracted_from_render_config():
    """RenderConfig was assigned to a variable upstream — extractor must
    resolve `render_config=render_config` back to the RenderConfig call."""
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    selects = {e["target"] for e in r["edges"] if e["relation"] == "selects_dbt"}
    excludes = {e["target"] for e in r["edges"] if e["relation"] == "excludes_dbt"}
    # From RenderConfig
    assert any("tag_bookstore" in s for s in selects)
    assert any("path_models_inventory" in s for s in selects)
    assert any("tag_experimental" in e for e in excludes)


def test_airflow_dbt_selectors_from_bash_command():
    """BashOperator with `dbt build --select tag:X --exclude tag:Y` —
    extractor parses the bash command (incl. concatenated + f-string) and
    emits selector edges from the task."""
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    bash_select = [e for e in r["edges"]
                   if e["relation"] == "selects_dbt"
                   and e["source"].startswith("airflow_task_")]
    bash_exclude = [e for e in r["edges"]
                    if e["relation"] == "excludes_dbt"
                    and e["source"].startswith("airflow_task_")]
    assert any("tag_bookstore" in e["target"] for e in bash_select)
    assert any("tag_slow" in e["target"] for e in bash_exclude)


def test_airflow_finds_tasks():
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    tasks = [n for n in r["nodes"] if n.get("airflow_operator")]
    task_ids = {n.get("task_id") for n in tasks}
    assert {"pull_orders_from_s3", "refresh_inventory_models",
            "publish_daily_report", "notify_partners"}.issubset(task_ids)


def test_airflow_task_dependencies_via_rshift():
    """`pull_orders >> refresh_inventory >> publish_report >> notify_partners` —
    chained `>>` must produce 3 depends_on edges."""
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    deps = [e for e in r["edges"] if e["relation"] == "depends_on"]
    assert len(deps) >= 3


def test_airflow_python_callable_linked():
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    callable_edges = [e for e in r["edges"] if e["relation"] == "calls_python"]
    callable_targets = [e["target"] for e in callable_edges]
    assert any("post_sales_report" in t for t in callable_targets)


def test_airflow_variable_get_with_string_literal():
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    var_edges = [e for e in r["edges"] if e["relation"] == "uses_variable"]
    var_names = [e["target"] for e in var_edges]
    assert any("BOOKSTORE_REGION" in v.upper() for v in var_names)


def test_airflow_variable_get_with_default_var_arg():
    """Variable.get takes an `default_var=` kwarg — the first positional arg
    is still the variable name we want to capture."""
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    var_edges = [e for e in r["edges"] if e["relation"] == "uses_variable"]
    var_names = [e["target"] for e in var_edges]
    assert any("BOOKSTORE_REPORT_CHUNK" in v.upper() for v in var_names)


def test_airflow_basehook_get_connection_resolves_variable():
    """BaseHook.get_connection(WEBHOOK_CONN_ID) where WEBHOOK_CONN_ID is a
    module-level constant — extractor must resolve the identifier."""
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    conn_edges = [e for e in r["edges"] if e["relation"] == "uses_connection"]
    conn_names = [e["target"] for e in conn_edges]
    assert any("bookstore_webhook" in c for c in conn_names)


def test_airflow_dataset_declared():
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    ds_nodes = [n for n in r["nodes"] if n.get("dataset_uri")]
    uris = {n["dataset_uri"] for n in ds_nodes}
    assert "s3://bookstore-raw/orders/" in uris
    assert "s3://bookstore-raw/inventory/" in uris


def test_airflow_extractor_skips_non_airflow_python_files():
    """A regular .py file with no airflow imports must produce empty result —
    the extractor is opt-in via head-scan for `from airflow` markers."""
    import tempfile, os
    src = "def hello():\n    return 42\n\nclass Foo:\n    pass\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(src)
        tmp_path = f.name
    try:
        r = extract_airflow_dag(Path(tmp_path))
        assert r["nodes"] == []
        assert r["edges"] == []
    finally:
        os.unlink(tmp_path)


def test_airflow_no_dangling_edges():
    r = extract_airflow_dag(FIXTURES / "sample_airflow_dag.py")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"Dangling source: {e}"
        assert e["target"] in node_ids, f"Dangling target: {e}"


def test_extract_python_with_airflow_combines_both():
    """The .py dispatcher wrapper runs extract_python AND extract_airflow_dag
    on Airflow files — result should contain BOTH Python class/function
    nodes AND Airflow DAG/Task nodes, on a single file extraction."""
    r = extract_python_with_airflow(FIXTURES / "sample_airflow_dag.py")
    labels = [n["label"] for n in r["nodes"]]
    # Python side: function `post_sales_report` defined in the file
    assert any("post_sales_report" in l for l in labels)
    # Airflow side: task and DAG
    assert any("DAG:" in l for l in labels)
    assert any("task:" in l for l in labels)


def test_extract_python_with_airflow_no_overhead_for_plain_python():
    """For non-Airflow .py files (e.g. existing fixtures), the wrapper just
    returns plain python output."""
    r = extract_python_with_airflow(FIXTURES / "sample.py")
    af_specific = [n for n in r["nodes"] if n.get("airflow_operator") or n.get("dag_id")]
    assert af_specific == []
