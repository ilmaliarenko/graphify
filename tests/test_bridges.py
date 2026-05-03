"""Tests for the cross-system bridges resolver in graphify.bridges.

The resolver runs as a post-pass after all per-file extractors finish. It
inspects the merged node/edge list looking for entities that were extracted
by DIFFERENT systems but represent the same underlying thing — e.g. an
Airflow Cosmos selector `tag:bookstore` and a dbt model that was tagged
`bookstore` in its `config()`. Those get bridged with `bridges_to` edges
labelled `INFERRED` (so reviewers can audit them).
"""
from __future__ import annotations
from pathlib import Path

import pytest

from graphify.bridges import (
    compute_bridges,
    _strip_selector_prefix,
    _path_under,
    _normalize_path,
)
from graphify.extract import (
    extract,
    extract_dbt_sql,
    extract_airflow_dag,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── Pure helper tests ────────────────────────────────────────────────────────


def test_strip_selector_prefix_tag_with_plus():
    assert _strip_selector_prefix("tag:bookstore+") == ("tag", "bookstore")


def test_strip_selector_prefix_path():
    assert _strip_selector_prefix("path:models/inventory") == ("path", "models/inventory")


def test_strip_selector_prefix_no_prefix():
    assert _strip_selector_prefix("just_a_name") == ("", "just_a_name")


def test_path_under_simple_match():
    assert _path_under("/usr/local/dbt/models/x.sql", "/usr/local/dbt") is True


def test_path_under_case_insensitive():
    assert _path_under("/USR/Local/Dbt/x.sql", "/usr/local/dbt") is True


def test_path_under_not_a_prefix():
    assert _path_under("/usr/local/other/x.sql", "/usr/local/dbt") is False


def test_normalize_path_strips_trailing_slash_and_lowers():
    assert _normalize_path("/USR/Local/dbt/") == "/usr/local/dbt"


# ── compute_bridges with synthetic node sets ────────────────────────────────


def test_compute_bridges_tag_selector_to_dbt_model():
    """An Airflow `tag:gold` selector on a Cosmos DAG should bridge to every
    dbt model node that has `gold` in its `dbt_tags`."""
    nodes = [
        {"id": "airflow_dag_x", "label": "DAG: x", "dag_id": "x"},
        {"id": "dbt_selector_tag_gold", "label": "dbt: tag:gold",
         "dbt_selector": "tag:gold"},
        {"id": "model_a", "label": "model_a", "dbt_tags": ["gold", "daily"]},
        {"id": "model_b", "label": "model_b", "dbt_tags": ["silver"]},
    ]
    edges = [
        {"source": "airflow_dag_x", "target": "dbt_selector_tag_gold",
         "relation": "selects_dbt"},
    ]
    new_nodes, new_edges = compute_bridges(nodes, edges)
    bridge_edges = [e for e in new_edges if e["relation"] == "bridges_to"]
    targets = {e["target"] for e in bridge_edges}
    assert "model_a" in targets        # gold-tagged → bridge fires
    assert "model_b" not in targets    # silver-only → no bridge
    # Bridges are INFERRED, not EXTRACTED
    for e in bridge_edges:
        assert e["confidence"] == "INFERRED"
        assert e.get("bridge_reason")


def test_compute_bridges_path_selector_to_dbt_model_under_path():
    nodes = [
        {"id": "dag_x", "dag_id": "x"},
        {"id": "sel_path", "dbt_selector": "path:models/inventory"},
        {"id": "model_in_inventory", "label": "x",
         "source_file": "/proj/dbt/models/inventory/sales.sql",
         "dbt_alias": "sales"},
        {"id": "model_elsewhere", "label": "y",
         "source_file": "/proj/dbt/models/marketing/leads.sql",
         "dbt_alias": "leads"},
    ]
    edges = [
        {"source": "dag_x", "target": "sel_path", "relation": "selects_dbt"},
    ]
    new_nodes, new_edges = compute_bridges(nodes, edges)
    bridge_targets = {e["target"] for e in new_edges if e["relation"] == "bridges_to"}
    assert "model_in_inventory" in bridge_targets
    assert "model_elsewhere" not in bridge_targets


def test_compute_bridges_cosmos_dag_to_models_in_project():
    """A Cosmos DAG that binds to project_path P should bridge to every dbt
    model whose source_file is under P."""
    nodes = [
        {"id": "cosmos_dag", "label": "DbtDag: x", "cosmos": True},
        {"id": "proj_root", "dbt_project_path": "/usr/local/airflow/dbt/myproject"},
        {"id": "model_under", "source_file": "/usr/local/airflow/dbt/myproject/models/m.sql",
         "dbt_alias": "m"},
        {"id": "model_outside", "source_file": "/other/place/x.sql", "dbt_alias": "x"},
    ]
    edges = [
        {"source": "cosmos_dag", "target": "proj_root", "relation": "uses_dbt_project"},
    ]
    _, new_edges = compute_bridges(nodes, edges)
    bridge_targets = {e["target"] for e in new_edges if e["relation"] == "bridges_to"}
    assert "model_under" in bridge_targets
    assert "model_outside" not in bridge_targets


def test_compute_bridges_idempotent():
    """Running compute_bridges twice on the same graph (with bridge edges
    already added) should NOT create duplicate edges."""
    nodes = [
        {"id": "dag_x"},
        {"id": "sel", "dbt_selector": "tag:gold"},
        {"id": "m", "dbt_tags": ["gold"]},
    ]
    edges = [
        {"source": "dag_x", "target": "sel", "relation": "selects_dbt"},
    ]
    _, first = compute_bridges(nodes, edges)
    edges_after_first = edges + first
    _, second = compute_bridges(nodes, edges_after_first)
    # Second call must add zero bridges (already present)
    assert second == [], f"Expected idempotent, got dupes: {second}"


def test_compute_bridges_no_self_loops():
    """A model that's BOTH a selector owner and a tag-match shouldn't bridge to itself."""
    nodes = [
        {"id": "x", "dbt_tags": ["gold"], "dbt_selector": "tag:gold"},
    ]
    edges = [
        {"source": "x", "target": "x", "relation": "selects_dbt"},
    ]
    _, new_edges = compute_bridges(nodes, edges)
    for e in new_edges:
        assert e["source"] != e["target"]


# ── End-to-end with real fixtures ────────────────────────────────────────────


def test_dbt_sql_extractor_captures_tags_from_config():
    """`config(tags=['bookstore', 'analytics'])` in sample_dbt.sql must put
    `dbt_tags: ['bookstore', 'analytics']` on the file node — needed for the
    bridge resolver to match Airflow `tag:bookstore` selectors."""
    r = extract_dbt_sql(FIXTURES / "sample_dbt.sql")
    file_node = r["nodes"][0]
    assert file_node.get("dbt_tags") == ["bookstore", "analytics"]


def test_end_to_end_airflow_to_dbt_bridge_via_tag(tmp_path):
    """Run the full extract() pipeline on both sample_airflow_dag.py and
    sample_dbt.sql together. The bridge resolver should emit a
    `bridges_to` edge from the Airflow DbtDag (which selects `tag:bookstore`)
    to the dbt model (tagged `bookstore` in its config)."""
    files = [FIXTURES / "sample_airflow_dag.py", FIXTURES / "sample_dbt.sql"]
    result = extract(files, cache_root=tmp_path / "cache")
    bridges = [e for e in result["edges"] if e["relation"] == "bridges_to"]
    assert len(bridges) > 0, "Expected at least one bridges_to edge"
    # The bridge target should be the dbt model node (sample_dbt)
    targets = {e["target"] for e in bridges}
    assert any("sample_dbt" in t for t in targets), f"Got bridge targets: {targets}"
    # And the source should be the Cosmos DAG or BashOperator that selects tag:bookstore
    sources = {e["source"] for e in bridges}
    assert any("dbt_refresh_inventory" in s or "bookstore_daily_report" in s
               or "task_" in s or "dag_" in s for s in sources), f"Got bridge sources: {sources}"


def test_end_to_end_bridge_carries_reason(tmp_path):
    """Every bridge edge has a human-readable `bridge_reason` for audit."""
    files = [FIXTURES / "sample_airflow_dag.py", FIXTURES / "sample_dbt.sql"]
    result = extract(files, cache_root=tmp_path / "cache")
    bridges = [e for e in result["edges"] if e["relation"] == "bridges_to"]
    for e in bridges:
        assert e.get("bridge_reason"), f"Bridge missing reason: {e}"


def test_end_to_end_no_bridges_for_unrelated_files(tmp_path):
    """Two `.py` and `.sql` files that DON'T share any tag/path/project must
    produce zero bridges."""
    # sample.py is a plain Python file (no airflow imports), sample.sql is plain SQL
    files = [FIXTURES / "sample.py", FIXTURES / "sample.sql"]
    result = extract(files, cache_root=tmp_path / "cache")
    bridges = [e for e in result["edges"] if e["relation"] == "bridges_to"]
    assert bridges == []
