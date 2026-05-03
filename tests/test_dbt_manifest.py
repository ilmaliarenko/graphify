"""Tests for the dbt manifest converter (`graphify.dbt_manifest`)."""
from __future__ import annotations
import json
from pathlib import Path

import pytest

from graphify.dbt_manifest import convert, stats, _node_id_from_unique_id, _make_id


# ── ID convention ───────────────────────────────────────────────────────────


def test_make_id_matches_extract_dbt_sql_convention():
    """The IDs `dbt_manifest.convert` produces must equal what
    `extract_dbt_sql` produces for the same model — that's how file-node
    and manifest-node merge in the graph."""
    assert _make_id("silver_tl__profile_audit") == "silver_tl_profile_audit"
    assert _make_id("source", "events", "page_views") == "source_events_page_views"


def test_node_id_for_model():
    nid = _node_id_from_unique_id("model.my_pkg.silver_x")
    assert nid == "silver_x"


def test_node_id_for_source():
    nid = _node_id_from_unique_id("source.my_pkg.events.clicks")
    assert nid == "source_events_clicks"


def test_node_id_for_seed():
    nid = _node_id_from_unique_id("seed.my_pkg.ref_genres")
    assert nid == "seed_ref_genres"


def test_node_id_for_snapshot():
    nid = _node_id_from_unique_id("snapshot.my_pkg.users_snapshot")
    assert nid == "snapshot_users_snapshot"


def test_node_id_for_test():
    nid = _node_id_from_unique_id("test.my_pkg.unique_silver_x_id")
    assert nid == "test_unique_silver_x_id"


# ── Synthetic manifest conversion ────────────────────────────────────────────


@pytest.fixture
def synthetic_manifest(tmp_path) -> Path:
    """A minimal but realistic dbt manifest with 2 models, 1 source, 1 seed,
    1 snapshot, 1 test, and a `parent_map` that links them."""
    manifest = {
        "metadata": {"project_name": "bookstore_dbt", "dbt_version": "1.8.0"},
        "nodes": {
            "model.bookstore_dbt.daily_book_sales": {
                "name": "daily_book_sales",
                "resource_type": "model",
                "original_file_path": "models/marts/daily_book_sales.sql",
                "config": {"materialized": "table", "tags": ["bookstore", "analytics"]},
                "database": "BOOKSTORE_DB",
                "schema": "marts",
                "alias": "daily_book_sales",
                "tags": ["bookstore", "analytics"],
            },
            "model.bookstore_dbt.stg_orders": {
                "name": "stg_orders",
                "resource_type": "model",
                "original_file_path": "models/staging/stg_orders.sql",
                "config": {"materialized": "view"},
                "database": "BOOKSTORE_DB",
                "schema": "staging",
                "alias": "stg_orders",
                "tags": ["staging"],
            },
            "seed.bookstore_dbt.ref_genres": {
                "name": "ref_genres",
                "resource_type": "seed",
                "original_file_path": "seeds/ref_genres.csv",
                "config": {},
                "database": "BOOKSTORE_DB",
                "schema": "seeds",
                "tags": [],
            },
            "snapshot.bookstore_dbt.orders_snapshot": {
                "name": "orders_snapshot",
                "resource_type": "snapshot",
                "original_file_path": "snapshots/orders_snapshot.sql",
                "config": {"strategy": "check"},
                "database": "BOOKSTORE_DB",
                "schema": "snapshots",
                "tags": [],
            },
            "test.bookstore_dbt.unique_daily_book_sales_book_isbn": {
                "name": "unique_daily_book_sales_book_isbn",
                "resource_type": "test",
                "original_file_path": "models/marts/schema.yml",
                "config": {},
                "tags": [],
            },
        },
        "sources": {
            "source.bookstore_dbt.warehouse.book_catalog": {
                "name": "book_catalog",
                "source_name": "warehouse",
                "original_file_path": "models/staging/sources.yml",
                "database": "BOOKSTORE_DB",
                "schema": "raw",
            },
        },
        "parent_map": {
            "model.bookstore_dbt.daily_book_sales": [
                "model.bookstore_dbt.stg_orders",
                "seed.bookstore_dbt.ref_genres",
            ],
            "model.bookstore_dbt.stg_orders": [
                "source.bookstore_dbt.warehouse.book_catalog",
            ],
            "test.bookstore_dbt.unique_daily_book_sales_book_isbn": [
                "model.bookstore_dbt.daily_book_sales",
            ],
        },
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return p


def test_convert_returns_expected_node_count(synthetic_manifest):
    g = convert(synthetic_manifest)
    # 5 from nodes + 1 from sources = 6
    assert len(g["nodes"]) == 6


def test_convert_emits_references_edges(synthetic_manifest):
    g = convert(synthetic_manifest)
    refs = [e for e in g["edges"] if e["relation"] == "references"]
    # daily_book_sales → stg_orders and → ref_genres (seed)
    # test → daily_book_sales
    assert len(refs) == 3


def test_convert_emits_references_source_edge(synthetic_manifest):
    g = convert(synthetic_manifest)
    src_edges = [e for e in g["edges"] if e["relation"] == "references_source"]
    assert len(src_edges) == 1
    assert src_edges[0]["source"] == "stg_orders"
    assert src_edges[0]["target"] == "source_warehouse_book_catalog"


def test_convert_carries_dbt_attributes(synthetic_manifest):
    g = convert(synthetic_manifest)
    sales = next(n for n in g["nodes"] if n["id"] == "daily_book_sales")
    assert sales["dbt_resource_type"] == "model"
    assert sales["dbt_materialized"] == "table"
    assert sales["dbt_database"] == "BOOKSTORE_DB"
    assert sales["dbt_schema"] == "marts"
    assert sales["dbt_alias"] == "daily_book_sales"
    assert "bookstore" in sales["dbt_tags"]


def test_convert_resolves_paths_against_project_root(synthetic_manifest, tmp_path):
    g = convert(synthetic_manifest, project_root=Path("/dbt/project"))
    sales = next(n for n in g["nodes"] if n["id"] == "daily_book_sales")
    # original_file_path is relative; should be resolved against project_root
    assert sales["source_file"] == "/dbt/project/models/marts/daily_book_sales.sql"


def test_convert_skips_macro_edges(synthetic_manifest, tmp_path):
    """parent_map can include macro.* refs — we exclude them to keep the
    graph focused on data lineage."""
    # Augment manifest with a macro parent
    m = json.loads(synthetic_manifest.read_text())
    m["parent_map"]["model.bookstore_dbt.daily_book_sales"].append(
        "macro.bookstore_dbt.compute_total"
    )
    p = tmp_path / "with_macro.json"
    p.write_text(json.dumps(m))
    g = convert(p)
    # Even though macro is now a parent, no edge to it (nor a node for it)
    macro_targets = [e["target"] for e in g["edges"] if "compute_total" in e["target"]]
    assert macro_targets == []


def test_convert_idempotent_no_duplicate_edges(synthetic_manifest):
    """parent_map can list a parent twice — output edges must dedupe."""
    g = convert(synthetic_manifest)
    edge_keys = [(e["source"], e["target"], e["relation"]) for e in g["edges"]]
    assert len(edge_keys) == len(set(edge_keys))


def test_stats_returns_breakdown(synthetic_manifest):
    g = convert(synthetic_manifest)
    s = stats(g)
    assert s["total_nodes"] == 6
    assert s["total_edges"] == 4
    assert s["by_resource_type"]["model"] == 2
    assert s["by_resource_type"]["seed"] == 1
    assert s["by_resource_type"]["snapshot"] == 1
    assert s["by_resource_type"]["test"] == 1
    assert s["by_resource_type"]["source"] == 1


def test_convert_empty_manifest_returns_empty_graph(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"nodes": {}, "sources": {}, "parent_map": {}}))
    g = convert(p)
    assert g == {"nodes": [], "edges": [], "hyperedges": []}
