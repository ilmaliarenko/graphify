"""Tests for the LLM-memory utilities (`graphify.memory`).

The memory module is plan-mode-safe: every function is a pure read against
a graph dict, no file writes, no LLM calls. Tests use synthetic graphs to
exercise behaviour deterministically.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import pytest

from graphify.memory import (
    load_graph, search, context, diff, reflect, _approx_tokens,
)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _mk_graph(nodes: list[dict], edges: list[dict]) -> dict:
    return {"nodes": nodes, "links": edges, "directed": False, "graph": {}}


# ── load_graph ──────────────────────────────────────────────────────────────


def test_load_graph_reads_json(tmp_path):
    g = _mk_graph([{"id": "a", "label": "Alpha"}], [])
    p = tmp_path / "graph.json"
    p.write_text(json.dumps(g))
    loaded = load_graph(p)
    assert loaded["nodes"][0]["id"] == "a"


def test_load_graph_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_graph(tmp_path / "missing.json")


# ── search ──────────────────────────────────────────────────────────────────


def test_search_exact_label_match_ranks_highest():
    g = _mk_graph([
        {"id": "a", "label": "auth"},
        {"id": "b", "label": "user_auth_service"},
        {"id": "c", "label": "completely_unrelated"},
    ], [])
    r = search(g, "auth")
    assert r["matches"][0]["id"] == "a"  # exact match wins
    assert r["matches"][1]["id"] == "b"  # substring match second


def test_search_returns_neighbors_by_default():
    g = _mk_graph(
        [{"id": "a", "label": "core"}, {"id": "b", "label": "other"}],
        [{"source": "a", "target": "b", "relation": "calls"}],
    )
    r = search(g, "core")
    assert r["matches"][0]["neighbors"] == [
        {"direction": "out", "relation": "calls", "id": "b", "label": "other"}
    ]


def test_search_no_neighbors_flag():
    g = _mk_graph(
        [{"id": "a", "label": "x"}, {"id": "b", "label": "y"}],
        [{"source": "a", "target": "b", "relation": "r"}],
    )
    r = search(g, "x", include_neighbors=False)
    assert "neighbors" not in r["matches"][0]


def test_search_attribute_match():
    """Match should fire on key attributes, not just label."""
    g = _mk_graph([
        {"id": "n1", "label": "model_42",
         "dbt_alias": "customer_summary"},
        {"id": "n2", "label": "other"},
    ], [])
    r = search(g, "customer_summary")
    assert r["matches"][0]["id"] == "n1"


def test_search_dbt_tag_membership():
    g = _mk_graph([
        {"id": "n1", "label": "x", "dbt_tags": ["bookstore", "daily"]},
        {"id": "n2", "label": "y", "dbt_tags": ["other"]},
    ], [])
    r = search(g, "bookstore")
    ids = [m["id"] for m in r["matches"]]
    assert "n1" in ids
    assert "n2" not in ids


def test_search_empty_query_returns_empty():
    g = _mk_graph([{"id": "a", "label": "anything"}], [])
    r = search(g, "")
    assert r["matches"] == []


def test_search_limit_caps_results():
    g = _mk_graph(
        [{"id": f"n{i}", "label": f"matchable_{i}"} for i in range(20)], []
    )
    r = search(g, "matchable", limit=5)
    assert len(r["matches"]) == 5


def test_search_ranks_by_degree_as_tiebreak():
    """Two nodes with identical label match scores — the more connected one
    ranks higher.  Use disjoint neighbor sets so degrees actually differ."""
    g = _mk_graph(
        [{"id": "lo", "label": "x"}, {"id": "hi", "label": "x"},
         {"id": "n1", "label": "n1"}, {"id": "n2", "label": "n2"},
         {"id": "n3", "label": "n3"}],
        [
            {"source": "hi", "target": "n1", "relation": "r"},
            {"source": "hi", "target": "n2", "relation": "r"},
            {"source": "hi", "target": "n3", "relation": "r"},
            {"source": "lo", "target": "n1", "relation": "r"},
        ],
    )
    r = search(g, "x")
    assert r["matches"][0]["id"] == "hi"  # degree 3 > degree 1


# ── context ─────────────────────────────────────────────────────────────────


def test_context_returns_empty_for_no_match():
    g = _mk_graph([{"id": "a", "label": "alpha"}], [])
    r = context(g, "nothing_matches_this", budget_tokens=1000)
    assert r["seeds"] == []
    assert r["nodes"] == []


def test_context_includes_seeds_and_neighbors_within_budget():
    g = _mk_graph([
        {"id": "core", "label": "core_module"},
        {"id": "dep1", "label": "dep_one"},
        {"id": "dep2", "label": "dep_two"},
    ], [
        {"source": "core", "target": "dep1", "relation": "imports", "confidence": "EXTRACTED"},
        {"source": "core", "target": "dep2", "relation": "imports", "confidence": "EXTRACTED"},
    ])
    r = context(g, "core_module", budget_tokens=2000, max_hops=1)
    ids = {n["id"] for n in r["nodes"]}
    assert "core" in ids
    # Both 1-hop neighbors should fit in 2000 tokens
    assert "dep1" in ids
    assert "dep2" in ids
    assert len(r["edges"]) == 2


def test_context_truncates_when_over_budget():
    """A tight budget that fits only the seed should not include neighbors."""
    g = _mk_graph(
        [{"id": "core", "label": "core_thing"}] +
        [{"id": f"d{i}", "label": "d" * 200} for i in range(20)],  # bulky neighbors
        [{"source": "core", "target": f"d{i}", "relation": "r",
          "confidence": "EXTRACTED"} for i in range(20)],
    )
    r = context(g, "core_thing", budget_tokens=100)
    # Should only include the seed itself
    ids = {n["id"] for n in r["nodes"]}
    assert "core" in ids
    # Some neighbors might be too big to fit individually
    assert len(r["nodes"]) < 21


def test_context_prioritizes_extracted_over_inferred():
    """When BFS expands, EXTRACTED edges should be added before INFERRED ones."""
    g = _mk_graph([
        {"id": "core", "label": "core"},
        {"id": "ex", "label": "extracted_target"},
        {"id": "inf", "label": "inferred_target"},
    ], [
        {"source": "core", "target": "inf", "relation": "r", "confidence": "INFERRED"},
        {"source": "core", "target": "ex", "relation": "r", "confidence": "EXTRACTED"},
    ])
    r = context(g, "core", budget_tokens=2000, max_hops=1)
    # Both should fit, but EXTRACTED edge should be added first
    assert r["edges"][0]["confidence"] == "EXTRACTED"


# ── diff ────────────────────────────────────────────────────────────────────


def test_diff_filters_by_extracted_at():
    now = int(time.time())
    g = _mk_graph([
        {"id": "old", "label": "old", "extracted_at": now - 100_000},
        {"id": "new", "label": "new", "extracted_at": now - 60},
    ], [])
    r = diff(g, since_seconds=300)  # 5 minutes
    assert r["total_new_nodes"] == 1
    new_ids = sum(([n["id"] for n in items]
                   for items in r["new_nodes_by_directory"].values()), [])
    assert "new" in new_ids
    assert "old" not in new_ids


def test_diff_groups_by_directory():
    now = int(time.time())
    g = _mk_graph([
        {"id": "a", "label": "a", "extracted_at": now,
         "source_file": "/proj/dbt/models/x.sql"},
        {"id": "b", "label": "b", "extracted_at": now,
         "source_file": "/proj/dbt/models/y.sql"},
        {"id": "c", "label": "c", "extracted_at": now,
         "source_file": "/proj/airflow/dags/z.py"},
    ], [])
    r = diff(g, since_seconds=600)
    dirs = list(r["new_nodes_by_directory"].keys())
    assert any("models" in d for d in dirs)
    assert any("dags" in d for d in dirs)


def test_diff_counts_relation_types():
    now = int(time.time())
    g = _mk_graph(
        [{"id": "a", "label": "a", "extracted_at": now},
         {"id": "b", "label": "b", "extracted_at": now}],
        [{"source": "a", "target": "b", "relation": "references", "extracted_at": now},
         {"source": "a", "target": "b", "relation": "calls", "extracted_at": now},
         {"source": "b", "target": "a", "relation": "references", "extracted_at": now}],
    )
    r = diff(g, since_seconds=600)
    assert r["edge_relations_added"]["references"] == 2
    assert r["edge_relations_added"]["calls"] == 1


def test_diff_returns_zero_for_no_recent_changes():
    g = _mk_graph(
        [{"id": "old", "label": "old", "extracted_at": 1}], []
    )
    r = diff(g, since_seconds=60)
    assert r["total_new_nodes"] == 0


# ── reflect ─────────────────────────────────────────────────────────────────


def test_reflect_god_nodes_ranked_by_degree():
    g = _mk_graph(
        [{"id": "a", "label": "a"}, {"id": "b", "label": "b"},
         {"id": "c", "label": "c"}, {"id": "d", "label": "d"}],
        # a has degree 3, b degree 2, c degree 1, d degree 0
        [{"source": "a", "target": "b", "relation": "r"},
         {"source": "a", "target": "c", "relation": "r"},
         {"source": "a", "target": "d", "relation": "r"},
         {"source": "b", "target": "c", "relation": "r"}],
    )
    r = reflect(g, top_k=4)
    god_ids = [g["id"] for g in r["god_nodes"]]
    assert god_ids[0] == "a"  # degree 3


def test_reflect_recently_added_sorted():
    g = _mk_graph(
        [{"id": f"n{i}", "label": f"n{i}", "extracted_at": 100 + i} for i in range(5)],
        [],
    )
    r = reflect(g, top_k=3)
    recent_ids = [x["id"] for x in r["recently_added"]]
    assert recent_ids == ["n4", "n3", "n2"]


def test_reflect_surfaces_ambiguous_edges():
    g = _mk_graph(
        [{"id": "a"}, {"id": "b"}],
        [{"source": "a", "target": "b", "relation": "r1", "confidence": "EXTRACTED"},
         {"source": "a", "target": "b", "relation": "r2", "confidence": "AMBIGUOUS",
          "source_file": "/foo.py"}],
    )
    r = reflect(g, top_k=10)
    assert r["totals"]["ambiguous_edges"] == 1
    assert len(r["ambiguous_edges_to_review"]) == 1
    assert r["ambiguous_edges_to_review"][0]["relation"] == "r2"


def test_reflect_counts_orphan_nodes():
    g = _mk_graph(
        [{"id": "lonely", "label": "lonely"},
         {"id": "a", "label": "a"}, {"id": "b", "label": "b"}],
        [{"source": "a", "target": "b", "relation": "r"}],
    )
    r = reflect(g)
    assert r["totals"]["orphan_nodes"] == 1


def test_reflect_breaks_down_by_dbt_resource_type():
    g = _mk_graph([
        {"id": "m1", "label": "m1", "dbt_resource_type": "model"},
        {"id": "m2", "label": "m2", "dbt_resource_type": "model"},
        {"id": "t1", "label": "t1", "dbt_resource_type": "test"},
        {"id": "x", "label": "x"},  # no dbt type
    ], [])
    r = reflect(g)
    assert r["by_dbt_resource_type"]["model"] == 2
    assert r["by_dbt_resource_type"]["test"] == 1
    assert "" not in r["by_dbt_resource_type"]


# ── recency stamp on extract output ──────────────────────────────────────────


def test_extract_stamps_extracted_at_on_every_node_and_edge(tmp_path):
    """End-to-end: running `extract()` on real fixtures must give every
    node and edge an `extracted_at` unix-seconds attribute."""
    from graphify.extract import extract

    files = [Path(__file__).parent / "fixtures" / "sample.py"]
    result = extract(files, cache_root=tmp_path / "cache")
    now = int(time.time())
    for n in result["nodes"]:
        assert "extracted_at" in n
        assert abs(n["extracted_at"] - now) < 5  # within 5 seconds
    for e in result["edges"]:
        assert "extracted_at" in e
