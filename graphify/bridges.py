"""Cross-system bridge resolver.

After per-file extractors finish and their nodes/edges are merged, this
module runs a post-pass that detects entities that exist in MULTIPLE
systems and emits `bridges_to` edges between them.

Each detector is small, conservative, and fully deterministic — no LLM,
no fuzzy matching beyond simple normalization (case, whitespace, quoting).
False positives are worse than false negatives here, so the heuristics
err on the side of skipping ambiguous matches.

Currently implemented bridges:

  1. **Airflow Cosmos selector → dbt model** — when a `DbtDag` /
     `BashOperator` declares `tag:X`, find dbt models in the same graph
     that carry `dbt_tags` containing `X` and emit a `bridges_to` edge.

  2. **Airflow `dbt_project_path` → dbt model files** — when a Cosmos DAG
     binds to a project at path P, every dbt model node whose
     `source_file` lives under P gets a `bridges_to` edge from the DAG.

  3. **Airflow path: selector → dbt model files under that path** — a
     Cosmos selector `path:models/inventory` matches dbt model nodes
     whose `source_file` ends with `models/inventory/...`.

Bridge edges always carry `confidence: "INFERRED"` and a reason string in
`bridge_reason` so reviewers can audit them.
"""
from __future__ import annotations
import os
from typing import Iterable


def _normalize_path(p: str) -> str:
    """Normalize a path for cross-system comparison: lowercase, forward
    slashes only, no trailing slash."""
    return p.replace("\\", "/").rstrip("/").lower()


def _path_under(child: str, parent: str) -> bool:
    """True if `child` path is at-or-under `parent` (after normalization)."""
    if not child or not parent:
        return False
    c = _normalize_path(child)
    p = _normalize_path(parent)
    return c == p or c.startswith(p + "/")


def _strip_selector_prefix(sel: str) -> tuple[str, str]:
    """Split `tag:bookstore+` → `("tag", "bookstore")`. Returns `("", sel)` if
    no prefix found. Trailing `+` / `1+` (depth markers) stripped."""
    if ":" in sel:
        prefix, _, rest = sel.partition(":")
        prefix = prefix.strip().lower()
    else:
        prefix, rest = "", sel
    rest = rest.strip().rstrip("+").rstrip("1+").lstrip("+1").strip()
    return prefix, rest


def _make_bridge_edge(source_id: str, target_id: str, *, reason: str,
                      relation: str = "bridges_to") -> dict:
    return {
        "source": source_id,
        "target": target_id,
        "relation": relation,
        "confidence": "INFERRED",
        "confidence_score": 0.85,
        "source_file": "",
        "source_location": "",
        "weight": 1.0,
        "bridge_reason": reason,
    }


def compute_bridges(nodes: list[dict], edges: list[dict]) -> tuple[list[dict], list[dict]]:
    """Run all bridge detectors. Returns (extra_nodes, extra_edges) ready
    to be appended to the existing extraction.

    Idempotent: dedupes against existing edges by (source, target, relation)
    so calling this twice doesn't multiply bridges.
    """
    new_nodes: list[dict] = []
    new_edges: list[dict] = []

    # ── Index pass ─────────────────────────────────────────────────────────
    # Node id → node (for fast lookup)
    by_id: dict[str, dict] = {n["id"]: n for n in nodes}

    # Airflow Cosmos selectors emitted by extract_airflow_dag.
    # Each `dbt_selector` node carries the raw selector string in its
    # `dbt_selector` attribute (e.g., "tag:bookstore" or "path:models/inv").
    selector_nodes = [n for n in nodes if n.get("dbt_selector")]

    # Airflow Cosmos `uses_dbt_project` edges → DAG node ↔ project_path
    cosmos_dag_to_project_path: dict[str, str] = {}
    for e in edges:
        if e.get("relation") != "uses_dbt_project":
            continue
        proj_node = by_id.get(e.get("target", ""))
        dag_node = by_id.get(e.get("source", ""))
        if proj_node is None or dag_node is None:
            continue
        path = proj_node.get("dbt_project_path")
        if path:
            cosmos_dag_to_project_path[dag_node["id"]] = path

    # dbt model nodes — produced by extract_dbt_sql for each .sql in a dbt
    # project. Identified by carrying `dbt_alias` or `dbt_materialized`,
    # OR being the source of any `references` / `references_source` edge.
    dbt_model_ids = set()
    for n in nodes:
        if n.get("dbt_alias") or n.get("dbt_materialized") or n.get("dbt_tags"):
            dbt_model_ids.add(n["id"])
    for e in edges:
        if e.get("relation") in ("references", "references_source", "uses_var"):
            dbt_model_ids.add(e.get("source", ""))
    dbt_model_ids.discard("")
    dbt_models = [by_id[i] for i in dbt_model_ids if i in by_id]

    # Existing edges set for dedup
    existing_pairs = {(e.get("source"), e.get("target"), e.get("relation"))
                      for e in edges}

    def _emit(src: str, tgt: str, *, reason: str) -> None:
        if not src or not tgt or src == tgt:
            return
        key = (src, tgt, "bridges_to")
        if key in existing_pairs:
            return
        existing_pairs.add(key)
        new_edges.append(_make_bridge_edge(src, tgt, reason=reason))

    # ── Bridge 1: Airflow `selects_dbt: tag:X` → dbt models with tag X ────

    # Index models by tag
    models_by_tag: dict[str, list[str]] = {}
    for m in dbt_models:
        for t in (m.get("dbt_tags") or []):
            models_by_tag.setdefault(t.lower(), []).append(m["id"])

    # Find which DAG/Task owns each selector via existing selects_dbt /
    # excludes_dbt edges
    selector_owners: dict[str, list[str]] = {}
    for e in edges:
        if e.get("relation") in ("selects_dbt", "excludes_dbt"):
            sel_node = by_id.get(e.get("target", ""))
            if sel_node and sel_node.get("dbt_selector"):
                selector_owners.setdefault(e["target"], []).append(e["source"])

    for sel_node in selector_nodes:
        sel_str = sel_node["dbt_selector"]
        prefix, body = _strip_selector_prefix(sel_str)
        if prefix == "tag":
            for model_id in models_by_tag.get(body.lower(), []):
                for owner_id in selector_owners.get(sel_node["id"], []):
                    _emit(owner_id, model_id,
                          reason=f"airflow selector '{sel_str}' matches dbt model tag '{body}'")
        elif prefix == "path":
            # Match dbt model nodes whose source_file contains this path
            target_path = body.strip("/")
            for m in dbt_models:
                sf = m.get("source_file", "")
                if not sf:
                    continue
                if target_path.lower() in _normalize_path(sf):
                    for owner_id in selector_owners.get(sel_node["id"], []):
                        _emit(owner_id, m["id"],
                              reason=f"airflow selector '{sel_str}' matches dbt model under '{body}'")

    # ── Bridge 2: Airflow `dbt_project_path` → dbt models in that project ──

    for dag_id, project_path in cosmos_dag_to_project_path.items():
        for m in dbt_models:
            sf = m.get("source_file", "")
            if _path_under(sf, project_path):
                _emit(dag_id, m["id"],
                      reason=f"airflow Cosmos DAG runs dbt project at '{project_path}'")

    return new_nodes, new_edges
