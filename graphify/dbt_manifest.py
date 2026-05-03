"""Convert a dbt `target/manifest.json` to graphify graph nodes/edges.

`dbt parse` (or any successful `dbt run`) writes `target/manifest.json` —
a complete, env-resolved view of every model / source / seed / snapshot /
test / macro plus the full `parent_map` of dependencies. This module reads
that file and emits nodes/edges in the standard graphify schema.

Why a separate extractor (in addition to `extract_dbt_sql` /
`extract_dbt_yaml` walking files)?

  • **Manifest is env-resolved.** `silver_x` becomes `db.schema.alias`,
    Jinja `{{ var(...) }}` becomes literal values, conditional
    materializations resolved. The file-walking extractor sees only
    source code — manifest sees what dbt actually compiled.

  • **Manifest is the source of truth for tests.** Every dbt test (built-in
    or custom) appears as its own node with a `parent_map` link to the
    target model + column.

  • **Cross-merging.** Manifest node IDs are produced via the same
    `_make_id` convention as the file extractors, so loading both into
    the same graph naturally collapses the file-node + manifest-node
    for each model into one merged entity.

CLI: `graphify dbt-manifest <manifest.json> --out graph.json --project-root <dir>`
"""
from __future__ import annotations
import json
import re
from pathlib import Path


def _make_id(*parts: str) -> str:
    """Match graphify's `_make_id` semantics so manifest IDs align with the
    IDs produced by the file-walking extractors (`extract_dbt_sql`, etc.)."""
    combined = "_".join(p.strip("_.") for p in parts if p)
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", combined).strip("_").lower()
    return cleaned


def _node_id_from_unique_id(unique_id: str) -> str:
    """Map a dbt `unique_id` (e.g. `model.pkg.silver_tl__profile_audit`)
    to the graphify ID convention used by the file extractors."""
    parts = unique_id.split(".")
    rt = parts[0] if parts else ""
    if rt == "source" and len(parts) >= 4:
        return _make_id("source", parts[2], parts[3])
    if rt == "seed":
        return _make_id("seed", parts[-1])
    if rt == "snapshot":
        return _make_id("snapshot", parts[-1])
    if rt == "test":
        return _make_id("test", parts[-1])
    # model, exposure, metric, semantic_model, analysis — last part is the name
    return _make_id(parts[-1])


def _label_from_node(node: dict, unique_id: str) -> str:
    rt = unique_id.split(".")[0]
    name = node.get("name", unique_id.split(".")[-1])
    if rt == "source":
        return f"source.{node.get('source_name', '?')}.{name}"
    return name


def convert(manifest_path: Path, project_root: Path | None = None) -> dict:
    """Read a dbt `manifest.json` and return `{nodes, edges, hyperedges}`.

    `project_root` is only used to compute absolute `source_file` paths from
    the manifest's relative `original_file_path`. If omitted, paths stay
    relative — graphify's downstream pipeline relativizes anyway.
    """
    m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    nodes_out: list[dict] = []
    edges_out: list[dict] = []
    seen_ids: set[str] = set()

    def _add_node(nid: str, label: str, source_file: str, extras: dict | None = None) -> None:
        if nid in seen_ids:
            return
        seen_ids.add(nid)
        n = {
            "id": nid, "label": label, "file_type": "code",
            "source_file": source_file, "source_location": "L1",
        }
        if extras:
            n.update({k: v for k, v in extras.items() if v not in (None, "", [])})
        nodes_out.append(n)

    def _resolve_path(rel: str) -> str:
        if not rel:
            return ""
        if project_root is None:
            return rel
        return str((Path(project_root) / rel).resolve())

    # 1) Models, snapshots, seeds, tests
    for unique_id, node in m.get("nodes", {}).items():
        nid = _node_id_from_unique_id(unique_id)
        label = _label_from_node(node, unique_id)
        abs_path = _resolve_path(node.get("original_file_path", ""))
        cfg = node.get("config", {}) or {}
        extras = {
            "dbt_unique_id": unique_id,
            "dbt_resource_type": node.get("resource_type"),
            "dbt_materialized": cfg.get("materialized"),
            "dbt_database": node.get("database"),
            "dbt_schema": node.get("schema"),
            "dbt_alias": node.get("alias"),
            "dbt_tags": node.get("tags") or cfg.get("tags") or [],
        }
        _add_node(nid, label, abs_path, extras)

    # 2) Sources
    for unique_id, src in m.get("sources", {}).items():
        nid = _node_id_from_unique_id(unique_id)
        label = _label_from_node(src, unique_id)
        abs_path = _resolve_path(src.get("original_file_path", ""))
        extras = {
            "dbt_unique_id": unique_id,
            "dbt_resource_type": "source",
            "dbt_database": src.get("database"),
            "dbt_schema": src.get("schema"),
            "dbt_source_name": src.get("source_name"),
            "dbt_table_name": src.get("name"),
        }
        _add_node(nid, label, abs_path, extras)

    # 3) parent_map → references / references_source edges (child → parent)
    parent_map = m.get("parent_map", {})
    edges_seen: set[tuple[str, str]] = set()
    for child_uid, parents in parent_map.items():
        if not parents:
            continue
        child_nid = _node_id_from_unique_id(child_uid)
        if child_nid not in seen_ids:
            continue
        for parent_uid in parents:
            # Skip macro→model edges (kept out to keep the graph focused on data lineage)
            if parent_uid.startswith("macro."):
                continue
            parent_nid = _node_id_from_unique_id(parent_uid)
            if parent_nid not in seen_ids:
                continue
            key = (child_nid, parent_nid)
            if key in edges_seen:
                continue
            edges_seen.add(key)
            relation = "references_source" if parent_uid.startswith("source.") else "references"
            edges_out.append({
                "source": child_nid,
                "target": parent_nid,
                "relation": relation,
                "confidence": "EXTRACTED",
                "confidence_score": 1.0,
                "source_file": "",
                "source_location": "",
                "weight": 1.0,
                "from_dbt_manifest": True,
            })

    return {"nodes": nodes_out, "edges": edges_out, "hyperedges": []}


def stats(graph: dict) -> dict:
    """Summary stats for a manifest-derived graph."""
    from collections import Counter
    rel = Counter(e["relation"] for e in graph["edges"])
    rt = Counter(n.get("dbt_resource_type", "?") for n in graph["nodes"])
    return {
        "total_nodes": len(graph["nodes"]),
        "total_edges": len(graph["edges"]),
        "by_resource_type": dict(rt),
        "by_relation": dict(rel),
    }
