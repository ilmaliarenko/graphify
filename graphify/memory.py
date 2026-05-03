"""LLM-memory utilities for graphify.

A small read-only API on top of `graph.json`, designed for AI agents that
treat the graph as persistent memory: fast lookup (`search`), token-budgeted
context retrieval (`context`), recency-based deltas (`diff`), and a one-shot
state summary (`reflect`).

All functions are pure read — they never write to the graph file, so they
are safe to call from a Claude Code plan-mode agent. The only side effect
is stdout (via the CLI wrappers in `__main__.py`).
"""
from __future__ import annotations
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


# ── Graph loading helper ─────────────────────────────────────────────────────


def load_graph(graph_path: str | Path) -> dict:
    """Load `graph.json` (NetworkX node-link format).  Returns the raw dict
    with `nodes` and `links` (or `edges`) keys."""
    p = Path(graph_path)
    if not p.exists():
        raise FileNotFoundError(f"graph.json not found at {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _edges(graph: dict) -> list[dict]:
    """NetworkX json_graph stores edges under `links`; some hand-built graphs
    use `edges`. Accept both."""
    return graph.get("links", graph.get("edges", []))


# ── search ──────────────────────────────────────────────────────────────────


def search(graph: dict, query: str, *, limit: int = 10,
           include_neighbors: bool = True) -> dict:
    """Substring + keyword-token match across node labels and select
    attributes (`dbt_alias`, `dbt_tags`, `task_id`, `dag_id`,
    `snowflake_qualified_name`, `dbt_selector`, `airflow_variable`,
    `airflow_connection`).

    Returns top-K matches ranked by:
      1. label exact match (case-insensitive)
      2. label substring match
      3. attribute match
      4. tie-breaker: node degree (more connected ranks higher)

    Each match optionally carries its 1-hop neighbors (id + relation),
    so an LLM can decide whether to dive deeper without a second call.
    """
    if not query:
        return {"query": query, "matches": [], "total": 0}

    nodes = graph.get("nodes", [])
    edges = _edges(graph)
    by_id: dict[str, dict] = {n["id"]: n for n in nodes}

    # Pre-compute degree
    degree: Counter = Counter()
    for e in edges:
        degree[e.get("source", "")] += 1
        degree[e.get("target", "")] += 1

    q_lower = query.lower()
    q_tokens = [t for t in q_lower.replace(":", " ").replace(".", " ").split() if t]

    _SEARCHABLE_ATTRS = (
        "dbt_alias", "dbt_unique_id", "task_id", "dag_id",
        "snowflake_qualified_name", "dbt_selector", "airflow_variable",
        "airflow_connection", "snowflake_object",
    )

    scored: list[tuple[float, dict]] = []
    for n in nodes:
        label = (n.get("label") or "").lower()
        score = 0.0
        # 1. Exact label match
        if label == q_lower:
            score += 100.0
        # 2. Substring match in label
        elif q_lower in label:
            score += 50.0
        # 3. Token overlap with label
        else:
            tokens_hit = sum(1 for t in q_tokens if t in label)
            if tokens_hit:
                score += 10.0 * tokens_hit
        # 4. Attribute matches
        for attr in _SEARCHABLE_ATTRS:
            v = n.get(attr)
            if v is None:
                continue
            v_str = str(v).lower()
            if q_lower in v_str:
                score += 30.0
            elif any(t in v_str for t in q_tokens):
                score += 5.0
        # 5. dbt_tags is a list — check membership
        for t in (n.get("dbt_tags") or []):
            if q_lower == str(t).lower():
                score += 40.0
        if score == 0:
            continue
        # Degree tie-breaker
        score += min(degree.get(n["id"], 0), 50) * 0.1
        scored.append((score, n))

    scored.sort(key=lambda x: -x[0])
    matches = scored[:limit]

    out = []
    for score, n in matches:
        entry = {
            "id": n["id"],
            "label": n.get("label"),
            "score": round(score, 2),
            "source_file": n.get("source_file", ""),
        }
        # Include a few key attrs that aid LLM disambiguation
        for k in ("dbt_resource_type", "dbt_materialized", "snowflake_object",
                  "airflow_operator", "extracted_at"):
            if n.get(k) is not None:
                entry[k] = n[k]
        if include_neighbors:
            neigh = []
            for e in edges:
                if e.get("source") == n["id"]:
                    tgt = by_id.get(e.get("target"))
                    if tgt:
                        neigh.append({"direction": "out",
                                      "relation": e.get("relation"),
                                      "id": tgt["id"], "label": tgt.get("label")})
                elif e.get("target") == n["id"]:
                    src = by_id.get(e.get("source"))
                    if src:
                        neigh.append({"direction": "in",
                                      "relation": e.get("relation"),
                                      "id": src["id"], "label": src.get("label")})
            entry["neighbors"] = neigh[:10]
        out.append(entry)

    return {"query": query, "matches": out, "total": len(scored)}


# ── context ─────────────────────────────────────────────────────────────────


def _approx_tokens(s: str) -> int:
    """Rough token estimate (~4 chars/token, GPT-style). Good enough for
    budget capping; the actual model might differ by ±20%."""
    return max(1, len(s) // 4)


def context(graph: dict, query: str, *, budget_tokens: int = 2000,
            max_hops: int = 2) -> dict:
    """Token-budgeted subgraph retrieval. Used by an LLM agent to fetch the
    smallest relevant subgraph for a question without blowing its context
    window.

    Strategy:
      1. Run `search()` to get seed nodes ranked by relevance.
      2. BFS-expand from seeds up to `max_hops` hops, prioritizing by
         (label match strength, edge confidence, recency, node degree).
      3. Stop adding nodes/edges once the rendered representation would
         exceed `budget_tokens` (approx).
      4. Return as JSON: {seeds, nodes, edges, truncated_at}.
    """
    nodes = graph.get("nodes", [])
    edges = _edges(graph)
    by_id: dict[str, dict] = {n["id"]: n for n in nodes}

    # Build adjacency for fast BFS
    out_adj: dict[str, list[dict]] = defaultdict(list)
    in_adj: dict[str, list[dict]] = defaultdict(list)
    for e in edges:
        out_adj[e.get("source", "")].append(e)
        in_adj[e.get("target", "")].append(e)

    # Seed via search (no neighbors needed, just ids)
    seeds = search(graph, query, limit=5, include_neighbors=False)["matches"]
    if not seeds:
        return {"query": query, "seeds": [], "nodes": [], "edges": [],
                "budget_tokens": budget_tokens, "truncated": False}

    seed_ids = [s["id"] for s in seeds]

    # BFS frontier
    visited_nodes: set[str] = set(seed_ids)
    visited_edges: set[tuple] = set()
    used_tokens = 0

    selected_nodes: list[dict] = []
    selected_edges: list[dict] = []

    def _node_token_cost(n: dict) -> int:
        return _approx_tokens(json.dumps(n, default=str))

    def _edge_token_cost(e: dict) -> int:
        return _approx_tokens(json.dumps(e, default=str))

    # Add seeds first (best matches)
    for sid in seed_ids:
        n = by_id.get(sid)
        if n is None:
            continue
        cost = _node_token_cost(n)
        if used_tokens + cost > budget_tokens:
            return {
                "query": query, "seeds": seeds, "nodes": selected_nodes,
                "edges": selected_edges, "budget_tokens": budget_tokens,
                "truncated": True, "stopped_at": "seed",
            }
        selected_nodes.append(n)
        used_tokens += cost

    # BFS expansion
    frontier = list(seed_ids)
    for hop in range(max_hops):
        next_frontier = []
        # Score each candidate edge by (confidence + recency + endpoint relevance)
        candidates = []
        for nid in frontier:
            for e in out_adj.get(nid, []):
                candidates.append(("out", nid, e))
            for e in in_adj.get(nid, []):
                candidates.append(("in", nid, e))
        # Rank candidates: EXTRACTED > INFERRED > AMBIGUOUS, then by recency
        def _edge_score(c):
            _, _, e = c
            conf = e.get("confidence", "")
            base = {"EXTRACTED": 3, "INFERRED": 2, "AMBIGUOUS": 1}.get(conf, 1)
            return -(base * 1000 + (e.get("extracted_at") or 0) / 1e9)
        candidates.sort(key=_edge_score)

        for direction, src_id, e in candidates:
            other_id = e.get("target") if direction == "out" else e.get("source")
            if not other_id:
                continue
            ek = (e.get("source"), e.get("target"), e.get("relation"))
            if ek in visited_edges:
                continue
            edge_cost = _edge_token_cost(e)
            node_cost = 0
            new_node = None
            if other_id not in visited_nodes:
                new_node = by_id.get(other_id)
                if new_node is None:
                    continue
                node_cost = _node_token_cost(new_node)
            if used_tokens + edge_cost + node_cost > budget_tokens:
                # Skip this expansion but try smaller ones
                continue
            visited_edges.add(ek)
            selected_edges.append(e)
            used_tokens += edge_cost
            if new_node is not None:
                visited_nodes.add(other_id)
                selected_nodes.append(new_node)
                used_tokens += node_cost
                next_frontier.append(other_id)
        frontier = next_frontier
        if not frontier:
            break

    return {
        "query": query,
        "seeds": [{"id": s["id"], "label": s["label"], "score": s["score"]}
                  for s in seeds],
        "nodes": selected_nodes,
        "edges": selected_edges,
        "budget_tokens": budget_tokens,
        "approx_tokens_used": used_tokens,
        "truncated": False,
    }


# ── diff ────────────────────────────────────────────────────────────────────


def diff(graph: dict, *, since_seconds: int) -> dict:
    """Return nodes and edges with `extracted_at >= now - since_seconds`.
    Useful for "what's new in my memory since yesterday" queries."""
    cutoff = int(time.time()) - since_seconds
    nodes = graph.get("nodes", [])
    edges = _edges(graph)

    new_nodes = [n for n in nodes if (n.get("extracted_at") or 0) >= cutoff]
    new_edges = [e for e in edges if (e.get("extracted_at") or 0) >= cutoff]

    # Group new nodes by source-file directory for compact reporting
    by_dir: dict[str, list[dict]] = defaultdict(list)
    for n in new_nodes:
        sf = n.get("source_file") or ""
        d = "/".join(sf.split("/")[:-1]) if sf else "(no source)"
        by_dir[d].append(n)

    return {
        "since_seconds": since_seconds,
        "cutoff_unix": cutoff,
        "total_new_nodes": len(new_nodes),
        "total_new_edges": len(new_edges),
        "new_nodes_by_directory": {
            d: [{"id": n["id"], "label": n.get("label")} for n in items[:10]]
            for d, items in sorted(by_dir.items(), key=lambda x: -len(x[1]))
        },
        "edge_relations_added": dict(
            Counter(e.get("relation", "?") for e in new_edges).most_common()
        ),
    }


# ── reflect ─────────────────────────────────────────────────────────────────


def reflect(graph: dict, *, top_k: int = 10) -> dict:
    """One-shot "state of my memory" summary. Surfaces the kinds of things
    an LLM agent should orient on at session start: what's central, what's
    suspicious, what's recent.
    """
    nodes = graph.get("nodes", [])
    edges = _edges(graph)
    by_id: dict[str, dict] = {n["id"]: n for n in nodes}

    # Degree → god nodes
    degree: Counter = Counter()
    for e in edges:
        degree[e.get("source", "")] += 1
        degree[e.get("target", "")] += 1
    god_nodes = [
        {"id": nid, "label": by_id[nid].get("label") if nid in by_id else nid,
         "degree": d}
        for nid, d in degree.most_common(top_k) if nid in by_id
    ]

    # Recency: top recently-added nodes
    recent = sorted(
        ((n.get("extracted_at") or 0, n) for n in nodes),
        key=lambda x: -x[0],
    )[:top_k]
    recent_out = [
        {"id": n["id"], "label": n.get("label"),
         "extracted_at": n.get("extracted_at")}
        for _, n in recent if n.get("extracted_at")
    ]

    # AMBIGUOUS edges that need human review
    ambiguous = [
        {"source": e.get("source"), "target": e.get("target"),
         "relation": e.get("relation"), "source_file": e.get("source_file")}
        for e in edges if e.get("confidence") == "AMBIGUOUS"
    ][:top_k]

    # Edge confidence breakdown
    conf = Counter(e.get("confidence", "?") for e in edges)
    rel = Counter(e.get("relation", "?") for e in edges)

    # Resource-type breakdown (for dbt-heavy graphs)
    dbt_rt = Counter(n.get("dbt_resource_type", "") for n in nodes
                     if n.get("dbt_resource_type"))
    sf_obj = Counter(n.get("snowflake_object", "") for n in nodes
                     if n.get("snowflake_object"))
    af_op = Counter(n.get("airflow_operator", "") for n in nodes
                    if n.get("airflow_operator"))

    # Weakest-connected (degree 0): orphan nodes
    orphans = [n for n in nodes if degree.get(n["id"], 0) == 0]

    return {
        "totals": {
            "nodes": len(nodes), "edges": len(edges),
            "ambiguous_edges": conf.get("AMBIGUOUS", 0),
            "inferred_edges": conf.get("INFERRED", 0),
            "extracted_edges": conf.get("EXTRACTED", 0),
            "orphan_nodes": len(orphans),
        },
        "god_nodes": god_nodes,
        "recently_added": recent_out,
        "ambiguous_edges_to_review": ambiguous,
        "edge_relations": dict(rel.most_common(10)),
        "by_dbt_resource_type": dict(dbt_rt.most_common()),
        "by_snowflake_object": dict(sf_obj.most_common()),
        "by_airflow_operator": dict(af_op.most_common()),
    }
