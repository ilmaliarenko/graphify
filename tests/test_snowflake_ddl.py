"""Tests for the Snowflake admin DDL extractor in graphify.extract."""
from __future__ import annotations
from pathlib import Path

import pytest

from graphify.extract import (
    extract_snowflake_ddl,
    extract_sql,
    _is_snowflake_ddl,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── Detection helper ────────────────────────────────────────────────────────


def test_is_snowflake_ddl_detects_masking_policy():
    assert _is_snowflake_ddl("CREATE OR REPLACE MASKING POLICY x AS (v) ...") is True


def test_is_snowflake_ddl_detects_pipe():
    assert _is_snowflake_ddl("CREATE PIPE my_pipe AUTO_INGEST = TRUE AS COPY INTO ...") is True


def test_is_snowflake_ddl_detects_grant():
    assert _is_snowflake_ddl("GRANT SELECT ON TABLE x TO ROLE r;") is True


def test_is_snowflake_ddl_negative_for_plain_sql():
    assert _is_snowflake_ddl("SELECT id, name FROM users WHERE created > '2025-01-01'") is False


def test_is_snowflake_ddl_negative_for_dbt_jinja():
    assert _is_snowflake_ddl("{{ config(materialized='table') }} SELECT * FROM {{ ref('x') }}") is False


# ── Real fixture extraction ──────────────────────────────────────────────────


def test_snowflake_ddl_no_error():
    r = extract_snowflake_ddl(FIXTURES / "sample_snowflake_ddl.sql")
    assert "error" not in r, r.get("error")


def test_snowflake_ddl_extracts_masking_policies():
    r = extract_snowflake_ddl(FIXTURES / "sample_snowflake_ddl.sql")
    policies = [n for n in r["nodes"] if n.get("snowflake_object") == "masking_policy"]
    qualified = {n.get("snowflake_qualified_name") for n in policies}
    assert "GOVERNANCE.mask_customer_email" in qualified
    assert "GOVERNANCE.mask_credit_card" in qualified


def test_snowflake_ddl_extracts_function():
    r = extract_snowflake_ddl(FIXTURES / "sample_snowflake_ddl.sql")
    funcs = [n for n in r["nodes"] if n.get("snowflake_object") == "function"]
    qualified = {n.get("snowflake_qualified_name") for n in funcs}
    assert "GOVERNANCE.mask_email" in qualified


def test_snowflake_ddl_extracts_pipe():
    r = extract_snowflake_ddl(FIXTURES / "sample_snowflake_ddl.sql")
    pipes = [n for n in r["nodes"] if n.get("snowflake_object") == "pipe"]
    qualified = {n.get("snowflake_qualified_name") for n in pipes}
    assert any("orders_pipe" in q for q in qualified)


def test_snowflake_ddl_extracts_role_definition():
    r = extract_snowflake_ddl(FIXTURES / "sample_snowflake_ddl.sql")
    roles = [n for n in r["nodes"] if n.get("snowflake_object") == "role"]
    role_labels = {n["label"] for n in roles}
    assert any("bookstore_analyst_role" in r for r in role_labels)


def test_snowflake_ddl_alter_table_emits_applies_masking_policy_edge():
    """`ALTER TABLE T ... SET MASKING POLICY P` produces a `applies_masking_policy`
    edge from the table to the policy node."""
    r = extract_snowflake_ddl(FIXTURES / "sample_snowflake_ddl.sql")
    apply_edges = [e for e in r["edges"] if e["relation"] == "applies_masking_policy"]
    assert len(apply_edges) == 2  # CUSTOMERS + ORDERS
    targets = {e["target"] for e in apply_edges}
    sources = {e["source"] for e in apply_edges}
    # Both policies should be applied
    policy_labels = [n["label"] for n in r["nodes"] if n["id"] in targets]
    assert any("mask_customer_email" in l for l in policy_labels)
    assert any("mask_credit_card" in l for l in policy_labels)
    # Both tables should appear as sources
    table_labels = [n["label"] for n in r["nodes"] if n["id"] in sources]
    assert any("CUSTOMERS" in l for l in table_labels)
    assert any("ORDERS" in l for l in table_labels)


def test_snowflake_ddl_grants_to_role():
    r = extract_snowflake_ddl(FIXTURES / "sample_snowflake_ddl.sql")
    grants = [e for e in r["edges"] if e["relation"] == "grants_to"]
    assert len(grants) >= 3  # USAGE + SELECT + APPLY MASKING POLICY
    # Each grant edge has snowflake_privileges + snowflake_grant_object attributes
    for e in grants:
        assert e.get("snowflake_privileges")
        assert e.get("snowflake_grant_object")


def test_snowflake_ddl_no_dangling_edges():
    r = extract_snowflake_ddl(FIXTURES / "sample_snowflake_ddl.sql")
    node_ids = {n["id"] for n in r["nodes"]}
    for e in r["edges"]:
        assert e["source"] in node_ids, f"Dangling source: {e}"
        assert e["target"] in node_ids, f"Dangling target: {e}"


# ── Routing through extract_sql ──────────────────────────────────────────────


def test_extract_sql_routes_to_snowflake_ddl_for_admin_files(tmp_path):
    """A pure Snowflake admin file (no Jinja, no dbt project) must reach
    `extract_snowflake_ddl`, not the plain tree-sitter-sql path."""
    f = tmp_path / "admin.sql"
    f.write_text(
        "CREATE OR REPLACE MASKING POLICY my_db.my_schema.mask_email AS (val STRING)\n"
        "RETURNS STRING -> CASE WHEN CURRENT_ROLE() = 'ADMIN' THEN val ELSE '***' END;\n"
    )
    r = extract_sql(f)
    # Snowflake DDL produces snowflake_object attributes; tree-sitter-sql does not
    masking = [n for n in r["nodes"] if n.get("snowflake_object") == "masking_policy"]
    assert len(masking) == 1


def test_extract_sql_keeps_dbt_path_for_jinja_sql(tmp_path):
    """A SQL file with Jinja markers must still go through dbt_sql, NOT
    snowflake_ddl, even if it accidentally mentions GRANT or MASKING POLICY
    in a comment."""
    f = tmp_path / "dbt_model.sql"
    f.write_text(
        "-- This model also uses GRANT in a comment but it's a dbt model\n"
        "{{ config(materialized='table') }}\n"
        "SELECT * FROM {{ ref('upstream') }}\n"
    )
    r = extract_sql(f)
    # dbt path emits `references` edges
    assert any(e["relation"] == "references" for e in r["edges"])
    # And NO snowflake_object attribute on any node
    assert all(n.get("snowflake_object") is None for n in r["nodes"])


def test_extract_sql_keeps_plain_sql_path_for_select(tmp_path):
    """A pure SELECT (no Jinja, no Snowflake DDL signatures) goes through
    the upstream tree-sitter-sql extractor."""
    f = tmp_path / "query.sql"
    f.write_text(
        "CREATE TABLE foo (id INT);\n"
        "SELECT id, name FROM bar JOIN baz ON bar.id = baz.bar_id;\n"
    )
    r = extract_sql(f)
    rels = {e["relation"] for e in r["edges"]}
    # tree-sitter-sql produces `contains` edges; snowflake_ddl doesn't
    assert "contains" in rels
