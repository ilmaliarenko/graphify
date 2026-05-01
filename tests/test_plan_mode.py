"""Tests for the plan-mode-aware install text and the `graphify ast-only` CLI.

Plan mode = the agent operating under read-only constraints (no writes outside
a plan file). graphify supports this with: (a) explicit guidance in the
CLAUDE.md / SKILL.md install text telling agents that READING the graph is
allowed, and (b) a `graphify ast-only` subcommand that runs AST-only
extraction with a temp cache root (no `graphify-out/` writes) and JSON-only
stdout output (no LLM calls).
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from graphify.__main__ import _CLAUDE_MD_SECTION, _SKILL_REGISTRATION


# ── Plan-mode text in install artefacts ──────────────────────────────────────


def test_claude_md_section_mentions_plan_mode():
    """The CLAUDE.md text written by `graphify claude install` must explicitly
    address plan mode so agents know READING the graph is safe."""
    assert "plan mode" in _CLAUDE_MD_SECTION.lower()


def test_claude_md_section_mentions_ast_only_command():
    """The plan-mode block must point to `graphify ast-only` as the safe way
    to refresh structural understanding without writing to graphify-out/."""
    assert "ast-only" in _CLAUDE_MD_SECTION


def test_claude_md_section_mentions_no_writes_in_plan_mode():
    """Plan-mode rules should make the no-write contract explicit."""
    assert "without writing" in _CLAUDE_MD_SECTION.lower() or "never writes" in _CLAUDE_MD_SECTION.lower()


def test_skill_registration_mentions_plan_mode():
    """Generic `_SKILL_REGISTRATION` (used by `graphify install` for non-Claude
    platforms too) also gets plan-mode guidance — agents on Codex/Cursor/Aider
    benefit equally."""
    assert "plan mode" in _SKILL_REGISTRATION.lower()
    assert "ast-only" in _SKILL_REGISTRATION


# ── `graphify ast-only` CLI ──────────────────────────────────────────────────


def _graphify_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    """Invoke the graphify CLI as a real subprocess so we exercise the same
    code path users hit. Uses the same Python interpreter pytest is using."""
    return subprocess.run(
        [sys.executable, "-m", "graphify", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def test_ast_only_outputs_valid_json_for_a_file(fixtures_dir, tmp_path):
    """`graphify ast-only <file>` prints a JSON dict with nodes/edges to stdout."""
    target = fixtures_dir / "sample.py"
    result = _graphify_cli("ast-only", str(target), "--cache-root", str(tmp_path / "cache"), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert "nodes" in payload and isinstance(payload["nodes"], list)
    assert "edges" in payload and isinstance(payload["edges"], list)
    assert payload["files_extracted"] == 1
    assert payload["scanned"] == str(target.resolve())


def test_ast_only_works_on_a_directory(fixtures_dir, tmp_path):
    """Passing a directory walks it via collect_files() and extracts every code file."""
    result = _graphify_cli("ast-only", str(fixtures_dir), "--cache-root", str(tmp_path / "cache"), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["files_extracted"] >= 5
    assert len(payload["nodes"]) > 0


def test_ast_only_does_not_create_graphify_out_in_cwd(fixtures_dir, tmp_path):
    """The whole point of plan mode: running `ast-only` from any directory must
    NOT create `graphify-out/` next to the user's project."""
    result = _graphify_cli("ast-only", str(fixtures_dir / "sample.py"),
                            "--cache-root", str(tmp_path / "cache"), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "graphify-out").exists()


def test_ast_only_writes_only_to_cache_root(fixtures_dir, tmp_path):
    """Cache writes go to the explicit --cache-root, nowhere else."""
    cache = tmp_path / "ast-cache"
    result = _graphify_cli("ast-only", str(fixtures_dir / "sample.py"),
                            "--cache-root", str(cache), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    # cache_root is created, sibling dirs are not
    assert cache.exists()
    siblings = [p.name for p in tmp_path.iterdir() if p.is_dir() and p.name != "ast-cache"]
    assert siblings == [], f"Unexpected dirs created next to cache: {siblings}"


def test_ast_only_no_cache_uses_fresh_tempdir(fixtures_dir, tmp_path):
    """`--no-cache` makes every run a cache miss (uses a fresh /tmp dir)."""
    target = fixtures_dir / "sample.py"
    r1 = _graphify_cli("ast-only", str(target), "--no-cache", cwd=tmp_path)
    r2 = _graphify_cli("ast-only", str(target), "--no-cache", cwd=tmp_path)
    assert r1.returncode == 0 and r2.returncode == 0
    p1 = json.loads(r1.stdout)
    p2 = json.loads(r2.stdout)
    # Same input → identical extraction (both are fresh, deterministic)
    assert p1["nodes"] == p2["nodes"]


def test_ast_only_missing_path_argument_exits_nonzero(tmp_path):
    """Calling `graphify ast-only` with no path prints usage and exits != 0."""
    result = _graphify_cli("ast-only", cwd=tmp_path)
    assert result.returncode != 0
    assert "usage" in result.stderr.lower()


def test_ast_only_missing_path_target_errors_cleanly(tmp_path):
    """Pointing at a non-existent path errors with a clear message, no traceback."""
    bogus = tmp_path / "no-such-file"
    result = _graphify_cli("ast-only", str(bogus), cwd=tmp_path)
    assert result.returncode != 0
    assert "path not found" in result.stderr.lower()
    # Should not be a Python traceback
    assert "Traceback" not in result.stderr


def test_ast_only_empty_dir_returns_warning_not_error(tmp_path):
    """An empty dir (no code files) returns a JSON warning but exits 0 — it's
    a valid plan-mode query: 'check what's here, return nothing if no code'."""
    empty = tmp_path / "empty"
    empty.mkdir()
    result = _graphify_cli("ast-only", str(empty),
                            "--cache-root", str(tmp_path / "cache"), cwd=tmp_path)
    assert result.returncode == 0
    payload = json.loads(result.stdout)
    assert payload["nodes"] == []
    assert payload["edges"] == []
    assert "warning" in payload


def test_ast_only_help_flag_shows_usage(tmp_path):
    """`graphify ast-only --help` prints usage and exits 2 (standard for help)."""
    result = _graphify_cli("ast-only", "--help", cwd=tmp_path)
    # CLI uses exit 2 for usage display by convention
    assert result.returncode == 2
    assert "usage" in result.stderr.lower()
    assert "--cache-root" in result.stderr
    assert "--no-cache" in result.stderr


# ── Side-effect: skill version warning goes to stderr ───────────────────────


def test_skill_version_warning_does_not_pollute_stdout(fixtures_dir, tmp_path):
    """Stale-skill warnings must go to stderr so JSON-emitting commands stay
    parseable — `graphify ast-only` must produce pure JSON on stdout even
    when the skill is out of date."""
    # Set up a stale skill version stamp at ~/.claude/skills/graphify/.graphify_version
    # Use a temp HOME so we don't clobber the real one
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    skill_dir = fake_home / ".claude" / "skills" / "graphify"
    skill_dir.mkdir(parents=True)
    (skill_dir / ".graphify_version").write_text("0.0.1-stale", encoding="utf-8")

    env = {**os.environ, "HOME": str(fake_home)}
    target = fixtures_dir / "sample.py"
    result = subprocess.run(
        [sys.executable, "-m", "graphify", "ast-only", str(target),
         "--cache-root", str(tmp_path / "cache")],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    # Warning should appear on stderr, not stdout
    assert "warning" in result.stderr.lower()
    # Stdout should be clean parseable JSON, no warning text
    payload = json.loads(result.stdout)
    assert "warning" not in result.stdout.lower() or payload.get("warning") is not None
    # More strict: the warning literal "skill is from graphify" must not be in stdout
    assert "skill is from graphify" not in result.stdout
