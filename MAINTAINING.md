# Maintaining this fork

This fork carries a stack of personal features on top of upstream
[`safishamsi/graphify`](https://github.com/safishamsi/graphify). The stable
"use this" branch is **`tl-stack`**. Install from it directly:

```bash
pip install 'graphifyy[all] @ git+https://github.com/ilmaliarenko/graphify.git@tl-stack'
# or for editable dev:
pip install -e '/path/to/graphify[all]'
```

## What's on top of upstream

`tl-stack` = `upstream/v6` + the following commits, replayed in order:

1. Terraform / HCL AST extractor
2. dbt-aware SQL extraction (Jinja routing on top of tree-sitter-sql)
3. `collect_files._EXTENSIONS` derived from `detect.CODE_EXTENSIONS`
4. Plan-mode awareness + `graphify ast-only` subcommand
5. dbt YAML extractor (`extract_dbt_yaml`) — opt-in, safe for k8s users
6. Airflow / Astronomer Cosmos DAG structural extractor
7. Cross-system bridges resolver + dbt SQL `tags=[...]` extraction
8. dbt-manifest CLI + Snowflake admin DDL extractor
9. LLM-memory utilities (`search` / `context` / `diff` / `reflect` + `extracted_at` recency)

Run `git log --oneline upstream/v6..tl-stack` for the live list.

## Syncing with upstream

Whenever upstream cuts a new release (or just lands commits on `v6`):

```bash
cd /path/to/graphify
git fetch upstream
git checkout tl-stack
git rebase upstream/v6
# Resolve conflicts. Most live in graphify/extract.py (_DISPATCH, extension
# lists), pyproject.toml (extras), and tests/test_languages.py. Keep upstream's
# refactors + re-apply our additions on top.
pytest tests/ -q   # must stay green
git push --force-with-lease origin tl-stack
```

That's the whole loop — one branch, one rebase.

## Conflict cheat sheet

Recurring spots where upstream and this fork tend to clash:

| File | What to expect |
|---|---|
| `graphify/extract.py` (`_DISPATCH`) | Upstream adds a new extractor — keep theirs and add our `.tf`/`.hcl`/`.yml`/`.yaml`/`.py → extract_python_with_airflow` entries back. |
| `graphify/extract.py` (extract loop) | Upstream tweaks the parallel/cache scaffold — keep theirs; our additions live in standalone functions, not in the loop. |
| `pyproject.toml` (`[project.optional-dependencies]`) | Union-merge — keep upstream's new extras + our `dbt`/`terraform`. |
| `tests/test_languages.py` | Upstream adds language tests — keep theirs and re-attach our extractor test classes below. |
| `README.md` | Append our extras to the supported-languages / extras tables; never overwrite upstream's section. |

If a rebase pulls in an upstream refactor that genuinely supersedes one of our
patches, just drop the now-redundant commit during the interactive rebase.

## When tests stay red after a rebase

- New silent-output requirements from upstream (e.g. `hook-check` must emit
  zero stderr) are likely; gate any newly-added warnings behind
  `_SILENT_COMMANDS` in `graphify/__main__.py`.
- Upstream sometimes adds dynamic-import gates for tree-sitter parsers; our
  extractor tests already use `pytest.importorskip(...)` for the same reason —
  follow that pattern for any new dependency.
