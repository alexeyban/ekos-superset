# EKOS additions to Superset

This fork of Apache Superset 4.1.1 adds **Ask (AI)**: text-to-SQL and text-to-dashboard.
Everything is in `superset/ekos_ai/` plus one registration block in `superset/initialization/__init__.py`.

## How it works

The model proposes; deterministic code decides. The model sees **table and column names, never row
values**, and answers JSON only.

* **Text to SQL** (`POST /api/v1/ekos_ai/sql`): the model proposes one SELECT. `guard.check_sql`
  accepts it only if it is a single read-only statement over tables on the allow-list, with no write
  nodes, no system schema and no file/sleep/dblink functions; a row limit is added. It runs with a
  15 s statement timeout. Anything else is refused with the reason.
* **Text to dashboard** (`POST /api/v1/ekos_ai/dashboard`): the model proposes up to 6 chart specs
  (`bar|line|pie|big_number|table`, columns, aggregate). Each chart's SQL goes through the same guard
  and is run; `planner.form_data` checks the spec against the result's columns. Only then are the
  dataset, chart and dashboard created with Superset's own commands, owned by the caller.
* The person's text is delimited data (`<question>`, angle brackets removed, 600 characters).

## Configuration (`superset_config.py`)

| Key | Meaning |
|---|---|
| `EKOS_AI_ENABLED` | off by default; nothing is registered when false |
| `EKOS_AI_DATABASE` | name of the Superset database connection to query (use a **read-only role**) |
| `EKOS_AI_SCHEMA` | schema to read |
| `EKOS_AI_TABLES` | comma list of tables the model may use (recommended) |
| `EKOS_AI_MODEL`, `EKOS_AI_BASE_URL` | Anthropic Messages API |
| `EKOS_AI_KEY_ENV` | name of the environment variable holding the key (default `ANTHROPIC_API_KEY`) |
| `EKOS_AI_MAX_ROWS` | row limit (default 200) |

Permissions: the `EkosAI` resource (`can read` for SQL, `can write` for dashboards); Admin has both.

## Not done yet

Per-user daily limits and the abuse/ban policy of EKOS RFC 0249 are not wired in here; until they
are, keep the instance behind an allow-list. Question text is sent to the model provider.

## Build

`Dockerfile.ekos` overlays the module on the official image. Tests run in the official image:
`tests/unit_tests/ekos_ai`.
