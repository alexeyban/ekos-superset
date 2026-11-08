# EKOS additions to Superset

This fork of Apache Superset 4.1.1 adds **Ask (AI)**: text-to-SQL and text-to-dashboard.
Everything is in `superset/ekos_ai/` plus one registration block in `superset/initialization/__init__.py`.

## Answers come from EKOS

Every question is answered from the knowledge EKOS compiled, read over MCP (`ekos_mcp.py`,
`knowledge.py`). Superset never reads the database's own catalog for this, and there is no
fallback: when EKOS cannot be reached, or has compiled no tables, nothing is answered.

For each question:

1. **Schema from EKOS.** `ekos_ossie_export` gives the compiled tables with their descriptions,
   synonyms, columns, primary keys and foreign-key joins, each with its ledger id and evidence
   (cached 5 minutes). `EKOS_AI_TABLES` narrows it; a table EKOS has not compiled cannot be queried.
2. **Relevant tables.** `ekos_search` ranks the question's words, combined with matches on names,
   synonyms and descriptions; tables joined by a foreign key to the best ones come along.
3. **Engine rules.** `ekos_sql_check` with only `dialect` returns the engine's rules (row limits,
   integer division, quoting, dates). They go into the prompt.
4. **SQL is written by EKOS** when the endpoint offers `ekos_sql_generate` (EKOS RFC 0250): EKOS's
   own text-to-SQL agent, measured on BIRD, writes the query from the compiled schema and probes the
   live database read-only; Superset only applies its guard and runs it. Otherwise (and for
   dashboard charts) **the model proposes** SQL or chart specs from that schema only, as JSON. It
   sees names and meanings, **never row values**.
5. **Checks.** `guard.check_sql` (one read-only SELECT over EKOS's tables, no system schema, no
   file/sleep/dblink functions, a row limit), then `ekos_sql_check` against the compiled schema
   (unknown columns with the nearest real names, joins off the foreign keys, dialect mistakes).
   An error goes back to the model once; if it is still there, the question is refused with EKOS's
   findings and nothing runs. For a dashboard, every chart's SQL goes through the same checks.
6. **The answer** carries an `ekos` block: the tables used (ledger id, description, evidence), the
   foreign-key joins between them, the check's warnings and its note. The page shows it under
   "From EKOS".

The query then runs with a 15 s statement timeout. Dashboards are built with Superset's own
commands, owned by the caller, after `planner.form_data` checks each spec against its result's
columns. The person's text is delimited data (`<question>`, angle brackets removed, 600 characters).

## Configuration (`superset_config.py`)

| Key | Meaning |
|---|---|
| `EKOS_AI_ENABLED` | off by default; nothing is registered when false |
| `EKOS_MCP_URL` | the EKOS MCP endpoint: `ekos mcp serve --http <addr>` (`http://host:port/mcp`) or a console project's gateway (`https://<console>/mcp/<tenant>/<project>`) |
| `EKOS_MCP_TOKEN_ENV` | name of the environment variable holding the bearer token (default `EKOS_MCP_TOKEN`); for the console, an API token with the **read** role |
| `EKOS_MCP_DATABASE` | only tables of this `[[databases]]` name (optional) |
| `EKOS_MCP_STATUS` | which business meanings to show the model: `confirmed` (default), `hypothesis` or `all` |
| `EKOS_MCP_TIMEOUT` | seconds per EKOS call (default 30) |
| `EKOS_AI_DIALECT` | the engine for EKOS's checks and the guard: `postgres` (default), `trino`, `oracle`, … |
| `EKOS_AI_DATABASE` | name of the Superset database connection to query (use a **read-only role**) |
| `EKOS_AI_SCHEMA` | schema the queries run in |
| `EKOS_AI_TABLES` | comma list of tables the model may use, out of those EKOS knows (recommended) |
| `EKOS_AI_MODEL`, `EKOS_AI_BASE_URL` | Anthropic Messages API |
| `EKOS_AI_KEY_ENV` | name of the environment variable holding the model key (default `ANTHROPIC_API_KEY`) |
| `EKOS_AI_MAX_ROWS` | row limit (default 200) |
| `EKOS_AI_PER_USER_PER_DAY` | questions (SQL or dashboard) per Superset user per UTC day (default 5) |
| `EKOS_AI_PER_DAY` | questions per day for the whole instance (default 50) |

Permissions: the `EkosAI` resource (`can read` for SQL, `can write` for dashboards); Admin has both.

The EKOS tools used are all read-only and in the console's scoped set: `ekos_ossie_export`,
`ekos_search`, `ekos_sql_check`, and `ekos_sql_generate` when the project turns it on
(it calls a model; EKOS and the console cap it too).

## Not done yet

The abuse/ban policy of EKOS RFC 0249 is not wired in here (the daily limits are); until it is,
keep the instance behind sign-in. The two limits are checked then counted, so two requests at the
same moment can pass one over the limit. Question text is sent to the model provider.
`ekos_sql_check` is static: it cannot see value spellings or row counts.

## Build

`Dockerfile.ekos` overlays the module on the official image. Tests run in the official image:
`tests/unit_tests/ekos_ai` (a fake EKOS and a scripted model; no network, no model calls).

## Public sign-in (EKOS RFC 0251)

Behind Azure Container Apps' authentication (require sign-in), `superset.ekos_ai.remote_user.PlatformUser`
turns `X-MS-CLIENT-PRINCIPAL-IDP` / `-NAME` into `REMOTE_USER` (`<idp>:<name>`, e.g. `github:octocat`):

```python
from flask_appbuilder.security.manager import AUTH_REMOTE_USER
from superset.ekos_ai.remote_user import PlatformUser
AUTH_TYPE = AUTH_REMOTE_USER
AUTH_USER_REGISTRATION = True
AUTH_USER_REGISTRATION_ROLE = "EkosPublic"
ADDITIONAL_MIDDLEWARE = [PlatformUser]
```

After `superset init`, `python -m superset.ekos_ai.public_role` creates `EkosPublic` (Gamma without SQL
Lab, plus Ask (AI) and access to `EKOS_AI_DATABASE`) and makes the `EKOS_ADMINS` sign-in names Admin.
Only behind that authentication: it strips the headers from outside requests.
