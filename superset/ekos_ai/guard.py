"""Accept a model's SQL only if it is one read-only SELECT over known tables (pure, no I/O)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

# Functions that read files, sleep, reach other servers or change session state.
_FORBIDDEN_FUNCTIONS = {
    "pg_sleep", "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "dblink", "dblink_exec", "set_config", "current_setting",
    "pg_terminate_backend", "pg_cancel_backend", "copy", "load_file", "sleep", "benchmark",
}
_SYSTEM_SCHEMAS = {"pg_catalog", "information_schema", "pg_toast", "mysql", "sys"}
_WRITE_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command,
    exp.Merge, exp.Set, exp.Use, exp.Grant, exp.TruncateTable, exp.Into,
)


class Refused(ValueError):
    """The SQL is not acceptable; the message says why, in words a person can act on."""


@dataclass(frozen=True)
class Checked:
    sql: str
    tables: tuple[str, ...]


def check_sql(
    sql: str, tables: dict[str, list[str]], *, dialect: str = "postgres", max_rows: int = 200,
    schema: str | None = None,
) -> Checked:
    """Return the SQL with a row limit if it is acceptable, else raise Refused.

    ``tables`` maps each table the person may query (lower case) to its columns.
    """
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise Refused("The model returned no SQL.")
    try:
        statements = [s for s in sqlglot.parse(text, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as err:
        raise Refused(f"The SQL does not parse: {str(err)[:160]}") from err
    if len(statements) != 1:
        raise Refused("Exactly one statement is allowed.")
    root = statements[0]
    if not isinstance(root, (exp.Select, exp.Union, exp.Subquery, exp.With)) and not isinstance(
        root, exp.Query
    ):
        raise Refused("Only a SELECT is allowed.")
    for node in root.walk():
        if isinstance(node, _WRITE_NODES):
            raise Refused(f"{type(node).__name__} is not allowed: this is a read-only query.")
        if isinstance(node, exp.Anonymous) and str(node.this).lower() in _FORBIDDEN_FUNCTIONS:
            raise Refused(f"The function {node.this} is not allowed.")
        if isinstance(node, exp.Func):
            name = (node.sql_name() or "").lower()
            if name in _FORBIDDEN_FUNCTIONS:
                raise Refused(f"The function {name} is not allowed.")
        if isinstance(node, exp.Anonymous) and str(node.this).lower().startswith("pg_"):
            raise Refused(f"The function {node.this} is not allowed.")
    requalify_to = schema  # the loop below names each table's own schema `schema`
    ctes = {c.alias_or_name.lower() for c in root.find_all(exp.CTE)}
    used: list[str] = []
    for table in root.find_all(exp.Table):
        name = (table.name or "").lower()
        schema = (table.db or "").lower()
        if name in ctes and not schema:
            continue
        if schema in _SYSTEM_SCHEMAS:
            raise Refused(f"The schema {schema} is not available.")
        if name not in tables:
            raise Refused(f"The table {name or '?'} is not one you may query.")
        used.append(name)
    if not used:
        raise Refused("The query reads no table.")
    if requalify_to:
        # The source holds these tables under another schema (an Iceberg copy): name it.
        for table in root.find_all(exp.Table):
            if (table.name or "").lower() in ctes and not table.db:
                continue
            table.set("db", exp.to_identifier(requalify_to))
    if root.args.get("limit") is None and isinstance(root, exp.Select):
        root = root.limit(max_rows)
    return Checked(sql=root.sql(dialect=dialect), tables=tuple(sorted(set(used))))


_TAG = re.compile(r"[<>]")


def quote_question(text: str, *, limit: int = 600) -> str:
    """The person's text as data: bounded, with angle brackets neutralised so it cannot close
    the tag it sits in."""
    return _TAG.sub(" ", (text or "")[:limit]).strip()
