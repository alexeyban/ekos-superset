"""Accept a model's SQL only if it is one read-only SELECT over known tables (pure, no I/O)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.tokens import TokenType

# Functions that read files, sleep, reach other servers or change session state.
_FORBIDDEN_FUNCTIONS = {
    "pg_sleep", "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "dblink", "dblink_exec", "set_config", "current_setting",
    "pg_terminate_backend", "pg_cancel_backend", "copy", "load_file", "sleep", "benchmark",
    # DuckDB (EKOS RFC 0259 S3): table functions that read files or the network, the environment,
    # and those that run SQL given as a string. DuckDB refuses most of these itself when the
    # connection has `enable_external_access = false` (S1); the guard is the second barrier.
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "read_json_objects", "read_json_objects_auto", "read_ndjson", "read_ndjson_auto",
    "read_ndjson_objects", "read_text", "read_blob", "read_xlsx", "glob", "iceberg_scan",
    "iceberg_metadata", "iceberg_snapshots", "delta_scan", "parquet_scan", "parquet_metadata",
    "parquet_schema", "parquet_file_metadata", "parquet_kv_metadata", "sniff_csv", "getenv",
    "query", "query_table", "duckdb_secrets", "which_secret", "duckdb_settings",
}
# Name prefixes refused as functions: PostgreSQL's server functions; DuckDB's file readers.
_FORBIDDEN_PREFIXES = ("pg_", "read_", "parquet_", "iceberg_", "delta_")
# Statements refused by their first word, before parsing (sqlglot reads several of DuckDB's as
# something else: `DETACH x` and `INSTALL x` as an alias, `EXPORT DATABASE` not at all).
_FORBIDDEN_STATEMENTS = {
    "ATTACH", "DETACH", "COPY", "EXPORT", "IMPORT", "INSTALL", "LOAD", "PRAGMA", "SET", "RESET",
    "CALL", "USE", "CHECKPOINT", "FORCE", "VACUUM",
}
_SYSTEM_SCHEMAS = {"pg_catalog", "information_schema", "pg_toast", "mysql", "sys", "system", "temp"}
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
    _refuse_statement(text, dialect)
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
        if isinstance(node, exp.Anonymous) and str(node.this).lower().startswith(_FORBIDDEN_PREFIXES):
            raise Refused(f"The function {node.this} is not allowed.")
    literal_tables = _string_literal_tables(text, dialect)
    requalify_to = schema  # the loop below names each table's own schema `schema`
    ctes = {c.alias_or_name.lower() for c in root.find_all(exp.CTE)}
    used: list[str] = []
    for table in root.find_all(exp.Table):
        name = (table.name or "").lower()
        schema = (table.db or "").lower()
        if isinstance(table.this, exp.Identifier) and table.this.quoted and table.name in literal_tables:
            raise Refused(f"A string cannot be read as a table ('{table.name[:80]}'): name a table.")
        if name in ctes and not schema:
            continue
        for part in (schema, (table.catalog or "").lower()):
            if part in _SYSTEM_SCHEMAS:
                raise Refused(f"The schema {part} is not available.")
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


def _tokens(text: str, dialect: str):
    try:
        return sqlglot.tokenize(text, read=dialect)
    except sqlglot.errors.TokenError as err:
        raise Refused(f"The SQL does not parse: {str(err)[:160]}") from err


def _refuse_statement(text: str, dialect: str) -> None:
    """Refuse a statement by its first word (comments skipped by the tokenizer)."""
    tokens = _tokens(text, dialect)
    first = tokens[0].text.upper() if tokens else ""
    if first in _FORBIDDEN_STATEMENTS:
        raise Refused(f"{first} is not allowed: this is a read-only query.")


def _string_literal_tables(text: str, dialect: str) -> set[str]:
    """The texts of the query's string literals. sqlglot reads DuckDB's `FROM '/etc/passwd'` (a
    file read by a replacement scan) as a quoted identifier, the same as `FROM "x"`; a quoted
    table name that is also one of these texts was written as a string."""
    return {t.text for t in _tokens(text, dialect) if t.token_type == TokenType.STRING}


def tables_in(sql: str, *, dialect: str = "postgres") -> set[str]:
    """The tables a query reads (lower case, without schema; CTE names left out). Raises Refused
    when it does not parse — used to fetch only those tables' schema before ``check_sql``."""
    text = (sql or "").strip().rstrip(";").strip()
    try:
        statements = [s for s in sqlglot.parse(text, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as err:
        raise Refused(f"The SQL does not parse: {str(err)[:160]}") from err
    names: set[str] = set()
    for root in statements:
        ctes = {c.alias_or_name.lower() for c in root.find_all(exp.CTE)}
        for table in root.find_all(exp.Table):
            name = (table.name or "").lower()
            if name and not (name in ctes and not table.db):
                names.add(name)
    return names


_TAG = re.compile(r"[<>]")


def quote_question(text: str, *, limit: int = 600) -> str:
    """The person's text as data: bounded, with angle brackets neutralised so it cannot close
    the tag it sits in."""
    return _TAG.sub(" ", (text or "")[:limit]).strip()
