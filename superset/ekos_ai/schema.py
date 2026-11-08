"""The tables the model may see: names and columns, never values."""

from __future__ import annotations

import re
import time

_CACHE: dict[tuple, tuple[float, dict[str, list[tuple[str, str]]]]] = {}
_TTL = 3600


def load(database, schema: str, allowed: set[str] | None) -> dict[str, list[tuple[str, str]]]:
    """table -> [(column, type)] for one schema, from information_schema, cached for an hour."""
    key = (database.id, schema, tuple(sorted(allowed or ())))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    sql = (
        "select table_name, column_name, data_type from information_schema.columns "
        "where table_schema = '" + schema.replace("'", "''") + "' "
        "order by table_name, ordinal_position"
    )
    frame = database.get_df(sql, schema=schema)
    tables: dict[str, list[tuple[str, str]]] = {}
    for table, column, kind in frame.itertuples(index=False):
        name = str(table).lower()
        if allowed and name not in allowed:
            continue
        tables.setdefault(name, []).append((str(column).lower(), str(kind)))
    _CACHE[key] = (time.time(), tables)
    return tables


_WORD = re.compile(r"[a-z0-9]+")


def rank(question: str, tables: dict[str, list[tuple[str, str]]], keep: int = 14) -> list[str]:
    """The tables most likely to matter: words of the question against table and column names."""
    words = {w.rstrip("s") for w in _WORD.findall(question.lower()) if len(w) > 2}
    scored = []
    for name, cols in tables.items():
        parts = {p.rstrip("s") for p in name.split("_")}
        score = 3 * len(words & parts) + sum(
            1 for c, _ in cols if words & {p.rstrip("s") for p in c.split("_")}
        )
        if score:
            scored.append((-score, name))
    scored.sort()
    return [n for _, n in scored[:keep]]


def describe(names: list[str], tables: dict[str, list[tuple[str, str]]]) -> str:
    return "\n".join(
        f"{n}({', '.join(f'{c} {t}' for c, t in tables[n][:60])})" for n in names
    )
