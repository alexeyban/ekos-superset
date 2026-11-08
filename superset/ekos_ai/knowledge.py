"""What the model may see, and every check on what it writes, from EKOS.

The schema is EKOS's compiled one (``ekos_ossie_export``): tables with their descriptions,
synonyms, columns, keys and foreign-key joins, each with its ledger id and evidence. The
tables relevant to a question are ranked with ``ekos_search`` plus those names and synonyms,
and a draft SQL is accepted only after ``ekos_sql_check`` finds no error in it. The database's
own catalog is never read: a table EKOS has not compiled cannot be queried.

Everything here except ``load`` / ``search_names`` / ``check`` is pure.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from superset.ekos_ai.ekos_mcp import EkosClient, EkosUnavailable

_TTL = 300
_CACHE: dict[tuple, tuple[float, "Catalog"]] = {}
_WORD = re.compile(r"[a-z0-9]+")
MAX_DATASETS = 500
KEEP = 14


@dataclass(frozen=True)
class Column:
    name: str
    datatype: str = ""
    description: str = ""
    synonyms: tuple[str, ...] = ()


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    description: str = ""
    synonyms: tuple[str, ...] = ()
    primary_key: tuple[str, ...] = ()
    ekos_id: str = ""
    evidence: tuple[str, ...] = ()
    status: str = ""


@dataclass(frozen=True)
class Join:
    left: str
    left_columns: tuple[str, ...]
    right: str
    right_columns: tuple[str, ...]
    ekos_id: str = ""

    def text(self) -> str:
        pairs = " AND ".join(
            f"{self.left}.{a} = {self.right}.{b}"
            for a, b in zip(self.left_columns, self.right_columns)
        )
        return pairs


@dataclass
class Catalog:
    tables: dict[str, Table] = field(default_factory=dict)
    joins: list[Join] = field(default_factory=list)
    instructions: str = ""
    truncated: bool = False

    def columns(self) -> dict[str, list[str]]:
        """table -> column names, the allow-list ``guard.check_sql`` takes."""
        return {n: [c.name for c in t.columns] for n, t in self.tables.items()}


# ── reading EKOS's answer (pure) ─────────────────────────────────────────────────────────────


def _ekos_ext(element: dict) -> dict:
    for ext in element.get("custom_extensions") or []:
        if ext.get("vendor_name") == "EKOS":
            try:
                value = json.loads(ext.get("data") or "{}")
            except (TypeError, json.JSONDecodeError):
                return {}
            return value if isinstance(value, dict) else {}
    return {}


def _synonyms(element: dict) -> tuple[str, ...]:
    context = element.get("ai_context")
    if isinstance(context, dict):
        return tuple(str(s) for s in context.get("synonyms") or [] if s)
    return ()


def _text(value: Any, limit: int = 300) -> str:
    return " ".join(str(value or "").split())[:limit]


def catalog_from_ossie(result: dict, allowed: set[str] | None = None) -> Catalog:
    """A Catalog from an ``ekos_ossie_export`` result; ``allowed`` narrows it to those tables."""
    model = result.get("model") or {}
    catalog = Catalog(truncated=bool(result.get("truncated")))
    context = model.get("ai_context")
    if isinstance(context, dict):
        catalog.instructions = _text(context.get("instructions"), 600)
    for ds in model.get("datasets") or []:
        name = str(ds.get("name") or "").lower()
        if not name or (allowed and name not in allowed):
            continue
        ext = _ekos_ext(ds)
        columns = tuple(
            Column(
                name=str(f.get("name") or "").lower(),
                datatype=str(f.get("datatype") or ""),
                description=_text(f.get("description"), 200),
                synonyms=_synonyms(f),
            )
            for f in ds.get("fields") or []
            if f.get("name")
        )
        catalog.tables[name] = Table(
            name=name,
            columns=columns,
            description=_text(ds.get("description")),
            synonyms=_synonyms(ds),
            primary_key=tuple(str(c).lower() for c in ds.get("primary_key") or []),
            ekos_id=str(ext.get("ekos_id") or ""),
            evidence=tuple(str(e) for e in (ext.get("evidence") or [])[:4]),
            status=str(ext.get("status") or ""),
        )
    for rel in model.get("relationships") or []:
        left, right = str(rel.get("from") or "").lower(), str(rel.get("to") or "").lower()
        if left in catalog.tables and right in catalog.tables:
            catalog.joins.append(Join(
                left=left,
                left_columns=tuple(str(c).lower() for c in rel.get("from_columns") or []),
                right=right,
                right_columns=tuple(str(c).lower() for c in rel.get("to_columns") or []),
                ekos_id=str(_ekos_ext(rel).get("ekos_id") or ""),
            ))
    return catalog


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if len(w) > 2}


def rank(question: str, catalog: Catalog, searched: list[str] | None = None,
         keep: int = KEEP) -> list[str]:
    """The tables a question most likely needs, best first.

    A table scores on the question's words against its name, synonyms, description and columns,
    and on EKOS's own search ranking (``searched``: names in ekos_search order). Tables joined by
    a foreign key to a chosen table follow, so a join the question needs is in view. The order is
    a pure function of the inputs.
    """
    words = _words(question)
    boost = {}
    for position, name in enumerate(searched or []):
        boost.setdefault(name.lower(), max(1, 6 - position))
    scored = []
    for name, table in catalog.tables.items():
        score = 4 * len(words & _words(name.replace("_", " ")))
        score += 3 * len(words & _words(" ".join(table.synonyms)))
        score += len(words & _words(table.description))
        for column in table.columns:
            column_words = _words(column.name.replace("_", " ") + " " + " ".join(column.synonyms))
            score += 1 if words & column_words else 0
        score += boost.get(name, 0)
        if score:
            scored.append((-score, name))
    scored.sort()
    chosen = [n for _, n in scored[:keep]]
    for join in sorted(catalog.joins, key=lambda j: (j.left, j.right, j.left_columns)):
        if len(chosen) >= keep:
            break
        for a, b in ((join.left, join.right), (join.right, join.left)):
            if a in chosen[: max(1, keep // 2)] and b not in chosen and len(chosen) < keep:
                chosen.append(b)
    return chosen


def describe(names: list[str], catalog: Catalog) -> str:
    """The schema block of the prompt: EKOS's tables, columns, meanings and joins."""
    lines = []
    for name in names:
        table = catalog.tables[name]
        head = f"TABLE {name}"
        if table.description:
            head += f" -- {table.description}"
        if table.synonyms:
            head += f" (also called: {', '.join(table.synonyms[:6])})"
        lines.append(head)
        for column in table.columns[:60]:
            line = f"  {column.name} {column.datatype}".rstrip()
            if column.name in table.primary_key:
                line += " PRIMARY KEY"
            notes = column.description
            if column.synonyms:
                notes = (notes + "; " if notes else "") + "also: " + ", ".join(column.synonyms[:4])
            if notes:
                line += f" -- {notes}"
            lines.append(line)
    shown = set(names)
    joins = [j for j in catalog.joins if j.left in shown and j.right in shown]
    if joins:
        lines.append("JOINS (foreign keys; join only on these):")
        lines.extend(f"  {j.text()}" for j in joins)
    if catalog.instructions:
        lines.append(f"NOTE: {catalog.instructions}")
    return "\n".join(lines)


def errors(report: dict) -> list[str]:
    """The blocking problems in an ``ekos_sql_check`` report."""
    found = []
    if report.get("parse_error"):
        found.append(f"parse error: {_text(report['parse_error'], 200)}")
    for item in report.get("findings") or []:
        if str(item.get("severity", "")).lower() == "error":
            found.append(f"{item.get('code', '')}: {_text(item.get('message'), 240)}")
    return found


def warnings(report: dict) -> list[dict]:
    return [
        {"code": str(i.get("code", "")), "message": _text(i.get("message"), 240)}
        for i in report.get("findings") or []
        if str(i.get("severity", "")).lower() != "error"
    ]


def grounding(catalog: Catalog, used: tuple[str, ...], report: dict | None) -> dict:
    """What the answer rests on, for the person: EKOS's tables, joins and the check's result."""
    used_set = set(used)
    return {
        "source": "EKOS",
        "tables": [
            {"name": n, "ekos_id": catalog.tables[n].ekos_id,
             "description": catalog.tables[n].description,
             "evidence": list(catalog.tables[n].evidence)}
            for n in used if n in catalog.tables
        ],
        "joins": [
            {"on": j.text(), "ekos_id": j.ekos_id}
            for j in catalog.joins if j.left in used_set and j.right in used_set
        ],
        "warnings": warnings(report or {}),
        "note": _text((report or {}).get("note"), 200),
    }


# ── calls to EKOS ────────────────────────────────────────────────────────────────────────────


def load(client: EkosClient, *, allowed: set[str] | None, database: str = "",
         status: str = "confirmed") -> Catalog:
    """EKOS's compiled schema, cached for a few minutes per endpoint."""
    key = (client.url, database, status, tuple(sorted(allowed or ())))
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        log = getattr(client, "log", None)
        if isinstance(log, list):
            log.append({"step": "read the compiled schema", "tool": "ekos_ossie_export", "ok": True,
                        "ms": 0, "summary": f"{len(hit[1].tables)} tables, {len(hit[1].joins)} "
                        f"joins — from EKOS {int(time.time() - hit[0])} s ago (cached)"})
        return hit[1]
    args: dict[str, Any] = {"status": status, "max_datasets": MAX_DATASETS}
    if database:
        args["database"] = database
    if allowed:
        # EKOS 1.4.3+: exactly these tables in one call (an older EKOS ignores `names` and the
        # per-table fallback below fills the gaps).
        args["names"] = sorted(allowed)
    first = client.call("ekos_ossie_export", args, step="read the compiled schema")
    catalog = catalog_from_ossie(first, allowed)
    if catalog.truncated and allowed:
        # The export stopped at MAX_DATASETS: fetch each allowed table it left out, then build
        # one catalog from every answer so a join to a late table is kept.
        results = [first] + [
            client.call("ekos_ossie_export",
                        {**{a: v for a, v in args.items() if a != "names"}, "match": name,
                         "max_datasets": 20},
                        step=f"read table {name}")
            for name in sorted(allowed - set(catalog.tables))
        ]
        catalog = catalog_from_ossie(_merge(results), allowed)
    if not catalog.tables:
        raise EkosUnavailable("EKOS has compiled no tables for questions yet.")
    _CACHE[key] = (time.time(), catalog)
    return catalog


def _merge(results: list[dict]) -> dict:
    """Several export answers as one: each dataset and relationship once, first answer wins."""
    datasets: dict[str, dict] = {}
    relationships: dict[tuple, dict] = {}
    for result in results:
        model = result.get("model") or {}
        for ds in model.get("datasets") or []:
            datasets.setdefault(str(ds.get("name") or "").lower(), ds)
        for rel in model.get("relationships") or []:
            key = (str(rel.get("from")).lower(), str(rel.get("to")).lower(),
                   tuple(rel.get("from_columns") or []))
            relationships.setdefault(key, rel)
    head = (results[0].get("model") or {}) if results else {}
    return {"truncated": False, "model": {
        "ai_context": head.get("ai_context"),
        "datasets": list(datasets.values()),
        "relationships": list(relationships.values()),
    }}


def search_names(client: EkosClient, question: str) -> list[str]:
    """EKOS's own ranking of the question's words (names only, best first)."""
    words = [w for w in _WORD.findall(question.lower()) if len(w) > 2][:6]
    if not words:
        return []
    result = client.call("ekos_search", {"query": " ".join(words), "limit": 30},
                         step="find the tables the question is about")
    return [str(m.get("name") or "") for m in result.get("matches") or [] if m.get("name")]


def check(client: EkosClient, sql: str, dialect: str) -> dict:
    return client.call("ekos_sql_check", {"sql": sql, "dialect": dialect},
                       step="check the SQL against the compiled schema")


def dialect_notes(client: EkosClient, dialect: str) -> list[str]:
    rules = client.call("ekos_sql_check", {"dialect": dialect}, step=f"read the {dialect} rules")
    return [str(n) for n in rules.get("dialect_notes") or []]
