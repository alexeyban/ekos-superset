"""EKOS RFC 0248 §2.4: what "Ask your data" publishes is also written as a bundle — the YAML file
EKOS Lakehouse keeps per dashboard in the pipeline repository (``dashboards/<key>.yaml``): per
chart the structured request, the resolved terms with the rule that applied, the digest of the
mart it reads, the SQL and the reconciliation with silver. ``lakehouse dashboards sync`` commits
it as the dashboard's first version and reads it instead of parsing the SQL; ``lakehouse
dashboards change`` changes the dashboard from it (RFC 0258).

Pure apart from ``write``: no Superset, no model, no EKOS call.
"""

from __future__ import annotations

import os
import re
import tempfile
from decimal import Decimal, InvalidOperation
from typing import Any

import yaml

FORMAT = 1
_PRESENTATION = ("title", "viz", "x", "group_by", "metric", "order", "number_format", "color_scheme")


def totals_equal(a: Any, b: Any) -> bool:
    """Two totals read as text are equal as decimals (``12.50`` = ``12.5``; both empty = equal)."""
    sa, sb = ("" if a is None else str(a).strip()), ("" if b is None else str(b).strip())
    if not sa or not sb:
        return sa == sb
    try:
        return Decimal(sa) == Decimal(sb)
    except InvalidOperation:
        return sa == sb


def key_for(dashboard_id: int) -> str:
    return f"dash-{int(dashboard_id)}"


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if v not in (None, "", [])}


def presentation(spec: dict, order: int) -> dict:
    """The chart's presentation as the bundle holds it (only the known fields)."""
    p = {k: spec.get(k) for k in _PRESENTATION if k != "order"}
    p["title"] = str(spec.get("title") or spec.get("name") or f"Chart {order + 1}")
    out = _clean(p)
    out["order"] = order
    return out


def chart_entry(number: int, compiled: dict, presentation_: dict, placed: dict,
                reconciled: dict) -> dict:
    """One chart: what EKOS compiled (request, terms, mart, digest, SQL), what was published
    (Superset ids, form data) and its reconciliation."""
    return {
        "key": f"c{number}",
        "presentation": presentation_,
        "request": _clean(dict(compiled["request"])) | {"measures": compiled["request"]["measures"]},
        "mart": compiled["mart"],
        "marts": [{"mart": compiled["mart"], "digest": compiled["digest"]}],
        "terms": compiled["terms"],
        "sql": compiled["sql"],
        "reconciled": reconciled,
        "superset": {"chart_id": placed["chart_id"], "dataset_id": placed["dataset_id"],
                     "dataset": placed["dataset"]},
        "params": placed["params"],
    }


def bundle(dashboard_id: int, title: str, author: str, question: str, charts: list[dict]) -> dict:
    return {
        "format": FORMAT,
        "dashboard": key_for(dashboard_id),
        "title": title,
        "author": author,
        "request_text": question,
        "superset": {"dashboard_id": int(dashboard_id), "url": f"/superset/dashboard/{dashboard_id}/"},
        "charts": charts,
    }


_KEY = re.compile(r"^[a-z0-9_-]{1,64}$")


def write(directory: str, value: dict) -> str:
    """Write the bundle as ``<directory>/<key>.yaml`` (atomically); returns the path. The key
    comes from the Superset id, never from text a person wrote."""
    key = value["dashboard"]
    if not _KEY.match(key):
        raise ValueError(f"{key!r} is not a dashboard key")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{key}.yaml")
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".bundle-", suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(value, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp, path)
    return path
