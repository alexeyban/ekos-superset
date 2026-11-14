"""Turn a model's chart specs into Superset form data and a dashboard layout (pure)."""

from __future__ import annotations

import json
import re
from typing import Any

import sqlglot
from sqlglot import exp

VIZ = {
    "bar": "echarts_timeseries_bar",
    "line": "echarts_timeseries_line",
    "pie": "pie",
    "big_number": "big_number_total",
    "table": "table",
}
AGG = {"sum": "SUM", "avg": "AVG", "min": "MIN", "max": "MAX", "count": "COUNT"}
MAX_CHARTS = 6
_NAME = re.compile(r"[^\w .,:%&()/+-]")


class BadSpec(ValueError):
    """A chart spec the model proposed cannot be built; the message says what is wrong."""


def clean_name(text: Any, default: str, limit: int = 80) -> str:
    value = _NAME.sub("", str(text or "")).strip()[:limit]
    return value or default


def _column(spec: dict, key: str, columns: list[str], *, required: bool) -> str | None:
    value = spec.get(key)
    if value in (None, ""):
        if required:
            raise BadSpec(f"The chart needs '{key}'.")
        return None
    by_lower = {c.lower(): c for c in columns}
    found = by_lower.get(str(value).lower())
    if found is None:
        raise BadSpec(f"'{value}' is not a column of the query (columns: {', '.join(columns)}).")
    return found


def _aggregates(sql: str, dialect: str = "postgres") -> bool:
    """Whether the query groups at its top level (its rows are already totals)."""
    try:
        root = sqlglot.parse_one(sql or "", read=dialect)
    except sqlglot.errors.ParseError:
        return False
    return isinstance(root, exp.Select) and root.args.get("group") is not None


_TEMPORAL = {
    exp.DataType.Type.DATE,
    exp.DataType.Type.DATETIME,
    exp.DataType.Type.TIMESTAMP,
    exp.DataType.Type.TIMESTAMPTZ,
    exp.DataType.Type.TIMESTAMPLTZ,
}


def _select(sql: str, dialect: str = "postgres"):
    try:
        root = sqlglot.parse_one(sql or "", read=dialect)
    except sqlglot.errors.ParseError:
        return None
    return root if isinstance(root, exp.Select) else None


def temporal_columns(sql: str, columns: list[str], dtypes: dict[str, str],
                     dialect: str = "postgres") -> set[str]:
    """The output columns that hold dates: a datetime type in the result, or a projection that
    casts to a date/timestamp or truncates one. A dataset Superset could not read columns for (a
    query with no rows yet) needs this to draw a time axis."""
    out = {c for c in columns if "datetime" in str(dtypes.get(c, "")).lower()}
    root = _select(sql, dialect)
    if root is None:
        return out
    by_lower = {c.lower(): c for c in columns}
    for projection in root.expressions:
        node = projection.this if isinstance(projection, exp.Alias) else projection
        temporal = (isinstance(node, exp.Cast) and node.to.this in _TEMPORAL) or isinstance(
            node, (exp.DateTrunc, exp.TimestampTrunc, exp.Date)
        )
        name = by_lower.get(projection.alias_or_name.lower())
        if temporal and name:
            out.add(name)
    return out


_NOW = (exp.CurrentDate, exp.CurrentTimestamp, exp.CurrentDatetime)


def latest_value_probe(sql: str, dialect: str = "postgres") -> tuple[str, str] | None:
    """For a query that compares a column with the current date (`>= DATE_TRUNC('YEAR',
    CURRENT_DATE)`), the query that tells how far the data goes: `SELECT MAX(col) FROM table`
    over its one table, and the column's name. `None` when there is no such comparison or more
    than one table. Built from the parsed identifiers, never from text."""
    root = _select(sql, dialect)
    if root is None:
        return None
    tables = list(root.find_all(exp.Table))
    where = root.args.get("where")
    if len(tables) != 1 or where is None:
        return None
    for cmp in where.find_all(exp.GT, exp.GTE, exp.EQ):
        column, other = cmp.this, cmp.expression
        if isinstance(column, exp.Column) and any(other.find_all(*_NOW)):
            table = tables[0]
            probe = exp.select(exp.Max(this=exp.column(column.name))).from_(
                exp.table_(table.name, db=table.db or None)
            )
            return probe.sql(dialect=dialect), f"{table.name}.{column.name}"
    return None


def form_data(spec: dict, columns: list[str], dataset_id: int,
              dialect: str = "postgres") -> tuple[str, dict]:
    """The viz type and params of one chart; raises BadSpec for anything not in the vocabulary."""
    viz = VIZ.get(str(spec.get("viz", "")).lower())
    if viz is None:
        raise BadSpec(f"Unknown chart type '{spec.get('viz')}' (use {', '.join(VIZ)}).")
    kind = str(spec.get("viz")).lower()
    agg = AGG.get(str(spec.get("agg", "sum")).lower())
    if agg is None:
        raise BadSpec(f"Unknown aggregate '{spec.get('agg')}'.")
    base: dict[str, Any] = {"datasource": f"{dataset_id}__table", "viz_type": viz}
    if kind == "table":
        return viz, {**base, "query_mode": "raw", "all_columns": columns[:20], "row_limit": 200}
    metric_col = _column(spec, "metric", columns, required=agg != "COUNT")
    if metric_col is None and agg == "COUNT" and _aggregates(str(spec.get("sql", "")), dialect):
        # The SQL already counted (one row per group): counting its rows would draw 1 for every
        # group. Its one value column is the measure.
        keys = {str(spec.get(k, "")).lower() for k in ("x", "group_by")}
        values = [c for c in columns if c.lower() not in keys]
        if len(values) != 1:
            raise BadSpec("The SQL already aggregates: name the column to chart as 'metric'.")
        metric_col, agg = values[0], "SUM"
    if metric_col is None:
        metric: dict[str, Any] = {
            "expressionType": "SQL", "sqlExpression": "COUNT(*)", "label": "Count",
        }
    else:
        metric = {
            "expressionType": "SIMPLE", "column": {"column_name": metric_col},
            "aggregate": agg, "label": f"{agg.title()} of {metric_col}"[:60],
        }
    if kind == "big_number":
        return viz, {**base, "metric": metric}
    group = _column(spec, "group_by", columns, required=False)
    if kind == "pie":
        dim = _column(spec, "x", columns, required=True)
        return viz, {**base, "groupby": [dim], "metric": metric, "row_limit": 50}
    x = _column(spec, "x", columns, required=True)
    return viz, {
        **base, "x_axis": x, "metrics": [metric], "groupby": [group] if group else [],
        "row_limit": 1000, "stack": bool(group), "x_axis_sort_asc": True,
    }


def layout(title: str, charts: list[tuple[int, str]]) -> str:
    """position_json for a dashboard: two charts to a row."""
    doc: dict[str, Any] = {
        "DASHBOARD_VERSION_KEY": "v2",
        "ROOT_ID": {"type": "ROOT", "id": "ROOT_ID", "children": ["GRID_ID"]},
        "GRID_ID": {"type": "GRID", "id": "GRID_ID", "children": [], "parents": ["ROOT_ID"]},
        "HEADER_ID": {"id": "HEADER_ID", "type": "HEADER", "meta": {"text": title}},
    }
    for start in range(0, len(charts), 2):
        row = f"ROW-{start // 2}"
        doc["GRID_ID"]["children"].append(row)
        doc[row] = {
            "type": "ROW", "id": row, "children": [],
            "parents": ["ROOT_ID", "GRID_ID"], "meta": {"background": "BACKGROUND_TRANSPARENT"},
        }
        for offset, (chart_id, name) in enumerate(charts[start : start + 2]):
            cid = f"CHART-{start + offset}"
            doc[row]["children"].append(cid)
            doc[cid] = {
                "type": "CHART", "id": cid, "children": [],
                "parents": ["ROOT_ID", "GRID_ID", row],
                "meta": {"width": 6, "height": 50, "chartId": chart_id, "sliceName": name},
            }
    return json.dumps(doc)
