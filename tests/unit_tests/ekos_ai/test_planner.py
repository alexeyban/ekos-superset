import json

import pytest

from superset.ekos_ai.planner import BadSpec, clean_name, form_data, layout

COLS = ["partner", "total", "day"]


def test_bar_chart():
    viz, p = form_data({"viz": "bar", "x": "partner", "metric": "total", "agg": "sum"}, COLS, 7)
    assert viz == "echarts_timeseries_bar" and p["x_axis"] == "partner" and p["datasource"] == "7__table"
    assert p["metrics"][0]["aggregate"] == "SUM"


def test_columns_match_case_insensitively():
    assert form_data({"viz": "pie", "x": "PARTNER", "metric": "Total"}, COLS, 1)[1]["groupby"] == ["partner"]


def test_count_needs_no_metric_column():
    assert form_data({"viz": "big_number", "agg": "count"}, COLS, 1)[1]["metric"]["sqlExpression"] == "COUNT(*)"


@pytest.mark.parametrize("spec", [
    {"viz": "scatter3d", "x": "partner"},
    {"viz": "bar", "x": "nope", "metric": "total"},
    {"viz": "bar", "x": "partner", "metric": "total", "agg": "median"},
    {"viz": "bar", "metric": "total"},
])
def test_bad_specs(spec):
    with pytest.raises(BadSpec):
        form_data(spec, COLS, 1)


def test_layout_two_per_row():
    doc = json.loads(layout("T", [(1, "a"), (2, "b"), (3, "c")]))
    assert doc["GRID_ID"]["children"] == ["ROW-0", "ROW-1"]
    assert doc["CHART-2"]["meta"]["chartId"] == 3


def test_names_are_cleaned():
    assert clean_name("<script>x</script> Sales", "d") == "scriptx/script Sales"
    assert clean_name("", "fallback") == "fallback"


def test_count_over_an_aggregated_query_sums_its_value_column():
    # 2026-11-09: "sales orders per month" drew 1 for every month — COUNT(*) over one row per month.
    spec = {"viz": "line", "x": "month", "agg": "count",
            "sql": "SELECT date_trunc('month', dateordered) AS month, COUNT(*) AS sales_orders "
                   "FROM c_order GROUP BY 1"}
    _, params = form_data(spec, ["month", "sales_orders"], 3)
    assert params["metrics"][0]["aggregate"] == "SUM"
    assert params["metrics"][0]["column"]["column_name"] == "sales_orders"


def test_count_over_raw_rows_stays_a_count():
    spec = {"viz": "bar", "x": "status", "agg": "count", "sql": "SELECT docstatus AS status FROM c_order"}
    _, params = form_data(spec, ["status"], 3)
    assert params["metrics"][0]["sqlExpression"] == "COUNT(*)"


def test_an_aggregated_query_with_several_values_needs_a_metric():
    spec = {"viz": "bar", "x": "m", "agg": "count",
            "sql": "SELECT m, COUNT(*) AS n, SUM(x) AS total FROM t GROUP BY m"}
    with pytest.raises(BadSpec):
        form_data(spec, ["m", "n", "total"], 3)
