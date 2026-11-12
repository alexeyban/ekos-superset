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


def test_date_projections_are_temporal_even_without_rows():
    from superset.ekos_ai.planner import temporal_columns

    sql = ("SELECT CAST(DATE_TRUNC('MONTH', datetrx) AS DATE) AS month, AVG(payamt) AS avg_cheque "
           "FROM adempiere.c_payment GROUP BY 1")
    assert temporal_columns(sql, ["month", "avg_cheque"], {"month": "object", "avg_cheque": "object"}) == {"month"}
    assert temporal_columns("SELECT d FROM t", ["d"], {"d": "datetime64[ns]"}) == {"d"}
    assert temporal_columns("not sql at all (", ["x"], {}) == set()


def test_a_query_about_the_current_period_gets_a_latest_value_probe():
    from superset.ekos_ai.planner import latest_value_probe

    sql = ("SELECT AVG(payamt) AS avg_cheque FROM adempiere.c_payment WHERE tendertype = 'K' "
           "AND datetrx >= DATE_TRUNC('YEAR', CURRENT_DATE) AND datetrx < DATE_TRUNC('YEAR', CURRENT_DATE) + INTERVAL '1 YEAR'")
    probe, column = latest_value_probe(sql)
    assert probe == "SELECT MAX(datetrx) FROM adempiere.c_payment" and column == "c_payment.datetrx"
    assert latest_value_probe("SELECT * FROM c_payment WHERE payamt > 10") is None
    assert latest_value_probe("SELECT * FROM a JOIN b ON a.id = b.id WHERE a.d >= CURRENT_DATE") is None
