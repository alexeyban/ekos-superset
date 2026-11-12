"""RFC 0248 §2.4 in the fork: a dashboard from terms (EKOS compiles the SQL), reconciled with
silver before anything is created, and written as the bundle EKOS Lakehouse reads."""

import pandas as pd
import pytest
import yaml
from flask import Flask

from superset.ekos_ai import bundle, service
from .fake_ekos import FakeEkos

VOCAB = {"marts": [{"mart": "sales_by_month",
                    "dimensions": [{"name": "month", "definition": "month of c_order.dateordered"}],
                    "measures": [{"name": "total", "definition": "sum of c_order.grandtotal"}],
                    "filters": ["c_order.issotrx is true"]}]}


def compiled(status="ok"):
    if status != "ok":
        return {"charts": [{"chart": 1, "status": "needs_clarification",
                            "questions": [{"text": "`c_order.weather` is not a column", "candidates": []}]}]}
    return {"charts": [{
        "chart": 1, "status": "ok", "mart": "sales_by_month", "digest": "e893c774d03c",
        "sql": 'select "month",\n       sum("total") as "revenue"\nfrom "iceberg"."gold_sales"."sales_by_month"\ngroup by 1\norder by 1',
        "request": {"measures": [{"name": "revenue", "definition": "sum of c_order.grandtotal"}],
                    "dimensions": [{"name": "month", "definition": "month of c_order.dateordered"}],
                    "filters": ["c_order.issotrx is true"]},
        "presentation": {"title": "Revenue by month", "viz": "bar", "x": "month", "metric": "revenue", "order": 0},
        "terms": [{"section": "measures", "name": "revenue", "written": "sum of c_order.grandtotal",
                   "rule": "sum over `C_ORDER.GRANDTOTAL`", "served_by": "sum of gold.sales_by_month.total"}],
        "reconcile": [{"measure": "revenue", "chart_sql": "select cast(sum(revenue) as varchar) from gold_sales.x",
                       "silver_sql": "select cast(sum(total) as varchar) from silver.c_order"}],
    }]}


class TermsEkos(FakeEkos):
    def __init__(self, answer):
        super().__init__()
        self.answer = answer

    def call(self, tool, arguments=None, *, step=""):
        if tool == service.COMPILE_TOOL:
            self.calls.append((tool, arguments or {}))
            return VOCAB if not (arguments or {}).get("charts") else self.answer
        return super().call(tool, arguments, step=step)


SPEC = {"status": "ok", "title": "Sales", "charts": [{
    "title": "Revenue by month", "viz": "bar", "x": "month", "metric": "revenue",
    "measures": [{"name": "revenue", "definition": "sum of c_order.grandtotal"}],
    "dimensions": [{"name": "month", "definition": "month of c_order.dateordered"}],
    "sql": "select 1  -- the model's SQL is never passed on"}]}


@pytest.fixture
def app(tmp_path):
    flask_app = Flask(__name__)
    flask_app.config.update(EKOS_AI_ENABLED=True, EKOS_AI_DATABASE="db", EKOS_AI_SCHEMA="gold_sales",
                            EKOS_AI_MODEL="m", EKOS_AI_BASE_URL="http://model", EKOS_MCP_URL="fake://ekos",
                            EKOS_AI_DIALECT="trino", EKOS_AI_DASHBOARD_MODE="terms",
                            EKOS_AI_BUNDLE_DIR=str(tmp_path / "dashboards"))
    with flask_app.app_context():
        yield flask_app


def wire(monkeypatch, ekos, totals=("100.50", "100.5")):
    ran, built = [], []
    monkeypatch.setattr(service, "_client", lambda: ekos)
    monkeypatch.setattr(service.llm, "complete", lambda system, user, **kw: dict(SPEC))
    monkeypatch.setattr(service, "_database", lambda: object())
    monkeypatch.setattr(service, "_spend_question", lambda: {"yours": 4})

    def run(db, sch, sql, rows):
        ran.append(sql)
        if "silver" in sql:
            return pd.DataFrame({"v": [totals[1]]})
        if sql.startswith("select cast"):
            return pd.DataFrame({"v": [totals[0]]})
        return pd.DataFrame({"month": ["2025-01-01"], "revenue": [1.0]})
    monkeypatch.setattr(service, "_run", run)

    def build(database, sch, proposal, prepared):
        built.append(prepared)
        return {"status": "ok", "dashboard_id": 12, "url": "/superset/dashboard/12/", "title": "Sales",
                "charts": ["Revenue by month"],
                "built": [{"chart_id": 30, "dataset_id": 40, "dataset": "ai_abcdef_1",
                           "params": {"viz_type": "echarts_timeseries_bar"}, "name": "Revenue by month"}]}
    monkeypatch.setattr(service, "_build", build)
    return ran, built


def test_terms_become_a_dashboard_and_a_bundle(app, monkeypatch, tmp_path):
    ekos = TermsEkos(compiled())
    ran, built = wire(monkeypatch, ekos)
    r = service.text_to_dashboard("revenue by month")
    assert r["status"] == "ok" and r["bundle"] == {"key": "dash-12", "written": True}
    # The request EKOS compiled holds terms only: the model's SQL went nowhere.
    sent = [a for t, a in ekos.calls if t == service.COMPILE_TOOL and a.get("charts")][0]
    assert "sql" not in str(sent["charts"][0]["request"])
    assert built[0][0][1].sql.startswith('select "month"'), "EKOS's SQL is the dataset's"
    assert all("select 1" not in s for s in ran)
    b = yaml.safe_load((tmp_path / "dashboards" / "dash-12.yaml").read_text())
    assert b["format"] == 1 and b["superset"]["dashboard_id"] == 12
    c = b["charts"][0]
    assert c["marts"] == [{"mart": "sales_by_month", "digest": "e893c774d03c"}]
    assert c["request"]["filters"] == ["c_order.issotrx is true"]
    assert c["terms"][0]["served_by"] == "sum of gold.sales_by_month.total"
    assert c["reconciled"]["ok"] is True and c["superset"]["chart_id"] == 30


def test_a_chart_that_does_not_reconcile_publishes_nothing(app, monkeypatch, tmp_path):
    ran, built = wire(monkeypatch, TermsEkos(compiled()), totals=("100.50", "99"))
    r = service.text_to_dashboard("revenue by month")
    assert r["status"] == "refused" and "does not reconcile" in r["explanation"]
    assert not built and not (tmp_path / "dashboards").exists()


def test_a_term_that_does_not_resolve_is_a_question(app, monkeypatch, tmp_path):
    ran, built = wire(monkeypatch, TermsEkos(compiled("needs_clarification")))
    r = service.text_to_dashboard("revenue by weather")
    assert r["status"] == "needs_clarification" and "weather" in r["question_back"]
    assert not built and not ran


def test_totals_and_keys():
    assert bundle.totals_equal("12.50", "12.5") and not bundle.totals_equal("1", "2")
    assert bundle.totals_equal("", "") and not bundle.totals_equal("", "0")
    with pytest.raises(ValueError):
        bundle.write("/tmp", {"dashboard": "../x"})


def test_a_term_named_without_its_definition_takes_the_vocabularys_only():
    marts = VOCAB["marts"] + [{"mart": "other", "dimensions": [{"name": "month", "definition": "month of x.d"}],
                               "measures": [{"name": "total", "definition": "sum of c_order.grandtotal"}]}]
    r = service._terms_request({"measures": ["total"], "dimensions": [{"name": "month"}]}, marts)
    assert r["measures"] == [{"name": "total", "definition": "sum of c_order.grandtotal"}]
    assert r["dimensions"] == [{"name": "month", "definition": ""}], "two definitions: none is chosen"


def test_only_the_drawing_is_completed():
    p = service._terms_presentation({"title": "Orders", "viz": "big_number", "metric": "orders",
                                     "dimensions": [{"name": "customer"}]}, 2)
    assert p == {"title": "Orders", "viz": "bar", "x": "customer", "metric": "orders", "order": 1}
    q = service._terms_presentation({"title": "T", "viz": "line", "x": "month", "metric": "m",
                                     "dimensions": [{"name": "month"}, {"name": "customer"}]}, 1)
    assert q["group_by"] == "customer"
