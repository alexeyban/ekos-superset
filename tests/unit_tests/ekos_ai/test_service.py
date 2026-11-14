"""The question path end to end with a fake EKOS and a scripted model: the schema comes from
EKOS, every SQL is checked by EKOS before it runs, and nothing is answered without EKOS."""

import pandas as pd
import pytest
from flask import Flask

from superset.ekos_ai import ekos_mcp, knowledge, service
from .fake_ekos import FakeEkos


@pytest.fixture
def app():
    flask_app = Flask(__name__)
    flask_app.config.update(EKOS_AI_ENABLED=True, EKOS_AI_DATABASE="db", EKOS_AI_SCHEMA="public",
                            EKOS_AI_MODEL="m", EKOS_AI_BASE_URL="http://model",
                            EKOS_MCP_URL="fake://ekos", EKOS_AI_DIALECT="postgres")
    knowledge._CACHE.clear()
    with flask_app.app_context():
        yield flask_app
    knowledge._CACHE.clear()


@pytest.fixture
def wired(monkeypatch):
    """Install a fake EKOS and a scripted model; returns (ekos, prompts, ran)."""
    def install(ekos, answers):
        prompts, ran = [], []
        def client():
            from flask import g
            g.ekos_client = ekos
            if not isinstance(getattr(ekos, "log", None), list):
                ekos.log = []
            return ekos
        monkeypatch.setattr(service, "_client", client)
        monkeypatch.setattr(service.llm, "complete",
                            lambda system, user, **kw: prompts.append(user) or answers.pop(0))
        monkeypatch.setattr(service, "_database", lambda: object())
        monkeypatch.setattr(service, "_spend_question", lambda: {"yours": 4, "everyone": 49})
        monkeypatch.setattr(service, "_run", lambda db, sch, sql, rows: ran.append(sql) or pd.DataFrame({"n": [3]}))
        return prompts, ran
    return install


OK_SQL = ("select c.c_last_name, sum(s.ss_sales_price) from store_sales s "
          "join customer c on c.c_customer_sk = s.ss_customer_sk group by 1")


def test_answer_is_grounded_in_ekos_and_checked_by_it(app, wired):
    ekos = FakeEkos()
    prompts, ran = wired(ekos, [{"status": "ok", "sql": OK_SQL, "explanation": "Revenue per customer."}])
    r = service.text_to_sql("revenue per customer")
    assert r["status"] == "ok" and r["rows"] == [["3"]]
    # The model saw EKOS's schema: descriptions, synonyms and the FK join.
    assert "TABLE customer -- Customer dimension" in prompts[0]
    assert "store_sales.ss_customer_sk = customer.c_customer_sk" in prompts[0]
    assert "Limit rows with LIMIT n." in prompts[0]
    # EKOS checked the exact SQL that ran, before it ran.
    checked = [a["sql"] for t, a in ekos.calls if t == "ekos_sql_check" and "sql" in a]
    assert checked == ran
    assert r["ekos"]["source"] == "EKOS"
    assert [t["ekos_id"] for t in r["ekos"]["tables"]] == ["id-c", "id-ss"]
    assert r["ekos"]["joins"][0]["ekos_id"] == "fk-1"


def test_an_ekos_error_is_sent_back_once_for_repair(app, wired):
    bad = {"findings": [{"code": "L001", "severity": "error", "message": "unknown column ss_price; did you mean ss_sales_price"}]}
    ekos = FakeEkos(check_reports=[bad])
    prompts, ran = wired(ekos, [
        {"status": "ok", "sql": "select sum(ss_price) from store_sales"},
        {"status": "ok", "sql": "select sum(ss_sales_price) from store_sales"},
    ])
    r = service.text_to_sql("total revenue")
    assert r["status"] == "ok"
    assert "did you mean ss_sales_price" in prompts[1] and "<previous_sql>" in prompts[1]
    assert len(ran) == 1 and "ss_sales_price" in ran[0]


def test_persistent_ekos_errors_refuse_and_nothing_runs(app, wired):
    bad = {"findings": [{"code": "L002", "severity": "error", "message": "join off the foreign keys"}]}
    ekos = FakeEkos(check_reports=[bad, bad])
    _, ran = wired(ekos, [{"status": "ok", "sql": "select 1 from store_sales"}] * 2)
    r = service.text_to_sql("total revenue")
    assert r["status"] == "refused" and "L002" in r["explanation"] and ran == []


def test_a_fan_out_join_is_repaired_or_refused_never_answered(app, wired):
    fan_out = {"findings": [{"code": "L002", "severity": "warn",
                             "message": "join a.month = b.month is no foreign key and neither side is a key"}]}
    ekos = FakeEkos(check_reports=[fan_out, fan_out])
    prompts, ran = wired(ekos, [{"status": "ok", "sql": "select 1 from store_sales"}] * 2)
    r = service.text_to_sql("revenue by customer and category")
    assert r["status"] == "refused" and "L002" in r["explanation"] and ran == []
    assert "impossible" in prompts[1]


def test_a_fan_out_join_the_model_cannot_avoid_becomes_impossible(app, wired):
    fan_out = {"findings": [{"code": "L002", "severity": "warn", "message": "join off the keys"}]}
    ekos = FakeEkos(check_reports=[fan_out])
    _, ran = wired(ekos, [
        {"status": "ok", "sql": "select 1 from store_sales"},
        {"status": "impossible", "explanation": "No table has revenue by customer."},
    ])
    r = service.text_to_sql("revenue by customer and category")
    assert r["status"] == "impossible" and "by customer" in r["explanation"] and ran == []


def test_a_table_ekos_does_not_know_is_refused_then_repaired(app, wired):
    ekos = FakeEkos()
    prompts, ran = wired(ekos, [
        {"status": "ok", "sql": "select count(*) from secret_payroll"},
        {"status": "ok", "sql": "select count(*) from customer"},
    ])
    r = service.text_to_sql("how many shoppers")
    assert r["status"] == "ok" and "secret_payroll" in prompts[1] and ran == [r["sql"]]


def test_no_matching_table_asks_back_without_a_model_call(app, wired, monkeypatch):
    # A small catalog is shown whole when nothing ranks; a larger one asks back instead.
    monkeypatch.setattr(knowledge, "KEEP", 2)
    ekos = FakeEkos()
    prompts, _ = wired(ekos, [])
    r = service.text_to_sql("what is the weather tomorrow")
    assert r["status"] == "needs_clarification" and prompts == []
    assert "customer" in r["question_back"]


def test_ekos_down_answers_nothing(app, wired):
    ekos = FakeEkos(down=True)
    prompts, ran = wired(ekos, [])
    with pytest.raises(ekos_mcp.EkosUnavailable):
        service.text_to_sql("revenue per customer")
    assert prompts == [] and ran == []


def test_model_asking_back_carries_ekos_tables(app, wired):
    ekos = FakeEkos()
    wired(ekos, [{"status": "needs_clarification", "question_back": "Which period?"}])
    r = service.text_to_sql("revenue per customer")
    assert r["status"] == "needs_clarification" and r["ekos"]["tables"]


def test_ekos_writes_the_sql_when_it_offers_the_generator(app, wired):
    ekos = FakeEkos(generated={"status": "ok", "sql": OK_SQL, "open_findings": [],
                               "checked": "checked against pg, read-only"})
    prompts, ran = wired(ekos, [])  # Superset's own model is never asked
    r = service.text_to_sql("revenue per customer")
    assert r["status"] == "ok" and prompts == []
    assert ("ekos_sql_generate", {"question": "revenue per customer", "dialect": "postgres"}) in ekos.calls
    assert r["ekos"]["author"] == "ekos" and r["ekos"]["note"] == "checked against pg, read-only"
    assert r["ekos"]["remaining"] == {"yours": 4, "everyone": 49}
    assert len(ran) == 1 and "LIMIT 200" in ran[0]


def test_generated_sql_still_passes_the_read_only_guard(app, wired):
    ekos = FakeEkos(generated={"status": "ok", "sql": "delete from customer"})
    prompts, ran = wired(ekos, [{"status": "ok", "sql": OK_SQL}])
    r = service.text_to_sql("revenue per customer")
    # A write is the answer, not a reason to try again: nothing runs, no model is asked.
    assert r["status"] == "refused" and "SELECT" in r["explanation"]
    assert prompts == [] and ran == []
    assert any(e["step"].startswith("refused") for e in ekos.log)


def test_a_direct_write_is_refused_plainly(app, wired):
    app.config["EKOS_AI_TABLES"] = "customer,store_sales"
    ekos = FakeEkos(generated={"status": "open_findings", "sql": "DELETE FROM customer"},
                    generate_takes_tables=True)
    prompts, ran = wired(ekos, [{"status": "ok", "sql": OK_SQL}])
    r = service.text_to_sql("delete every customer")
    assert r["status"] == "refused" and prompts == [] and ran == []
    assert not any("outside the allowed" in e["step"] for e in ekos.log)


def test_an_ekos_refusal_is_the_answer(app, wired):
    app.config["EKOS_AI_TABLES"] = "customer,store_sales"
    ekos = FakeEkos(generated={"status": "refused", "sql": None,
                               "reason": "EKOS writes read-only SELECT queries only"},
                    generate_takes_tables=True)
    prompts, ran = wired(ekos, [])
    r = service.text_to_sql("delete every customer")
    assert r["status"] == "refused" and "read-only" in r["explanation"]
    assert prompts == [] and ran == [] and "ekos_ossie_export" not in ekos.called()


def test_a_refused_ekos_query_falls_back_and_a_failed_draft_says_why(app, wired):
    ekos = FakeEkos(generated={"status": "ok", "sql": "select * from ad_user"})
    _, ran = wired(ekos, [{"status": "impossible", "sql": None, "explanation": "no such data"}])
    r = service.text_to_sql("who buys most")
    assert r["status"] != "ok" and ran == []


def test_the_allowed_tables_go_to_ekos_when_its_generator_takes_them(app, wired):
    app.config["EKOS_AI_TABLES"] = "customer,store_sales"
    ekos = FakeEkos(generated={"status": "ok", "sql": OK_SQL, "open_findings": []},
                    generate_takes_tables=True)
    wired(ekos, [])
    service.text_to_sql("revenue per customer")
    sent = [a for t, a in ekos.calls if t == "ekos_sql_generate"][0]
    assert sent["tables"] == ["customer", "store_sales"]

    older = FakeEkos(generated={"status": "ok", "sql": OK_SQL, "open_findings": []})
    wired(older, [])
    service.text_to_sql("revenue per customer")
    assert "tables" not in [a for t, a in older.calls if t == "ekos_sql_generate"][0]


def test_ekos_limit_and_no_sql_are_plain_answers(app, wired):
    _, ran = wired(FakeEkos(generated={"status": "limit", "sql": None, "reason": "the daily limit of 50"}), [])
    assert "daily limit of 50" in service.text_to_sql("revenue per customer")["explanation"]
    _, ran = wired(FakeEkos(generated={"status": "no_sql", "sql": None}), [])
    assert service.text_to_sql("revenue per customer")["status"] == "impossible" and ran == []


def test_the_quota_is_spent_only_after_ekos_answered(app, wired, monkeypatch):
    spent = []
    ekos = FakeEkos(down=True)
    wired(ekos, [])
    monkeypatch.setattr(service, "_spend_question", lambda: spent.append(1) or {})
    with pytest.raises(ekos_mcp.EkosUnavailable):
        service.text_to_sql("revenue per customer")
    assert spent == []


def test_the_answer_carries_the_log_of_ekos_calls(app, wired):
    from flask import g

    ekos = FakeEkos()
    ekos.log = []
    wired(ekos, [{"status": "ok", "sql": OK_SQL}])
    service.text_to_sql("revenue per customer")
    assert g.ekos_client is ekos
    steps = [e["step"] for e in ekos.log]
    assert any("model drafts" in s for s in steps)  # Superset's model, marked as not EKOS
    assert any(s.startswith("run the checked SQL") for s in steps)


def test_a_source_overrides_database_schema_and_dialect(app, wired):
    app.config["EKOS_AI_SOURCES"] = [
        {"id": "postgres", "label": "PostgreSQL", "database": "pg", "schema": "adempiere", "dialect": "postgres"},
        {"id": "iceberg", "label": "Iceberg", "database": "trino", "schema": "bronze_adempiere",
         "dialect": "trino", "requalify": True},
    ]
    ekos = FakeEkos()
    _, ran = wired(ekos, [{"status": "ok", "sql": OK_SQL}])
    service.select_source("iceberg")
    assert service.cfg("EKOS_AI_DATABASE") == "trino" and service.cfg("EKOS_AI_DIALECT") == "trino"
    r = service.text_to_sql("revenue per customer")
    assert r["status"] == "ok" and "bronze_adempiere.customer" in ran[0]
    assert ("ekos_sql_check", "trino") in [(t, a.get("dialect")) for t, a in ekos.calls if "sql" in a]
    with pytest.raises(service.guard.Refused):
        service.select_source("nope")


def test_the_model_and_the_database_steps_are_announced_before_they_end(app, wired):
    ekos = FakeEkos()
    ekos.log = ekos_mcp.StepLog()
    heard = []
    ekos.log.listener = heard.append
    wired(ekos, [{"status": "ok", "sql": OK_SQL}])
    service.text_to_sql("revenue per customer")
    for label in ("model drafts", "run the checked SQL"):
        kinds = [e["type"] for e in heard if label in e["entry"].get("step", "")]
        assert kinds[:2] == ["start", "step"], (label, kinds)


def test_a_direct_question_skips_search_and_reads_only_the_tables_used(app, wired):
    app.config["EKOS_AI_TABLES"] = "customer,store_sales,item"
    ekos = FakeEkos(generated={"status": "ok", "sql": OK_SQL, "open_findings": [],
                               "checked": "checked against pg, read-only"},
                    generate_takes_tables=True)
    prompts, ran = wired(ekos, [])
    r = service.text_to_sql("revenue per customer")
    assert r["status"] == "ok" and prompts == [] and len(ran) == 1
    tools = ekos.called()
    assert "ekos_search" not in tools, "no search when EKOS writes the SQL"
    assert tools.index("ekos_sql_generate") < tools.index("ekos_ossie_export"), "SQL first"
    sent = [a for t, a in ekos.calls if t == "ekos_sql_generate"][0]
    assert sent["tables"] == ["customer", "item", "store_sales"]
    export = [a for t, a in ekos.calls if t == "ekos_ossie_export"][0]
    assert export["names"] == ["customer", "store_sales"], "only the tables the SQL uses"
    assert r["ekos"]["author"] == "ekos"
    assert {t["name"] for t in r["ekos"]["tables"]} == {"customer", "store_sales"}


def test_a_direct_query_outside_the_allowed_tables_falls_back_and_never_runs(app, wired):
    app.config["EKOS_AI_TABLES"] = "customer,store_sales"
    ekos = FakeEkos(generated={"status": "ok", "sql": "select * from item"},
                    generate_takes_tables=True)
    prompts, ran = wired(ekos, [{"status": "ok", "sql": OK_SQL}])
    r = service.text_to_sql("revenue per customer")
    assert r["status"] == "ok" and len(prompts) == 1
    assert len(ran) == 1 and "item" not in ran[0].lower()
    assert any("outside the allowed" in e["step"] for e in ekos.log)


def test_direct_limit_and_no_sql_are_plain_answers(app, wired):
    app.config["EKOS_AI_TABLES"] = "customer,store_sales"
    _, ran = wired(FakeEkos(generated={"status": "limit", "sql": None, "reason": "the daily limit of 50"},
                            generate_takes_tables=True), [])
    assert "daily limit of 50" in service.text_to_sql("revenue per customer")["explanation"]
    _, ran = wired(FakeEkos(generated={"status": "no_sql", "sql": None}, generate_takes_tables=True), [])
    r = service.text_to_sql("revenue per customer")
    assert r["status"] == "impossible" and ran == []


def test_an_unlimited_user_spends_no_question(app, monkeypatch):
    from types import SimpleNamespace
    from flask import g

    app.config["EKOS_AI_UNLIMITED_USERS"] = "google:owner@example.com"
    spent = []
    monkeypatch.setattr(service.quota, "spend", lambda *a, **k: spent.append(1) or {})
    g.user = SimpleNamespace(id=1, username="google:owner@example.com")
    assert service._spend_question() == {"yours": "no limit", "everyone": "not counted"}
    assert spent == []


def test_a_dataset_superset_read_no_columns_for_gets_the_checked_querys(app, monkeypatch):
    """A query with no rows yet (this year over older data): Superset leaves the virtual dataset
    without columns and every chart on it fails with "Columns missing in dataset"."""

    class Col:
        def __init__(self, column_name, is_dttm=False):
            self.column_name, self.is_dttm = column_name, is_dttm

    class Dataset:
        def __init__(self, cols):
            self.columns, self.main_dttm_col = list(cols), None

    committed = []
    # Stand-ins for Superset's model and extension modules: importing the real ones needs a
    # running Superset app (Flask-Babel), which this unit test does not start.
    import sys
    import types

    models = types.ModuleType("superset.connectors.sqla.models")
    models.TableColumn = Col
    extensions = types.ModuleType("superset.extensions")
    extensions.db = types.SimpleNamespace(
        session=types.SimpleNamespace(commit=lambda: committed.append(True))
    )
    monkeypatch.setitem(sys.modules, "superset.connectors.sqla.models", models)
    monkeypatch.setitem(sys.modules, "superset.extensions", extensions)
    empty = Dataset([])
    assert service._complete_columns(empty, ["month", "avg_cheque"], {"month"}) == 2
    assert [(c.column_name, c.is_dttm) for c in empty.columns] == [("month", True), ("avg_cheque", False)]
    assert empty.main_dttm_col == "month" and committed == [True]
    full = Dataset([Col("month"), Col("avg_cheque")])
    assert service._complete_columns(full, ["month", "avg_cheque"], {"month"}) == 0
    assert committed == [True], "nothing to add, nothing written"


def test_the_model_is_told_the_date_and_to_compute_relative_periods(app, wired, monkeypatch):
    """"This year" is today's year: the prompt carries the date, and the rules say a relative
    period is CURRENT_DATE arithmetic, never a year the model writes (llama3 wrote 2023)."""
    monkeypatch.setattr(service.quota, "today", lambda: "2026-11-14")
    ekos = FakeEkos()
    prompts, _ = wired(ekos, [{"status": "ok", "sql": OK_SQL, "explanation": "Revenue per customer."}])
    service.text_to_sql("revenue per customer this year")
    assert "<today>2026-11-14</today>" in prompts[0]
    assert "CURRENT_DATE" in service.SQL_SYSTEM and "CURRENT_DATE" in service.DASH_SYSTEM
