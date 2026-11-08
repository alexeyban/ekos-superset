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
        monkeypatch.setattr(service, "_client", lambda: ekos)
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
    _, ran = wired(ekos, [])
    r = service.text_to_sql("revenue per customer")
    assert r["status"] == "refused" and ran == []


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
