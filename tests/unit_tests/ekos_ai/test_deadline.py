"""EKOS RFC 0259 S4: a DuckDB query is interrupted at its deadline (DuckDB has no statement
timeout). Runs a real in-process DuckDB through duckdb-engine, as the image does."""

import time

import pytest

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("duckdb_engine")

from sqlalchemy import create_engine, text  # noqa: E402

from superset.ekos_ai.deadline import DeadlinePassed, query_deadline  # noqa: E402

LONG = "select count(*) from range(100000000000) a"


@pytest.fixture
def engine(tmp_path):
    path = tmp_path / "marts.duckdb"
    con = duckdb.connect(str(path))
    con.execute("create table sales as select range as id from range(10)")
    con.close()
    return create_engine(f"duckdb:///{path}", connect_args={
        "read_only": True,
        "config": {"enable_external_access": False, "lock_configuration": True},
    })


def test_a_long_query_is_stopped_at_the_deadline(engine):
    started = time.monotonic()
    with engine.connect() as conn, pytest.raises(DeadlinePassed, match="longer than 0.5 s"):
        with query_deadline(conn, 0.5):
            conn.execute(text(LONG)).fetchall()
    assert time.monotonic() - started < 10


def test_a_quick_query_is_not_touched_and_the_connection_stays_usable(engine):
    with engine.connect() as conn:
        with query_deadline(conn, 5):
            assert conn.execute(text("select count(*) from sales")).scalar() == 10
        time.sleep(0.1)
        assert conn.execute(text("select max(id) from sales")).scalar() == 9


def test_after_an_interrupt_the_next_query_runs(engine):
    with engine.connect() as conn:
        with pytest.raises(DeadlinePassed), query_deadline(conn, 0.3):
            conn.execute(text(LONG)).fetchall()
        with query_deadline(conn, 5):
            assert conn.execute(text("select count(*) from sales")).scalar() == 10


def test_a_failing_query_keeps_its_own_error(engine):
    with engine.connect() as conn, pytest.raises(Exception) as err:
        with query_deadline(conn, 5):
            conn.execute(text("select * from no_such_table")).fetchall()
    assert not isinstance(err.value, DeadlinePassed)


def test_postgres_gets_a_statement_timeout():
    sent = []

    class Conn:
        class dialect:
            name = "postgresql"

        def execute(self, stmt):
            sent.append(str(stmt))

    with query_deadline(Conn(), 15):
        pass
    assert sent == ["SET statement_timeout = 15000"]
