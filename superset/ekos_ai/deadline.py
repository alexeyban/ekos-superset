"""A deadline for one query, whatever the engine (kept apart from ``service`` so it is tested alone).

PostgreSQL stops a statement itself (``statement_timeout``). DuckDB has no statement timeout
(EKOS RFC 0259 S4): a timer thread calls the connection's ``interrupt()`` once the time is up,
and the query fails with DuckDB's interrupt error, which is turned into ``DeadlinePassed``."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Iterator

DEFAULT_SECONDS = 15.0


class DeadlinePassed(RuntimeError):
    """The query ran past its deadline and was stopped."""


def _raw(conn):
    """The DB-API connection under a SQLAlchemy 1.4 ``Connection``."""
    fairy = conn.connection
    return getattr(fairy, "dbapi_connection", None) or fairy.connection


@contextmanager
def query_deadline(conn, seconds: float = DEFAULT_SECONDS) -> Iterator[None]:
    """Stop what runs on ``conn`` inside this block after ``seconds``.

    PostgreSQL: ``SET statement_timeout``. DuckDB: ``interrupt()`` from a timer thread; an
    interrupted query raises ``DeadlinePassed``. Other engines: no deadline is set here."""
    from sqlalchemy import text

    seconds = max(float(seconds), 0.001)
    name = conn.dialect.name
    if name == "postgresql":
        conn.execute(text(f"SET statement_timeout = {int(seconds * 1000)}"))
        yield
        return
    if name != "duckdb":
        yield
        return
    raw = _raw(conn)
    fired = threading.Event()

    def stop() -> None:
        fired.set()
        raw.interrupt()

    timer = threading.Timer(seconds, stop)
    timer.daemon = True
    timer.start()
    try:
        yield
    except Exception as err:
        if fired.is_set():
            raise DeadlinePassed(
                f"The query ran longer than {seconds:g} s and was stopped."
            ) from err
        raise
    finally:
        timer.cancel()
