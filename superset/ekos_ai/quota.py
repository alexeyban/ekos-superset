"""Daily question limits (EKOS RFC 0250): per Superset user and for the whole instance.

Every Ask (AI) request, SQL or dashboard, spends one of the person's questions and one of the
instance's before any model is called — except for the people in ``EKOS_AI_UNLIMITED_USERS``. The counts live in Superset's metadata database, in one
small table this module creates (``ekos_ai_usage``; ``user_key`` ``*`` is the instance's row).
A day is a UTC day.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import text

TABLE = "ekos_ai_usage"
INSTANCE = "*"

_CREATE = (
    f"CREATE TABLE IF NOT EXISTS {TABLE} ("
    "user_key VARCHAR(64) NOT NULL, day VARCHAR(10) NOT NULL, calls INTEGER NOT NULL, "
    "PRIMARY KEY (user_key, day))"
)


class LimitReached(RuntimeError):
    """A daily limit is spent; the message says which and when it resets."""


def unlimited(username: str, listed: str) -> bool:
    """Whether this sign-in name (``<idp>:<name>``, e.g. ``google:someone@example.com``) is one of
    the comma-separated ``EKOS_AI_UNLIMITED_USERS``: its questions are neither limited nor counted."""
    names = {n.strip().lower() for n in listed.split(",") if n.strip()}
    return bool(username) and username.strip().lower() in names


def today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _calls(conn, key: str, day: str) -> int:
    row = conn.execute(
        text(f"SELECT calls FROM {TABLE} WHERE user_key = :k AND day = :d"), {"k": key, "d": day}
    ).fetchone()
    return int(row[0]) if row else 0


def _bump(conn, key: str, day: str) -> None:
    conn.execute(
        text(
            f"INSERT INTO {TABLE} (user_key, day, calls) VALUES (:k, :d, 1) "
            "ON CONFLICT (user_key, day) DO UPDATE SET calls = "
            f"{TABLE}.calls + 1"
        ),
        {"k": key, "d": day},
    )


def spend(engine, user_key: str, *, per_user: int, per_instance: int, day: str | None = None) -> dict:
    """Spend one question for ``user_key`` and the instance, or raise LimitReached and spend
    nothing. Returns what is left today."""
    day = day or today()
    with engine.begin() as conn:
        conn.execute(text(_CREATE))
        mine, everyone = _calls(conn, user_key, day), _calls(conn, INSTANCE, day)
        if mine >= per_user:
            raise LimitReached(
                f"You have used your {per_user} questions for today; they reset at 00:00 UTC."
            )
        if everyone >= per_instance:
            raise LimitReached(
                f"This demo has answered its {per_instance} questions for today; "
                "it resets at 00:00 UTC."
            )
        _bump(conn, user_key, day)
        _bump(conn, INSTANCE, day)
    return {"yours": per_user - mine - 1, "everyone": per_instance - everyone - 1}
