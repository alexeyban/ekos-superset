import pytest
from sqlalchemy import create_engine

from superset.ekos_ai import quota


@pytest.fixture
def engine():
    return create_engine("sqlite://")


def test_per_user_limit_then_reset_next_day(engine):
    left = [quota.spend(engine, "u1", per_user=2, per_instance=50, day="2026-11-08") for _ in range(2)]
    assert left[-1] == {"yours": 0, "everyone": 48}
    with pytest.raises(quota.LimitReached, match="your 2 questions"):
        quota.spend(engine, "u1", per_user=2, per_instance=50, day="2026-11-08")
    assert quota.spend(engine, "u2", per_user=2, per_instance=50, day="2026-11-08")["yours"] == 1
    assert quota.spend(engine, "u1", per_user=2, per_instance=50, day="2026-11-09")["yours"] == 1


def test_instance_limit_and_a_refusal_spends_nothing(engine):
    for user in ("a", "b", "c"):
        quota.spend(engine, user, per_user=5, per_instance=3, day="d")
    with pytest.raises(quota.LimitReached, match="its 3 questions"):
        quota.spend(engine, "d", per_user=5, per_instance=3, day="d")
    # The refused person spent nothing: with room again they have all five.
    assert quota.spend(engine, "d", per_user=5, per_instance=10, day="d")["yours"] == 4


def test_listed_sign_ins_are_unlimited_case_insensitive():
    listed = "github:alexeyban, google:Alexeyban@gmail.com"
    assert quota.unlimited("google:alexeyban@gmail.com", listed)
    assert quota.unlimited("GITHUB:alexeyban", listed)
    assert not quota.unlimited("google:someone@example.com", listed)
    assert not quota.unlimited("", listed) and not quota.unlimited("github:alexeyban", "")
