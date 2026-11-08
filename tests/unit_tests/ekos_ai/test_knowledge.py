import copy

import pytest

from superset.ekos_ai import knowledge
from superset.ekos_ai.ekos_mcp import EkosUnavailable
from .fake_ekos import OSSIE, FakeEkos


@pytest.fixture(autouse=True)
def no_cache():
    knowledge._CACHE.clear()
    yield
    knowledge._CACHE.clear()


def test_catalog_carries_ekos_ids_evidence_keys_and_joins():
    cat = knowledge.catalog_from_ossie(OSSIE)
    assert set(cat.tables) == {"store_sales", "customer", "item"}
    customer = cat.tables["customer"]
    assert customer.ekos_id == "id-c" and customer.evidence == ("db/tpcds.sql:26",)
    assert customer.synonyms == ("shoppers", "buyers") and customer.primary_key == ("c_customer_sk",)
    assert [j.ekos_id for j in cat.joins] == ["fk-1", "fk-2"]
    assert cat.columns()["item"] == ["i_item_sk", "i_brand"]


def test_allow_list_narrows_tables_and_drops_dangling_joins():
    cat = knowledge.catalog_from_ossie(OSSIE, {"store_sales", "item"})
    assert set(cat.tables) == {"store_sales", "item"}
    assert [j.right for j in cat.joins] == ["item"]


def test_synonyms_rank_a_table_and_its_fk_partner_follows():
    cat = knowledge.catalog_from_ossie(OSSIE)
    names = knowledge.rank("how many shoppers", cat, [])
    assert names[0] == "customer"
    assert "store_sales" in names  # joined to customer by a foreign key


def test_column_synonym_and_ekos_search_count():
    cat = knowledge.catalog_from_ossie(OSSIE)
    assert knowledge.rank("total revenue", cat, [])[0] == "store_sales"
    assert knowledge.rank("zzz", cat, ["item"])[0] == "item"


def test_rank_does_not_depend_on_ekos_answer_order():
    reversed_ossie = copy.deepcopy(OSSIE)
    reversed_ossie["model"]["datasets"].reverse()
    reversed_ossie["model"]["relationships"].reverse()
    a = knowledge.rank("brand revenue per customer", knowledge.catalog_from_ossie(OSSIE), [])
    b = knowledge.rank("brand revenue per customer", knowledge.catalog_from_ossie(reversed_ossie), [])
    assert a == b


def test_no_match_ranks_nothing():
    assert knowledge.rank("weather forecast", knowledge.catalog_from_ossie(OSSIE), []) == []


def test_describe_shows_meanings_keys_and_only_shown_joins():
    cat = knowledge.catalog_from_ossie(OSSIE)
    text = knowledge.describe(["store_sales", "customer"], cat)
    assert "TABLE customer -- Customer dimension (also called: shoppers, buyers)" in text
    assert "c_customer_sk Integer PRIMARY KEY" in text
    assert "ss_sales_price Decimal -- Price paid; also: revenue" in text
    assert "store_sales.ss_customer_sk = customer.c_customer_sk" in text
    assert "i_item_sk" not in text  # item not shown, nor its join


def test_errors_and_warnings_split_by_severity():
    report = {"parse_error": None, "findings": [
        {"code": "L001", "severity": "error", "message": "unknown column x"},
        {"code": "L005", "severity": "warn", "message": "integer division"}]}
    assert knowledge.errors(report) == ["L001: unknown column x"]
    assert knowledge.warnings(report) == [{"code": "L005", "message": "integer division"}]
    assert knowledge.errors({"parse_error": "bad"}) == ["parse error: bad"]


def test_grounding_names_only_used_tables_and_their_joins():
    cat = knowledge.catalog_from_ossie(OSSIE)
    g = knowledge.grounding(cat, ("customer", "store_sales"), {"note": "static"})
    assert g["source"] == "EKOS"
    assert [t["ekos_id"] for t in g["tables"]] == ["id-c", "id-ss"]
    assert g["joins"] == [{"on": "store_sales.ss_customer_sk = customer.c_customer_sk", "ekos_id": "fk-1"}]


def test_load_is_cached_and_empty_is_unavailable():
    fake = FakeEkos()
    knowledge.load(fake, allowed=None)
    knowledge.load(fake, allowed=None)
    assert fake.called().count("ekos_ossie_export") == 1
    empty = FakeEkos(ossie={"model": {"datasets": []}})
    empty.url = "fake://empty"  # the cache is per endpoint
    with pytest.raises(EkosUnavailable, match="no tables"):
        knowledge.load(empty, allowed=None)


def test_truncated_export_fetches_missing_allowed_tables():
    first = copy.deepcopy(OSSIE)
    first["truncated"] = True
    first["model"]["datasets"] = first["model"]["datasets"][:1]

    class Paged(FakeEkos):
        def call(self, tool, arguments=None, *, step=""):
            self.calls.append((tool, arguments))
            if arguments.get("match"):
                return OSSIE
            return first

    fake = Paged()
    cat = knowledge.load(fake, allowed={"store_sales", "item"})
    assert set(cat.tables) == {"store_sales", "item"}
    assert [a.get("match") for t, a in fake.calls] == [None, "item"]
    assert [j.right for j in cat.joins] == ["item"]


def test_an_allow_list_is_asked_for_by_exact_names_in_one_call():
    fake = FakeEkos()
    knowledge.load(fake, allowed={"store_sales", "item"})
    exports = [a for t, a in fake.calls if t == "ekos_ossie_export"]
    assert len(exports) == 1 and exports[0]["names"] == ["item", "store_sales"]
