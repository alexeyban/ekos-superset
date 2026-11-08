"""A stand-in for an EKOS MCP server: the TPC-DS slice of an ``ekos_ossie_export`` answer, a
search over it, and a scripted ``ekos_sql_check``."""

import json

from superset.ekos_ai.ekos_mcp import EkosUnavailable


def ext(**data):
    return [{"vendor_name": "EKOS", "data": json.dumps(data)}]


def field(name, datatype="Integer", description="", synonyms=()):
    out = {"name": name, "datatype": datatype, "expression": {"sql": name}}
    if description:
        out["description"] = description
    if synonyms:
        out["ai_context"] = {"synonyms": list(synonyms)}
    return out


OSSIE = {
    "datasets": 3, "truncated": False, "status_filter": "confirmed",
    "model": {
        "ai_context": {"instructions": "Meanings marked unreviewed are hypotheses."},
        "datasets": [
            {"name": "store_sales", "description": "One row per item sold in a store",
             "custom_extensions": ext(ekos_id="id-ss", evidence=["db/tpcds.sql:40"], status="observed"),
             "fields": [field("ss_item_sk"), field("ss_customer_sk"),
                        field("ss_sales_price", "Decimal", "Price paid", ["revenue"])]},
            {"name": "customer", "description": "Customer dimension",
             "ai_context": {"synonyms": ["shoppers", "buyers"]},
             "custom_extensions": ext(ekos_id="id-c", evidence=["db/tpcds.sql:26"]),
             "primary_key": ["c_customer_sk"],
             "fields": [field("c_customer_sk"), field("c_last_name", "String", "Last name")]},
            {"name": "item", "description": "Products",
             "custom_extensions": ext(ekos_id="id-i", evidence=["db/tpcds.sql:5"]),
             "primary_key": ["i_item_sk"],
             "fields": [field("i_item_sk"), field("i_brand", "String", "Brand")]},
        ],
        "relationships": [
            {"from": "store_sales", "from_columns": ["ss_customer_sk"], "to": "customer",
             "to_columns": ["c_customer_sk"], "custom_extensions": ext(ekos_id="fk-1")},
            {"from": "store_sales", "from_columns": ["ss_item_sk"], "to": "item",
             "to_columns": ["i_item_sk"], "custom_extensions": ext(ekos_id="fk-2")},
        ],
    },
}


class FakeEkos:
    url = "fake://ekos"

    def __init__(self, check_reports=None, ossie=OSSIE, down=False):
        self.calls = []
        self.check_reports = list(check_reports or [])
        self.ossie = ossie
        self.down = down

    def call(self, tool, arguments=None):
        arguments = arguments or {}
        self.calls.append((tool, arguments))
        if self.down:
            raise EkosUnavailable("EKOS could not be reached.")
        if tool == "ekos_ossie_export":
            return self.ossie
        if tool == "ekos_search":
            words = arguments["query"].split()
            names = [d["name"] for d in self.ossie["model"]["datasets"]
                     if any(w in d["name"] or w in d.get("description", "").lower() for w in words)]
            return {"matches": [{"id": n, "name": n} for n in names]}
        if tool == "ekos_sql_check":
            if "sql" not in arguments:
                return {"dialect_notes": ["Limit rows with LIMIT n."], "findings": []}
            if self.check_reports:
                return self.check_reports.pop(0)
            return {"findings": [], "parse_error": None, "note": "static checks only"}
        raise AssertionError(f"unexpected tool {tool}")

    def tools(self):
        return [t for t, _ in self.calls]
