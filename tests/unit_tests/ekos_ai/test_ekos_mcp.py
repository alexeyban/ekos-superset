import json

import pytest
import requests

from superset.ekos_ai.ekos_mcp import EkosClient, EkosUnavailable, parse_body


class Resp:
    def __init__(self, body, status=200, content_type="application/json"):
        self.text = body if isinstance(body, str) else json.dumps(body)
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.encoding = None


def tool_result(value, is_error=False):
    return {"jsonrpc": "2.0", "id": 1, "result": {
        "content": [{"type": "text", "text": json.dumps(value)}], "isError": is_error}}


class FakePost:
    def __init__(self, answer):
        self.answer = answer
        self.sent = []

    def __call__(self, url, headers, json, timeout):
        self.sent.append((url, headers, json))
        if json["method"] == "initialize":
            return Resp({"jsonrpc": "2.0", "id": json["id"], "result": {"protocolVersion": "2025-06-18"}})
        if "id" not in json:
            return Resp("", status=202)
        return self.answer(json)


def test_initializes_once_and_returns_the_tool_json():
    post = FakePost(lambda m: Resp(tool_result({"matches": [{"name": "c_order"}]})))
    client = EkosClient("http://ekos/mcp", token="t0k", post=post)
    assert client.call("ekos_search", {"query": "order"}) == {"matches": [{"name": "c_order"}]}
    client.call("ekos_search", {"query": "x"})
    methods = [m["method"] for _, _, m in post.sent]
    assert methods == ["initialize", "notifications/initialized", "tools/call", "tools/call"]
    assert post.sent[0][1]["Authorization"] == "Bearer t0k"
    assert post.sent[2][2]["params"] == {"name": "ekos_search", "arguments": {"query": "order"}}


def test_no_token_sends_no_authorization_header():
    post = FakePost(lambda m: Resp(tool_result({})))
    EkosClient("http://ekos/mcp", post=post).call("ekos_status")
    assert "Authorization" not in post.sent[0][1]


def test_sse_answer_is_read():
    body = 'event: message\ndata: {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"{\\"a\\": 1}"}]}}\n\n'
    post = FakePost(lambda m: Resp(body, content_type="text/event-stream"))
    assert EkosClient("http://ekos/mcp", post=post).call("ekos_status") == {"a": 1}


def test_tool_error_is_unavailable():
    post = FakePost(lambda m: Resp(tool_result("no such table", is_error=True)))
    with pytest.raises(EkosUnavailable, match="ekos_state"):
        EkosClient("http://ekos/mcp", post=post).call("ekos_state", {"id": "x"})


@pytest.mark.parametrize("status,match", [(401, "token"), (403, "token"), (502, "502")])
def test_http_errors_are_unavailable(status, match):
    post = FakePost(lambda m: Resp("nope", status=status))
    with pytest.raises(EkosUnavailable, match=match):
        EkosClient("http://ekos/mcp", post=post).call("ekos_status")


def test_unreachable_is_unavailable():
    def post(*a, **k):
        raise requests.ConnectionError("refused")

    with pytest.raises(EkosUnavailable, match="reached"):
        EkosClient("http://ekos/mcp", post=post).call("ekos_status")


def test_no_url_is_unavailable():
    with pytest.raises(EkosUnavailable, match="not configured") as err:
        EkosClient("")
    assert "EKOS_MCP_URL" in err.value.detail["cause"]


def test_jsonrpc_error_is_unavailable():
    post = FakePost(lambda m: Resp({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "method not found"}}))
    with pytest.raises(EkosUnavailable) as err:
        EkosClient("http://ekos/mcp", post=post).call("ekos_status")
    assert "method not found" in err.value.detail["cause"]


def test_parse_body_rejects_garbage():
    with pytest.raises(EkosUnavailable):
        parse_body("<html>", "text/html")
    with pytest.raises(EkosUnavailable):
        parse_body("data: {}\n", "text/event-stream")


def test_every_request_is_logged_with_a_summary():
    ossie = {"model": {"datasets": [{"name": "a"}, {"name": "b"}], "relationships": [{}]}}
    post = FakePost(lambda m: Resp(tool_result(ossie)))
    client = EkosClient("http://ekos/mcp?x=1", token="t0k", post=post)
    client.call("ekos_ossie_export", {"status": "confirmed"}, step="read the compiled schema")
    steps = [e["step"] for e in client.log]
    assert steps == ["connect", "read the compiled schema"]
    last = client.log[-1]
    assert last["ok"] is True and last["tool"] == "ekos_ossie_export"
    assert last["summary"] == "2 tables, 1 foreign-key joins"
    assert '"status": "confirmed"' in last["arguments"] and isinstance(last["ms"], int)
    assert "t0k" not in str(client.log)


def test_a_timeout_says_which_step_where_and_what_to_do():
    def post(url, headers, json, timeout):
        if json["method"] == "initialize":
            return Resp({"jsonrpc": "2.0", "id": 1, "result": {}})
        if "id" not in json:
            return Resp("", status=202)
        raise requests.Timeout("read timed out")

    client = EkosClient("https://console.example/mcp/default/idempiere", token="secret", post=post, timeout=30)
    with pytest.raises(EkosUnavailable) as err:
        client.call("ekos_search", {"query": "orders"}, step="find the tables")
    d = err.value.detail
    assert d["step"] == "find the tables" and d["cause"] == "no answer within 30 s"
    assert d["endpoint"] == "https://console.example/mcp/default/idempiere"
    assert "loading" in d["hint"] and "secret" not in str(d)
    assert client.log[-1]["ok"] is False


def test_a_gateway_error_is_explained():
    post = FakePost(lambda m: Resp({"jsonrpc": "2.0", "id": 2, "error": {
        "code": -32603, "message": "the project's server is not ready: never became ready"}}))
    with pytest.raises(EkosUnavailable) as err:
        EkosClient("http://ekos/mcp", post=post).call("ekos_status")
    assert "not ready" in err.value.detail["cause"] and "starting" in err.value.detail["hint"]


def test_a_note_after_the_json_is_kept_apart():
    # Found on Azure 2026-11-08: EKOS adds a freshness note as a second text item; joining the
    # items made the JSON unparsable and Superset saw no tables at all.
    answer = {"jsonrpc": "2.0", "id": 1, "result": {"content": [
        {"type": "text", "text": json.dumps({"datasets": 1})},
        {"type": "text", "text": "Note: the ledger may be behind the sources (3 files changed)."},
    ], "isError": False}}
    post = FakePost(lambda m: Resp(answer))
    client = EkosClient("http://ekos/mcp", post=post)
    assert client.call("ekos_ossie_export") == {"datasets": 1}
    assert client.log[-1]["notes"] == ["Note: the ledger may be behind the sources (3 files changed)."]


def test_each_step_is_heard_when_it_starts_and_when_it_ends():
    post = FakePost(lambda m: Resp(tool_result({"matches": [{"name": "c_order"}]})))
    client = EkosClient("http://ekos/mcp", post=post)
    heard = []
    client.log.listener = heard.append
    client.call("ekos_search", {"query": "order"}, step="find tables")
    assert [(e["type"], e["i"]) for e in heard] == [
        ("start", 0), ("step", 0),            # connect
        ("start", 1), ("step", 1), ("step", 1),  # the call, then its summary
    ]
    assert heard[2]["entry"]["step"] == "find tables" and "ok" not in heard[2]["entry"]
    assert heard[-1]["entry"]["ok"] is True and "summary" in heard[-1]["entry"]


def test_a_failing_listener_never_stops_the_answer():
    post = FakePost(lambda m: Resp(tool_result({"ok": 1})))
    client = EkosClient("http://ekos/mcp", post=post)

    def broken(event):
        raise RuntimeError("the page went away")

    client.log.listener = broken
    assert client.call("ekos_status") == {"ok": 1}
    assert client.log.listener is None and len(client.log) == 2
