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
    with pytest.raises(EkosUnavailable, match="EKOS_MCP_URL"):
        EkosClient("")


def test_jsonrpc_error_is_unavailable():
    post = FakePost(lambda m: Resp({"jsonrpc": "2.0", "id": 1, "error": {"code": -32601, "message": "method not found"}}))
    with pytest.raises(EkosUnavailable, match="method not found"):
        EkosClient("http://ekos/mcp", post=post).call("ekos_status")


def test_parse_body_rejects_garbage():
    with pytest.raises(EkosUnavailable):
        parse_body("<html>", "text/html")
    with pytest.raises(EkosUnavailable):
        parse_body("data: {}\n", "text/event-stream")
