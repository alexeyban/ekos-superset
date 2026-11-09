"""The live answer: NDJSON events from a thread that sees what the request put in `g`."""

import json

from flask import Flask, g

from superset.ekos_ai.live import stream


def lines(response):
    return [json.loads(line) for line in "".join(response.response).splitlines()]


def test_steps_then_the_result_with_the_requests_g_carried_over():
    app = Flask(__name__)
    with app.test_request_context(json={"question": "q", "stream": True}):
        g.user = "alice"
        g.ekos_source = {"id": "postgres"}

        def outcome():
            g.ekos_listener({"type": "start", "i": 0, "entry": {"step": "find tables"}})
            g.ekos_listener({"type": "step", "i": 0, "entry": {"step": "find tables", "ok": True}})
            return {"status": "ok", "user": g.user, "source": g.ekos_source["id"]}

        response = stream(outcome)
        assert response.mimetype == "application/x-ndjson"
        events = lines(response)
    assert [e["type"] for e in events] == ["start", "step", "result"]
    assert events[-1]["result"] == {"status": "ok", "user": "alice", "source": "postgres"}


def test_a_failure_still_ends_with_a_result():
    app = Flask(__name__)
    with app.test_request_context(json={"question": "q", "stream": True}):
        def outcome():
            raise ValueError("boom")

        events = lines(stream(outcome))
    assert events == [{"type": "result", "result": {
        "status": "failed", "explanation": "The question could not be answered."}}]
