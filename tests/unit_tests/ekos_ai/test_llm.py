"""The model client: a cut-off reply is asked once more with more room; every failure says why."""

import pytest

from superset.ekos_ai import llm


class Reply:
    def __init__(self, text, stop):
        self.status_code = 200
        self._body = {"content": [{"type": "text", "text": text}], "stop_reason": stop}

    def json(self):
        return self._body


def scripted(monkeypatch, replies):
    sent = []

    def post(url, headers, json, timeout):
        sent.append(json["max_tokens"])
        return replies.pop(0)

    monkeypatch.setenv("K", "key")
    monkeypatch.setattr(llm.requests, "post", post)
    return sent


def call():
    return llm.complete("s", "u", model="m", base_url="http://model", key_env="K", max_tokens=100)


def test_a_cut_off_reply_is_asked_once_more_with_twice_the_room(monkeypatch):
    sent = scripted(monkeypatch, [Reply('```json\n{"charts": [{"sql": "select', "max_tokens"),
                                  Reply('{"status": "ok", "charts": []}', "end_turn")])
    assert call() == {"status": "ok", "charts": [], "_attempts": 2}
    assert sent == [100, 200]


def test_a_failure_says_why_without_the_text(monkeypatch):
    scripted(monkeypatch, [Reply("", "end_turn")])
    with pytest.raises(llm.ModelError) as err:
        call()
    assert "stop reason: end_turn; 0 characters" in str(err.value)


def test_a_second_cut_off_is_an_error_not_a_loop(monkeypatch):
    sent = scripted(monkeypatch, [Reply("{\"charts\": [", "max_tokens"), Reply("{\"charts\": [", "max_tokens")])
    with pytest.raises(llm.ModelError) as err:
        call()
    assert sent == [100, 200] and "asked twice" in str(err.value)


def test_a_model_that_does_not_answer_in_time_is_a_model_error(monkeypatch):
    import requests
    from superset.ekos_ai import llm

    def slow(*a, **kw):
        raise requests.Timeout("read timed out")
    monkeypatch.setenv("K", "x")
    monkeypatch.setattr(llm.requests, "post", slow)
    try:
        llm.complete("s", "u", model="m", base_url="http://m", key_env="K", timeout=7)
    except llm.ModelError as err:
        assert "within 7 s" in str(err)
    else:
        raise AssertionError("no error")
