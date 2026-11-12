"""The one model call, over the Anthropic Messages API. The key is read from the environment
variable named in the config; it is never logged and never sent to the browser."""

from __future__ import annotations

import json
import os
import re

import requests


class ModelError(RuntimeError):
    pass


def complete(system: str, user: str, *, model: str, base_url: str, key_env: str,
             max_tokens: int = 1500, timeout: float = 45.0) -> dict:
    """The model's JSON answer. A reply cut off at ``max_tokens`` is asked once more with twice
    the room (the answer then carries ``_attempts: 2``); every failure says why — the stop
    reason and how much text came back — never the text itself."""
    key = os.environ.get(key_env, "")
    if not key:
        raise ModelError("No model key is configured.")
    attempts, limit = 0, max_tokens
    while True:
        attempts += 1
        try:
            response = requests.post(
                base_url.rstrip("/") + "/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json={"model": model, "max_tokens": limit, "system": system,
                      "messages": [{"role": "user", "content": user}]},
                timeout=timeout,
            )
        except requests.Timeout as err:
            raise ModelError(f"The model service did not answer within {timeout:.0f} s.") from err
        except requests.RequestException as err:
            raise ModelError("The model service could not be reached.") from err
        if response.status_code >= 300:
            raise ModelError(f"The model service answered {response.status_code}.")
        body = response.json()
        text = "".join(b.get("text", "") for b in body.get("content", []) if b.get("type") == "text")
        stop = str(body.get("stop_reason") or "unknown")
        try:
            value = parse_json(text)
        except ModelError as err:
            if stop == "max_tokens" and attempts == 1:
                limit *= 2
                continue
            raise ModelError(f"{err} (stop reason: {stop}; {len(text)} characters"
                             f"{'; asked twice' if attempts > 1 else ''}; limit {limit} tokens)") from err
        if attempts > 1:
            value["_attempts"] = attempts
        return value


def parse_json(text: str) -> dict:
    """The first JSON object in the model's text; anything else is an error, not a guess."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ModelError("The model did not answer with JSON.")
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError as err:
        raise ModelError("The model's JSON did not parse.") from err
    if not isinstance(value, dict):
        raise ModelError("The model's answer was not an object.")
    return value
