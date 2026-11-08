"""The EKOS knowledge ledger over MCP (Streamable HTTP, JSON-RPC 2.0).

Ask (AI) answers only from what EKOS has compiled: tables, columns, descriptions and foreign keys
come from EKOS, and every SQL the model drafts is checked by ``ekos_sql_check``. There is no
fallback to the database's own catalog: when EKOS cannot be reached the question is not answered.

The endpoint is either ``ekos mcp serve --http <addr>`` (``http://host:port/mcp``) or the EKOS
console's gateway for one project (``https://console/mcp/<tenant>/<project>``, an API token with
the read role). The token is read from the environment variable named in the config and is never
logged or sent to the browser.
"""

from __future__ import annotations

import itertools
import json
import os
from typing import Any, Callable

import requests

PROTOCOL_VERSION = "2025-06-18"


class EkosUnavailable(RuntimeError):
    """EKOS could not answer; the message says why in words a person can act on."""


Post = Callable[..., Any]


class EkosClient:
    """A thin MCP client: ``call(tool, arguments)`` returns the tool's JSON result."""

    def __init__(self, url: str, *, token: str = "", timeout: float = 30.0,
                 post: Post | None = None) -> None:
        if not url:
            raise EkosUnavailable("EKOS is not configured (EKOS_MCP_URL).")
        self.url = url
        self.token = token
        self.timeout = timeout
        self._post = post or requests.post
        self._ids = itertools.count(1)
        self._initialized = False

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": PROTOCOL_VERSION,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _send(self, method: str, params: dict | None = None, *, notify: bool = False) -> Any:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notify:
            message["id"] = next(self._ids)
        try:
            response = self._post(self.url, headers=self._headers(), json=message,
                                  timeout=self.timeout)
        except requests.RequestException as err:
            raise EkosUnavailable("EKOS could not be reached.") from err
        if notify:
            return None
        if response.status_code in (401, 403):
            raise EkosUnavailable("EKOS refused the token.")
        if response.status_code >= 300:
            raise EkosUnavailable(f"EKOS answered {response.status_code}.")
        # EKOS always answers UTF-8; without a charset `requests` would decode Latin-1.
        response.encoding = "utf-8"
        body = parse_body(response.text, response.headers.get("Content-Type", ""))
        if "error" in body:
            message = str((body.get("error") or {}).get("message", "error"))[:200]
            raise EkosUnavailable(f"EKOS: {message}")
        return body.get("result")

    def initialize(self) -> None:
        if self._initialized:
            return
        self._send("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "ekos-superset", "version": "1"},
        })
        self._send("notifications/initialized", notify=True)
        self._initialized = True

    def tools(self) -> set[str]:
        """The names of the tools this endpoint offers the caller."""
        self.initialize()
        result = self._send("tools/list", {}) or {}
        return {str(t.get("name")) for t in result.get("tools", []) if t.get("name")}

    def call(self, tool: str, arguments: dict | None = None) -> dict:
        """The tool's result as JSON. A tool error raises EkosUnavailable."""
        self.initialize()
        result = self._send("tools/call", {"name": tool, "arguments": arguments or {}}) or {}
        text = "".join(
            c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"
        )
        if result.get("isError"):
            raise EkosUnavailable(f"EKOS {tool}: {text[:200] or 'error'}")
        try:
            value = json.loads(text) if text else {}
        except json.JSONDecodeError:
            return {"text": text}
        return value if isinstance(value, dict) else {"value": value}


def parse_body(text: str, content_type: str) -> dict:
    """One JSON-RPC response from a JSON body or an SSE stream (the last ``data:`` event that
    carries a result or an error)."""
    if "text/event-stream" in content_type.lower():
        found: dict | None = None
        for line in text.splitlines():
            if not line.startswith("data:"):
                continue
            try:
                value = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and ("result" in value or "error" in value):
                found = value
        if found is None:
            raise EkosUnavailable("EKOS sent no answer.")
        return found
    try:
        value = json.loads(text)
    except json.JSONDecodeError as err:
        raise EkosUnavailable("EKOS's answer did not parse.") from err
    if isinstance(value, list):  # a batch answer: the first response
        value = next((v for v in value if isinstance(v, dict)), {})
    if not isinstance(value, dict):
        raise EkosUnavailable("EKOS's answer was not an object.")
    return value


def from_config(cfg: Callable[[str, Any], Any]) -> EkosClient:
    token = os.environ.get(cfg("EKOS_MCP_TOKEN_ENV", "EKOS_MCP_TOKEN") or "", "")
    return EkosClient(cfg("EKOS_MCP_URL", ""), token=token,
                      timeout=float(cfg("EKOS_MCP_TIMEOUT", 30)))
