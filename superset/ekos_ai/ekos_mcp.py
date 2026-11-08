"""The EKOS knowledge ledger over MCP (Streamable HTTP, JSON-RPC 2.0).

Ask (AI) answers only from what EKOS has compiled: tables, columns, descriptions and foreign keys
come from EKOS, and every SQL is written or checked by EKOS. There is no fallback to the
database's own catalog: when EKOS cannot be reached the question is not answered.

The endpoint is either ``ekos mcp serve --http <addr>`` (``http://host:port/mcp``) or the EKOS
console's gateway for one project (``https://console/mcp/<tenant>/<project>``, an API token with
the read role). The token is read from the environment variable named in the config and is never
logged or sent to the browser.

Every request is recorded in ``EkosClient.log`` — the step, the tool and its arguments, how long it
took and what came back — so the page can show how the answer was built from EKOS.
"""

from __future__ import annotations

import itertools
import json
import os
import time
from typing import Any, Callable
from urllib.parse import urlsplit

import requests

PROTOCOL_VERSION = "2025-06-18"
_ARGS_SHOWN = 300


class EkosUnavailable(RuntimeError):
    """EKOS could not answer. The message is one line for the person; ``detail`` says which
    step failed, where, why, and what to do."""

    def __init__(self, message: str, detail: dict | None = None) -> None:
        super().__init__(message)
        self.detail = detail or {}


Post = Callable[..., Any]


def _endpoint(url: str) -> str:
    """The address without credentials or query: safe to show."""
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.hostname or ''}{parts.path}"


def _summary(tool: str, value: dict) -> str:
    """One line about a tool's result, for the log."""
    if tool == "ekos_ossie_export":
        model = value.get("model") or {}
        return (f"{len(model.get('datasets') or [])} tables, "
                f"{len(model.get('relationships') or [])} foreign-key joins"
                + (" (truncated)" if value.get("truncated") else ""))
    if tool == "ekos_search":
        names = [str(m.get("name")) for m in value.get("matches") or []]
        return f"{len(names)} matches" + (f": {', '.join(names[:8])}" if names else "")
    if tool == "ekos_sql_check":
        if "sql" not in value and value.get("dialect_notes") is not None and not value.get("findings"):
            return f"{len(value.get('dialect_notes') or [])} {value.get('dialect', '')} rules"
        findings = value.get("findings") or []
        codes = [f"{f.get('code')} {f.get('severity')}" for f in findings]
        out = f"{value.get('tables_known', '?')} tables known, {len(findings)} findings"
        if codes:
            out += f": {', '.join(codes[:6])}"
        if value.get("parse_error"):
            out += f"; parse error: {str(value['parse_error'])[:120]}"
        return out
    if tool == "ekos_sql_generate":
        sql = " ".join(str(value.get("sql") or "").split())
        return (f"status {value.get('status')}, {value.get('turns', '?')} model replies, "
                f"open findings {value.get('open_findings') or []}"
                + (f"; SQL: {sql[:200]}" if sql else ""))
    return ", ".join(f"{k}={str(v)[:40]}" for k, v in list(value.items())[:4])


class EkosClient:
    """A thin MCP client: ``call(tool, arguments)`` returns the tool's JSON result."""

    def __init__(self, url: str, *, token: str = "", timeout: float = 30.0,
                 post: Post | None = None) -> None:
        self.log: list[dict] = []
        if not url:
            raise EkosUnavailable("EKOS is not configured.", {
                "step": "configuration", "cause": "EKOS_MCP_URL is empty",
                "hint": "Set EKOS_MCP_URL to the EKOS project's MCP endpoint.",
            })
        self.url = url
        self.endpoint = _endpoint(url)
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

    def _fail(self, entry: dict, started: float, message: str, cause: str, hint: str) -> EkosUnavailable:
        entry.update(ms=round((time.monotonic() - started) * 1000), ok=False, error=cause)
        self.log.append(entry)
        return EkosUnavailable(message, {
            "step": entry["step"], "endpoint": self.endpoint, "cause": cause, "hint": hint,
            "elapsed_ms": entry["ms"],
        })

    def _send(self, method: str, params: dict | None = None, *, notify: bool = False,
              step: str = "") -> Any:
        message: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notify:
            message["id"] = next(self._ids)
        entry: dict[str, Any] = {"step": step or method, "method": method}
        if method == "tools/call" and params:
            entry["tool"] = params.get("name")
            entry["arguments"] = json.dumps(params.get("arguments") or {}, ensure_ascii=False)[:_ARGS_SHOWN]
        started = time.monotonic()
        try:
            response = self._post(self.url, headers=self._headers(), json=message,
                                  timeout=self.timeout)
        except requests.Timeout as err:
            raise self._fail(
                entry, started, "EKOS did not answer in time.",
                f"no answer within {self.timeout:.0f} s",
                "The project's EKOS server may still be loading its knowledge (a large project "
                "takes a few minutes after a restart). Ask again in a minute.",
            ) from err
        except requests.ConnectionError as err:
            raise self._fail(
                entry, started, "EKOS could not be reached.",
                f"connection failed: {str(err)[:160]}",
                "The EKOS console is down or restarting, or its address is wrong.",
            ) from err
        except requests.RequestException as err:
            raise self._fail(entry, started, "EKOS could not be reached.",
                             str(err)[:200], "Try again; if it persists, tell the operator.") from err
        if notify:
            return None
        if response.status_code in (401, 403):
            raise self._fail(entry, started, "EKOS refused the token.",
                             f"HTTP {response.status_code}",
                             "The API token is missing, expired, or has no read role on the project.")
        if response.status_code >= 300:
            raise self._fail(entry, started, f"EKOS answered {response.status_code}.",
                             f"HTTP {response.status_code}: {response.text[:160]}",
                             "The EKOS console had an error; see its log.")
        # EKOS always answers UTF-8; without a charset `requests` would decode Latin-1.
        response.encoding = "utf-8"
        try:
            body = parse_body(response.text, response.headers.get("Content-Type", ""))
        except EkosUnavailable as err:
            raise self._fail(entry, started, str(err), "the answer was not JSON-RPC",
                             "Check that EKOS_MCP_URL is an MCP endpoint.") from err
        if "error" in body:
            text = str((body.get("error") or {}).get("message", "error"))[:300]
            hint = ("The project's EKOS server is starting or failed to start; on the console, "
                    "check the project and its log." if "server" in text else
                    "EKOS reported an error for this request.")
            raise self._fail(entry, started, "EKOS could not answer.", text, hint)
        entry.update(ms=round((time.monotonic() - started) * 1000), ok=True)
        self.log.append(entry)
        return body.get("result")

    def initialize(self) -> None:
        if self._initialized:
            return
        self._send("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "ekos-superset", "version": "1"},
        }, step="connect")
        self._send("notifications/initialized", notify=True, step="connect")
        self._initialized = True

    def tools(self) -> set[str]:
        """The names of the tools this endpoint offers the caller."""
        self.initialize()
        result = self._send("tools/list", {}, step="list tools") or {}
        names = {str(t.get("name")) for t in result.get("tools", []) if t.get("name")}
        self.log[-1]["summary"] = f"{len(names)} tools" + (
            "; ekos_sql_generate offered" if "ekos_sql_generate" in names else "")
        return names

    def call(self, tool: str, arguments: dict | None = None, *, step: str = "") -> dict:
        """The tool's result as JSON. A tool error raises EkosUnavailable."""
        self.initialize()
        result = self._send("tools/call", {"name": tool, "arguments": arguments or {}},
                            step=step or tool) or {}
        entry = self.log[-1]
        text = "".join(
            c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"
        )
        if result.get("isError"):
            entry.update(ok=False, error=text[:300] or "error")
            raise EkosUnavailable(f"EKOS could not answer ({tool}).", {
                "step": entry["step"], "endpoint": self.endpoint, "cause": text[:300] or "error",
                "hint": "EKOS reported an error for this request.", "elapsed_ms": entry.get("ms"),
            })
        try:
            value = json.loads(text) if text else {}
        except json.JSONDecodeError:
            value = {"text": text}
        if not isinstance(value, dict):
            value = {"value": value}
        entry["summary"] = _summary(tool, value)
        return value


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
