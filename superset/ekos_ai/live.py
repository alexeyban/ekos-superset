"""The live answer: a question's steps streamed to the page as they happen (NDJSON).

Kept apart from ``api`` (which needs a running Superset) so that it can be tested alone."""

from __future__ import annotations

import json
import logging
import queue
import threading

from flask import Response, copy_current_request_context, g

logger = logging.getLogger(__name__)

_KEEPALIVE_SECONDS = 10


def stream(outcome) -> Response:
    """NDJSON, one event per line: each step as it starts (``start``) and as it is recorded
    (``step``), then ``{"type": "result", "result": …}``; ``ping`` lines keep a quiet stream open.

    ``outcome()`` runs in a thread with a copy of this request. The copy gets a fresh ``g``, so
    what the request had put there (the person, the chosen source) is carried over, and
    ``g.ekos_listener`` is where the EKOS client reports its steps (``service._client``)."""
    events: queue.Queue = queue.Queue()
    carried = {k: g.get(k) for k in ("user", "ekos_source") if g.get(k) is not None}

    @copy_current_request_context
    def work() -> None:
        for key, value in carried.items():
            setattr(g, key, value)
        g.ekos_listener = events.put
        try:
            events.put({"type": "result", "result": outcome()})
        except Exception:  # noqa: BLE001 - outcome() answers every failure itself; belt and braces
            logger.exception("ekos_ai stream failed")
            events.put({"type": "result", "result": {
                "status": "failed", "explanation": "The question could not be answered."}})
        finally:
            events.put(None)

    threading.Thread(target=work, name="ekos-ai-question", daemon=True).start()

    def lines():
        while True:
            try:
                event = events.get(timeout=_KEEPALIVE_SECONDS)
            except queue.Empty:
                event = {"type": "ping"}
            if event is None:
                return
            yield json.dumps(event, default=str, ensure_ascii=False) + "\n"

    return Response(lines(), mimetype="application/x-ndjson",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
