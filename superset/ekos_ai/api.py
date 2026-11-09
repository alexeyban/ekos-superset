"""REST endpoints and the page."""

from __future__ import annotations

import logging
import os

from flask import g, request
from flask_appbuilder import BaseView, expose as view_expose
from flask_appbuilder.api import expose, protect, safe
from flask_appbuilder.security.decorators import has_access

from superset.ekos_ai import ekos_mcp, guard, llm, quota, service
from superset.ekos_ai.live import stream
from superset.views.base_api import BaseSupersetApi, statsd_metrics

logger = logging.getLogger(__name__)


class EkosAIRestApi(BaseSupersetApi):
    method_permission_name = {"sql": "read", "dashboard": "write", "sources": "read"}
    allow_browser_login = True
    class_permission_name = "EkosAI"
    resource_name = "ekos_ai"
    openapi_spec_tag = "EKOS AI"

    def _question(self):
        body = request.get_json(silent=True) or {}
        text = str(body.get("question", "")).strip()
        if not service.enabled():
            return None, self.response(404, message="Not enabled.")
        if len(text) < 4:
            return None, self.response(400, message="Ask a question.")
        try:
            service.select_source(str(body.get("source") or "") or None)
        except guard.Refused as err:
            return None, self.response(400, message=str(err))
        return text, None

    @expose("/sources", methods=("GET",))
    @protect()
    @safe
    def sources(self):
        """The sources a question may run on, for the page's picker."""
        if not service.enabled():
            return self.response(404, message="Not enabled.")
        return self.response(200, result=[
            {"id": s["id"], "label": s.get("label") or s["id"]} for s in service.sources()
        ])

    @staticmethod
    def _with_log(result: dict) -> dict:
        """Every answer carries the log of the EKOS calls behind it (and failures too)."""
        client = getattr(g, "ekos_client", None)
        if client is not None:
            result["ekos_log"] = client.log
        return result

    def _outcome(self, fn, text) -> dict:
        """The answer, or why there is none — always a result the page can show."""
        try:
            return self._with_log(fn(text))
        except (llm.ModelError, guard.Refused, quota.LimitReached) as err:
            return self._with_log({"status": "failed", "explanation": str(err)})
        except ekos_mcp.EkosUnavailable as err:
            # No fallback to the database's own catalog: questions are answered from EKOS only.
            logger.warning("ekos_ai: EKOS unavailable: %s %s", err, err.detail)
            return self._with_log({
                "status": "failed",
                "explanation": f"{err} Questions are answered only from EKOS's compiled "
                               "knowledge, so this one was not answered.",
                "ekos_error": err.detail,
            })
        except Exception:  # noqa: BLE001 - logged, and the person gets a plain message
            logger.exception("ekos_ai failed")
            return self._with_log({"status": "failed",
                                   "explanation": "The question could not be answered."})

    def _answer(self, fn, text):
        if (request.get_json(silent=True) or {}).get("stream"):
            return stream(lambda: self._outcome(fn, text))
        return self.response(200, result=self._outcome(fn, text))

    @expose("/sql", methods=("POST",))
    @protect()
    @safe
    @statsd_metrics
    def sql(self):
        """Answer a question with a checked SELECT and its rows."""
        text, error = self._question()
        return error or self._answer(service.text_to_sql, text)

    @expose("/dashboard", methods=("POST",))
    @protect()
    @safe
    @statsd_metrics
    def dashboard(self):
        """Build a dashboard from a request."""
        text, error = self._question()
        return error or self._answer(service.text_to_dashboard, text)


class EkosAIView(BaseView):
    route_base = "/ekos-ai"
    default_view = "ask"
    # Absolute: Flask-AppBuilder resolves a relative template folder against its own package,
    # so `templates/ekos_ai/ask.html` beside this module was never found (the page answered 500).
    template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

    @view_expose("/")
    @has_access
    def ask(self):
        return self.render_template("ekos_ai/ask.html")
