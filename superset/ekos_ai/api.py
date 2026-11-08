"""REST endpoints and the page."""

from __future__ import annotations

import logging

from flask import request
from flask_appbuilder import BaseView, expose as view_expose
from flask_appbuilder.api import expose, protect, safe
from flask_appbuilder.security.decorators import has_access

from superset.ekos_ai import ekos_mcp, guard, llm, quota, service
from superset.views.base_api import BaseSupersetApi, statsd_metrics

logger = logging.getLogger(__name__)


class EkosAIRestApi(BaseSupersetApi):
    method_permission_name = {"sql": "read", "dashboard": "write"}
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
        return text, None

    def _answer(self, fn, text):
        try:
            return self.response(200, result=fn(text))
        except (llm.ModelError, guard.Refused, quota.LimitReached) as err:
            return self.response(200, result={"status": "failed", "explanation": str(err)})
        except ekos_mcp.EkosUnavailable as err:
            # No fallback to the database's own catalog: questions are answered from EKOS only.
            logger.warning("ekos_ai: EKOS unavailable: %s", err)
            return self.response(200, result={
                "status": "failed",
                "explanation": f"{err} Questions are answered only from EKOS's compiled knowledge.",
            })
        except Exception:  # noqa: BLE001 - logged, and the person gets a plain message
            logger.exception("ekos_ai failed")
            return self.response(500, message="The question could not be answered.")

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

    @view_expose("/")
    @has_access
    def ask(self):
        return self.render_template("ekos_ai/ask.html")
