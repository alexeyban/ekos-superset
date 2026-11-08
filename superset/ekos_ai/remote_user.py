"""Sign-in done by Azure Container Apps in front of Superset (EKOS RFC 0251).

With Container Apps' authentication set to require sign-in, every request that reaches Superset
carries ``X-MS-CLIENT-PRINCIPAL-IDP`` and ``X-MS-CLIENT-PRINCIPAL-NAME``, which Container Apps strips
from outside requests. This WSGI middleware turns them into ``REMOTE_USER`` (``<idp>:<name>``, lower
case) for Flask-AppBuilder's ``AUTH_REMOTE_USER``; anything a client sent as ``REMOTE_USER`` is
dropped first. Without both headers there is no user.

``superset_config.py``::

    from flask_appbuilder.security.manager import AUTH_REMOTE_USER
    from superset.ekos_ai.remote_user import PlatformUser
    AUTH_TYPE = AUTH_REMOTE_USER
    AUTH_USER_REGISTRATION = True
    AUTH_USER_REGISTRATION_ROLE = "EkosPublic"
    ADDITIONAL_MIDDLEWARE = [PlatformUser]
"""

from __future__ import annotations

import re

_PART = re.compile(r"^[A-Za-z0-9._@+-]{1,200}$")


def platform_user(environ: dict) -> str | None:
    """``<idp>:<name>`` from the platform's headers, or None when either is missing or odd."""
    idp = str(environ.get("HTTP_X_MS_CLIENT_PRINCIPAL_IDP", "")).strip().lower()
    name = str(environ.get("HTTP_X_MS_CLIENT_PRINCIPAL_NAME", "")).strip().lower()
    if not idp or not name or not _PART.match(idp) or not _PART.match(name):
        return None
    return f"{idp}:{name}"


class PlatformUser:
    def __init__(self, app):
        self.app = app

    def __call__(self, environ, start_response):
        environ.pop("REMOTE_USER", None)
        user = platform_user(environ)
        if user is not None:
            environ["REMOTE_USER"] = user
        return self.app(environ, start_response)
