from superset.ekos_ai.public_role import admin_names, public_permissions
from superset.ekos_ai.remote_user import PlatformUser, platform_user


def run(environ):
    seen = {}

    def app(env, start):
        seen.update(env)
        return []

    PlatformUser(app)(environ, lambda *a: None)
    return seen


def test_both_headers_make_the_remote_user():
    env = run({"HTTP_X_MS_CLIENT_PRINCIPAL_IDP": "GitHub", "HTTP_X_MS_CLIENT_PRINCIPAL_NAME": "Alice"})
    assert env["REMOTE_USER"] == "github:alice"


def test_a_client_sent_remote_user_is_dropped():
    assert "REMOTE_USER" not in run({"REMOTE_USER": "admin"})
    assert "REMOTE_USER" not in run({"REMOTE_USER": "admin", "HTTP_X_MS_CLIENT_PRINCIPAL_NAME": "x"})


def test_odd_headers_are_nobody():
    assert platform_user({"HTTP_X_MS_CLIENT_PRINCIPAL_IDP": "github",
                          "HTTP_X_MS_CLIENT_PRINCIPAL_NAME": "a b"}) is None


def test_public_role_is_gamma_without_sql_lab_plus_ask():
    gamma = [("can_read", "Dashboard"), ("menu_access", "SQL Lab"), ("can_sql_lab", "Superset"),
             ("can_read", "Chart")]
    pairs = public_permissions(gamma)
    assert ("can_read", "Dashboard") in pairs and ("can_read", "Chart") in pairs
    assert ("menu_access", "SQL Lab") not in pairs and ("can_sql_lab", "Superset") not in pairs
    assert ("can_read", "EkosAI") in pairs and ("can_write", "EkosAI") in pairs
    assert ("menu_access", "Ask (AI)") in pairs  # the page is in the menu, not only at its URL


def test_admin_names():
    assert admin_names(" GitHub:AlexeyBan , ,google:a@b.c") == ["github:alexeyban", "google:a@b.c"]


def test_the_ask_page_template_is_where_the_view_looks():
    # Found 2026-11-08: a relative template folder was resolved against Flask-AppBuilder, and the
    # Ask (AI) page answered 500 (TemplateNotFound). The view's folder is the module's own.
    import os

    import superset.ekos_ai as pkg

    here = os.path.dirname(os.path.abspath(pkg.__file__))
    assert os.path.isfile(os.path.join(here, "templates", "ekos_ai", "ask.html"))
    src = open(os.path.join(here, "api.py")).read()
    assert 'template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")' in src
