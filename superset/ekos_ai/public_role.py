"""The public demo's role and owner (EKOS RFC 0251), set up at start-up after ``superset init``.

``python -m superset.ekos_ai.public_role``:

* ``EKOS_PUBLIC_ROLE`` (default ``EkosPublic``): Gamma's permissions, plus Ask (AI) (``EkosAI``
  read and write) and access to the database questions run on (``EKOS_AI_DATABASE``), without
  SQL Lab. A person who signs in for the first time gets this role.
* ``EKOS_ADMINS``: comma-separated ``<idp>:<name>`` sign-in names (e.g. ``github:alexeyban``) that
  are made, or kept, ``Admin`` users.
"""

from __future__ import annotations

import os

SQL_LAB_VIEWS = {"SQL Lab", "SQLLab", "Query", "SavedQuery", "TabStateView", "TableSchemaView"}


def public_permissions(gamma: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Gamma's (permission, view) pairs minus SQL Lab, plus Ask (AI). Pure."""
    keep = [(p, v) for p, v in gamma if v not in SQL_LAB_VIEWS and "sql_lab" not in p]
    for pair in (("can_read", "EkosAI"), ("can_write", "EkosAI"), ("can_ask", "EkosAIView"),
                 ("menu_access", "Ask (AI)")):
        if pair not in keep:
            keep.append(pair)
    return keep


def admin_names(raw: str) -> list[str]:
    return [n.strip().lower() for n in raw.split(",") if n.strip()]


def main() -> None:  # pragma: no cover - runs against a real metadata database
    from superset.app import create_app

    app = create_app()
    with app.app_context():
        from superset.extensions import db, security_manager as sm

        role_name = os.environ.get("EKOS_PUBLIC_ROLE", "EkosPublic")
        gamma = sm.find_role("Gamma")
        role = sm.find_role(role_name) or sm.add_role(role_name)
        pairs = public_permissions(
            [(pv.permission.name, pv.view_menu.name) for pv in gamma.permissions]
        )
        # The Iceberg lakehouse through Trino, as a second source for questions (EKOS_TRINO_URI).
        trino_uri = os.environ.get("EKOS_TRINO_URI", "")
        trino_name = os.environ.get("EKOS_TRINO_DATABASE", "Iceberg (Trino)")
        if trino_uri:
            from superset.models.core import Database

            existing = db.session.query(Database).filter_by(database_name=trino_name).one_or_none()
            if existing is None:
                db.session.add(Database(database_name=trino_name, sqlalchemy_uri=trino_uri,
                                        expose_in_sqllab=False, allow_ctas=False, allow_cvas=False,
                                        allow_dml=False))
                db.session.flush()
                print(f"database {trino_name!r} created")
        for name in [os.environ.get("EKOS_AI_DATABASE", "")] + ([trino_name] if trino_uri else []):
            if not name:
                continue
            from superset.models.core import Database

            found = db.session.query(Database).filter_by(database_name=name).one_or_none()
            if found is not None:
                pairs.append(("database_access", found.perm))
            else:
                print(f"database {name!r} not found: the role gets no access to it")
        database = ""
        if database:
            # Not DatabaseDAO: its lookup applies the signed-in user's access filter, and a
            # start-up script has no user (it failed with AttributeError: user, 2026-11-08).
            from superset.models.core import Database

            found = db.session.query(Database).filter_by(database_name=database).one_or_none()
            if found is not None:
                pairs.append(("database_access", found.perm))
            else:
                print(f"database {database!r} not found: the role gets no database access")
        for perm, view in pairs:
            pv = sm.find_permission_view_menu(perm, view) or sm.add_permission_view_menu(perm, view)
            if pv is not None and pv not in role.permissions:
                sm.add_permission_role(role, pv)
        admin = sm.find_role("Admin")
        for name in admin_names(os.environ.get("EKOS_ADMINS", "")):
            user = sm.find_user(username=name)
            if user is None:
                sm.add_user(name, name.split(":", 1)[-1], "admin", f"{name.replace(':', '.')}@admin.invalid", admin)
            elif admin not in user.roles:
                user.roles.append(admin)
        db.session.commit()
        print(f"role {role_name}: {len(role.permissions)} permissions; admins: {os.environ.get('EKOS_ADMINS', '')}")


if __name__ == "__main__":  # pragma: no cover
    main()
