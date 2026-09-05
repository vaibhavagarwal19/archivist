"""Startup database preparation.

Schema is owned by Alembic (`alembic upgrade head`), not by create_all().
Creating tables implicitly at startup meant the schema drifted silently and
there was no way to evolve a deployed database. Startup only verifies that the
database is reachable and up to date, and says so loudly when it is not.
"""
import logging
from pathlib import Path

from sqlalchemy import inspect

from app.core.config import get_settings
from app.db.session import engine

logger = logging.getLogger(__name__)


def init_db() -> None:
    settings = get_settings()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)

    tables = set(inspect(engine).get_table_names())
    expected = {"users", "documents", "reports", "workflow_actions"}
    missing = expected - tables

    if missing:
        logger.warning(
            "Database is missing tables %s. Run `alembic upgrade head` to create the schema.",
            ", ".join(sorted(missing)),
        )
    elif "alembic_version" not in tables:
        logger.warning(
            "Schema exists but is not tracked by Alembic. Run `alembic stamp head` to adopt it."
        )
