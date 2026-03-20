from pathlib import Path

from app.core.config import get_settings
from app.db.base import Base
from app.db.session import engine
from app.models import document, report, user, workflow_action  # noqa: F401


def init_db() -> None:
    settings = get_settings()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.faiss_index_path).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
