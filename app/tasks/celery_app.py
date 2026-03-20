from celery import Celery

from app.core.config import get_settings


settings = get_settings()

celery_app = Celery("internal_ai_agents", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(task_track_started=True, timezone="UTC")
