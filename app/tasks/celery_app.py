from celery import Celery

from app.core.config import get_settings

settings = get_settings()

celery_app = Celery("archivist", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_track_started=True,
    timezone="UTC",
    # Publishing a task opens a broker connection. Celery's defaults retry with
    # backoff for ~19s before giving up, which blocks the HTTP request that is
    # enqueueing the job. Background processing is optional here, so fail fast
    # and let the caller carry on instead of stalling the upload response.
    broker_connection_timeout=2.0,
    broker_connection_retry=False,
    broker_connection_retry_on_startup=False,
    broker_transport_options={"max_retries": 0},
)
