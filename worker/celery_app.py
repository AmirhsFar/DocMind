from celery import Celery

from api.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "docmind",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)


@celery_app.task(name="worker.ping")
def ping() -> str:
    """Sanity-check task: run `celery -A worker.celery_app call worker.ping`
    or trigger it from a Python shell to confirm the worker is processing jobs.

    Real ingestion tasks (extract -> chunk -> embed) arrive in Phase 3.
    """
    return "pong"
