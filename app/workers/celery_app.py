"""
Minimal Celery Application for PlanBookAI
"""
from celery import Celery
from app.core.config import settings

# Create Celery instance
celery_app = Celery(
    "planbook_ai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

# Basic configuration
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
)
