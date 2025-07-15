"""
Celery Application Configuration for PlanBookAI
Cấu hình Celery chuẩn chỉnh với Redis broker và auto-discovery tasks
"""

import os
import sys
from celery import Celery
from kombu import Queue

# Add project root to Python path for imports
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.core.config import settings

# Tạo Celery instance
celery_app = Celery("planbook_ai")

# Debug logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"🔧 Celery app created with broker: {settings.CELERY_BROKER_URL}")
logger.info(f"🔧 Celery result backend: {settings.CELERY_RESULT_BACKEND}")

# Cấu hình Celery từ settings
celery_app.conf.update(
    # Broker và Result Backend
    broker_url=settings.CELERY_BROKER_URL,
    result_backend=settings.CELERY_RESULT_BACKEND,
    # Serialization
    task_serializer=settings.CELERY_TASK_SERIALIZER,
    result_serializer=settings.CELERY_RESULT_SERIALIZER,
    accept_content=settings.CELERY_ACCEPT_CONTENT,
    # Timezone
    timezone=settings.CELERY_TIMEZONE,
    enable_utc=settings.CELERY_ENABLE_UTC,
    # Task routing và queues
    task_routes={
        "app.tasks.pdf_tasks.*": {"queue": "pdf_queue"},
        "app.tasks.embeddings_tasks.*": {"queue": "embeddings_queue"},
        "app.tasks.cv_tasks.*": {"queue": "cv_queue"},
        "app.tasks.slide_generation_tasks.*": {"queue": "slide_generation_queue"},
    },
    # Định nghĩa queues
    task_queues=(
        Queue("default", routing_key="default"),
        Queue("pdf_queue", routing_key="pdf_queue"),
        Queue("embeddings_queue", routing_key="embeddings_queue"),
        Queue("cv_queue", routing_key="cv_queue"),
        Queue("slide_generation_queue", routing_key="slide_generation_queue"),
    ),
    # Task execution settings - Đơn giản hóa để tránh lỗi
    task_acks_late=False,  # Đổi thành False để tránh lỗi
    worker_prefetch_multiplier=1,
    task_reject_on_worker_lost=False,  # Đổi thành False
    # Result settings
    result_expires=3600,  # 1 hour
    result_persistent=False,  # Đổi thành False để tránh lỗi
    # Task time limits
    task_soft_time_limit=1800,  # 30 minutes
    task_time_limit=2400,  # 40 minutes
    # Worker settings
    worker_max_tasks_per_child=1000,
    worker_disable_rate_limits=True,
    # Monitoring - Tắt để tránh lỗi
    worker_send_task_events=False,  # Đổi thành False
    task_send_sent_event=False,  # Đổi thành False
    # Include tasks - auto-discovery
    include=[
        "app.tasks.pdf_tasks",
        "app.tasks.embeddings_tasks",
        "app.tasks.cv_tasks",
        "app.tasks.lesson_plan_tasks",
        "app.tasks.smart_exam_tasks",
        "app.tasks.slide_generation_tasks",
    ],
)

# Auto-discover tasks từ installed apps
celery_app.autodiscover_tasks(
    [
        "app.tasks",
    ]
)


# Task decorator với default settings
def task_with_retry(
    bind=True,
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 3, "countdown": 60},
):
    """Decorator cho tasks với retry logic"""

    def decorator(func):
        return celery_app.task(
            bind=bind, autoretry_for=autoretry_for, retry_kwargs=retry_kwargs
        )(func)

    return decorator


# Health check task
@celery_app.task(name="app.tasks.health_check")
def health_check():
    """Health check task để test Celery worker"""
    return {
        "status": "healthy",
        "message": "Celery worker is running",
        "broker": settings.CELERY_BROKER_URL,
        "backend": settings.CELERY_RESULT_BACKEND,
    }


# Task để test connection
@celery_app.task(name="app.tasks.test_task")
def test_task(message: str = "Hello from Celery!"):
    """Simple test task"""
    return {
        "success": True,
        "message": message,
        "worker_info": {
            "broker": celery_app.conf.broker_url,
            "backend": celery_app.conf.result_backend,
        },
    }


if __name__ == "__main__":
    celery_app.start()
