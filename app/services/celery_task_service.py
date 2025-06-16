"""
Celery Task Service - Service tích hợp Celery với background task processing
Thay thế asyncio.create_task bằng Celery tasks để chạy trong background workers
"""

import logging
import time
from typing import Dict, Any, Optional, List
from celery.exceptions import WorkerLostError, Retry
from app.core.celery_app import celery_app
from app.services.mongodb_task_service import mongodb_task_service, TaskType

logger = logging.getLogger(__name__)


class CeleryTaskService:
    """Service quản lý Celery tasks cho PlanBookAI"""

    def __init__(self):
        self.celery_app = celery_app

    async def create_and_dispatch_task(
        self, task_type: str, task_data: Dict[str, Any], task_id: Optional[str] = None
    ) -> str:
        """
        Tạo task trong MongoDB và dispatch đến Celery worker

        Args:
            task_type: Loại task (quick_analysis, process_textbook, etc.)
            task_data: Dữ liệu task
            task_id: ID task (optional, sẽ tự tạo nếu không có)

        Returns:
            str: Task ID
        """
        try:
            # Tạo task trong MongoDB
            if not task_id:
                task_id = await mongodb_task_service.create_task(
                    task_type=TaskType(task_type), task_data=task_data
                )

            # Dispatch task đến Celery worker dựa trên task_type
            celery_task_name = self._get_celery_task_name(task_type)

            if celery_task_name:
                # Gửi task đến Celery worker
                celery_result = self.celery_app.send_task(
                    celery_task_name,
                    args=[task_id],
                    queue=self._get_queue_for_task(task_type),
                )

                logger.info(
                    f"Dispatched task {task_id} to Celery worker: {celery_result.id}"
                )
            else:
                logger.warning(
                    f"No Celery task mapping found for task_type: {task_type}"
                )

            return task_id

        except Exception as e:
            logger.error(f"Error creating and dispatching task: {e}")
            if task_id:
                await mongodb_task_service.mark_task_failed(task_id, str(e))
            raise

    def _get_celery_task_name(self, task_type: str) -> Optional[str]:
        """
        Mapping từ task_type sang Celery task name

        Args:
            task_type: Loại task

        Returns:
            str: Tên Celery task hoặc None nếu không tìm thấy
        """
        task_mapping = {
            "quick_analysis": "app.tasks.pdf_tasks.process_pdf_quick_analysis",
            "process_textbook": "app.tasks.pdf_tasks.process_pdf_textbook",
            "process_textbook_auto": "app.tasks.pdf_tasks.process_pdf_textbook_auto",
            "process_cv": "app.tasks.cv_tasks.process_cv_task",
            "create_embeddings": "app.tasks.embeddings_tasks.create_embeddings_task",
            "update_embeddings": "app.tasks.embeddings_tasks.update_embeddings_task",
        }

        return task_mapping.get(task_type)

    def _get_queue_for_task(self, task_type: str) -> str:
        """
        Xác định queue cho task dựa trên task_type

        Args:
            task_type: Loại task

        Returns:
            str: Tên queue
        """
        queue_mapping = {
            "quick_analysis": "pdf_queue",
            "process_textbook": "pdf_queue",
            "process_textbook_auto": "pdf_queue",
            "process_cv": "cv_queue",
            "create_embeddings": "embeddings_queue",
            "update_embeddings": "embeddings_queue",
        }

        return queue_mapping.get(task_type, "default")

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy status của task từ MongoDB

        Args:
            task_id: ID của task

        Returns:
            Dict: Thông tin task hoặc None nếu không tìm thấy
        """
        return await mongodb_task_service.get_task_status(task_id)

    async def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy kết quả của task từ MongoDB

        Args:
            task_id: ID của task

        Returns:
            Dict: Kết quả task hoặc None nếu không tìm thấy
        """
        task = await mongodb_task_service.get_task_status(task_id)
        if task and task.get("status") == "completed":
            return task.get("result")
        return None

    def get_celery_task_info(self, celery_task_id: str) -> Dict[str, Any]:
        """
        Lấy thông tin task từ Celery result backend

        Args:
            celery_task_id: ID của Celery task

        Returns:
            Dict: Thông tin task từ Celery
        """
        try:
            result = self.celery_app.AsyncResult(celery_task_id)
            return {
                "id": result.id,
                "status": result.status,
                "result": result.result if result.ready() else None,
                "traceback": result.traceback,
                "info": result.info,
            }
        except Exception as e:
            logger.error(f"Error getting Celery task info: {e}")
            return {"error": str(e)}

    async def cancel_task(self, task_id: str) -> bool:
        """
        Hủy task (cả MongoDB và Celery)

        Args:
            task_id: ID của task

        Returns:
            bool: True nếu thành công
        """
        try:
            # Cập nhật status trong MongoDB
            await mongodb_task_service.mark_task_failed(
                task_id, "Task cancelled by user"
            )

            # TODO: Implement Celery task cancellation if needed
            # Celery không hỗ trợ cancel task đang chạy một cách dễ dàng

            return True
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False

    def _is_valid_task_type(self, task_type: str) -> bool:
        """Kiểm tra task_type có hợp lệ không"""
        valid_types = {
            "quick_analysis",
            "process_textbook",
            "process_textbook_auto",
            "process_cv",
            "create_embeddings",
            "update_embeddings"
        }
        return task_type in valid_types

    def _check_worker_availability(self) -> bool:
        """Kiểm tra có worker nào đang active không"""
        try:
            inspect = self.celery_app.control.inspect()
            active_workers = inspect.active()
            return bool(active_workers)
        except Exception as e:
            logger.warning(f"Cannot check worker availability: {e}")
            return True  # Assume workers are available

    async def get_worker_stats(self) -> Dict[str, Any]:
        """Lấy thống kê workers"""
        try:
            inspect = self.celery_app.control.inspect()

            stats = {
                "active_workers": 0,
                "active_tasks": 0,
                "queues": {},
                "workers": {}
            }

            # Active workers và tasks
            active = inspect.active()
            if active:
                stats["active_workers"] = len(active)
                for worker, tasks in active.items():
                    stats["active_tasks"] += len(tasks)
                    stats["workers"][worker] = {
                        "active_tasks": len(tasks),
                        "tasks": [task["name"] for task in tasks]
                    }

            # Queue lengths (nếu có thể lấy được)
            try:
                reserved = inspect.reserved()
                if reserved:
                    for worker, tasks in reserved.items():
                        if worker in stats["workers"]:
                            stats["workers"][worker]["reserved_tasks"] = len(tasks)
            except:
                pass

            return stats

        except Exception as e:
            logger.error(f"Error getting worker stats: {e}")
            return {"error": str(e)}

    async def health_check(self) -> Dict[str, Any]:
        """Health check cho Celery service"""
        try:
            # Test Redis connection
            import redis
            from app.core.config import settings

            r = redis.from_url(settings.CELERY_BROKER_URL)
            r.ping()
            redis_ok = True
        except Exception as e:
            redis_ok = False
            redis_error = str(e)

        # Test worker availability
        workers_ok = self._check_worker_availability()

        # Test task dispatch
        try:
            result = self.celery_app.send_task('app.tasks.health_check')
            dispatch_ok = True
            test_task_id = result.id
        except Exception as e:
            dispatch_ok = False
            dispatch_error = str(e)
            test_task_id = None

        return {
            "status": "healthy" if (redis_ok and workers_ok and dispatch_ok) else "unhealthy",
            "redis_connection": redis_ok,
            "redis_error": redis_error if not redis_ok else None,
            "workers_available": workers_ok,
            "task_dispatch": dispatch_ok,
            "dispatch_error": dispatch_error if not dispatch_ok else None,
            "test_task_id": test_task_id,
            "timestamp": time.time()
        }


# Singleton instance
celery_task_service = CeleryTaskService()
