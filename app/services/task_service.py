"""
Task Service - Quản lý background tasks và async processing
"""

import asyncio
import logging
import uuid
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Trạng thái của task"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskType(str, Enum):
    """Loại task"""
    PROCESS_TEXTBOOK = "process_textbook"
    PROCESS_CV = "process_cv"
    CREATE_EMBEDDINGS = "create_embeddings"
    GENERATE_LESSON_PLAN = "generate_lesson_plan"


class TaskService:
    """Service quản lý background tasks"""

    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.processing_tasks = set()
        self._lock = threading.Lock()

    def create_task(
        self, 
        task_type: TaskType, 
        task_data: Dict[str, Any], 
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Tạo task mới
        
        Args:
            task_type: Loại task
            task_data: Dữ liệu task
            task_id: ID tùy chỉnh (optional)
            metadata: Metadata bổ sung (optional)
            
        Returns:
            Task ID
        """
        with self._lock:
            if not task_id:
                task_id = str(uuid.uuid4())

            task = {
                "task_id": task_id,
                "task_type": task_type.value,
                "status": TaskStatus.PENDING,
                "created_at": datetime.now().isoformat(),
                "started_at": None,
                "completed_at": None,
                "progress": 0,
                "message": "Task created",
                "data": task_data,
                "metadata": metadata or {},
                "result": None,
                "error": None,
                "estimated_duration": self._estimate_duration(task_type, task_data)
            }

            self.tasks[task_id] = task
            logger.info(f"Created task {task_id} of type {task_type.value}")

            return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Lấy trạng thái task"""
        with self._lock:
            return self.tasks.get(task_id)

    def get_all_tasks(self, limit: int = 100, status_filter: Optional[TaskStatus] = None) -> Dict[str, Any]:
        """
        Lấy danh sách tất cả tasks
        
        Args:
            limit: Số lượng tasks tối đa
            status_filter: Lọc theo status
            
        Returns:
            Dict chứa thống kê và danh sách tasks
        """
        with self._lock:
            tasks = list(self.tasks.values())
            
            # Filter by status if provided
            if status_filter:
                tasks = [t for t in tasks if t["status"] == status_filter.value]
            
            # Sort by created_at desc
            tasks.sort(key=lambda x: x["created_at"], reverse=True)
            
            # Limit results
            tasks = tasks[:limit]
            
            return {
                "tasks": tasks,
                "total_tasks": len(self.tasks),
                "processing_tasks": len(self.processing_tasks),
                "completed_tasks": len([t for t in self.tasks.values() if t["status"] == TaskStatus.COMPLETED]),
                "failed_tasks": len([t for t in self.tasks.values() if t["status"] == TaskStatus.FAILED]),
                "pending_tasks": len([t for t in self.tasks.values() if t["status"] == TaskStatus.PENDING]),
            }

    def update_task_progress(self, task_id: str, progress: int, message: str = None):
        """Cập nhật tiến độ task"""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id]["progress"] = progress
                if message:
                    self.tasks[task_id]["message"] = message
                logger.info(f"Task {task_id}: {progress}% - {message}")

    def mark_task_processing(self, task_id: str):
        """Đánh dấu task đang xử lý"""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = TaskStatus.PROCESSING
                self.tasks[task_id]["started_at"] = datetime.now().isoformat()
                self.processing_tasks.add(task_id)

    def mark_task_completed(self, task_id: str, result: Dict[str, Any]):
        """Đánh dấu task hoàn thành"""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = TaskStatus.COMPLETED
                self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
                self.tasks[task_id]["progress"] = 100
                self.tasks[task_id]["message"] = "Task completed successfully"
                self.tasks[task_id]["result"] = result
                self.processing_tasks.discard(task_id)

    def mark_task_failed(self, task_id: str, error: str):
        """Đánh dấu task thất bại"""
        with self._lock:
            if task_id in self.tasks:
                self.tasks[task_id]["status"] = TaskStatus.FAILED
                self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
                self.tasks[task_id]["message"] = f"Task failed: {error}"
                self.tasks[task_id]["error"] = error
                self.processing_tasks.discard(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """
        Hủy task (chỉ có thể hủy task pending)
        
        Returns:
            True nếu hủy thành công, False nếu không thể hủy
        """
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task["status"] == TaskStatus.PENDING:
                    task["status"] = TaskStatus.FAILED
                    task["completed_at"] = datetime.now().isoformat()
                    task["message"] = "Task cancelled by user"
                    task["error"] = "Cancelled"
                    return True
        return False

    def delete_task(self, task_id: str) -> bool:
        """
        Xóa task (chỉ có thể xóa task completed hoặc failed)
        
        Returns:
            True nếu xóa thành công, False nếu không thể xóa
        """
        with self._lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task["status"] in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    del self.tasks[task_id]
                    self.processing_tasks.discard(task_id)
                    return True
        return False

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Dọn dẹp tasks cũ"""
        from datetime import datetime, timedelta

        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        tasks_to_remove = []

        with self._lock:
            for task_id, task in self.tasks.items():
                created_at = datetime.fromisoformat(task["created_at"])
                if created_at < cutoff_time and task["status"] in [
                    TaskStatus.COMPLETED,
                    TaskStatus.FAILED,
                ]:
                    tasks_to_remove.append(task_id)

            for task_id in tasks_to_remove:
                del self.tasks[task_id]
                self.processing_tasks.discard(task_id)

        logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")

    def _estimate_duration(self, task_type: TaskType, task_data: Dict[str, Any]) -> str:
        """Ước tính thời gian xử lý"""
        if task_type == TaskType.PROCESS_TEXTBOOK:
            file_size = len(task_data.get("file_content", b""))
            if file_size < 1024 * 1024:  # < 1MB
                return "1-2 minutes"
            elif file_size < 10 * 1024 * 1024:  # < 10MB
                return "2-5 minutes"
            else:
                return "5-10 minutes"
        elif task_type == TaskType.PROCESS_CV:
            return "30-60 seconds"
        elif task_type == TaskType.CREATE_EMBEDDINGS:
            return "1-3 minutes"
        elif task_type == TaskType.GENERATE_LESSON_PLAN:
            return "2-4 minutes"
        else:
            return "Unknown"

    def get_task_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê tasks"""
        with self._lock:
            total = len(self.tasks)
            if total == 0:
                return {
                    "total_tasks": 0,
                    "status_distribution": {},
                    "type_distribution": {},
                    "average_processing_time": 0
                }

            status_counts = {}
            type_counts = {}
            processing_times = []

            for task in self.tasks.values():
                # Status distribution
                status = task["status"]
                status_counts[status] = status_counts.get(status, 0) + 1

                # Type distribution
                task_type = task["task_type"]
                type_counts[task_type] = type_counts.get(task_type, 0) + 1

                # Processing time calculation
                if task["started_at"] and task["completed_at"]:
                    start = datetime.fromisoformat(task["started_at"])
                    end = datetime.fromisoformat(task["completed_at"])
                    duration = (end - start).total_seconds()
                    processing_times.append(duration)

            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0

            return {
                "total_tasks": total,
                "status_distribution": status_counts,
                "type_distribution": type_counts,
                "average_processing_time": round(avg_processing_time, 2),
                "processing_tasks_count": len(self.processing_tasks)
            }


# Singleton instance
task_service = TaskService()
