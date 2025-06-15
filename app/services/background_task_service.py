"""
Background Task Service - Xử lý PDF bất đồng bộ
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum
import json

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Trạng thái của task"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BackgroundTaskService:
    """Service quản lý background tasks"""

    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.processing_tasks = set()

    def create_task(
        self, task_type: str, task_data: Dict[str, Any], task_id: Optional[str] = None
    ) -> str:
        """Tạo task mới"""

        if not task_id:
            task_id = str(uuid.uuid4())

        task = {
            "task_id": task_id,
            "task_type": task_type,
            "status": TaskStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "progress": 0,
            "message": "Task created",
            "data": task_data,
            "result": None,
            "error": None,
        }

        self.tasks[task_id] = task
        logger.info(f"Created task {task_id} of type {task_type}")

        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Lấy trạng thái task"""
        return self.tasks.get(task_id)

    def update_task_progress(self, task_id: str, progress: int, message: str = None):
        """Cập nhật tiến độ task"""
        if task_id in self.tasks:
            self.tasks[task_id]["progress"] = progress
            if message:
                self.tasks[task_id]["message"] = message
            logger.info(f"Task {task_id}: {progress}% - {message}")

    def mark_task_processing(self, task_id: str):
        """Đánh dấu task đang xử lý"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = TaskStatus.PROCESSING
            self.tasks[task_id]["started_at"] = datetime.now().isoformat()
            self.processing_tasks.add(task_id)

    def mark_task_completed(self, task_id: str, result: Dict[str, Any]):
        """Đánh dấu task hoàn thành"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = TaskStatus.COMPLETED
            self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
            self.tasks[task_id]["progress"] = 100
            self.tasks[task_id]["message"] = "Task completed successfully"
            self.tasks[task_id]["result"] = result
            self.processing_tasks.discard(task_id)

    def mark_task_failed(self, task_id: str, error: str):
        """Đánh dấu task thất bại"""
        if task_id in self.tasks:
            self.tasks[task_id]["status"] = TaskStatus.FAILED
            self.tasks[task_id]["completed_at"] = datetime.now().isoformat()
            self.tasks[task_id]["message"] = f"Task failed: {error}"
            self.tasks[task_id]["error"] = error
            self.processing_tasks.discard(task_id)

    async def process_pdf_task(self, task_id: str):
        """Xử lý PDF task bất đồng bộ"""

        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        try:
            self.mark_task_processing(task_id)

            # Lấy dữ liệu task
            file_content = task["data"]["file_content"]
            filename = task["data"]["filename"]
            metadata = task["data"]["metadata"]
            create_embeddings = task["data"].get("create_embeddings", True)

            self.update_task_progress(task_id, 10, "Starting PDF processing...")

            # Import services
            from app.services.enhanced_textbook_service import enhanced_textbook_service
            from app.services.qdrant_service import qdrant_service

            # Bước 1: OCR và phân tích cấu trúc
            self.update_task_progress(task_id, 20, "Extracting text with OCR...")

            enhanced_result = (
                await textbook_service.process_textbook_to_structure(
                    pdf_content=file_content, filename=filename, book_metadata=metadata
                )
            )

            if not enhanced_result.get("success"):
                raise Exception(
                    f"PDF processing failed: {enhanced_result.get('error')}"
                )

            self.update_task_progress(task_id, 60, "PDF structure analysis completed")

            # Bước 2: Tạo embeddings nếu được yêu cầu
            embeddings_result = None
            if create_embeddings:
                self.update_task_progress(task_id, 70, "Creating embeddings...")

                book_structure_dict = enhanced_result["book"]
                if isinstance(book_structure_dict, str):
                    book_structure_dict = json.loads(book_structure_dict)

                embeddings_result = await qdrant_service.process_textbook(
                    book_id=metadata.get("id"), book_structure=book_structure_dict
                )

                if embeddings_result.get("success"):
                    self.update_task_progress(
                        task_id, 90, "Embeddings created successfully"
                    )
                else:
                    logger.warning(
                        f"Embeddings creation failed: {embeddings_result.get('error')}"
                    )

            # Tạo kết quả cuối cùng
            result = {
                "success": True,
                "book_id": metadata.get("id"),
                "filename": filename,
                "book_structure": enhanced_result["book"],
                "statistics": {
                    "total_pages": enhanced_result.get("total_pages", 0),
                    "total_chapters": enhanced_result.get("total_chapters", 0),
                    "total_lessons": enhanced_result.get("total_lessons", 0),
                },
                "processing_info": {
                    "ocr_applied": True,
                    "llm_analysis": True,
                    "processing_method": "enhanced_ocr_async",
                },
                "embeddings_created": embeddings_result.get("success", False)
                if embeddings_result
                else False,
                "embeddings_info": {
                    "collection_name": embeddings_result.get("collection_name")
                    if embeddings_result
                    else None,
                    "vector_count": embeddings_result.get("total_chunks", 0)
                    if embeddings_result
                    else 0,
                    "vector_dimension": embeddings_result.get("vector_dimension")
                    if embeddings_result
                    else None,
                },
            }

            self.mark_task_completed(task_id, result)

        except Exception as e:
            logger.error(f"Error processing PDF task {task_id}: {e}")
            self.mark_task_failed(task_id, str(e))

    def get_all_tasks(self) -> Dict[str, Any]:
        """Lấy danh sách tất cả tasks"""
        return {
            "tasks": list(self.tasks.values()),
            "total_tasks": len(self.tasks),
            "processing_tasks": len(self.processing_tasks),
            "completed_tasks": len(
                [t for t in self.tasks.values() if t["status"] == TaskStatus.COMPLETED]
            ),
            "failed_tasks": len(
                [t for t in self.tasks.values() if t["status"] == TaskStatus.FAILED]
            ),
        }

    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Dọn dẹp tasks cũ"""
        from datetime import datetime, timedelta

        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        tasks_to_remove = []

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


# Singleton instance
background_task_service = BackgroundTaskService()
