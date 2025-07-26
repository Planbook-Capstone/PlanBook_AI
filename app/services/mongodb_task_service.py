"""
MongoDB Task Service - Quản lý background tasks với MongoDB persistence
"""

import asyncio
import logging
import uuid
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from enum import Enum
import json

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING
from app.core.config import settings

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
    PROCESS_TEXTBOOK_AUTO = "process_textbook_auto"
    QUICK_ANALYSIS = "quick_analysis"
    PROCESS_CV = "process_cv"
    CREATE_EMBEDDINGS = "create_embeddings"
    GENERATE_LESSON_PLAN = "generate_lesson_plan"
    LESSON_PLAN_CONTENT_GENERATION = "lesson_plan_content_generation"
    SMART_EXAM_GENERATION = "smart_exam_generation"
    SLIDE_GENERATION = "slide_generation"
    JSON_TEMPLATE_PROCESSING = "json_template_processing"
    GUIDE_IMPORT = "guide_import"


class MongoDBTaskService:
    """Service quản lý background tasks với MongoDB persistence"""

    def __init__(self):
        self.client = None
        self.db = None
        self.tasks_collection = None
        self.processing_tasks = set()  # Keep in-memory for performance
        self._lock = threading.Lock()
        self._initialized = False

        # Task status cache để tránh query liên tục
        self._task_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timeout = 0
        self._cache_lock = threading.Lock()

    def _get_cached_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Lấy task từ cache nếu còn hợp lệ"""
        with self._cache_lock:
            cached = self._task_cache.get(task_id)
            if (
                cached
                and time.time() - cached.get("cached_at", 0) < self._cache_timeout
            ):
                return cached.get("data")
        return None

    def _cache_task(self, task_id: str, task_data: Dict[str, Any]):
        """Cache task data"""
        with self._cache_lock:
            self._task_cache[task_id] = {
                "data": task_data.copy(),
                "cached_at": time.time(),
            }

    def _clear_task_cache(self, task_id: str):
        """Xóa cache của task khi có update"""
        with self._cache_lock:
            self._task_cache.pop(task_id, None)

    async def initialize(self):
        """Khởi tạo kết nối MongoDB"""
        if self._initialized:
            return

        try:
            # Tối ưu connection pooling
            self.client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                maxPoolSize=20,  # Tăng connection pool
                minPoolSize=5,
                maxIdleTimeMS=30000,  # 30 seconds
                waitQueueTimeoutMS=5000,  # 5 seconds timeout
                connectTimeoutMS=5000,  # 5 seconds connect timeout
                serverSelectionTimeoutMS=5000,  # 5 seconds server selection timeout
            )
            self.db = self.client[settings.MONGODB_DATABASE]
            self.tasks_collection = self.db.tasks

            # Tạo indexes để tối ưu performance
            await self.tasks_collection.create_index("task_id", unique=True)
            await self.tasks_collection.create_index("status")
            await self.tasks_collection.create_index("task_type")
            await self.tasks_collection.create_index("created_at")

            self._initialized = True
            logger.info("MongoDB Task Service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize MongoDB Task Service: {e}")
            raise

    async def create_task(
        self,
        task_type: TaskType,
        task_data: Dict[str, Any],
        task_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Tạo task mới và lưu vào MongoDB"""
        await self.initialize()

        if not task_id:
            task_id = str(uuid.uuid4())

        # Tạo initial progress step
        initial_progress_step = {
            "progress": 0,
            "message": "Task created",
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat()
        }

        task = {
            "task_id": task_id,
            "task_type": task_type.value,
            "status": TaskStatus.PENDING.value,
            "created_at": datetime.now(),
            "started_at": None,
            "completed_at": None,
            "updated_at": datetime.now(),
            "progress": 0,
            "message": "Task created",
            "data": task_data,
            "metadata": metadata or {},
            "result": None,
            "error": None,
            "estimated_duration": self._estimate_duration(task_type, task_data),
            "progress_history": [initial_progress_step],
        }

        try:
            await self.tasks_collection.insert_one(task)
            logger.info(f"Created task {task_id} of type {task_type.value}")
            return task_id
        except Exception as e:
            logger.error(f"Error creating task {task_id}: {e}")
            raise

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Lấy trạng thái task từ MongoDB"""
        await self.initialize()

        # Kiểm tra cache trước - nhưng không cache task đang processing hoặc pending để đảm bảo real-time updates
        cached_task = self._get_cached_task(task_id)
        if cached_task and cached_task.get("status") not in [TaskStatus.PROCESSING.value, TaskStatus.PENDING.value]:
            return cached_task

        try:
            task = await self.tasks_collection.find_one({"task_id": task_id})
            if task:
                # Convert MongoDB ObjectId to string và datetime to ISO string
                task["_id"] = str(task["_id"])
                if task["created_at"]:
                    task["created_at"] = task["created_at"].isoformat()
                if task["started_at"]:
                    task["started_at"] = task["started_at"].isoformat()
                if task["completed_at"]:
                    task["completed_at"] = task["completed_at"].isoformat()

                # Cache task status - nhưng không cache task đang processing hoặc pending để đảm bảo real-time updates
                if task.get("status") not in [TaskStatus.PROCESSING.value, TaskStatus.PENDING.value]:
                    self._cache_task(task_id, task)

            return task
        except Exception as e:
            logger.error(f"Error getting task {task_id}: {e}")
            return None

    async def get_all_tasks(
        self, limit: int = 100, status_filter: Optional[TaskStatus] = None
    ) -> Dict[str, Any]:
        """Lấy danh sách tất cả tasks từ MongoDB"""
        await self.initialize()

        try:
            # Build query filter
            query = {}
            if status_filter:
                query["status"] = status_filter.value

            # Get tasks with pagination and sorting
            cursor = (
                self.tasks_collection.find(query)
                .sort("created_at", DESCENDING)
                .limit(limit)
            )
            tasks = await cursor.to_list(length=limit)

            # Convert MongoDB documents to JSON-serializable format
            for task in tasks:
                task["_id"] = str(task["_id"])
                if task["created_at"]:
                    task["created_at"] = task["created_at"].isoformat()
                if task["started_at"]:
                    task["started_at"] = task["started_at"].isoformat()
                if task["completed_at"]:
                    task["completed_at"] = task["completed_at"].isoformat()

            # Get statistics
            total_tasks = await self.tasks_collection.count_documents({})
            processing_tasks = await self.tasks_collection.count_documents(
                {"status": TaskStatus.PROCESSING.value}
            )
            completed_tasks = await self.tasks_collection.count_documents(
                {"status": TaskStatus.COMPLETED.value}
            )
            failed_tasks = await self.tasks_collection.count_documents(
                {"status": TaskStatus.FAILED.value}
            )
            pending_tasks = await self.tasks_collection.count_documents(
                {"status": TaskStatus.PENDING.value}
            )

            return {
                "tasks": tasks,
                "total_tasks": total_tasks,
                "processing_tasks": processing_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "pending_tasks": pending_tasks,
            }
        except Exception as e:
            logger.error(f"Error getting all tasks: {e}")
            return {
                "tasks": [],
                "total_tasks": 0,
                "processing_tasks": 0,
                "completed_tasks": 0,
                "failed_tasks": 0,                "pending_tasks": 0,
            }

    async def update_task_progress(
        self, task_id: str, progress: int, message: Optional[str] = None
    ):
        """Cập nhật tiến độ task và lưu progress history, đồng thời gửi qua Kafka"""
        await self.initialize()

        try:
            # Tạo progress step cho history
            progress_step = {
                "progress": progress,
                "message": message or f"Progress: {progress}%",
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat()
            }

            update_data = {
                "progress": progress,
                "updated_at": datetime.now()
            }
            if message:
                update_data["message"] = message

            # Update task và thêm vào progress_history
            await self.tasks_collection.update_one(
                {"task_id": task_id},
                {
                    "$set": update_data,
                    "$push": {"progress_history": progress_step}
                }
            )

            # Clear cache để force refresh lần query sau
            self._clear_task_cache(task_id)

            # Skip Kafka notification from MongoDB service
            # Kafka notifications will be handled directly in Celery tasks

            logger.info(f"Task {task_id}: {progress}% - {message}")
        except Exception as e:
            logger.error(f"Error updating task progress {task_id}: {e}")

    async def update_task_progress_with_result(
        self, task_id: str, progress: int, message: Optional[str] = None, partial_result: Optional[Dict[str, Any]] = None
    ):
        """Cập nhật tiến độ task và result từng phần (cho slide generation)"""
        await self.initialize()

        try:
            # Tạo progress step cho history
            progress_step = {
                "progress": progress,
                "message": message or f"Progress: {progress}%",
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat()
            }

            update_data = {
                "progress": progress,
                "updated_at": datetime.now()
            }
            if message:
                update_data["message"] = message

            # Cập nhật result từng phần nếu có
            if partial_result is not None:
                update_data["result"] = partial_result
                logger.info(f"🔄 Setting partial result for task {task_id} with {len(partial_result.get('processed_template', {}).get('slides', []))} slides")

            # Update task và thêm vào progress_history
            result = await self.tasks_collection.update_one(
                {"task_id": task_id},
                {
                    "$set": update_data,
                    "$push": {"progress_history": progress_step}
                }
            )

            # Clear cache để force refresh lần query sau
            self._clear_task_cache(task_id)

            logger.info(f"Task {task_id}: {progress}% - {message}")
            logger.info(f"MongoDB update result: modified_count={result.modified_count}")

            if partial_result:
                slides_count = len(partial_result.get("processed_template", {}).get("slides", []))
                logger.info(f"✅ Task {task_id}: Successfully updated with {slides_count} completed slides in result field")

        except Exception as e:
            logger.error(f"Error updating task progress with result {task_id}: {e}")

    async def mark_task_processing(self, task_id: str):
        """Đánh dấu task đang xử lý và lưu vào history"""
        await self.initialize()

        try:
            # Tạo progress step cho processing
            progress_step = {
                "progress": 5,
                "message": "Task started processing",
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat()
            }
            
            await self.tasks_collection.update_one(
                {"task_id": task_id},
                {
                    "$set": {
                        "status": TaskStatus.PROCESSING.value,
                        "started_at": datetime.now(),
                        "updated_at": datetime.now(),
                        "progress": 5,
                        "message": "Processing started"
                    },
                    "$push": {"progress_history": progress_step}
                },
            )

            with self._lock:
                self.processing_tasks.add(task_id)
                
            # Clear cache
            self._clear_task_cache(task_id)

            # Skip Kafka notification from MongoDB service
            # Kafka notifications will be handled directly in Celery tasks
            logger.info(f"✅ Task {task_id} marked as processing in MongoDB")

        except Exception as e:
            logger.error(f"Error marking task processing {task_id}: {e}")

    async def mark_task_completed(self, task_id: str, result: Dict[str, Any]):
        """Đánh dấu task hoàn thành"""
        await self.initialize()

        try:
            # Tạo progress step cho completion
            progress_step = {
                "progress": 100,
                "message": "Task completed successfully",
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat()
            }
            
            await self.tasks_collection.update_one(
                {"task_id": task_id},
                {
                    "$set": {
                        "status": TaskStatus.COMPLETED.value,
                        "completed_at": datetime.now(),
                        "updated_at": datetime.now(),
                        "progress": 100,
                        "message": "Task completed successfully",
                        "result": result,
                    },
                    "$push": {"progress_history": progress_step}
                },
            )

            with self._lock:
                self.processing_tasks.discard(task_id)

            # Xóa cache khi task hoàn thành
            self._clear_task_cache(task_id)

            # Skip Kafka notification from MongoDB service
            # Kafka notifications will be handled directly in Celery tasks
            logger.info(f"✅ Task {task_id} marked as completed in MongoDB")

        except Exception as e:
            logger.error(f"Error marking task completed {task_id}: {e}")

    async def mark_task_failed(self, task_id: str, error: str):
        """Đánh dấu task thất bại và lưu vào history"""
        await self.initialize()

        try:
            # Tạo progress step cho failure
            progress_step = {
                "progress": -1,  # -1 để đánh dấu failed
                "message": f"Task failed: {error}",
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat()
            }
            
            await self.tasks_collection.update_one(
                {"task_id": task_id},
                {
                    "$set": {
                        "status": TaskStatus.FAILED.value,
                        "completed_at": datetime.now(),
                        "updated_at": datetime.now(),
                        "message": f"Task failed: {error}",
                        "error": error,
                    },
                    "$push": {"progress_history": progress_step}
                },
            )

            with self._lock:
                self.processing_tasks.discard(task_id)

            # Xóa cache khi task thất bại
            self._clear_task_cache(task_id)

            # Skip Kafka notification from MongoDB service
            # Kafka notifications will be handled directly in Celery tasks
            logger.info(f"✅ Task {task_id} marked as failed in MongoDB")

        except Exception as e:
            logger.error(f"Error marking task failed {task_id}: {e}")

    async def cancel_task(self, task_id: str) -> bool:
        """Hủy task (chỉ có thể hủy task pending)"""
        await self.initialize()

        try:
            result = await self.tasks_collection.update_one(
                {"task_id": task_id, "status": TaskStatus.PENDING.value},
                {
                    "$set": {
                        "status": TaskStatus.FAILED.value,
                        "completed_at": datetime.now(),
                        "message": "Task cancelled by user",
                        "error": "Cancelled",
                    }
                },
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False

    async def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Dọn dẹp tasks cũ"""
        await self.initialize()

        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

            result = await self.tasks_collection.delete_many(
                {
                    "created_at": {"$lt": cutoff_time},
                    "status": {
                        "$in": [TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]
                    },
                }
            )

            logger.info(f"Cleaned up {result.deleted_count} old tasks")
            return result.deleted_count
        except Exception as e:
            logger.error(f"Error cleaning up old tasks: {e}")
            return 0

    def _estimate_duration(self, task_type: TaskType, task_data: Dict[str, Any]) -> str:
        """Ước tính thời gian xử lý"""
        if task_type in [TaskType.PROCESS_TEXTBOOK, TaskType.PROCESS_TEXTBOOK_AUTO]:
            file_size = len(task_data.get("file_content", b""))
            if file_size < 1024 * 1024:  # < 1MB
                return "1-2 minutes"
            elif file_size < 10 * 1024 * 1024:  # < 10MB
                return "2-5 minutes"
            else:
                return "5-10 minutes"
        elif task_type == TaskType.QUICK_ANALYSIS:
            file_size = len(task_data.get("file_content", b""))
            if file_size < 1024 * 1024:  # < 1MB
                return "30-60 seconds"
            elif file_size < 10 * 1024 * 1024:  # < 10MB
                return "1-2 minutes"
            else:
                return "2-3 minutes"
        elif task_type == TaskType.PROCESS_CV:
            return "30-60 seconds"
        elif task_type == TaskType.CREATE_EMBEDDINGS:
            return "1-3 minutes"
        elif task_type == TaskType.GENERATE_LESSON_PLAN:
            return "2-4 minutes"
        elif task_type == TaskType.JSON_TEMPLATE_PROCESSING:
            slides_count = len(task_data.get("template_json", {}).get("slides", []))
            if slides_count <= 3:
                return "1-2 minutes"
            elif slides_count <= 6:
                return "2-4 minutes"
            else:
                return "4-8 minutes"
        elif task_type == TaskType.SLIDE_GENERATION:
            return "3-6 minutes"
        else:
            return "Unknown"

    async def get_task_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê tasks"""
        await self.initialize()

        try:
            # Aggregate statistics
            pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]

            status_stats = {}
            async for doc in self.tasks_collection.aggregate(pipeline):
                status_stats[doc["_id"]] = doc["count"]

            # Type distribution
            type_pipeline = [{"$group": {"_id": "$task_type", "count": {"$sum": 1}}}]

            type_stats = {}
            async for doc in self.tasks_collection.aggregate(type_pipeline):
                type_stats[doc["_id"]] = doc["count"]

            # Average processing time
            avg_pipeline = [
                {
                    "$match": {
                        "started_at": {"$ne": None},
                        "completed_at": {"$ne": None},
                    }
                },
                {
                    "$project": {
                        "duration": {"$subtract": ["$completed_at", "$started_at"]}
                    }
                },
                {"$group": {"_id": None, "avg_duration": {"$avg": "$duration"}}},
            ]

            avg_duration = 0
            async for doc in self.tasks_collection.aggregate(avg_pipeline):
                avg_duration = doc["avg_duration"] / 1000  # Convert to seconds

            total_tasks = await self.tasks_collection.count_documents({})

            return {
                "total_tasks": total_tasks,
                "status_distribution": status_stats,
                "type_distribution": type_stats,
                "average_processing_time": round(avg_duration, 2),
                "processing_tasks_count": len(self.processing_tasks),
            }
        except Exception as e:
            logger.error(f"Error getting task statistics: {e}")
            return {
                "total_tasks": 0,
                "status_distribution": {},
                "type_distribution": {},
                "average_processing_time": 0,
                "processing_tasks_count": 0,
            }


# Factory function để tạo MongoDBTaskService instance
def get_mongodb_task_service() -> MongoDBTaskService:
    """
    Tạo MongoDBTaskService instance mới

    Returns:
        MongoDBTaskService: Fresh instance
    """
    return MongoDBTaskService()
