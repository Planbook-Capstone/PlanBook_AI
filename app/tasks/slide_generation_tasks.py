"""
Celery tasks cho JSON Template Processing
Xử lý JSON template bất đồng bộ với progress tracking
"""

import logging
import asyncio
from typing import Dict, Any
from datetime import datetime

from app.core.celery_app import celery_app
from app.services.mongodb_task_service import get_mongodb_task_service
from app.services.json_template_service import get_json_template_service
from app.services.kafka_service import kafka_service, safe_kafka_call
from app.constants.kafka_message_types import PROGRESS_TYPE, RESULT_TYPE

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.slide_generation_tasks.process_json_template_task")
def process_json_template_task(self, task_id: str, lesson_id: str, template_json: Dict[str, Any], config_prompt: str = None, user_id: str = None, book_id: str = None):
    """
    Celery task để xử lý JSON template bất đồng bộ với progress tracking

    Args:
        task_id: ID của task trong MongoDB
        lesson_id: ID của bài học
        template_json: JSON template đã được phân tích sẵn
        config_prompt: Prompt cấu hình tùy chỉnh (optional)
        user_id: ID của user (optional, for Kafka notifications)
        book_id: ID của sách giáo khoa (optional)
    """

    logger.info(f"🚀 CELERY TASK STARTED: process_json_template_task")
    logger.info(f"   Task ID: {task_id}")
    logger.info(f"   Lesson ID: {lesson_id}")
    logger.info(f"   Slides count: {len(template_json.get('slides', []))}")
    logger.info(f"   Config Prompt: {config_prompt}")

    async def _async_process_json_template():
        """Async wrapper cho JSON template processing"""
        logger.info(f"🔄 ASYNC FUNCTION STARTED for task: {task_id}")

        try:
            logger.info("🔄 Getting MongoDB task service...")
            task_service = get_mongodb_task_service()
            logger.info("✅ MongoDB task service obtained")
        except Exception as e:
            logger.error(f"❌ Failed to get MongoDB task service: {e}")
            raise

        try:
            logger.info("🔄 Getting JSON template service...")
            json_service = get_json_template_service()

            if not json_service.is_available():
                raise Exception("JSON template service not available")

            logger.info("✅ JSON template service obtained and available")

            # Xử lý JSON template với progress tracking
            result = await json_service.process_json_template_with_progress(
                lesson_id=lesson_id,
                template_json=template_json,
                config_prompt=config_prompt,
                task_id=task_id,
                task_service=task_service,
                user_id=user_id,
                book_id=book_id
            )

            if result["success"]:
                # Hoàn thành thành công
                final_result = {
                    "success": True,
                    "output": result.get("processed_template", {}),
                    "created_at": datetime.now().isoformat()
                }

                await task_service.mark_task_completed(task_id, result=final_result)

                # Send final Kafka notification
                if user_id:
                    safe_kafka_call(
                        kafka_service.send_final_result_sync,
                        task_id=task_id,
                        user_id=user_id,
                        result=final_result,
                        tool_log_id=None
                    )

                logger.info(f"✅ Task {task_id} hoàn thành thành công")

            else:
                # Xử lý lỗi
                error_message = result.get("error", "Unknown error")
                error_result = {
                    "success": False,
                    "error": f"Lỗi xử lý JSON template: {error_message}",
                    "output": {
                        "task_id": task_id,
                        "error_details": {
                            "error_message": error_message,
                            "task_stage": "json_template_processing"
                        }
                    }
                }

                await task_service.mark_task_failed(task_id, error=f"Lỗi xử lý JSON template: {error_message}")

                # Send Kafka error notification
                if user_id:
                    safe_kafka_call(
                        kafka_service.send_final_result_sync,
                        task_id=task_id,
                        user_id=user_id,
                        result=error_result,
                        tool_log_id=None
                    )

                logger.error(f"❌ Task {task_id} thất bại: {error_message}")

        except Exception as e:
            logger.error(f"❌ Lỗi không mong muốn trong task {task_id}: {e}")

            # Create error result
            error_result = {
                "success": False,
                "error": f"Lỗi hệ thống: {str(e)}",
                "output": {
                    "task_id": task_id,
                    "error_details": {
                        "error_message": str(e),
                        "task_stage": "json_template_system_error"
                    }
                }
            }

            try:
                await task_service.mark_task_failed(
                    task_id,
                    error=f"Lỗi hệ thống: {str(e)}"
                )

                # Send Kafka error notification
                if user_id:
                    safe_kafka_call(
                        kafka_service.send_final_result_sync,
                        task_id=task_id,
                        user_id=user_id,
                        result=error_result,
                        tool_log_id=None
                    )

            except Exception as update_error:
                logger.error(f"❌ Không thể cập nhật trạng thái task: {update_error}")

    # Chạy async function
    logger.info(f"🔄 Starting Celery task execution for task_id: {task_id}")

    try:
        logger.info("🔄 Running async JSON template processing function...")
        asyncio.run(_async_process_json_template())
        logger.info(f"✅ Celery task completed successfully for task_id: {task_id}")

    except Exception as e:
        logger.error(f"❌ Lỗi chạy async task {task_id}: {e}")
        logger.error(f"❌ Exception type: {type(e).__name__}")
        logger.error(f"❌ Exception details: {str(e)}")

        # Fallback: cập nhật trạng thái lỗi trực tiếp
        try:
            logger.info("🔄 Attempting to update task status to FAILURE...")
            task_service = get_mongodb_task_service()

            # Create a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(task_service.mark_task_failed(
                    task_id,
                    error=f"Lỗi hệ thống: {str(e)}"
                ))
                logger.info("✅ Task status updated to FAILURE")
            finally:
                loop.close()
        except Exception as update_error:
            logger.error(f"❌ Failed to update task status: {update_error}")

    return {
        "success": True,
        "task_id": task_id,
        "message": "JSON template processing completed"
    }


# Helper functions
async def trigger_json_template_task(
    lesson_id: str,
    template_json: Dict[str, Any],
    config_prompt: str = None,
    user_id: str = None,
    book_id: str = None
) -> str:
    """
    Trigger Celery task cho JSON template processing và tạo task trong MongoDB

    Args:
        lesson_id: ID của bài học
        template_json: JSON template đã được phân tích sẵn
        config_prompt: Prompt cấu hình tùy chỉnh (optional)
        user_id: ID của user (optional)
        book_id: ID của sách giáo khoa (optional)

    Returns:
        str: Task ID để theo dõi tiến trình
    """
    try:
        # Import TaskType enum
        from app.services.mongodb_task_service import TaskType

        # Tạo task trong MongoDB trước
        task_service = get_mongodb_task_service()
        task_data = {
            "lesson_id": lesson_id,
            "template_json": template_json,
            "config_prompt": config_prompt,
            "user_id": user_id,
            "book_id": book_id
        }

        metadata = {
            "lesson_id": lesson_id,
            "slides_count": len(template_json.get("slides", [])),
            "config_prompt": config_prompt,
            "user_id": user_id,
            "book_id": book_id
        }

        task_id = await task_service.create_task(
            task_type=TaskType.JSON_TEMPLATE_PROCESSING,
            task_data=task_data,
            metadata=metadata
        )

        logger.info(f"✅ Created MongoDB task: {task_id}")

        # Trigger Celery task
        celery_task = process_json_template_task.delay(
            task_id=task_id,
            lesson_id=lesson_id,
            template_json=template_json,
            config_prompt=config_prompt,
            user_id=user_id,
            book_id=book_id
        )

        logger.info(f"✅ Triggered Celery task: {celery_task.id}")
        logger.info(f"📋 Task mapping: MongoDB={task_id}, Celery={celery_task.id}")

        return task_id

    except Exception as e:
        logger.error(f"❌ Error triggering JSON template task: {e}")
        raise


async def get_json_template_task_status(task_id: str) -> Dict[str, Any]:
    """
    Lấy trạng thái của JSON template processing task

    Args:
        task_id: ID của task trong MongoDB

    Returns:
        Dict chứa thông tin trạng thái task
    """
    try:
        task_service = get_mongodb_task_service()
        task_info = await task_service.get_task_status(task_id)

        if not task_info:
            return {
                "success": False,
                "error": "Task not found",
                "task_id": task_id
            }

        return {
            "success": True,
            "task_id": task_id,
            "status": task_info.get("status", "UNKNOWN"),
            "progress": task_info.get("progress", 0),
            "message": task_info.get("message", ""),
            "result": task_info.get("result"),
            "error": task_info.get("error"),
            "created_at": task_info.get("created_at"),
            "updated_at": task_info.get("updated_at")
        }

    except Exception as e:
        logger.error(f"❌ Error getting task status: {e}")
        return {
            "success": False,
            "error": str(e),
            "task_id": task_id
        }
