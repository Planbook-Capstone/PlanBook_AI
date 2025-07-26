"""
Kafka Progress Notification Service for PlanBook AI
Service để gửi progress updates qua Kafka cho SpringBoot
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from app.services.kafka_service import kafka_service
from app.core.kafka_config import get_responses_topic

logger = logging.getLogger(__name__)


class KafkaProgressService:
    """Service để gửi progress updates qua Kafka"""
    
    def __init__(self):
        self.kafka_service = kafka_service
    
    async def send_progress_update(
        self,
        task_id: str,
        user_id: str,
        progress: int,
        message: str,
        status: str = "processing",
        additional_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Gửi progress update qua Kafka cho SpringBoot
        
        Args:
            task_id: ID của task
            user_id: ID của user
            progress: Phần trăm hoàn thành (0-100)
            message: Thông điệp mô tả trạng thái
            status: Trạng thái task (processing, completed, failed)
            additional_data: Dữ liệu bổ sung (optional)
            
        Returns:
            bool: True nếu gửi thành công
        """
        try:
            # Chuẩn bị message data với routing information
            progress_data = {
                "task_id": task_id,
                "user_id": user_id,
                "progress": progress,
                "message": message,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
            
            # Thêm dữ liệu bổ sung nếu có
            if additional_data:
                progress_data.update(additional_data)
            
            # Tạo Kafka message
            kafka_message = {
                "type": "lesson_plan_content_generation_progress",
                "data": progress_data
            }
            
            # Gửi message qua Kafka responses topic với user_id làm key
            # Kafka sẽ route message đến partition dựa trên user_id
            logger.info(f"📤 Sending Kafka message - Topic: {get_responses_topic()}, Key: {user_id}, Type: {kafka_message['type']}")

            success = await self.kafka_service.send_message_async(
                message=kafka_message,
                topic=get_responses_topic(),
                key=user_id  # Sử dụng user_id làm key để route message
            )

            if success:
                logger.info(f"✅ Sent progress update via Kafka - Task: {task_id}, Progress: {progress}%, User: {user_id}, Status: {status}")
            else:
                logger.error(f"❌ Failed to send progress update via Kafka - Task: {task_id}")

            return success
            
        except Exception as e:
            logger.error(f"❌ Error sending progress update via Kafka: {e}")
            return False
    
    async def send_task_started(
        self,
        task_id: str,
        user_id: str,
        lesson_id: Optional[str] = None
    ) -> bool:
        """Gửi thông báo task bắt đầu"""
        additional_data = {}
        if lesson_id:
            additional_data["lesson_id"] = lesson_id
            
        return await self.send_progress_update(
            task_id=task_id,
            user_id=user_id,
            progress=5,
            message="Tác vụ đã bắt đầu xử lý",
            status="processing",
            additional_data=additional_data
        )
    
    async def send_task_completed(
        self,
        task_id: str,
        user_id: str,
        result: Dict[str, Any],
        lesson_id: Optional[str] = None
    ) -> bool:
        """Gửi thông báo task hoàn thành"""
        logger.info(f"📤 Sending task completion notification - Task: {task_id}, User: {user_id}, Success: {result.get('success', 'unknown')}")

        additional_data = {
            "result": result
        }
        if lesson_id:
            additional_data["lesson_id"] = lesson_id

        success = await self.send_progress_update(
            task_id=task_id,
            user_id=user_id,
            progress=100,
            message="Tác vụ đã hoàn thành thành công",
            status="completed",
            additional_data=additional_data
        )

        if success:
            logger.info(f"✅ Task completion notification sent successfully for task {task_id}")
        else:
            logger.error(f"❌ Failed to send task completion notification for task {task_id}")

        return success
    
    async def send_task_failed(
        self,
        task_id: str,
        user_id: str,
        error: str,
        lesson_id: Optional[str] = None
    ) -> bool:
        """Gửi thông báo task thất bại"""
        logger.info(f"📤 Sending task failure notification - Task: {task_id}, User: {user_id}, Error: {error}")

        additional_data = {
            "error": error
        }
        if lesson_id:
            additional_data["lesson_id"] = lesson_id

        success = await self.send_progress_update(
            task_id=task_id,
            user_id=user_id,
            progress=0,
            message=f"Tác vụ thất bại: {error}",
            status="failed",
            additional_data=additional_data
        )

        if success:
            logger.info(f"✅ Task failure notification sent successfully for task {task_id}")
        else:
            logger.error(f"❌ Failed to send task failure notification for task {task_id}")

        return success

    async def send_final_result(
        self,
        task_id: str,
        user_id: str,
        result: Dict[str, Any],
        lesson_id: Optional[str] = None
    ) -> bool:
        """
        Gửi kết quả cuối cùng của task về SpringBoot
        Bất kể thành công hay thất bại đều gửi message
        """
        try:
            is_success = result.get("success", False)
            error_msg = result.get("error", "")

            logger.info(f"📤 Sending final result - Task: {task_id}, User: {user_id}, Success: {is_success}")

            # Tạo message với format chuẩn cho SpringBoot
            message_type = "lesson_plan_content_generation_result"

            progress_data = {
                "task_id": task_id,
                "user_id": user_id,
                "success": is_success,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

            if lesson_id:
                progress_data["lesson_id"] = lesson_id

            if not is_success and error_msg:
                progress_data["error"] = error_msg
                progress_data["message"] = f"Tác vụ hoàn thành với lỗi: {error_msg}"
                progress_data["status"] = "completed_with_error"
            else:
                progress_data["message"] = "Tác vụ đã hoàn thành thành công"
                progress_data["status"] = "completed"

            # Tạo Kafka message
            kafka_message = {
                "type": message_type,
                "data": progress_data
            }

            # Gửi message qua Kafka responses topic
            logger.info(f"📤 Sending final result Kafka message - Topic: {get_responses_topic()}, Key: {user_id}")

            success = await self.kafka_service.send_message_async(
                message=kafka_message,
                topic=get_responses_topic(),
                key=user_id
            )

            if success:
                logger.info(f"✅ Final result sent successfully for task {task_id}")
            else:
                logger.error(f"❌ Failed to send final result for task {task_id}")

            return success

        except Exception as e:
            logger.error(f"❌ Error sending final result for task {task_id}: {e}")
            return False


class SyncKafkaProgressService:
    """
    Sync wrapper cho KafkaProgressService để sử dụng trong Celery tasks
    """

    def __init__(self):
        from app.services.kafka_service import kafka_service
        self.kafka_service = kafka_service

    def send_progress_update_sync(
        self,
        tool_log_id: str,
        task_id: str,
        user_id: str,
        progress: int,
        message: str,
        status: str = "processing",
        additional_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Gửi progress update qua Kafka (sync version cho Celery)
        Với fallback mechanism để không block task processing
        """
        try:
            # Chuẩn bị message data
            progress_data = {
                "tool_log_id": tool_log_id,
                "task_id": task_id,
                "user_id": user_id,
                "progress": progress,
                "message": message,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }

            # Thêm dữ liệu bổ sung nếu có
            if additional_data:
                progress_data.update(additional_data)

            # Tạo Kafka message
            kafka_message = {
                "type": "lesson_plan_content_generation_progress",
                "data": progress_data
            }

            # Gửi message qua sync Kafka producer
            logger.info(f"📤 [SYNC] Sending Kafka message - Topic: {get_responses_topic()}, Key: {user_id}, Type: {kafka_message['type']}")

            success = self.kafka_service.send_message_sync(
                message=kafka_message,
                topic=get_responses_topic(),
                key=user_id
            )

            if success:
                logger.info(f"✅ [SYNC] Sent progress update - Task: {task_id}, Progress: {progress}%, User: {user_id}, Status: {status}")
            else:
                logger.warning(f"⚠️ [SYNC] Failed to send progress update - Task: {task_id}, but continuing processing...")

            # Always return True to not block task processing
            # Even if Kafka fails, the task should continue
            return True

        except Exception as e:
            logger.warning(f"⚠️ [SYNC] Error sending progress update: {e}, but continuing processing...")
            # Don't let Kafka errors block the main task processing
            return True

    def send_task_started_sync(
        self,
        task_id: str,
        user_id: str,
        lesson_id: Optional[str] = None,
        tool_log_id: Optional[str] = None
    ) -> bool:
        """Gửi thông báo task bắt đầu (sync)"""
        additional_data = {}
        if lesson_id:
            additional_data["lesson_id"] = lesson_id

        return self.send_progress_update_sync(
            tool_log_id=tool_log_id or "",
            task_id=task_id,
            user_id=user_id,
            progress=5,
            message="Tác vụ đã bắt đầu xử lý",
            status="processing",
            additional_data=additional_data
        )

    def send_task_completed_sync(
        self,
        task_id: str,
        user_id: str,
        result: Dict[str, Any],
        lesson_id: Optional[str] = None,
        tool_log_id: Optional[str] = None
    ) -> bool:
        """Gửi thông báo task hoàn thành (sync)"""
        logger.info(f"📤 [SYNC] Sending task completion notification - Task: {task_id}, User: {user_id}, Success: {result.get('success', 'unknown')}")

        additional_data = {
            "result": result
        }
        if lesson_id:
            additional_data["lesson_id"] = lesson_id

        success = self.send_progress_update_sync(
            tool_log_id=tool_log_id or "",
            task_id=task_id,
            user_id=user_id,
            progress=100,
            message="Tác vụ đã hoàn thành thành công",
            status="completed",
            additional_data=additional_data
        )

        if success:
            logger.info(f"✅ [SYNC] Task completion notification sent successfully for task {task_id}")
        else:
            logger.error(f"❌ [SYNC] Failed to send task completion notification for task {task_id}")

        return success

    def send_task_failed_sync(
        self,
        task_id: str,
        user_id: str,
        error: str,
        lesson_id: Optional[str] = None,
        tool_log_id: Optional[str] = None
    ) -> bool:
        """Gửi thông báo task thất bại (sync)"""
        logger.info(f"📤 [SYNC] Sending task failure notification - Task: {task_id}, User: {user_id}, Error: {error}")

        additional_data = {
            "error": error
        }
        if lesson_id:
            additional_data["lesson_id"] = lesson_id

        success = self.send_progress_update_sync(
            tool_log_id=tool_log_id or "",
            task_id=task_id,
            user_id=user_id,
            progress=0,
            message=f"Tác vụ thất bại: {error}",
            status="failed",
            additional_data=additional_data
        )

        if success:
            logger.info(f"✅ [SYNC] Task failure notification sent successfully for task {task_id}")
        else:
            logger.error(f"❌ [SYNC] Failed to send task failure notification for task {task_id}")

        return success

    def send_final_result_sync(
        self,
        task_id: str,
        user_id: str,
        result: Dict[str, Any],
        lesson_id: Optional[str] = None,
        tool_log_id:Optional[Any] = None
    ) -> bool:
        """
        Gửi kết quả cuối cùng của task về SpringBoot (sync version)
        Với fallback mechanism để không block task completion
        """
        try:
            is_success = result.get("success", False)
            error_msg = result.get("error", "")

            logger.info(f"📤 [SYNC] Sending final result - Task: {task_id}, User: {user_id}, Success: {is_success}")

            # Tạo message với format chuẩn cho SpringBoot
            message_type = "lesson_plan_content_generation_result"

            progress_data = {
                "tool_log_id": tool_log_id,
                "task_id": task_id,
                "user_id": user_id,
                "success": is_success,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }

            if lesson_id:
                progress_data["lesson_id"] = lesson_id

            if not is_success and error_msg:
                progress_data["error"] = error_msg
                progress_data["message"] = f"Tác vụ hoàn thành với lỗi: {error_msg}"
                progress_data["status"] = "completed_with_error"
            else:
                progress_data["message"] = "Tác vụ đã hoàn thành thành công"
                progress_data["status"] = "completed"

            # Tạo Kafka message
            kafka_message = {
                "type": message_type,
                "data": progress_data
            }

            # Gửi message qua sync Kafka producer
            logger.info(f"📤 [SYNC] Sending final result Kafka message - Topic: {get_responses_topic()}, Key: {user_id}")

            success = self.kafka_service.send_message_sync(
                message=kafka_message,
                topic=get_responses_topic(),
                key=user_id
            )

            if success:
                logger.info(f"✅ [SYNC] Final result sent successfully for task {task_id}")
            else:
                logger.warning(f"⚠️ [SYNC] Failed to send final result for task {task_id}, but task completed")

            # Always return True to not block task completion
            # Even if Kafka fails, the task itself completed successfully
            return True

        except Exception as e:
            logger.warning(f"⚠️ [SYNC] Error sending final result for task {task_id}: {e}, but task completed")
            # Don't let Kafka errors affect task completion status
            return True


# Global instances
kafka_progress_service = KafkaProgressService()
sync_kafka_progress_service = SyncKafkaProgressService()
