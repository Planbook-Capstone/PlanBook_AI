"""
Kafka API Endpoints for PlanBook AI
API endpoints để test và quản lý Kafka producer/consumer
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.services.kafka_service import kafka_service
from app.core.kafka_config import get_topic_name, get_kafka_servers, kafka_settings

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class KafkaMessage(BaseModel):
    """Model for Kafka message"""
    type: str = Field(..., description="Message type")
    data: Dict[str, Any] = Field(..., description="Message data")
    key: Optional[str] = Field(None, description="Message key")
    topic: Optional[str] = Field(None, description="Target topic (optional)")


class KafkaResponse(BaseModel):
    """Model for Kafka response"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Model for health check response"""
    status: str
    connected: bool
    producer_ready: bool
    consumer_ready: bool
    servers: list
    topic: str
    error: Optional[str] = None


# Background task for message consumption
async def start_message_consumer():
    """Start Kafka message consumer in background"""
    try:
        logger.info("🔄 Starting Kafka message consumer...")
        await kafka_service.consume_messages_async(kafka_service.process_message)
    except Exception as e:
        logger.error(f"❌ Error in message consumer: {e}")


@router.get("/health", response_model=HealthResponse)
async def kafka_health_check():
    """
    Kiểm tra trạng thái kết nối Kafka
    """
    try:
        health_data = await kafka_service.health_check()
        return HealthResponse(**health_data)
    except Exception as e:
        logger.error(f"❌ Kafka health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.post("/send", response_model=KafkaResponse)
async def send_kafka_message(message: KafkaMessage):
    """
    Gửi message đến Kafka topic
    """
    try:
        # Prepare message data
        message_data = {
            "type": message.type,
            "timestamp": datetime.now().isoformat(),
            "data": message.data
        }
        
        # Send message
        success = await kafka_service.send_message_async(
            message=message_data,
            topic=message.topic,
            key=message.key
        )
        
        if success:
            return KafkaResponse(
                success=True,
                message="Message sent successfully",
                data={
                    "topic": message.topic or get_topic_name(),
                    "key": message.key,
                    "type": message.type
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send message")
            
    except Exception as e:
        logger.error(f"❌ Error sending Kafka message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")


@router.post("/send-sync", response_model=KafkaResponse)
async def send_kafka_message_sync(message: KafkaMessage):
    """
    Gửi message đến Kafka topic (synchronous)
    """
    try:
        # Prepare message data
        message_data = {
            "type": message.type,
            "timestamp": datetime.now().isoformat(),
            "data": message.data
        }
        
        # Send message synchronously
        success = kafka_service.send_message_sync(
            message=message_data,
            topic=message.topic,
            key=message.key
        )
        
        if success:
            return KafkaResponse(
                success=True,
                message="Message sent successfully (sync)",
                data={
                    "topic": message.topic or get_topic_name(),
                    "key": message.key,
                    "type": message.type
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send message")
            
    except Exception as e:
        logger.error(f"❌ Error sending Kafka message (sync): {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")


@router.post("/start-consumer", response_model=KafkaResponse)
async def start_kafka_consumer(background_tasks: BackgroundTasks):
    """
    Bắt đầu Kafka consumer để nhận messages
    """
    try:
        # Add consumer task to background
        background_tasks.add_task(start_message_consumer)
        
        return KafkaResponse(
            success=True,
            message="Kafka consumer started in background",
            data={
                "topic": get_topic_name(),
                "servers": get_kafka_servers()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error starting Kafka consumer: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start consumer: {str(e)}")


@router.get("/config", response_model=Dict[str, Any])
async def get_kafka_config():
    """
    Lấy thông tin cấu hình Kafka hiện tại
    """
    try:
        return {
            "servers": get_kafka_servers(),
            "topic": get_topic_name(),
            "connected": kafka_service.is_connected,
            "producer_ready": kafka_service.async_producer is not None,
            "consumer_ready": kafka_service.async_consumer is not None
        }
    except Exception as e:
        logger.error(f"❌ Error getting Kafka config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get config: {str(e)}")


# Test endpoints for SpringBoot integration
@router.post("/test/lesson-plan", response_model=KafkaResponse)
async def test_lesson_plan_message():
    """
    Test gửi message lesson plan request đến SpringBoot
    """
    try:
        test_message = {
            "type": "lesson_plan_request",
            "data": {
                "subject": "Hóa học",
                "grade": 12,
                "lesson_id": "test_lesson_001",
                "requirements": ["Hiểu về phản ứng hóa học", "Áp dụng công thức"],
                "timestamp": datetime.now().isoformat()
            }
        }
        
        success = await kafka_service.send_message_async(test_message)
        
        if success:
            return KafkaResponse(
                success=True,
                message="Test lesson plan message sent to SpringBoot",
                data=test_message
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send test message")
            
    except Exception as e:
        logger.error(f"❌ Error sending test lesson plan message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send test message: {str(e)}")


@router.post("/test/exam-generation", response_model=KafkaResponse)
async def test_exam_generation_message():
    """
    Test gửi message exam generation request đến SpringBoot
    """
    try:
        test_message = {
            "type": "exam_generation_request",
            "data": {
                "subject": "Hóa học",
                "grade": 12,
                "exam_type": "multiple_choice",
                "question_count": 40,
                "difficulty": "medium",
                "topics": ["Phản ứng hóa học", "Cân bằng hóa học"],
                "timestamp": datetime.now().isoformat()
            }
        }

        success = await kafka_service.send_message_async(test_message)

        if success:
            return KafkaResponse(
                success=True,
                message="Test exam generation message sent to SpringBoot",
                data=test_message
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send test message")

    except Exception as e:
        logger.error(f"❌ Error sending test exam generation message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send test message: {str(e)}")


@router.get("/consumer/status", response_model=KafkaResponse)
async def get_consumer_status():
    """
    Kiểm tra trạng thái Kafka consumer
    """
    try:
        # Check if consumer is initialized
        consumer_status = {
            "async_consumer_initialized": kafka_service.async_consumer is not None,
            "sync_consumer_initialized": kafka_service.consumer is not None,
            "topic": get_topic_name(),
            "servers": get_kafka_servers(),
            "consumer_group": kafka_settings.AIOKAFKA_CONSUMER_CONFIG.get("group_id"),
        }

        return KafkaResponse(
            success=True,
            message="Consumer status retrieved successfully",
            data=consumer_status
        )

    except Exception as e:
        logger.error(f"❌ Error getting consumer status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get consumer status: {str(e)}")


@router.post("/test/simulate-external-message", response_model=KafkaResponse)
async def simulate_external_message():
    """
    Simulate message từ service khác để test consumer
    """
    try:
        # Simulate message from SpringBoot service
        external_message = {
            "timestamp": datetime.now().isoformat(),
            "source": "springboot-service",
            "data": {
                "type": "lesson_plan_request",
                "subject": "Hóa học",
                "grade": 12,
                "lesson_id": "lesson_001",
                "requirements": ["Hiểu về phản ứng hóa học", "Áp dụng công thức"],
                "request_id": "req_12345",
                "user_id": "teacher_001"
            }
        }

        success = await kafka_service.send_message_async(external_message)

        if success:
            return KafkaResponse(
                success=True,
                message="Simulated external message sent successfully - check logs for consumer processing",
                data=external_message
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send simulated message")

    except Exception as e:
        logger.error(f"❌ Error sending simulated external message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send simulated message: {str(e)}")


@router.post("/test/simulate-grading-request", response_model=KafkaResponse)
async def simulate_grading_request():
    """
    Simulate grading request từ service khác
    """
    try:
        # Simulate grading request from SpringBoot service
        grading_message = {
            "timestamp": datetime.now().isoformat(),
            "source": "springboot-service",
            "data": {
                "type": "grading_request",
                "exam_id": "exam_001",
                "student_id": "student_001",
                "student_answers": [
                    {"question_id": 1, "answer": "A"},
                    {"question_id": 2, "answer": "B"},
                    {"question_id": 3, "answer": "C"}
                ],
                "request_id": "grading_req_001"
            }
        }

        success = await kafka_service.send_message_async(grading_message)

        if success:
            return KafkaResponse(
                success=True,
                message="Simulated grading request sent successfully - check logs for consumer processing",
                data=grading_message
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send simulated grading request")

    except Exception as e:
        logger.error(f"❌ Error sending simulated grading request: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send simulated grading request: {str(e)}")


@router.post("/test/grading", response_model=KafkaResponse)
async def test_grading_message():
    """
    Test gửi message grading request đến SpringBoot
    """
    try:
        test_message = {
            "type": "grading_request",
            "data": {
                "exam_id": "exam_001",
                "student_id": "student_001",
                "answers": ["A", "B", "C", "D", "A"],
                "exam_type": "multiple_choice",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        success = await kafka_service.send_message_async(test_message)
        
        if success:
            return KafkaResponse(
                success=True,
                message="Test grading message sent to SpringBoot",
                data=test_message
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to send test message")
            
    except Exception as e:
        logger.error(f"❌ Error sending test grading message: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send test message: {str(e)}")


@router.get("/messages/recent")
async def get_recent_messages():
    """
    Lấy danh sách messages gần đây (demo)
    """
    # This is a placeholder - in real implementation, you might want to store
    # recent messages in Redis or database for monitoring
    return {
        "message": "Recent messages endpoint - implement based on your monitoring needs",
        "suggestion": "Consider storing recent messages in Redis for monitoring"
    }
