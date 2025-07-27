"""
Kafka Service for PlanBook AI
Service để handle Kafka producer và consumer
"""
import json
import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, List, Union, Awaitable
from datetime import datetime

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError, KafkaTimeoutError
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaConnectionError

from app.core.kafka_config import (
    get_kafka_servers,
    get_topic_name,
    get_requests_topic,
    get_responses_topic,
    get_producer_config,
    get_consumer_config,
    get_aiokafka_producer_config,
    get_aiokafka_consumer_config,
    get_send_timeout,
    is_kafka_enabled,
    get_max_retries,
    get_retry_backoff_ms
)
from app.constants.kafka_message_types import PROGRESS_TYPE, RESULT_TYPE

logger = logging.getLogger(__name__)


def safe_kafka_call(func, *args, **kwargs):
    """
    Simplified Kafka call without threading to avoid deadlocks
    """
    try:
        logger.info("📤 Attempting Kafka operation...")
        result = func(*args, **kwargs)
        logger.info("✅ Kafka operation completed")
        return result
    except Exception as kafka_error:
        logger.warning(f"⚠️ Kafka operation failed: {kafka_error}, continuing...")
        return False


class KafkaService:
    """Kafka service for handling producer and consumer operations"""
    
    def __init__(self):
        self.producer: Optional[KafkaProducer] = None
        self.async_producer: Optional[AIOKafkaProducer] = None
        self.consumer: Optional[KafkaConsumer] = None
        self.async_consumer: Optional[AIOKafkaConsumer] = None
        self.is_connected = False
        self.message_handlers: Dict[str, Callable] = {}
    

    
    async def _initialize_async_producer(self):
        """Initialize async Kafka producer"""
        try:
            config = get_aiokafka_producer_config()
            self.async_producer = AIOKafkaProducer(**config)
            await self.async_producer.start()
            logger.info("✅ Async Kafka producer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize async producer: {e}")
            raise
    
    async def _initialize_async_consumer(self):
        """Initialize async Kafka consumer"""
        try:
            config = get_aiokafka_consumer_config()
            self.async_consumer = AIOKafkaConsumer(
                get_requests_topic(),  # Listen on requests topic for messages from SpringBoot
                **config
            )
            await self.async_consumer.start()
            logger.info("✅ Async Kafka consumer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize async consumer: {e}")
            raise
    
    def _initialize_sync_producer(self):
        """Initialize synchronous Kafka producer with improved configuration"""
        try:
            # Close existing producer if any
            if self.producer:
                try:
                    self.producer.close(timeout=2)
                except:
                    pass
                self.producer = None

            config = get_producer_config()
            config['bootstrap_servers'] = get_kafka_servers()

            # Add improved configuration for reliability
            config.update({
                'request_timeout_ms': get_send_timeout() * 1000,  # Convert to ms
                'delivery_timeout_ms': (get_send_timeout() + 2) * 1000,  # Slightly longer
                'connections_max_idle_ms': 30000,  # 30 seconds
                'reconnect_backoff_ms': 100,
                'reconnect_backoff_max_ms': 1000,
                'max_in_flight_requests_per_connection': 1,  # Ensure ordering
                'enable_idempotence': True,  # Prevent duplicates
                'retries': get_max_retries(),
                'retry_backoff_ms': get_retry_backoff_ms()
            })

            logger.info(f"🔄 Initializing Kafka producer with config: {config}")
            self.producer = KafkaProducer(**config)
            logger.info("✅ Sync Kafka producer initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize sync producer: {e}")
            self.producer = None
            raise
    
    def _initialize_sync_consumer(self):
        """Initialize synchronous Kafka consumer"""
        try:
            config = get_consumer_config()
            self.consumer = KafkaConsumer(
                get_topic_name(),
                **config
            )
            logger.info("✅ Sync Kafka consumer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize sync consumer: {e}")
            raise
    
    async def send_message_async(
        self,
        message: Dict[str, Any],
        topic: Optional[str] = None,
        key: Optional[str] = None
    ) -> bool:
        """Send message asynchronously"""
        try:
            if not self.async_producer:
                await self._initialize_async_producer()

            topic = topic or get_topic_name()

            # Prepare message
            message_data = {
                "timestamp": datetime.now().isoformat(),
                "source": "planbook-fastapi",
                "data": message
            }

            # Convert to JSON string
            message_json = json.dumps(message_data, ensure_ascii=False)

            # Send message
            await self.async_producer.send_and_wait(
                topic=topic,
                value=message_json,
                key=key
            )

            logger.info(f"✅ Message sent to topic '{topic}': {message}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            return False
    
    def send_message_sync(
        self,
        message: Dict[str, Any],
        topic: Optional[str] = None,
        key: Optional[str] = None
    ) -> bool:
        """Send message synchronously with improved error handling and timeout"""
        # Check if Kafka is enabled
        if not is_kafka_enabled():
            logger.warning("⚠️ Kafka is disabled, skipping message send")
            return True  # Return True to not block the process

        max_retries = get_max_retries()
        retry_backoff = get_retry_backoff_ms() / 1000.0  # Convert to seconds
        send_timeout = get_send_timeout()

        for attempt in range(max_retries + 1):
            try:
                # Always create a fresh producer for each attempt to avoid stale connections
                logger.info(f"🔄 Creating fresh Kafka producer for attempt {attempt + 1}...")
                self._initialize_sync_producer()

                topic = topic or get_topic_name()

                # Prepare message
                message_data = {
                    "timestamp": datetime.now().isoformat(),
                    "source": "planbook-fastapi",
                    "data": message
                }

                # Convert to JSON string
                message_json = json.dumps(message_data, ensure_ascii=False)

                logger.info(f"📤 Sending message to topic '{topic}' (attempt {attempt + 1}/{max_retries + 1})")

                # Send message with shorter timeout
                future = self.producer.send(
                    topic=topic,
                    value=message_json,
                    key=key
                )

                # Wait for result with configurable timeout
                record_metadata = future.get(timeout=send_timeout)

                logger.info(f"✅ Message sent to topic '{topic}' partition {record_metadata.partition}")

                # Close producer after successful send to prevent stale connections
                try:
                    self.producer.close(timeout=1)
                except:
                    pass
                self.producer = None

                return True

            except KafkaTimeoutError as e:
                logger.warning(f"⏰ Kafka timeout on attempt {attempt + 1}: {e}")
                # Always clean up producer on error
                try:
                    if self.producer:
                        self.producer.close(timeout=1)
                except:
                    pass
                self.producer = None

                if attempt < max_retries:
                    logger.info(f"🔄 Retrying in {retry_backoff} seconds...")
                    time.sleep(retry_backoff)
                else:
                    logger.error("❌ All Kafka send attempts failed due to timeout")
                    return False

            except Exception as e:
                logger.warning(f"⚠️ Kafka error on attempt {attempt + 1}: {e}")
                # Always clean up producer on error
                try:
                    if self.producer:
                        self.producer.close(timeout=1)
                except:
                    pass
                self.producer = None

                if attempt < max_retries:
                    logger.info(f"🔄 Retrying in {retry_backoff} seconds...")
                    time.sleep(retry_backoff)
                else:
                    logger.error(f"❌ All Kafka send attempts failed: {e}")
                    return False

        return False



    async def consume_messages_async(self, handler: Union[Callable[[Dict[str, Any]], None], Callable[[Dict[str, Any]], Awaitable[None]]]):
        """Consume messages asynchronously"""
        try:
            if not self.async_consumer:
                await self._initialize_async_consumer()

            logger.info("🔄 Starting async message consumption...")
            async for message in self.async_consumer:
                try:
                    # Decode message
                    if message.value:
                        message_data = json.loads(message.value)

                        logger.info(f"📨 Received message from topic '{message.topic}': {message_data}")

                        # Call handler
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message_data)
                        else:
                            handler(message_data)

                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to decode message: {e}")
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")

        except Exception as e:
            logger.error(f"❌ Error in message consumption: {e}")
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a handler for specific message type"""
        self.message_handlers[message_type] = handler
        logger.info(f"✅ Registered handler for message type: {message_type}")
    
    async def process_message(self, message_data: Dict[str, Any]):
        """Process incoming message based on type"""
        try:
            message_type = message_data.get("data", {}).get("type")
            
            if message_type and message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                
                if asyncio.iscoroutinefunction(handler):
                    await handler(message_data)
                else:
                    handler(message_data)
            else:
                logger.warning(f"⚠️ No handler found for message type: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Kafka connection health"""
        try:
            # Test async producer
            if self.async_producer:
                test_message = {
                    "type": "health_check",
                    "timestamp": datetime.now().isoformat()
                }
                
                success = await self.send_message_async(test_message)
                
                return {
                    "status": "healthy" if success else "unhealthy",
                    "connected": self.is_connected,
                    "producer_ready": self.async_producer is not None,
                    "consumer_ready": self.async_consumer is not None,
                    "servers": get_kafka_servers(),
                    "topic": get_topic_name()
                }
            else:
                return {
                    "status": "unhealthy",
                    "connected": False,
                    "error": "Producer not initialized"
                }
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e)
            }
    
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
                "type": PROGRESS_TYPE,
                "data": progress_data
            }

            # Gửi message qua sync Kafka producer
            logger.info(f"📤 [SYNC] Sending progress - Task: {task_id}, Progress: {progress}%")

            success = self.send_message_sync(
                message=kafka_message,
                topic=get_responses_topic(),
                key=user_id
            )

            if success:
                logger.info(f"✅ [SYNC] Sent progress update - Task: {task_id}")
            else:
                logger.warning(f"⚠️ [SYNC] Failed to send progress update - Task: {task_id}, but continuing...")

            # Always return True to not block task processing
            return True

        except Exception as e:
            logger.warning(f"⚠️ [SYNC] Error sending progress update: {e}, but continuing...")
            return True

    def send_final_result_sync(
        self,
        task_id: str,
        user_id: str,
        result: Dict[str, Any],
        lesson_id: Optional[str] = None,
        tool_log_id: Optional[Any] = None
    ) -> bool:
        """
        Gửi kết quả cuối cùng của task về SpringBoot (sync version)
        Với fallback mechanism để không block task completion
        """
        try:
            is_success = result.get("success", False)
            error_msg = result.get("error", "")

            logger.info(f"📤 [SYNC] Sending final result - Task: {task_id}, Success: {is_success}")

            # Tạo formatted result với output field
            formatted_result = {
                "success": is_success,
                "output": result.get("output", result)  # Nếu đã có output thì giữ nguyên, không thì wrap toàn bộ result
            }

            # Nếu có error, thêm vào formatted_result
            if not is_success and error_msg:
                formatted_result["error"] = error_msg

            # Tạo message data
            progress_data = {
                "tool_log_id": tool_log_id,
                "task_id": task_id,
                "user_id": user_id,
                "success": is_success,
                "result": formatted_result,
                "timestamp": datetime.now().isoformat()
            }

            if lesson_id:
                progress_data["lesson_id"] = lesson_id

            if not is_success and error_msg:
                progress_data["message"] = f"Tác vụ hoàn thành với lỗi: {error_msg}"
                progress_data["status"] = "completed_with_error"
            else:
                progress_data["message"] = "Tác vụ đã hoàn thành thành công"
                progress_data["status"] = "completed"

            # Tạo Kafka message
            kafka_message = {
                "type": RESULT_TYPE,
                "data": progress_data
            }

            # Gửi message
            success = self.send_message_sync(
                message=kafka_message,
                topic=get_responses_topic(),
                key=user_id
            )

            if success:
                logger.info(f"✅ [SYNC] Final result sent - Task: {task_id}")
            else:
                logger.warning(f"⚠️ [SYNC] Failed to send final result - Task: {task_id}, but task completed")

            # Always return True to not block task completion
            return True

        except Exception as e:
            logger.warning(f"⚠️ [SYNC] Error sending final result: {e}, but task completed")
            return True




# Global Kafka service instance
kafka_service = KafkaService()


# Note: Message handling is done directly in app/api/api.py via handle_incoming_message function
