"""
Kafka Service for PlanBook AI
Service để handle Kafka producer và consumer
"""
import json
import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError, KafkaTimeoutError
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
from aiokafka.errors import KafkaConnectionError

from app.core.kafka_config import (
    kafka_settings,
    get_kafka_servers,
    get_topic_name,
    get_producer_config,
    get_consumer_config,
    get_aiokafka_producer_config,
    get_aiokafka_consumer_config,
)

logger = logging.getLogger(__name__)


class KafkaService:
    """Kafka service for handling producer and consumer operations"""
    
    def __init__(self):
        self.producer: Optional[KafkaProducer] = None
        self.async_producer: Optional[AIOKafkaProducer] = None
        self.consumer: Optional[KafkaConsumer] = None
        self.async_consumer: Optional[AIOKafkaConsumer] = None
        self.is_connected = False
        self.message_handlers: Dict[str, Callable] = {}
    
    async def initialize(self):
        """Initialize Kafka connections"""
        try:
            await self._initialize_async_producer()
            await self._initialize_async_consumer()
            self.is_connected = True
            logger.info("✅ Kafka service initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Kafka service: {e}")
            logger.info("ℹ️ Kafka service will be available when Kafka server is running")
            self.is_connected = False
            # Don't raise exception to allow app to start without Kafka
    
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
                get_topic_name(),
                **config
            )
            await self.async_consumer.start()
            logger.info("✅ Async Kafka consumer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize async consumer: {e}")
            raise
    
    def _initialize_sync_producer(self):
        """Initialize synchronous Kafka producer"""
        try:
            config = get_producer_config()
            self.producer = KafkaProducer(**config)
            logger.info("✅ Sync Kafka producer initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize sync producer: {e}")
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
        """Send message synchronously"""
        try:
            if not self.producer:
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

            # Send message
            future = self.producer.send(
                topic=topic,
                value=message_json,
                key=key
            )

            # Wait for result
            record_metadata = future.get(timeout=10)

            logger.info(f"✅ Message sent to topic '{topic}' partition {record_metadata.partition}")
            return True

        except KafkaTimeoutError:
            logger.error("❌ Kafka timeout while sending message")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to send message: {e}")
            return False
    
    async def consume_messages_async(self, handler: Callable[[Dict[str, Any]], None]):
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
    
    async def close(self):
        """Close all Kafka connections"""
        try:
            if self.async_producer:
                await self.async_producer.stop()
                logger.info("✅ Async producer closed")
            
            if self.async_consumer:
                await self.async_consumer.stop()
                logger.info("✅ Async consumer closed")
            
            if self.producer:
                self.producer.close()
                logger.info("✅ Sync producer closed")
            
            if self.consumer:
                self.consumer.close()
                logger.info("✅ Sync consumer closed")
            
            self.is_connected = False
            logger.info("✅ Kafka service closed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error closing Kafka service: {e}")


# Global Kafka service instance
kafka_service = KafkaService()


# Message handlers
async def handle_planbook_message(message_data: Dict[str, Any]):
    """Handle planbook-specific messages"""
    try:
        data = message_data.get("data", {})
        message_type = data.get("type")
        
        logger.info(f"📋 Processing planbook message type: {message_type}")
        
        # Add your message processing logic here
        # For example:
        if message_type == "lesson_plan_request":
            # Handle lesson plan request from SpringBoot
            pass
        elif message_type == "exam_generation_request":
            # Handle exam generation request
            pass
        elif message_type == "grading_request":
            # Handle grading request
            pass
        else:
            logger.info(f"ℹ️ Unknown message type: {message_type}")
            
    except Exception as e:
        logger.error(f"❌ Error handling planbook message: {e}")


# Register default handlers
kafka_service.register_message_handler("planbook", handle_planbook_message)
