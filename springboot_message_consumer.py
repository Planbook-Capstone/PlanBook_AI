"""
Consumer chuyên dụng để nhận messages từ SpringBoot service qua Kafka
"""
import json
import asyncio
import logging
from datetime import datetime
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaConnectionError

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpringBootMessageConsumer:
    """Consumer để nhận messages từ SpringBoot service"""
    
    def __init__(self):
        self.consumer = None
        self.is_running = False
        
    async def initialize(self):
        """Khởi tạo consumer"""
        try:
            self.consumer = AIOKafkaConsumer(
                'planbook',
                bootstrap_servers='14.225.210.212:9092',
                group_id='planbook-springboot-consumer',
                auto_offset_reset='earliest',  # Đọc từ đầu để debug
                value_deserializer=None,  # Không deserialize tự động, để xử lý thủ công
                key_deserializer=None,    # Không deserialize key tự động
                session_timeout_ms=30000,
                heartbeat_interval_ms=3000,
                max_poll_records=10,
                enable_auto_commit=True,
                auto_commit_interval_ms=1000
            )
            
            await self.consumer.start()
            logger.info("✅ SpringBoot message consumer initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize consumer: {e}")
            return False
    
    async def consume_messages(self):
        """Lắng nghe và xử lý messages từ SpringBoot"""
        if not self.consumer:
            logger.error("❌ Consumer not initialized")
            return
            
        logger.info("🔄 Starting to consume messages from SpringBoot...")
        logger.info("🎯 Listening on topic: planbook")
        logger.info("=" * 60)
        
        self.is_running = True
        message_count = 0
        
        try:
            async for message in self.consumer:
                if not self.is_running:
                    break
                    
                try:
                    message_count += 1
                    
                    # Log message info
                    logger.info(f"\n📨 Message #{message_count} received:")
                    logger.info(f"   Topic: {message.topic}")
                    logger.info(f"   Partition: {message.partition}")
                    logger.info(f"   Offset: {message.offset}")
                    logger.info(f"   Key: {message.key.decode('utf-8') if message.key else None}")
                    logger.info(f"   Timestamp: {message.timestamp}")
                    
                    # Debug: Log raw message value
                    logger.info(f"   Raw Value Type: {type(message.value)}")
                    logger.info(f"   Raw Value Length: {len(message.value) if message.value else 0}")
                    logger.info(f"   Raw Value (first 100 chars): {repr(message.value[:100]) if message.value else 'None'}")
                    
                    # Parse message
                    if message.value:
                        # Decode bytes to string first
                        try:
                            if isinstance(message.value, bytes):
                                message_str = message.value.decode('utf-8')
                            else:
                                message_str = str(message.value)
                                
                            logger.info(f"   Decoded String: {repr(message_str)}")
                            
                            # Check if value is empty or just whitespace
                            if not message_str.strip():
                                logger.warning("⚠️ Message string is empty or whitespace")
                                continue
                                
                            try:
                                message_data = json.loads(message_str)
                                await self.process_springboot_message(message_data)
                            except json.JSONDecodeError as e:
                                logger.error(f"❌ JSON decode error: {e}")
                                logger.error(f"   Raw message content: {repr(message_str)}")
                                # Try to handle as plain text message
                                logger.info(f"   Treating as plain text: {message_str}")
                                
                        except UnicodeDecodeError as e:
                            logger.error(f"❌ Unicode decode error: {e}")
                            logger.error(f"   Raw bytes: {repr(message.value)}")
                    else:
                        logger.warning("⚠️ Message value is None")
                    
                    logger.info("=" * 60)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to decode JSON message: {e}")
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
                    
        except KafkaConnectionError as e:
            logger.error(f"❌ Kafka connection error: {e}")
        except Exception as e:
            logger.error(f"❌ Error in message consumption: {e}")
            
    async def process_springboot_message(self, message_data: dict):
        """Xử lý message từ SpringBoot"""
        try:
            # Extract message info
            source = message_data.get("source", "unknown")
            timestamp = message_data.get("timestamp", "")
            data = message_data.get("data", {})
            message_type = data.get("type", "unknown")
            
            logger.info(f"📋 Processing SpringBoot message:")
            logger.info(f"   Source: {source}")
            logger.info(f"   Type: {message_type}")
            logger.info(f"   Timestamp: {timestamp}")
            
            # Filter chỉ messages từ SpringBoot
            if source != "springboot-service":
                logger.info(f"ℹ️ Ignoring message from source: {source}")
                return
            
            # Process based on message type
            if message_type == "lesson_plan_request":
                await self.handle_lesson_plan_request(data)
            elif message_type == "exam_generation_request":
                await self.handle_exam_generation_request(data)
            elif message_type == "grading_request":
                await self.handle_grading_request(data)
            elif message_type == "textbook_processing_request":
                await self.handle_textbook_processing_request(data)
            else:
                logger.warning(f"⚠️ Unknown message type from SpringBoot: {message_type}")
                
        except Exception as e:
            logger.error(f"❌ Error processing SpringBoot message: {e}")
    
    async def handle_lesson_plan_request(self, data: dict):
        """Xử lý lesson plan request từ SpringBoot"""
        logger.info("📚 Processing lesson plan request from SpringBoot")
        
        # Extract data
        subject = data.get("subject", "")
        grade = data.get("grade", "")
        lesson_id = data.get("lesson_id", "")
        requirements = data.get("requirements", [])
        request_id = data.get("request_id", "")
        user_id = data.get("user_id", "")
        
        logger.info(f"   Subject: {subject}")
        logger.info(f"   Grade: {grade}")
        logger.info(f"   Lesson ID: {lesson_id}")
        logger.info(f"   Requirements: {requirements}")
        logger.info(f"   Request ID: {request_id}")
        logger.info(f"   User ID: {user_id}")
        
        # TODO: Implement lesson plan generation logic
        # Có thể call service để tạo lesson plan
        logger.info("✅ Lesson plan request processed successfully")
    
    async def handle_exam_generation_request(self, data: dict):
        """Xử lý exam generation request từ SpringBoot"""
        logger.info("📝 Processing exam generation request from SpringBoot")
        
        # Extract data
        subject = data.get("subject", "")
        grade = data.get("grade", "")
        exam_type = data.get("exam_type", "")
        question_count = data.get("question_count", 0)
        difficulty = data.get("difficulty", "")
        topics = data.get("topics", [])
        request_id = data.get("request_id", "")
        
        logger.info(f"   Subject: {subject}")
        logger.info(f"   Grade: {grade}")
        logger.info(f"   Exam Type: {exam_type}")
        logger.info(f"   Question Count: {question_count}")
        logger.info(f"   Difficulty: {difficulty}")
        logger.info(f"   Topics: {topics}")
        logger.info(f"   Request ID: {request_id}")
        
        # TODO: Implement exam generation logic
        logger.info("✅ Exam generation request processed successfully")
    
    async def handle_grading_request(self, data: dict):
        """Xử lý grading request từ SpringBoot"""
        logger.info("🎯 Processing grading request from SpringBoot")
        
        # Extract data
        exam_id = data.get("exam_id", "")
        student_id = data.get("student_id", "")
        student_answers = data.get("student_answers", [])
        request_id = data.get("request_id", "")
        
        logger.info(f"   Exam ID: {exam_id}")
        logger.info(f"   Student ID: {student_id}")
        logger.info(f"   Number of answers: {len(student_answers)}")
        logger.info(f"   Request ID: {request_id}")
        
        # TODO: Implement grading logic
        logger.info("✅ Grading request processed successfully")
    
    async def handle_textbook_processing_request(self, data: dict):
        """Xử lý textbook processing request từ SpringBoot"""
        logger.info("📖 Processing textbook processing request from SpringBoot")
        
        # Extract data
        textbook_path = data.get("textbook_path", "")
        processing_type = data.get("processing_type", "")
        request_id = data.get("request_id", "")
        processing_options = data.get("processing_options", {})
        
        logger.info(f"   Textbook Path: {textbook_path}")
        logger.info(f"   Processing Type: {processing_type}")
        logger.info(f"   Request ID: {request_id}")
        logger.info(f"   Processing Options: {processing_options}")
        
        # TODO: Implement textbook processing logic
        logger.info("✅ Textbook processing request processed successfully")
    
    async def stop(self):
        """Dừng consumer"""
        logger.info("🛑 Stopping SpringBoot message consumer...")
        self.is_running = False
        
        if self.consumer:
            await self.consumer.stop()
            logger.info("✅ SpringBoot message consumer stopped")

# Main function để chạy consumer
async def main():
    """Main function để chạy SpringBoot message consumer"""
    consumer = SpringBootMessageConsumer()
    
    try:
        # Initialize consumer
        if not await consumer.initialize():
            logger.error("❌ Failed to initialize consumer")
            return
        
        logger.info("🚀 SpringBoot Message Consumer Started")
        logger.info("📱 This consumer will receive messages from SpringBoot service")
        logger.info("⏹️ Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        # Start consuming messages
        await consumer.consume_messages()
        
    except KeyboardInterrupt:
        logger.info("\n⏹️ Consumer stopped by user")
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")
    finally:
        await consumer.stop()

if __name__ == "__main__":
    asyncio.run(main())
