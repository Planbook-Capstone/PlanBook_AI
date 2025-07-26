from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
import json

from app.api.endpoints.auto_grading import auto_grading_endpoint
from app.core.config import settings
from app.api.endpoints import (
    pdf_endpoints,
    tasks,
    celery_health,
    lesson_plan,
    exam_generation,
    exam_import,
    kafka_endpoints,
    llm_endpoints,
    slide_generation,
    auth_endpoints,
    protected_demo,
    rag_endpoints,
)

from app.services.kafka_service import kafka_service
from app.core.kafka_config import get_responses_topic
from app.api.endpoints import auto_grading

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
    debug=settings.DEBUG,
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create API router
api_router = APIRouter()

# Include routers from endpoints
api_router.include_router(pdf_endpoints.router, prefix="/pdf", tags=["Books Services"])
api_router.include_router(tasks.router, prefix="/tasks", tags=["Task Management"])
api_router.include_router(celery_health.router, tags=["Celery Health"])
api_router.include_router(
    lesson_plan.router, prefix="/lesson", tags=["Lesson Planning"]
)
api_router.include_router(
    auto_grading.router, prefix="/auto_grading", tags=["Auto Grading"]
)
api_router.include_router(
    exam_generation.router, prefix="/exam", tags=["Exam Generation"]
)
api_router.include_router(
    exam_import.router, prefix="/exam", tags=["Exam Import"]
)
api_router.include_router(
    kafka_endpoints.router, prefix="/kafka", tags=["Kafka Integration"]
)
api_router.include_router(
    llm_endpoints.router, prefix="/llm", tags=["LLM Services"]
)
api_router.include_router(
    slide_generation.router, prefix="/slides", tags=["Slide Generation"]
)
api_router.include_router(
    auth_endpoints.router, prefix="/auth", tags=["Authentication"]
)
api_router.include_router(
    protected_demo.router, prefix="/demo", tags=["Authentication Demo"]
)
api_router.include_router(
    rag_endpoints.router, prefix="/rag", tags=["RAG Services"]
)
# Add API router to app
app.include_router(api_router, prefix=settings.API_PREFIX)

@app.get("/")
async def root():
    return {
        "message": "Welcome to PlanBook AI Service",
        "docs": f"{settings.API_PREFIX}/docs",
    }


@app.on_event("startup")
async def startup_event():
    """Initialize only essential services on startup"""
    # Note: Services now use lazy initialization - they will initialize when first used
    # Only initialize services that absolutely need to be ready at startup

    try:
        # Kafka service cần khởi tạo ngay để listen messages từ external services
        await kafka_service.initialize()
        print("[OK] Kafka Service initialized successfully")

        # Start Kafka consumer in background to listen for messages from other services
        import asyncio
        asyncio.create_task(start_kafka_consumer_background())
        print("[OK] Kafka consumer started in background")

    except Exception as e:
        print(f"[ERROR] Failed to initialize Kafka Service: {e}")
        print("[INFO] Kafka service will be available when Kafka server is running")

    # Other services (LessonPlanFrameworkService, etc.) will initialize lazily when first used
    print("[OK] Application startup completed - services will initialize when first used")


async def start_kafka_consumer_background():
    """Start Kafka consumer in background to receive messages from other services"""
    try:
        await kafka_service.consume_messages_async(handle_incoming_message)
    except Exception as e:
        print(f"[ERROR] Kafka consumer error: {e}")


async def handle_incoming_message(data: dict):
    """Handle incoming messages from other services via Kafka"""
    try:
        message_data_str = data.get("payload", "{}")
        message_data = json.loads(message_data_str)
        print(f"[KAFKA] 📨 Received message from other service: {message_data}")

        # Extract message information
        source = message_data.get("source", "unknown")
        timestamp = message_data.get("timestamp", "")
        data = message_data.get("data", {})

        # Message type có thể ở top level hoặc trong data
        message_type = message_data.get("type") or data.get("type", "unknown")

        print(f"[KAFKA] 📋 Message details:")
        print(f"  - Source: {source}")
        print(f"  - Type: {message_type}")
        print(f"  - Timestamp: {timestamp}")
        print(f"  - Data: {data}")

        # Process message based on type
        if message_type == "lesson_plan_request":
            await handle_lesson_plan_request(data)
        elif message_type == "Tạo giáo án":
            await handle_lesson_plan_content_generation_request(data)
        elif message_type == "Tạo Slide":
            await handle_slide_generation_request(data)
        elif message_type == "Tạo đề thi thông minh":
            await handle_smart_exam_generation_request(data)
        elif message_type == "exam_generation_request":
            await handle_exam_generation_request(data)
        elif message_type == "grading_request":
            await handle_grading_request(data)
        elif message_type == "textbook_processing_request":
            await handle_textbook_processing_request(data)
        else:
            print(f"[KAFKA] ⚠️ Unknown message type: {message_type}")

    except Exception as e:
        print(f"[KAFKA] ❌ Error handling incoming message: {e}")


async def handle_lesson_plan_request(data: dict):
    """Handle lesson plan request from other service"""
    try:
        print(f"[KAFKA] 📚 Processing lesson plan request: {data}")

        # Extract request parameters
        subject = data.get("subject", "")
        grade = data.get("grade", "")
        lesson_id = data.get("lesson_id", "")

        # Process the lesson plan request
        # You can call your existing lesson plan service here
        # For now, just log the request
        print(f"[KAFKA] 📝 Lesson plan request for {subject} grade {grade}, lesson {lesson_id}")

        # Send response back via Kafka if needed
        response_message = {
            "type": "lesson_plan_response",
            "data": {
                "status": "received",
                "lesson_id": lesson_id,
                "message": "Lesson plan request received and being processed",
                "timestamp": data.get("timestamp", "")
            }
        }

        await kafka_service.send_message_async(response_message, topic=get_responses_topic())
        print(f"[KAFKA] ✅ Sent response for lesson plan request")

    except Exception as e:
        print(f"[KAFKA] ❌ Error handling lesson plan request: {e}")


async def _send_error_response(user_id: str, error_message: str, timestamp: str):
    """Send error response back to SpringBoot via Kafka"""
    try:
        error_response = {
            "type": "lesson_plan_content_generation_response",
            "data": {
                "status": "error",
                "user_id": user_id,
                "error": error_message,
                "message": "Failed to process lesson plan content generation request",
                "timestamp": timestamp
            }
        }

        await kafka_service.send_message_async(error_response, topic=get_responses_topic(), key=user_id)
        print(f"[KAFKA] ✅ Sent error response for user {user_id}: {error_message}")

    except Exception as e:
        print(f"[KAFKA] ❌ Failed to send error response: {e}")


async def handle_lesson_plan_content_generation_request(data: dict):
    """Handle lesson plan content generation request from SpringBoot"""
    try:
        print(f"[KAFKA] 📝 Processing lesson plan content generation request: {data}")

        # Extract request parameters
        user_id = data.get("user_id", "")
        lesson_plan_json = data.get("input", {})
        lesson_id = data.get("lesson_id", "")
        # book_id = data.get("book_id", "")  # Lấy bookID từ SpringBoot message
        tool_log_id = data.get("tool_log_id","")
        if not user_id:
            print(f"[KAFKA] ❌ Missing user_id in lesson plan content generation request")
            return

        if not lesson_plan_json:
            print(f"[KAFKA] ❌ Missing lesson_plan_json in lesson plan content generation request")
            await _send_error_response(user_id, "Missing lesson_plan_json in request", data.get("timestamp", ""))
            return

        # Remove validation for 'id' field since SpringBoot lesson_plan_json has different format
        # The lesson_plan_json from SpringBoot contains title and sections, not id/type/status
        print(f"[KAFKA] 📋 Lesson plan content generation for user {user_id}, lesson {lesson_id}")

        print(f"[KAFKA] 📋 Lesson plan content generation for user {user_id}, lesson {lesson_id}")

        # Import the lesson plan endpoint function
        from app.api.endpoints.lesson_plan import LessonPlanContentRequest
        from app.services.background_task_processor import get_background_task_processor

        # Create request object
        request_obj = LessonPlanContentRequest(
            lesson_plan_json=lesson_plan_json,
            lesson_id=lesson_id,
            user_id=user_id,
            tool_log_id=tool_log_id
            # book_id=book_id
        )

        # Create task using background task processor
        task_id = await get_background_task_processor().create_lesson_plan_content_task(
            lesson_plan_json=request_obj.lesson_plan_json,
            lesson_id=request_obj.lesson_id,
            user_id=request_obj.user_id,
            tool_log_id=request_obj.tool_log_id
            # book_id=request_obj.book_id
        )

        # Send initial response back via Kafka
        response_message = {
            "type": "lesson_plan_content_generation_response",
            "data": {
                "status": "accepted",
                "tool_log_id": tool_log_id,
                "task_id": task_id,
                "user_id": user_id,
                "message": "Lesson plan content generation task created successfully",
                "timestamp": data.get("timestamp", "")
            }
        }

        await kafka_service.send_message_async(response_message, topic=get_responses_topic(), key=user_id)
        print(f"[KAFKA] ✅ Sent response for lesson plan content generation request - Task ID: {task_id}")

    except Exception as e:
        print(f"[KAFKA] ❌ Error handling lesson plan content generation request: {e}")

        # Extract user_id for error response
        user_id = data.get("user_id", "")
        timestamp = data.get("timestamp", "")

        # Send error response back via Kafka using the helper function
        if user_id:
            await _send_error_response(user_id, str(e), timestamp)
        else:
            print(f"[KAFKA] ❌ Cannot send error response: missing user_id in data")


async def _send_smart_exam_error_response(user_id: str, error_message: str, timestamp: str, tool_log_id: str = ""):
    """Send error response for smart exam generation back to SpringBoot via Kafka"""
    try:
        error_response = {
            "type": "smart_exam_generation_response",
            "data": {
                "status": "error",
                "user_id": user_id,
                "tool_log_id": tool_log_id,
                "error": error_message,
                "message": "Failed to process smart exam generation request",
                "timestamp": timestamp
            }
        }

        await kafka_service.send_message_async(error_response, topic=get_responses_topic(), key=user_id)
        print(f"[KAFKA] ✅ Sent smart exam error response for user {user_id}: {error_message}")

    except Exception as e:
        print(f"[KAFKA] ❌ Failed to send smart exam error response: {e}")


async def _send_slide_generation_error_response(user_id: str, error_message: str, timestamp: str, tool_log_id: str = ""):
    """Send error response for slide generation back to SpringBoot via Kafka"""
    try:
        error_response = {
            "type": "slide_generation_response",
            "data": {
                "status": "error",
                "user_id": user_id,
                "tool_log_id": tool_log_id,
                "error": error_message,
                "message": "Failed to process slide generation request",
                "timestamp": timestamp
            }
        }

        await kafka_service.send_message_async(error_response, topic=get_responses_topic(), key=user_id)
        print(f"[KAFKA] ✅ Sent slide generation error response for user {user_id}: {error_message}")

    except Exception as e:
        print(f"[KAFKA] ❌ Failed to send slide generation error response: {e}")


async def handle_smart_exam_generation_request(data: dict):
    """Handle smart exam generation request from SpringBoot"""
    try:
        print(f"[KAFKA] 📝 Processing smart exam generation request: {data}")

        # Extract request parameters
        user_id = data.get("user_id", "")
        exam_request_data = data.get("exam_request", {})
        tool_log_id = data.get("tool_log_id", "")

        if not user_id:
            print(f"[KAFKA] ❌ Missing user_id in smart exam generation request")
            return

        if not exam_request_data:
            print(f"[KAFKA] ❌ Missing exam_request in smart exam generation request")
            await _send_smart_exam_error_response(user_id, "Missing exam_request in request", data.get("timestamp", ""), tool_log_id)
            return

        print(f"[KAFKA] 📋 Smart exam generation for user {user_id}")

        # Add user_id to exam_request_data for processing
        exam_request_data["user_id"] = user_id

        # Import background task processor
        from app.services.background_task_processor import get_background_task_processor

        # Create task using background task processor
        background_processor = get_background_task_processor()
        task_result = await background_processor.create_smart_exam_task(
            request_data=exam_request_data
        )

        if not task_result.get("success", False):
            error_msg = f"Không thể tạo task: {task_result.get('error', 'Lỗi không xác định')}"
            await _send_smart_exam_error_response(user_id, error_msg, data.get("timestamp", ""), tool_log_id)
            return

        task_id = task_result.get("task_id")

        # Send initial response back via Kafka
        response_message = {
            "type": "smart_exam_generation_response",
            "data": {
                "status": "accepted",
                "tool_log_id": tool_log_id,
                "task_id": task_id,
                "user_id": user_id,
                "message": "Smart exam generation task created successfully",
                "timestamp": data.get("timestamp", "")
            }
        }

        await kafka_service.send_message_async(response_message, topic=get_responses_topic(), key=user_id)
        print(f"[KAFKA] ✅ Sent response for smart exam generation request - Task ID: {task_id}")

    except Exception as e:
        print(f"[KAFKA] ❌ Error handling smart exam generation request: {e}")

        # Extract user_id for error response
        user_id = data.get("user_id", "")
        timestamp = data.get("timestamp", "")
        tool_log_id = data.get("tool_log_id", "")

        # Send error response back via Kafka
        if user_id:
            await _send_smart_exam_error_response(user_id, str(e), timestamp, tool_log_id)
        else:
            print(f"[KAFKA] ❌ Cannot send error response: missing user_id in data")


async def handle_slide_generation_request(data: dict):
    """Handle slide generation request from SpringBoot"""
    try:
        print(f"[KAFKA] 🎨 Processing slide generation request: {data}")

        # Extract request parameters
        user_id = data.get("user_id", "")
        slides_data = data.get("slides", [])
        lesson_id = data.get("lesson_id", "")
        tool_log_id = data.get("tool_log_id", "")

        if not user_id:
            print(f"[KAFKA] ❌ Missing user_id in slide generation request")
            return

        if not slides_data:
            print(f"[KAFKA] ❌ Missing slides data in slide generation request")
            await _send_slide_generation_error_response(user_id, "Missing slides data in request", data.get("timestamp", ""), tool_log_id)
            return

        print(f"[KAFKA] 📋 Slide generation for user {user_id}, lesson {lesson_id}")

        # Import the slide generation task function
        from app.tasks.slide_generation_tasks import trigger_json_template_task

        # Create template_json from slides data
        template_json = {
            "slides": slides_data,
            "version": "1.0",
            "slideFormat": "16:9"
        }

        # Trigger Celery task for JSON template processing
        task_id = await trigger_json_template_task(
            lesson_id=lesson_id,
            template_json=template_json,
            config_prompt=None,
            user_id=user_id
        )

        # Send initial response back via Kafka
        response_message = {
            "type": "slide_generation_response",
            "data": {
                "status": "accepted",
                "tool_log_id": tool_log_id,
                "task_id": task_id,
                "user_id": user_id,
                "message": "Slide generation task created successfully",
                "timestamp": data.get("timestamp", "")
            }
        }

        await kafka_service.send_message_async(response_message, topic=get_responses_topic(), key=user_id)
        print(f"[KAFKA] ✅ Sent response for slide generation request - Task ID: {task_id}")

    except Exception as e:
        print(f"[KAFKA] ❌ Error handling slide generation request: {e}")

        # Extract user_id for error response
        user_id = data.get("user_id", "")
        timestamp = data.get("timestamp", "")
        tool_log_id = data.get("tool_log_id", "")

        # Send error response back via Kafka
        if user_id:
            await _send_slide_generation_error_response(user_id, str(e), timestamp, tool_log_id)
        else:
            print(f"[KAFKA] ❌ Cannot send error response: missing user_id in data")


async def handle_exam_generation_request(data: dict):
    """Handle exam generation request from other service"""
    try:
        print(f"[KAFKA] 📝 Processing exam generation request: {data}")

        # Extract request parameters
        subject = data.get("subject", "")
        grade = data.get("grade", "")
        exam_type = data.get("exam_type", "")
        question_count = data.get("question_count", 0)

        print(f"[KAFKA] 📊 Exam generation request for {subject} grade {grade}, {question_count} questions")

        # Send response back via Kafka if needed
        response_message = {
            "type": "exam_generation_response",
            "data": {
                "status": "received",
                "exam_type": exam_type,
                "message": "Exam generation request received and being processed",
                "timestamp": data.get("timestamp", "")
            }
        }

        await kafka_service.send_message_async(response_message, topic=get_responses_topic())
        print(f"[KAFKA] ✅ Sent response for exam generation request")

    except Exception as e:
        print(f"[KAFKA] ❌ Error handling exam generation request: {e}")


async def handle_grading_request(data: dict):
    """Handle grading request from other service"""
    try:
        print(f"[KAFKA] 🎯 Processing grading request: {data}")

        # Extract request parameters
        exam_id = data.get("exam_id", "")

        print(f"[KAFKA] 📊 Grading request for exam {exam_id}")

        # Send response back via Kafka if needed
        response_message = {
            "type": "grading_response",
            "data": {
                "status": "received",
                "exam_id": exam_id,
                "message": "Grading request received and being processed",
                "timestamp": data.get("timestamp", "")
            }
        }

        await kafka_service.send_message_async(response_message, topic=get_responses_topic())
        print(f"[KAFKA] ✅ Sent response for grading request")

    except Exception as e:
        print(f"[KAFKA] ❌ Error handling grading request: {e}")


async def handle_textbook_processing_request(data: dict):
    """Handle textbook processing request from other service"""
    try:
        print(f"[KAFKA] 📖 Processing textbook processing request: {data}")

        # Extract request parameters
        textbook_path = data.get("textbook_path", "")

        print(f"[KAFKA] 📚 Textbook processing request for {textbook_path}")

        # Send response back via Kafka if needed
        response_message = {
            "type": "textbook_processing_response",
            "data": {
                "status": "received",
                "textbook_path": textbook_path,
                "message": "Textbook processing request received and being processed",
                "timestamp": data.get("timestamp", "")
            }
        }

        await kafka_service.send_message_async(response_message, topic=get_responses_topic())
        print(f"[KAFKA] ✅ Sent response for textbook processing request")

    except Exception as e:
        print(f"[KAFKA] ❌ Error handling textbook processing request: {e}")
