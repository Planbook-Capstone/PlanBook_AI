from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware

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
    auth_endpoints,
    protected_demo,
)
from app.services.lesson_plan_framework_service import lesson_plan_framework_service
from app.services.kafka_service import kafka_service
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
    auth_endpoints.router, prefix="/auth", tags=["Authentication"]
)
api_router.include_router(
    protected_demo.router, prefix="/demo", tags=["Authentication Demo"]
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


async def handle_incoming_message(message_data: dict):
    """Handle incoming messages from other services via Kafka"""
    try:
        print(f"[KAFKA] 📨 Received message from other service: {message_data}")

        # Extract message information
        source = message_data.get("source", "unknown")
        timestamp = message_data.get("timestamp", "")
        data = message_data.get("data", {})
        message_type = data.get("type", "unknown")

        print(f"[KAFKA] 📋 Message details:")
        print(f"  - Source: {source}")
        print(f"  - Type: {message_type}")
        print(f"  - Timestamp: {timestamp}")
        print(f"  - Data: {data}")

        # Process message based on type
        if message_type == "lesson_plan_request":
            await handle_lesson_plan_request(data)
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
        requirements = data.get("requirements", [])

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

        await kafka_service.send_message_async(response_message)
        print(f"[KAFKA] ✅ Sent response for lesson plan request")

    except Exception as e:
        print(f"[KAFKA] ❌ Error handling lesson plan request: {e}")


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

        await kafka_service.send_message_async(response_message)
        print(f"[KAFKA] ✅ Sent response for exam generation request")

    except Exception as e:
        print(f"[KAFKA] ❌ Error handling exam generation request: {e}")


async def handle_grading_request(data: dict):
    """Handle grading request from other service"""
    try:
        print(f"[KAFKA] 🎯 Processing grading request: {data}")

        # Extract request parameters
        exam_id = data.get("exam_id", "")
        student_answers = data.get("student_answers", [])

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

        await kafka_service.send_message_async(response_message)
        print(f"[KAFKA] ✅ Sent response for grading request")

    except Exception as e:
        print(f"[KAFKA] ❌ Error handling grading request: {e}")


async def handle_textbook_processing_request(data: dict):
    """Handle textbook processing request from other service"""
    try:
        print(f"[KAFKA] 📖 Processing textbook processing request: {data}")

        # Extract request parameters
        textbook_path = data.get("textbook_path", "")
        processing_type = data.get("processing_type", "")

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

        await kafka_service.send_message_async(response_message)
        print(f"[KAFKA] ✅ Sent response for textbook processing request")

    except Exception as e:
        print(f"[KAFKA] ❌ Error handling textbook processing request: {e}")
