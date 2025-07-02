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
    """Initialize services on startup"""
    try:
        await lesson_plan_framework_service.initialize()
        print("[OK] Lesson Plan Framework Service initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Lesson Plan Framework Service: {e}")

    try:
        await kafka_service.initialize()
        print("[OK] Kafka Service initialized successfully")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Kafka Service: {e}")
        print("[INFO] Kafka service will be available when Kafka server is running")
