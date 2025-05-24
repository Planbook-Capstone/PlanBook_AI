from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.schemas.ai_schemas import (
    LessonPlanRequest,
    LessonPlanResponse,
    SlideCreationRequest,
    SlideCreationResponse,
    TestGenerationRequest,
    TestGenerationResponse
)
from app.services.ai_service import AIService
from app.core.config import settings

router = APIRouter()
ai_service = AIService()

class ApiCheckResponse(BaseModel):
    status: str
    message: str
    api_key_configured: bool

@router.get("/check-api", response_model=ApiCheckResponse)
async def check_api_connection():
    """
    Check the connection to the Gemini API and verify the API key is configured correctly.
    """
    try:
        # Kiểm tra API key
        api_key_configured = bool(settings.GEMINI_API_KEY and len(settings.GEMINI_API_KEY) > 10)
        print(f"API Key configured: {api_key_configured}")
        print(f"API Key value: {settings.GEMINI_API_KEY[:5]}...{settings.GEMINI_API_KEY[-5:] if settings.GEMINI_API_KEY else ''}")

        if not api_key_configured:
            return ApiCheckResponse(
                status="error",
                message="API key is not configured. Please update your .env file with a valid Gemini API key.",
                api_key_configured=False
            )

        # Kiểm tra kết nối với API
        response = ai_service._call_gemini_api("Hello, this is a test message. Please respond with 'OK'.")

        if response and "OK" in response:
            return ApiCheckResponse(
                status="success",
                message="Successfully connected to Gemini API",
                api_key_configured=True
            )
        else:
            return ApiCheckResponse(
                status="warning",
                message=f"Connected to API but received unexpected response: {response[:100]}...",
                api_key_configured=True
            )
    except Exception as e:
        import traceback
        error_detail = f"Failed to connect to Gemini API: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_detail)

        return ApiCheckResponse(
            status="error",
            message=f"Failed to connect to Gemini API: {str(e)}",
            api_key_configured=api_key_configured if 'api_key_configured' in locals() else False
        )

@router.post("/generate-lesson-plan", response_model=LessonPlanResponse)
async def generate_lesson_plan(request: LessonPlanRequest):
    """
    Generate a lesson plan based on the provided subject, grade level, and topic.
    """
    try:
        # Thêm log để debug
        print(f"Generating lesson plan for subject: {request.subject}, grade: {request.grade_level}")
        print(f"Topic: {request.topic}, Duration: {request.duration} minutes")
        if request.objectives:
            print(f"Objectives: {request.objectives}")

        result = ai_service.generate_lesson_plan(
            subject=request.subject,
            grade_level=request.grade_level,
            topic=request.topic,
            duration=request.duration,
            objectives=request.objectives
        )

        # Kiểm tra kết quả
        if not result:
            raise ValueError("Empty result returned from AI service")

        return LessonPlanResponse(
            lesson_plan=result,
            message="Lesson plan generated successfully"
        )
    except ValueError as e:
        # Xử lý lỗi cụ thể từ service
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log lỗi chi tiết
        import traceback
        error_detail = f"Failed to generate lesson plan: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"Failed to generate lesson plan: {str(e)}")

@router.post("/create-slides", response_model=SlideCreationResponse)
async def create_slides(request: SlideCreationRequest):
    """
    Create presentation slides based on the provided content and preferences.
    """
    try:
        # Thêm log để debug
        print(f"Creating slides with style: {request.style}, num_slides: {request.num_slides}")
        print(f"Content length: {len(request.content)} characters")

        result = ai_service.create_slides(
            content=request.content,
            style=request.style,
            num_slides=request.num_slides
        )

        # Kiểm tra kết quả
        if not result:
            raise ValueError("Empty result returned from AI service")

        return SlideCreationResponse(
            slides=result,
            message="Slides created successfully"
        )
    except ValueError as e:
        # Xử lý lỗi cụ thể từ service
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log lỗi chi tiết
        import traceback
        error_detail = f"Failed to create slides: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"Failed to create slides: {str(e)}")

@router.post("/generate-test", response_model=TestGenerationResponse)
async def generate_test(request: TestGenerationRequest):
    """
    Generate a test paper based on the provided subject, topic, and difficulty level.
    """
    try:
        # Thêm log để debug
        print(f"Generating test for subject: {request.subject}, topic: {request.topic}")
        print(f"Difficulty: {request.difficulty}, Questions: {request.num_questions}")
        print(f"Question types: {request.question_types}")

        result = ai_service.generate_test(
            subject=request.subject,
            topic=request.topic,
            difficulty=request.difficulty,
            num_questions=request.num_questions,
            question_types=request.question_types
        )

        # Kiểm tra kết quả
        if not result:
            raise ValueError("Empty result returned from AI service")

        return TestGenerationResponse(
            test_paper=result,
            message="Test paper generated successfully"
        )
    except ValueError as e:
        # Xử lý lỗi cụ thể từ service
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log lỗi chi tiết
        import traceback
        error_detail = f"Failed to generate test: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"Failed to generate test: {str(e)}")
