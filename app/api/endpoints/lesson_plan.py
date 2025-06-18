from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from pydantic import BaseModel
from typing import Optional, List
import logging

from app.services.lesson_plan_framework_service import lesson_plan_framework_service

logger = logging.getLogger(__name__)
router = APIRouter()


# Pydantic models for request/response
class LessonPlanRequest(BaseModel):
    subject: str
    grade: str
    topic: str
    duration: int  # minutes
    learning_objectives: List[str]
    materials_needed: List[str]
    student_level: str
    special_requirements: Optional[str] = None
    framework_id: Optional[str] = None  # ID của khung giáo án đã upload


class LessonPlanResponse(BaseModel):
    lesson_plan_id: str
    content: dict
    framework_used: str
    created_at: str


@router.post("/lesson-plan-framework")
async def upload_lesson_plan_framework(
    framework_name: str = Form(...), framework_file: UploadFile = File(...)
):
    """
    Upload khung giáo án template
    """
    try:
        # Validate file type
        if not framework_file.filename or not framework_file.filename.endswith(
            (".pdf", ".docx", ".doc", ".txt")
        ):
            raise HTTPException(
                status_code=400, detail="Chỉ hỗ trợ file PDF, Word hoặc Text"
            )

        # Process framework file using service
        result = await lesson_plan_framework_service.process_framework_file(
            framework_name, framework_file
        )

        return {
            "message": "Khung giáo án đã được upload thành công",
            "framework_name": result["name"],
            "framework_id": result["framework_id"],
            "filename": result["filename"],
            "structure": result["structure"],
            "created_at": result["created_at"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Lỗi khi upload khung giáo án: {str(e)}"
        )


@router.post("/lesson-plan-generate", response_model=LessonPlanResponse)
async def generate_lesson_plan(request: LessonPlanRequest):
    """
    Tạo giáo án dựa trên dữ liệu người dùng và khung giáo án
    """
    try:
        # Validate required fields
        if not request.subject or not request.topic:
            raise HTTPException(status_code=400, detail="Môn học và chủ đề là bắt buộc")

        # TODO: Implement lesson plan generation logic
        # 1. Load framework template if framework_id provided
        # 2. Process user input data
        # 3. Generate lesson plan using AI/LLM
        # 4. Format according to framework

        # Mock response for now
        generated_plan = {
            "title": f"Giáo án {request.subject} - {request.topic}",
            "grade": request.grade,
            "duration": request.duration,
            "objectives": request.learning_objectives,
            "materials": request.materials_needed,
            "activities": [
                {
                    "phase": "Khởi động",
                    "duration": 10,
                    "content": "Hoạt động khởi động...",
                },
                {
                    "phase": "Phát triển",
                    "duration": request.duration - 20,
                    "content": "Nội dung chính...",
                },
                {"phase": "Củng cố", "duration": 10, "content": "Hoạt động củng cố..."},
            ],
        }

        return LessonPlanResponse(
            lesson_plan_id="plan_123",
            content=generated_plan,
            framework_used=request.framework_id or "default",
            created_at="2025-06-18T10:00:00Z",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi tạo giáo án: {str(e)}")


@router.get("/frameworks")
async def get_available_frameworks():
    """
    Lấy danh sách các khung giáo án có sẵn
    """
    try:
        logger.info("Getting all frameworks...")
        frameworks = await lesson_plan_framework_service.get_all_frameworks()
        logger.info(f"Found {len(frameworks)} frameworks")
        return {"frameworks": frameworks}
    except Exception as e:
        logger.error(f"Error getting frameworks: {e}")
        raise HTTPException(
            status_code=500, detail=f"Lỗi khi lấy danh sách frameworks: {str(e)}"
        )


@router.get("/frameworks/{framework_id}")
async def get_framework_by_id(framework_id: str):
    """
    Lấy thông tin chi tiết của một framework
    """
    try:
        framework = await lesson_plan_framework_service.get_framework_by_id(
            framework_id
        )
        if not framework:
            raise HTTPException(status_code=404, detail="Không tìm thấy framework")
        return framework
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy framework: {str(e)}")


@router.delete("/frameworks/{framework_id}")
async def delete_framework(framework_id: str):
    """
    Xóa một framework
    """
    try:
        success = await lesson_plan_framework_service.delete_framework(framework_id)
        if not success:
            raise HTTPException(status_code=404, detail="Không tìm thấy framework")
        return {"message": "Framework đã được xóa thành công"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi xóa framework: {str(e)}")
