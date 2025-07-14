"""
Slide Generation API Endpoints
API tạo slide tự động từ LessonID và TemplateID
"""

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from datetime import datetime

from app.models.slide_generation_models import (
    SlideGenerationRequest,
    SlideGenerationResponse,
    SlideGenerationTaskRequest,
    SlideGenerationTaskResponse,
    SlideGenerationError,
    SlideGenerationErrorCodes
)
from app.services.slide_generation_service import get_slide_generation_service
from app.services.mongodb_task_service import get_mongodb_task_service
from app.tasks.slide_generation_tasks import trigger_slide_generation_task

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate-slides", response_model=SlideGenerationResponse)
async def generate_slides_sync(request: SlideGenerationRequest):
    """
    Tạo slide tự động từ lesson_id và template_id (Synchronous)
    
    Endpoint này nhận lesson_id và template_id, sau đó:
    1. Lấy nội dung bài học từ Qdrant
    2. Phân tích cấu trúc template Google Slides
    3. Sử dụng LLM để sinh nội dung slide phù hợp
    4. Tạo Google Slides mới từ template và nội dung đã sinh
    5. Trả về link Google Slides để xem/chỉnh sửa
    
    Args:
        request: SlideGenerationRequest với lesson_id, template_id và config tùy chọn
        
    Returns:
        SlideGenerationResponse: Thông tin slide đã tạo hoặc lỗi nếu thất bại
    """
    try:
        logger.info(f"=== GENERATE-SLIDES SYNC ENDPOINT CALLED ===")
        logger.info(f"Request: lesson_id={request.lesson_id}, template_id={request.template_id}")
        
        # Lấy slide generation service
        slide_service = get_slide_generation_service()
        
        if not slide_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Slide generation service not available"
            )
        
        # Tạo slides
        result = await slide_service.generate_slides_from_lesson(
            lesson_id=request.lesson_id,
            template_id=request.template_id,
            config_prompt=request.config_prompt
        )
        
        if result["success"]:
            logger.info(f"✅ Slides generated successfully: {result['presentation_id']}")
            return SlideGenerationResponse(
                success=True,
                lesson_id=result["lesson_id"],
                template_id=result["original_template_id"],
                presentation_id=result["presentation_id"],
                presentation_title=result["presentation_title"],
                web_view_link=result["web_view_link"],
                slides_created=result["slides_created"],
                template_info=result["template_info"]
            )
        else:
            logger.error(f"❌ Slide generation failed: {result['error']}")
            
            # Xác định error code dựa trên error message
            error_code = SlideGenerationErrorCodes.UNKNOWN_ERROR
            if "lesson content" in result["error"].lower():
                error_code = SlideGenerationErrorCodes.LESSON_NOT_FOUND
            elif "template" in result["error"].lower():
                error_code = SlideGenerationErrorCodes.TEMPLATE_ANALYSIS_FAILED
            elif "llm" in result["error"].lower():
                error_code = SlideGenerationErrorCodes.LLM_GENERATION_FAILED
            elif "slides" in result["error"].lower():
                error_code = SlideGenerationErrorCodes.SLIDES_CREATION_FAILED
            
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": error_code,
                    "error_message": result["error"],
                    "success": False
                }
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in generate_slides_sync: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": SlideGenerationErrorCodes.UNKNOWN_ERROR,
                "error_message": str(e),
                "success": False
            }
        )


@router.post("/generate-slides-async", response_model=SlideGenerationTaskResponse)
async def generate_slides_async(request: SlideGenerationTaskRequest, background_tasks: BackgroundTasks):
    """
    Tạo slide tự động từ lesson_id và template_id (Asynchronous với Celery)
    
    Endpoint này tạo Celery task để xử lý slide generation bất đồng bộ:
    1. Tạo task trong MongoDB với trạng thái PENDING
    2. Khởi chạy Celery task xử lý slide generation
    3. Trả về task_id để client theo dõi progress
    
    Args:
        request: SlideGenerationTaskRequest với lesson_id, template_id và config
        background_tasks: FastAPI BackgroundTasks để xử lý async
        
    Returns:
        SlideGenerationTaskResponse: task_id và trạng thái để theo dõi
    """
    try:
        logger.info(f"=== GENERATE-SLIDES ASYNC ENDPOINT CALLED ===")
        logger.info(f"Request: lesson_id={request.lesson_id}, template_id={request.template_id}")
        
        # Trigger Celery task cho slide generation
        task_id = await trigger_slide_generation_task(
            lesson_id=request.lesson_id,
            template_id=request.template_id,
            config_prompt=request.config_prompt,
            presentation_title=request.presentation_title
        )

        logger.info(f"✅ Slide generation task created: {task_id}")
        
        return SlideGenerationTaskResponse(
            task_id=task_id,
            status="PENDING",
            message="Task tạo slide đã được khởi tạo. Sử dụng task_id để theo dõi tiến trình."
        )
        
    except Exception as e:
        logger.error(f"❌ Error creating slide generation task: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": SlideGenerationErrorCodes.UNKNOWN_ERROR,
                "error_message": f"Failed to create task: {str(e)}",
                "success": False
            }
        )


@router.get("/template-info/{template_id}")
async def get_template_info(template_id: str):
    """
    Lấy thông tin cấu trúc của Google Slides template (sử dụng analyze_template_structure cũ)

    Args:
        template_id: ID của Google Slides template

    Returns:
        Dict chứa thông tin slides và elements của template
    """
    try:
        logger.info(f"=== GET-TEMPLATE-INFO ENDPOINT CALLED ===")
        logger.info(f"Template ID: {template_id}")

        # Lấy slide generation service
        slide_service = get_slide_generation_service()

        if not slide_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Slide generation service not available"
            )

        # Bước 1: Copy template để phân tích (giống workflow chính)
        new_title = f"Template Analysis - {template_id}"
        copy_and_analyze_result = await slide_service.slides_service.copy_and_analyze_template(template_id, new_title)

        if not copy_and_analyze_result["success"]:
            logger.error(f"❌ Template copy failed: {copy_and_analyze_result['error']}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": SlideGenerationErrorCodes.TEMPLATE_ANALYSIS_FAILED,
                    "error_message": copy_and_analyze_result["error"],
                    "success": False
                }
            )

        # Bước 2: Phân tích template với placeholder detection (workflow mới)
        analyzed_template = slide_service._analyze_template_with_placeholders(copy_and_analyze_result)

        # Bước 3: Xóa temporary copy
        try:
            await slide_service.slides_service.delete_presentation(copy_and_analyze_result["copied_presentation_id"])
            logger.info(f"✅ Temporary copy deleted: {copy_and_analyze_result['copied_presentation_id']}")
        except Exception as e:
            logger.warning(f"⚠️ Could not delete temporary copy: {e}")

        # Tạo response với thông tin đầy đủ
        result = {
            "success": True,
            "template_id": template_id,
            "title": copy_and_analyze_result.get("presentation_title", "Unknown"),
            "slide_count": analyzed_template.get("total_slides", 0),
            "slides": analyzed_template.get("slides", []),
            "analysis_info": {
                "placeholder_detection": "enabled",
                "workflow": "new_2_phase_ai",
                "features": [
                    "Type detection",
                    "max_length calculation",
                    "font_size extraction",
                    "font_family extraction",
                    "font_style extraction",
                    "slide_description generation"
                ]
            }
        }

        logger.info(f"✅ Template analyzed successfully with new workflow: {result.get('title', 'Unknown')}")
        logger.info(f"📊 Found {result['slide_count']} slides with placeholder detection")

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in get_template_info: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": SlideGenerationErrorCodes.UNKNOWN_ERROR,
                "error_message": str(e),
                "success": False
            }
        )


@router.post("/copy-and-analyze-template")
async def copy_and_analyze_template(request: dict):
    """
    Test endpoint: Copy template và phân tích cấu trúc (QUY TRÌNH MỚI)

    Args:
        request: {"template_id": "...", "new_title": "..."}

    Returns:
        Dict chứa thông tin file đã copy và cấu trúc
    """
    try:
        logger.info(f"=== COPY-AND-ANALYZE-TEMPLATE ENDPOINT CALLED ===")

        template_id = request.get("template_id")
        new_title = request.get("new_title", f"Test Copy - {datetime.now().strftime('%Y%m%d_%H%M%S')}")

        if not template_id:
            raise HTTPException(
                status_code=400,
                detail="template_id is required"
            )

        logger.info(f"Template ID: {template_id}")
        logger.info(f"New Title: {new_title}")

        # Lấy slide generation service
        slide_service = get_slide_generation_service()

        if not slide_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Slide generation service not available"
            )

        # Copy và phân tích template
        result = await slide_service.slides_service.copy_and_analyze_template(template_id, new_title)

        if result["success"]:
            logger.info(f"✅ Template copied and analyzed successfully: {result['copied_presentation_id']}")
            return result
        else:
            logger.error(f"❌ Copy and analyze failed: {result['error']}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": SlideGenerationErrorCodes.TEMPLATE_ANALYSIS_FAILED,
                    "error_message": result["error"],
                    "success": False
                }
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in copy_and_analyze_template: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": SlideGenerationErrorCodes.UNKNOWN_ERROR,
                "error_message": str(e),
                "success": False
            }
        )


@router.get("/health")
async def health_check():
    """
    Health check endpoint cho slide generation service
    
    Returns:
        Dict chứa trạng thái các service liên quan
    """
    try:
        slide_service = get_slide_generation_service()
        
        return {
            "status": "healthy" if slide_service.is_available() else "unhealthy",
            "services": {
                "slide_generation": slide_service.is_available(),
                "llm_service": slide_service.llm_service.is_available() if slide_service.llm_service else False,
                "google_slides": slide_service.slides_service.is_available() if slide_service.slides_service else False
            },
            "available_endpoints": [
                "/generate-slides",  # Sync slide generation
                "/generate-slides-async",  # Async slide generation
                "/template-info/{template_id}",  # Template analysis
                "/health"  # Health check
            ]
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "services": {
                "slide_generation": False,
                "llm_service": False,
                "google_slides": False
            }
        }
