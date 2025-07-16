"""
Slide Generation API Endpoints
API tạo slide tự động từ LessonID và TemplateID
"""

import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any
from datetime import datetime
from googleapiclient.errors import HttpError

from app.models.slide_generation_models import (
    SlideGenerationRequest,
    SlideGenerationResponse,
    SlideGenerationTaskRequest,
    SlideGenerationTaskResponse,
    SlideGenerationError,
    SlideGenerationErrorCodes,
    SlideInfoRequest,
    SlideInfoResponse
)
from app.services.slide_generation_service import get_slide_generation_service
from app.services.mongodb_task_service import get_mongodb_task_service
from app.tasks.slide_generation_tasks import trigger_slide_generation_task, test_slide_generation_task
from app.services.google_slides_service import get_google_slides_service

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
            config_prompt=request.config_prompt,
            presentation_title=request.presentation_title
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
                "/slide-info/{presentation_id}",  # Get full slide information by ID
                "/slide-info-by-url?url=...",  # Get full slide information by URL
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


@router.get("/slide-info/{presentation_id}", response_model=SlideInfoResponse)
async def get_slide_info(presentation_id: str):
    """
    Lấy thông tin chi tiết của Google Slides presentation

    Endpoint này nhận presentation_id (có thể là ID hoặc URL đầy đủ) và trả về thông tin chi tiết về presentation, bao gồm:
    1. Thông tin cơ bản (tiêu đề, số lượng slides, etc.)
    2. Thông tin chi tiết về từng slide (ID, layout, elements)
    3. Thông tin chi tiết về từng element trong slide (ID, loại, nội dung text, etc.)
    4. Thông tin style chi tiết của từng element (font, màu sắc, kích thước, etc.)
    5. Thông tin transform chi tiết của từng element (vị trí, scale, shear, etc.)
    6. Metadata từ Google Drive (thời gian tạo, chỉnh sửa, etc.)

    Thông tin style bao gồm:
    - Shape: font, màu chữ, kích thước chữ, màu nền, đường viền, etc.
    - Image: crop, brightness, contrast, transparency, etc.
    - Table: màu nền cell, alignment, nội dung cell, etc.
    - Video: autoplay, start/end time, mute, etc.
    - Line: weight, dash style, màu sắc, arrow type, etc.

    Thông tin transform bao gồm:
    - translateX, translateY: vị trí của element
    - scaleX, scaleY: tỷ lệ scale của element
    - shearX, shearY: độ nghiêng của element
    - unit: đơn vị đo lường (EMU, PT, etc.)

    Args:
        presentation_id: ID của Google Slides presentation hoặc URL đầy đủ

    Returns:
        SlideInfoResponse: Thông tin chi tiết của presentation
    """
    try:
        logger.info(f"=== GET-SLIDE-INFO ENDPOINT CALLED ===")
        logger.info(f"Request: presentation_id={presentation_id}")

        # Sử dụng presentation_id trực tiếp
        final_presentation_id = presentation_id

        # Lấy Google Slides service
        slides_service = get_google_slides_service()

        if not slides_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Google Slides service not available"
            )

        # Xử lý presentation_id nếu là URL
        if 'docs.google.com/presentation/d/' in final_presentation_id:
            try:
                parts = final_presentation_id.split('/d/')
                if len(parts) > 1:
                    final_presentation_id = parts[1].split('/')[0]
            except:
                pass

        # Lấy thông tin chi tiết
        result = await slides_service.get_presentation_details(final_presentation_id)

        if not result.get("success", False):
            error_message = result.get("error", "Unknown error")
            logger.error(f"❌ Error getting slide info: {error_message}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": SlideGenerationErrorCodes.UNKNOWN_ERROR,
                    "error_message": error_message,
                    "success": False
                }
            )

        logger.info(f"✅ Successfully retrieved slide info for presentation: {final_presentation_id}")
        return result

    except HttpError as e:
        logger.error(f"❌ Google API error: {e}")
        status_code = e.resp.status
        error_code = SlideGenerationErrorCodes.UNKNOWN_ERROR

        if status_code == 404:
            error_code = SlideGenerationErrorCodes.TEMPLATE_NOT_ACCESSIBLE
        elif status_code == 403:
            error_code = SlideGenerationErrorCodes.PERMISSION_DENIED
        elif status_code == 429:
            error_code = SlideGenerationErrorCodes.QUOTA_EXCEEDED

        raise HTTPException(
            status_code=status_code,
            detail={
                "error_code": error_code,
                "error_message": str(e),
                "success": False
            }
        )

    except Exception as e:
        logger.error(f"❌ Error in get_slide_info: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": SlideGenerationErrorCodes.UNKNOWN_ERROR,
                "error_message": str(e),
                "success": False
            }
        )


@router.get("/slide-info-by-url", response_model=SlideInfoResponse)
async def get_slide_info_by_url(url: str):
    """
    Lấy thông tin chi tiết của Google Slides presentation bằng URL

    Endpoint này nhận URL đầy đủ của Google Slides và trả về thông tin chi tiết về presentation, bao gồm:
    1. Thông tin cơ bản (tiêu đề, số lượng slides, etc.)
    2. Thông tin chi tiết về từng slide (ID, layout, elements)
    3. Thông tin chi tiết về từng element trong slide (ID, loại, nội dung text, etc.)
    4. Thông tin style chi tiết của từng element (font, màu sắc, kích thước, etc.)
    5. Thông tin transform chi tiết của từng element (vị trí, scale, shear, etc.)
    6. Metadata từ Google Drive (thời gian tạo, chỉnh sửa, etc.)

    Thông tin style bao gồm:
    - Shape: font, màu chữ, kích thước chữ, màu nền, đường viền, etc.
    - Image: crop, brightness, contrast, transparency, etc.
    - Table: màu nền cell, alignment, nội dung cell, etc.
    - Video: autoplay, start/end time, mute, etc.
    - Line: weight, dash style, màu sắc, arrow type, etc.

    Thông tin transform bao gồm:
    - translateX, translateY: vị trí của element
    - scaleX, scaleY: tỷ lệ scale của element
    - shearX, shearY: độ nghiêng của element
    - unit: đơn vị đo lường (EMU, PT, etc.)

    Args:
        url: URL đầy đủ của Google Slides presentation

    Returns:
        SlideInfoResponse: Thông tin chi tiết của presentation
    """
    try:
        logger.info(f"=== GET-SLIDE-INFO-BY-URL ENDPOINT CALLED ===")
        logger.info(f"Request: url={url}")

        # Lấy Google Slides service
        slides_service = get_google_slides_service()

        if not slides_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="Google Slides service not available"
            )

        # Xử lý URL để lấy presentation_id
        presentation_id = url
        if 'docs.google.com/presentation/d/' in url:
            try:
                parts = url.split('/d/')
                if len(parts) > 1:
                    presentation_id = parts[1].split('/')[0]
            except:
                pass

        # Lấy thông tin chi tiết
        result = await slides_service.get_presentation_details(presentation_id)

        if not result.get("success", False):
            error_message = result.get("error", "Unknown error")
            logger.error(f"❌ Error getting slide info: {error_message}")
            raise HTTPException(
                status_code=500,
                detail={
                    "error_code": SlideGenerationErrorCodes.UNKNOWN_ERROR,
                    "error_message": error_message,
                    "success": False
                }
            )

        logger.info(f"✅ Successfully retrieved slide info for presentation: {presentation_id}")
        return result

    except HttpError as e:
        logger.error(f"❌ Google API error: {e}")
        status_code = e.resp.status
        error_code = SlideGenerationErrorCodes.UNKNOWN_ERROR

        if status_code == 404:
            error_code = SlideGenerationErrorCodes.TEMPLATE_NOT_ACCESSIBLE
        elif status_code == 403:
            error_code = SlideGenerationErrorCodes.PERMISSION_DENIED
        elif status_code == 429:
            error_code = SlideGenerationErrorCodes.QUOTA_EXCEEDED

        raise HTTPException(
            status_code=status_code,
            detail={
                "error_code": error_code,
                "error_message": str(e),
                "success": False
            }
        )

    except Exception as e:
        logger.error(f"❌ Error in get_slide_info_by_url: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": SlideGenerationErrorCodes.UNKNOWN_ERROR,
                "error_message": str(e),
                "success": False
            }
        )


@router.post("/test-celery")
async def test_celery():
    """
    Test endpoint để kiểm tra Celery worker có hoạt động không
    """
    try:
        logger.info("🧪 Testing Celery worker...")

        # Trigger test task
        result = test_slide_generation_task.delay("Test from API endpoint")

        return {
            "success": True,
            "message": "Celery test task triggered",
            "task_id": result.id,
            "status": "Task sent to worker"
        }

    except Exception as e:
        logger.error(f"❌ Celery test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Celery worker may not be running"
        }
