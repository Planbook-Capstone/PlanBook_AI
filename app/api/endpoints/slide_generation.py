"""
Slide Generation API Endpoints
API xử lý JSON template từ frontend
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from app.models.slide_generation_models import (
    SlideGenerationErrorCodes,
    JsonTemplateRequest,
    JsonTemplateResponse,
    SlideGenerationTaskResponse
)
from app.tasks.slide_generation_tasks import trigger_json_template_task
from app.services.json_template_service import get_json_template_service

logger = logging.getLogger(__name__)
router = APIRouter()





@router.post("/process-json-template", response_model=JsonTemplateResponse)
async def process_json_template(request: JsonTemplateRequest):
    """
    Xử lý JSON template từ frontend với nội dung bài học

    Endpoint này nhận danh sách slides đã được phân tích sẵn từ frontend và lesson_id, sau đó:
    1. Lấy nội dung bài học từ Qdrant
    2. Sử dụng trực tiếp slides đã được phân tích (có sẵn description trong mỗi slide)
    3. Sử dụng LLM để sinh nội dung phù hợp với template
    4. Map nội dung vào slides và trả về

    Args:
        request: JsonTemplateRequest với lesson_id, danh sách slides đã phân tích và config tùy chọn

    Returns:
        JsonTemplateResponse: JSON template đã được xử lý với nội dung
    """
    try:
        logger.info(f"=== PROCESS-JSON-TEMPLATE ENDPOINT CALLED ===")
        logger.info(f"Request: lesson_id={request.lesson_id}")
        logger.info(f"Slides count: {len(request.slides)}")

        # Lấy JSON template service
        json_service = get_json_template_service()

        if not json_service.is_available():
            raise HTTPException(
                status_code=503,
                detail="JSON template service not available"
            )

        # Tạo template_json từ request format mới
        template_json = {
            "slides": request.slides,
            "version": "1.0",
            "slideFormat": "16:9"
        }

        # Xử lý JSON template
        result = await json_service.process_json_template(
            lesson_id=request.lesson_id,
            template_json=template_json,
            config_prompt=request.config_prompt,
            book_id=None  # No book_id for HTTP endpoint calls
        )

        if result["success"]:
            logger.info(f"✅ JSON template processed successfully: {result['slides_created']} slides")
            return JsonTemplateResponse(
                success=True,
                lesson_id=result["lesson_id"],
                processed_template=result["processed_template"],
                slides_created=result["slides_created"]
            )
        else:
            logger.error(f"❌ JSON template processing failed: {result['error']}")

            # Xác định error code dựa trên error message
            error_code = SlideGenerationErrorCodes.UNKNOWN_ERROR
            if "lesson content" in result["error"].lower():
                error_code = SlideGenerationErrorCodes.LESSON_NOT_FOUND
            elif "llm" in result["error"].lower():
                error_code = SlideGenerationErrorCodes.LLM_GENERATION_FAILED

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
        logger.error(f"❌ Unexpected error in process_json_template: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": SlideGenerationErrorCodes.UNKNOWN_ERROR,
                "error_message": str(e),
                "success": False
            }
        )





@router.post("/process-json-template-async", response_model=SlideGenerationTaskResponse)
async def process_json_template_async(request: JsonTemplateRequest):
    """
    Xử lý JSON template từ frontend với nội dung bài học (Asynchronous với Celery)

    Endpoint này tạo Celery task để xử lý JSON template bất đồng bộ:
    1. Tạo task trong MongoDB với trạng thái PENDING
    2. Khởi chạy Celery task xử lý JSON template
    3. Trả về task_id để client theo dõi progress
    4. Cập nhật progress theo từng slide hoàn thành

    Args:
        request: JsonTemplateRequest với lesson_id, slides đã phân tích và config tùy chọn

    Returns:
        SlideGenerationTaskResponse: task_id và trạng thái để theo dõi
    """
    try:
        logger.info(f"=== PROCESS-JSON-TEMPLATE-ASYNC ENDPOINT CALLED ===")
        logger.info(f"Request: lesson_id={request.lesson_id}")
        logger.info(f"Slides count: {len(request.slides)}")

        # Tạo template_json từ request format mới
        template_json = {
            "slides": request.slides,
            "version": "1.0",
            "slideFormat": "16:9"
        }

        # Trigger Celery task cho JSON template processing
        task_id = await trigger_json_template_task(
            lesson_id=request.lesson_id,
            template_json=template_json,
            config_prompt=request.config_prompt,
            user_id=None,  
            book_id=None  
        )

        logger.info(f"✅ JSON template processing task created: {task_id}")

        return SlideGenerationTaskResponse(
            task_id=task_id,
            status="PENDING",
            message="Task xử lý JSON template đã được khởi tạo. Sử dụng task_id để theo dõi tiến trình."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in process_json_template_async: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error_code": SlideGenerationErrorCodes.UNKNOWN_ERROR,
                "error_message": str(e),
                "success": False
            }
        )
