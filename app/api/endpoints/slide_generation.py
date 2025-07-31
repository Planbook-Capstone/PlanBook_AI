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
