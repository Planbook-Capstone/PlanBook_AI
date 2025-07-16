"""
CV Processing Tasks for Celery Workers
Các task xử lý CV chạy trong background workers
"""

import asyncio
import logging
from typing import Dict, Any
from celery import current_task

from app.core.celery_app import celery_app, task_with_retry
from app.services.mongodb_task_service import get_mongodb_task_service

logger = logging.getLogger(__name__)

def run_async_task(coro):
    """Helper để chạy async function trong Celery task"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@task_with_retry()
def process_cv_task(self, task_id: str) -> Dict[str, Any]:
    """
    Celery task xử lý CV/Resume
    
    Args:
        task_id: ID của task trong MongoDB
        
    Returns:
        Dict kết quả xử lý
    """
    print(f"Starting CV processing task: {task_id}")
    
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 0, 'message': 'Starting CV processing...'}
        )
        
        result = run_async_task(_process_cv_async(task_id))
        print(f"CV processing completed: {task_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error in CV processing task {task_id}: {e}")
        run_async_task(get_mongodb_task_service().mark_task_failed(task_id, str(e)))
        raise

async def _process_cv_async(task_id: str) -> Dict[str, Any]:
    """Async implementation của CV processing"""
    
    mongodb_service = get_mongodb_task_service()
    task = await mongodb_service.get_task_status(task_id)
    if not task:
        raise Exception(f"Task {task_id} not found in MongoDB")

    try:
        await mongodb_service.mark_task_processing(task_id)
        
        # Lấy dữ liệu task
        file_content = task["data"]["file_content"]
        filename = task["data"]["filename"]
        
        await mongodb_service.update_task_progress(task_id, 20, "Processing CV with OCR...")
        
        # Import CV parser service
        from app.services.cv_parser_service import cv_parser_service
        
        # Xử lý CV - cần extract text từ PDF trước
        from app.services.simple_ocr_service import simple_ocr_service
        
        # Extract text từ PDF
        extracted_text, ocr_metadata = await simple_ocr_service.extract_text_from_pdf(
            file_content, filename
        )
        
        # Parse CV từ text
        cv_result = await cv_parser_service.parse_cv_to_structured_data(
            cv_text=extracted_text
        )
        
        if not cv_result.get("success"):
            raise Exception(f"CV processing failed: {cv_result.get('error')}")
        
        await mongodb_service.update_task_progress(task_id, 80, "CV processing completed")
        
        # Tạo kết quả
        result = {
            "success": True,
            "filename": filename,
            "cv_data": cv_result.get("cv_data", {}),
            "processing_info": {
                "ocr_applied": True,
                "processing_method": "cv_ocr_celery",
            },
        }
        
        await mongodb_service.mark_task_completed(task_id, result)
        return result

    except Exception as e:
        logger.error(f"Error in CV async processing {task_id}: {e}")
        await mongodb_service.mark_task_failed(task_id, str(e))
        raise
