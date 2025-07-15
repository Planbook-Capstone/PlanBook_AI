"""
Embeddings Processing Tasks for Celery Workers
Các task xử lý embeddings chạy trong background workers
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
def create_embeddings_task(self, task_id: str) -> Dict[str, Any]:
    """
    Celery task tạo embeddings cho textbook
    
    Args:
        task_id: ID của task trong MongoDB
        
    Returns:
        Dict kết quả xử lý
    """
    print(f"Starting embeddings creation task: {task_id}")
    
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 0, 'message': 'Starting embeddings creation...'}
        )
        
        result = run_async_task(_create_embeddings_async(task_id))
        print(f"Embeddings creation completed: {task_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error in embeddings creation task {task_id}: {e}")
        run_async_task(get_mongodb_task_service().mark_task_failed(task_id, str(e)))
        raise

async def _create_embeddings_async(task_id: str) -> Dict[str, Any]:
    """Async implementation của embeddings creation"""
    
    task = await get_mongodb_task_service().get_task_status(task_id)
    if not task:
        raise Exception(f"Task {task_id} not found in MongoDB")
    
    try:
        await get_mongodb_task_service().mark_task_processing(task_id)
        
        # Lấy dữ liệu task
        book_id = task["data"]["book_id"]
        book_structure = task["data"]["book_structure"]
        
        await get_mongodb_task_service().update_task_progress(task_id, 20, "Creating embeddings...")
        
        # Import Qdrant service
        from app.services.qdrant_service import qdrant_service
        
        # Tạo embeddings - sử dụng parameter content thống nhất
        embeddings_result = await qdrant_service.process_textbook(
            book_id=book_id,
            content=book_structure,  # Sử dụng parameter content thống nhất
            content_type="textbook"
        )
        
        if not embeddings_result.get("success"):
            raise Exception(f"Embeddings creation failed: {embeddings_result.get('error')}")
        
        await get_mongodb_task_service().update_task_progress(
            task_id, 90, "Embeddings created successfully"
        )
        
        # Tạo kết quả cuối cùng
        result = {
            "success": True,
            "book_id": book_id,
            "embeddings_count": embeddings_result.get("embeddings_count", 0),
            "processing_info": {
                "method": "embeddings_celery",
                "embeddings_created": True,
            },
            "embeddings_result": embeddings_result,
        }
        
        await get_mongodb_task_service().mark_task_completed(task_id, result)
        return result
        
    except Exception as e:
        logger.error(f"Error in embeddings async processing {task_id}: {e}")
        await get_mongodb_task_service().mark_task_failed(task_id, str(e))
        raise

@task_with_retry()
def update_embeddings_task(self, task_id: str) -> Dict[str, Any]:
    """
    Celery task cập nhật embeddings cho textbook
    
    Args:
        task_id: ID của task trong MongoDB
        
    Returns:
        Dict kết quả xử lý
    """
    print(f"Starting embeddings update task: {task_id}")
    
    try:
        current_task.update_state(
            state='PROGRESS',
            meta={'progress': 0, 'message': 'Starting embeddings update...'}
        )
        
        result = run_async_task(_update_embeddings_async(task_id))
        print(f"Embeddings update completed: {task_id}")
        return result
        
    except Exception as e:
        logger.error(f"Error in embeddings update task {task_id}: {e}")
        run_async_task(get_mongodb_task_service().mark_task_failed(task_id, str(e)))
        raise

async def _update_embeddings_async(task_id: str) -> Dict[str, Any]:
    """Async implementation của embeddings update"""
    
    task = await get_mongodb_task_service().get_task_status(task_id)
    if not task:
        raise Exception(f"Task {task_id} not found in MongoDB")
    
    try:
        await get_mongodb_task_service().mark_task_processing(task_id)
        
        # Lấy dữ liệu task
        book_id = task["data"]["book_id"]
        updated_content = task["data"]["updated_content"]
        
        await get_mongodb_task_service().update_task_progress(task_id, 20, "Updating embeddings...")
        
        # Import Qdrant service
        from app.services.qdrant_service import qdrant_service
        
        # Cập nhật embeddings
        update_result = await qdrant_service.update_textbook_embeddings(
            book_id=book_id, updated_content=updated_content
        )
        
        if not update_result.get("success"):
            raise Exception(f"Embeddings update failed: {update_result.get('error')}")
        
        await get_mongodb_task_service().update_task_progress(
            task_id, 90, "Embeddings updated successfully"
        )
        
        # Tạo kết quả cuối cùng
        result = {
            "success": True,
            "book_id": book_id,
            "updated_count": update_result.get("updated_count", 0),
            "processing_info": {
                "method": "embeddings_update_celery",
                "embeddings_updated": True,
            },
            "update_result": update_result,
        }
        
        await get_mongodb_task_service().mark_task_completed(task_id, result)
        return result
        
    except Exception as e:
        logger.error(f"Error in embeddings update async processing {task_id}: {e}")
        await get_mongodb_task_service().mark_task_failed(task_id, str(e))
        raise
