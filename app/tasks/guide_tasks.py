"""
Celery tasks cho xử lý guide import từ file DOCX
"""

import logging
from celery import Celery
from app.core.celery_app import celery_app
from app.services.mongodb_task_service import mongodb_task_service

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.guide_tasks.process_guide_import")
def process_guide_import(self, task_id: str):
    """
    Celery task để xử lý guide import từ file DOCX
    
    Args:
        task_id: ID của task trong MongoDB
    """
    import asyncio
    
    logger.info(f"Starting Celery task for guide import: {task_id}")
    
    try:
        # Chạy async function trong Celery worker
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Import background processor
        from app.services.background_task_processor import background_task_processor
        
        # Chạy task processing
        loop.run_until_complete(
            background_task_processor.process_guide_import_task(task_id)
        )
        
        logger.info(f"Completed Celery task for guide import: {task_id}")
        return {"status": "completed", "task_id": task_id}
        
    except Exception as e:
        logger.error(f"Error in Celery guide import task {task_id}: {e}")
        
        # Mark task as failed in MongoDB
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                mongodb_task_service.mark_task_failed(task_id, str(e))
            )
        except Exception as mark_error:
            logger.error(f"Failed to mark task as failed: {mark_error}")
        
        # Re-raise để Celery biết task failed
        raise e
    
    finally:
        # Cleanup event loop
        try:
            loop.close()
        except:
            pass
