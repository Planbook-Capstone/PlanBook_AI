"""
Celery tasks cho xử lý guide import từ file DOCX
"""

import asyncio
import logging
from celery import Celery
from app.core.celery_app import celery_app
from app.services.mongodb_task_service import mongodb_task_service

logger = logging.getLogger(__name__)


def run_async_task(coro):
    """Helper để chạy async function trong Celery task"""
    try:
        # Always create a new event loop for each task to avoid "Event loop is closed" error
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(coro)
        finally:
            # Always close the loop after use
            try:
                # Cancel all pending tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()

                # Wait for all tasks to be cancelled
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                loop.close()
            except Exception as cleanup_error:
                logger.warning(f"Error during loop cleanup: {cleanup_error}")

    except Exception as e:
        logger.error(f"Error in run_async_task: {e}")
        raise e


@celery_app.task(bind=True, name="app.tasks.guide_tasks.process_guide_import")
def process_guide_import(self, task_id: str):
    """
    Celery task để xử lý guide import từ file DOCX

    Args:
        task_id: ID của task trong MongoDB
    """
    logger.info(f"Starting Celery task for guide import: {task_id}")

    try:
        # Import background processor
        from app.services.background_task_processor import background_task_processor

        # Chạy task processing với helper function
        run_async_task(
            background_task_processor.process_guide_import_task(task_id)
        )

        logger.info(f"Completed Celery task for guide import: {task_id}")
        return {"status": "completed", "task_id": task_id}

    except Exception as e:
        logger.error(f"Error in Celery guide import task {task_id}: {e}")

        # Mark task as failed in MongoDB
        try:
            run_async_task(
                mongodb_task_service.mark_task_failed(task_id, str(e))
            )
        except Exception as mark_error:
            logger.error(f"Failed to mark task as failed: {mark_error}")

        # Re-raise để Celery biết task failed
        raise e
