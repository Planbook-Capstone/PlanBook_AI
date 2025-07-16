"""
PDF Tasks - Các task Celery liên quan đến xử lý PDF
"""

import asyncio
import logging
from typing import Dict, Any
import time

from app.core.celery_app import celery_app
from app.services.mongodb_task_service import get_mongodb_task_service

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


@celery_app.task(name="app.tasks.pdf_tasks.health_check")
def pdf_health_check() -> Dict[str, Any]:
    """Health check task cho PDF processing"""
    try:
        return {
            "status": "healthy",
            "message": "PDF tasks worker is running",
            "service": "pdf_processing",
            "queue": "pdf_queue",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"PDF tasks worker error: {str(e)}",
            "service": "pdf_processing",
            "queue": "pdf_queue",
        }


@celery_app.task(name="app.tasks.pdf_tasks.test_task")
def test_task(message: str = "Hello from PDF tasks!") -> Dict[str, Any]:
    """Simple test task"""
    return {"success": True, "message": message, "service": "pdf_processing"}


@celery_app.task(name="app.tasks.pdf_tasks.process_pdf_quick_analysis", bind=True)
def process_pdf_quick_analysis(self, task_id: str) -> Dict[str, Any]:
    """
    Celery task xử lý phân tích nhanh PDF sách giáo khoa
    """
    logger.info(f"Starting PDF quick analysis task: {task_id}")

    try:
        # Update Celery state
        self.update_state(
            state="PROGRESS",
            meta={"progress": 0, "message": "Starting PDF quick analysis..."},
        )

        # Run async implementation
        result = run_async_task(_process_pdf_quick_analysis_async(task_id))
        logger.info(f"PDF quick analysis task {task_id} completed successfully")
        return result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in PDF quick analysis task {task_id}: {error_msg}")

        # Mark task as failed in MongoDB
        run_async_task(get_mongodb_task_service().mark_task_failed(task_id, error_msg))

        # Update Celery state
        self.update_state(state="FAILURE", meta={"error": error_msg})

        raise


async def _process_pdf_quick_analysis_async(task_id: str) -> Dict[str, Any]:
    """Async implementation của PDF quick analysis - Tối ưu hóa"""

    # Get task from MongoDB
    task = await get_mongodb_task_service().get_task_status(task_id)
    if not task:
        raise Exception(f"Task {task_id} not found in MongoDB")

    try:
        # Mark task as processing
        await get_mongodb_task_service().mark_task_processing(task_id)

        # Extract task data
        task_data = task.get("data") or task.get("task_data")
        if not task_data:
            raise Exception(f"No task data found. Available keys: {list(task.keys())}")

        # Get required parameters
        file_content = task_data["file_content"]
        filename = task_data["filename"]
        book_id = task_data["book_id"]
        lesson_id = task_data.get("lesson_id")

        logger.info(f"🔍 PDF Task Debug:")
        logger.info(f"   - filename: {filename}")
        logger.info(f"   - book_id: {book_id}")
        logger.info(f"   - lesson_id: {lesson_id} (type: {type(lesson_id)})")
        logger.info(f"   - task_data keys: {list(task_data.keys())}")
        logger.info(f"Processing {filename} (book_id: {book_id}, lesson_id: {lesson_id})")

        # Import services
        from app.services.enhanced_textbook_service import get_enhanced_textbook_service
        from app.services.qdrant_service import get_qdrant_service

        enhanced_textbook_service = get_enhanced_textbook_service()
        qdrant_service = get_qdrant_service()

        # BƯỚC KIỂM TRA LESSON_ID TRƯỚC KHI XỬ LÝ
        if lesson_id:
            logger.info(f"🔍 Checking if lesson_id '{lesson_id}' already exists...")
            await get_mongodb_task_service().update_task_progress(
                task_id, 10, f"Checking lesson_id '{lesson_id}' existence..."
            )

            lesson_check = await qdrant_service.check_lesson_id_exists(lesson_id)

            if not lesson_check.get("success"):
                error_msg = f"Failed to check lesson_id: {lesson_check.get('error')}"
                logger.error(f"❌ {error_msg}")
                raise Exception(error_msg)

            if lesson_check.get("exists"):
                existing_book_id = lesson_check.get("existing_book_id", "unknown")
                error_msg = f"Lesson ID '{lesson_id}' already exists in book '{existing_book_id}'. Please use a different lesson_id or delete the existing lesson first."
                logger.error(f"❌ {error_msg}")

                # Trả về lỗi conflict với thông tin chi tiết
                conflict_result = {
                    "success": False,
                    "error": error_msg,
                    "lesson_id": lesson_id,
                    "existing_book_id": existing_book_id,
                    "conflict": True,
                    "message": "Lesson ID conflict detected"
                }

                # Mark task as failed với thông tin conflict
                await get_mongodb_task_service().mark_task_failed(task_id, error_msg)
                return conflict_result

            logger.info(f"✅ Lesson ID '{lesson_id}' is available for use")
        else:
            logger.info("ℹ️ No lesson_id provided, skipping lesson_id validation")

        # Process textbook
        await get_mongodb_task_service().update_task_progress(
            task_id, 40, "Processing textbook content..."
        )

        book_metadata = {
            "id": book_id,
            "title": filename.replace(".pdf", ""),
            "language": "vi",
        }

        processing_result = await enhanced_textbook_service.process_textbook_to_structure(
            file_content, filename, book_metadata, lesson_id
        )

        if not processing_result.get("success"):
            raise Exception(f"Textbook processing failed: {processing_result.get('error')}")

        # Auto create embeddings
        await get_mongodb_task_service().update_task_progress(
            task_id, 70, "Creating embeddings..."
        )

        try:
            # Use the clean book structure from processing result
            book_content = processing_result.get("clean_book_structure")
            logger.info(f"Book content: {book_content}")
            if not book_content:
                raise Exception("No book book_content found for embedding creation")

            # Truyền đầy đủ tham số cần thiết cho process_textbook - sử dụng parameter content thống nhất
            logger.info(f"🔍 Calling process_textbook with:")
            logger.info(f"   - book_id: {book_id}")
            logger.info(f"   - lesson_id: {lesson_id} (type: {type(lesson_id)})")
            logger.info(f"   - content type: {type(book_content)}")
            
            embeddings_result = await qdrant_service.process_textbook(
                book_id=book_id,
                content=book_content,  # Sử dụng parameter content thống nhất
                lesson_id=lesson_id,
                content_type="textbook"
            )

            if embeddings_result and embeddings_result.get("success"):
                await get_mongodb_task_service().update_task_progress(
                    task_id, 90, "Embeddings created successfully"
                )
                embeddings_created = True
            else:
                logger.warning("Embeddings creation failed")
                embeddings_created = False
                embeddings_result = None

        except Exception as e:
            logger.warning(f"Embeddings creation failed: {str(e)}")
            embeddings_created = False
            embeddings_result = None

        # Create simplified result
        result = {
            "success": True,
            "book_id": book_id,
            "filename": filename,
            "lesson_id": lesson_id,
            "book_structure": processing_result.get("clean_book_structure"),
            "processing_info": {
                "task_id": task_id,
                "processing_method": "enhanced_textbook_service",
            },
            "embeddings_created": embeddings_created,
            "embeddings_info": {
                "collection_name": embeddings_result.get("collection_name") if embeddings_result else None,
                "vector_count": embeddings_result.get("total_chunks", 0) if embeddings_result else 0,
            },
            "message": "PDF processing completed successfully"
        }

        # Mark task as completed
        await get_mongodb_task_service().mark_task_completed(task_id, result)
        logger.info(f"PDF quick analysis task {task_id} completed successfully")
        return result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing PDF quick analysis task {task_id}: {error_msg}")

        # Mark task as failed
        await get_mongodb_task_service().mark_task_failed(task_id, error_msg)
        raise


@celery_app.task(name="app.tasks.pdf_tasks.simple_test")
def simple_test(message: str = "Hello from worker!") -> Dict[str, Any]:
    """Simple sync test task without async dependencies"""
    logger.info(f"Simple test task received: {message}")

    time.sleep(2)  # Simulate some work

    result = {
        "success": True,
        "message": f"Processed: {message}",
        "worker": "celery@HongThinh",
        "timestamp": time.time(),
    }

    logger.info(f"Simple test task completed: {result}")
    return result
