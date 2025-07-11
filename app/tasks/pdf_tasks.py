"""
PDF Tasks - Các task Celery liên quan đến xử lý PDF
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, Optional
import time

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
        run_async_task(mongodb_task_service.mark_task_failed(task_id, error_msg))

        # Update Celery state
        self.update_state(state="FAILURE", meta={"error": error_msg})

        raise


async def _process_pdf_quick_analysis_async(task_id: str) -> Dict[str, Any]:
    """Async implementation của PDF quick analysis"""

    # Get task from MongoDB
    task = await mongodb_task_service.get_task_status(task_id)
    if not task:
        raise Exception(f"Task {task_id} not found in MongoDB")

    try:
        # Mark task as processing
        await mongodb_task_service.mark_task_processing(task_id)

        # Debug: Log task structure
        logger.info(f"Task structure: {json.dumps(task, indent=2, default=str)}")

        # Extract task data - check different possible structures
        if "data" in task:
            task_data = task["data"]
        elif "task_data" in task:
            task_data = task["task_data"]
        else:
            raise Exception(
                f"No task data found in task structure. Available keys: {list(task.keys())}"
            )

        file_content = task_data["file_content"]
        filename = task_data["filename"]
        create_embeddings = task_data.get("create_embeddings", True)
        lesson_id = task_data.get(
            "lesson_id"
        )  # Extract lesson_id if provided        logger.info(f"Processing file: {filename} (size: {len(file_content)} bytes)")
        if lesson_id:
            logger.info(f"📍 Associated with lesson_id: {lesson_id}")
        else:
            logger.info("📍 No lesson_id provided")

        # Update progress: Start OCR
        await mongodb_task_service.update_task_progress(
            task_id, 10, "Extracting pages with OCR..."
        )

        try:
            # Import services here to avoid circular imports
            from app.services.enhanced_textbook_service import enhanced_textbook_service
            from app.services.qdrant_service import qdrant_service

            # Skip separate OCR step - it's handled inside process_textbook_to_structure
            await mongodb_task_service.update_task_progress(
                task_id, 35, "Skipping image analysis for faster processing..."
            )  # Create metadata
            book_metadata = {
                "id": str(uuid.uuid4())[:8],
                "title": filename.replace(".pdf", ""),
                "subject": "Chưa xác định",
                "grade": "Chưa xác định",
                "author": "Chưa xác định",
                "language": "vi",
            }

            # Process textbook with enhanced service
            await mongodb_task_service.update_task_progress(
                task_id, 50, "Analyzing book structure and refining content with AI..."
            )

            logger.info("🧠 Using enhanced textbook processing...")
            # Use the main processing method that returns full structure
            processing_result = (
                await enhanced_textbook_service.process_textbook_to_structure(
                    file_content,
                    filename,
                    book_metadata,
                    lesson_id,  # Pass lesson_id
                )
            )

            if not processing_result.get("success"):
                raise Exception(
                    f"Textbook processing failed: {processing_result.get('error', 'Unknown error')}"
                )

            # Extract the clean text content directly
            clean_book_structure = processing_result.get("clean_book_structure", "")
           
            await mongodb_task_service.update_task_progress(
                task_id, 60, "Book structure analysis and content refinement with OpenRouter LLM completed"
            )

            # Create embeddings if requested
            embeddings_result = None
            embeddings_created = False

            if create_embeddings:
                await mongodb_task_service.update_task_progress(
                    task_id, 70, "Creating embeddings..."
                )

                try:
                    # Use the simplified processing method for text content only
                    embeddings_result = await qdrant_service.process_textbook(
                        book_id=book_metadata.get("id", "unknown"),
                        text_content=clean_book_structure,  # Clean text content
                        lesson_id=lesson_id or "1",
                        book_title=book_metadata.get("title", "Unknown"),
                    )

                    if embeddings_result and embeddings_result.get("success"):
                        embeddings_created = True
                        await mongodb_task_service.update_task_progress(
                            task_id, 90, "Embeddings created successfully"
                        )
                        logger.info(f"Embeddings created successfully")
                    else:
                        logger.warning(
                            f"Embeddings creation failed: {embeddings_result.get('error') if embeddings_result else 'Unknown error'}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Embeddings creation failed: {str(e)}"
                    )  # Calculate simple statistics for 1 lesson per PDF
            total_chapters = 1  # Always 1 chapter for simplified structure
            total_lessons = 1   # Always 1 lesson per PDF

            # Create final result
            result = {
                "success": True,
                "book_id": book_metadata.get("id"),
                "filename": filename,
                "book_structure": {"text": clean_book_structure},  # Simple structure with text content
                "lesson_id": lesson_id,  # Include lesson_id if provided
                "statistics": {
                    "total_pages": processing_result.get("total_pages", 0),
                    "total_chapters": total_chapters,
                    "total_lessons": total_lessons,
                    "total_images": 0,  # No images processed
                },
                "processing_info": {
                    "ocr_applied": True,
                    "llm_analysis": True,
                    "processing_method": "celery_enhanced_analysis",
                    "task_id": task_id,
                    "associated_lesson_id": lesson_id,
                    "images_processed": False,  # No images processed
                },
                "message": "Enhanced textbook analysis completed successfully with LLM content refinement",
                "embeddings_created": embeddings_created,
                "embeddings_info": {
                    "collection_name": embeddings_result.get("embeddings", {}).get(
                        "collection_name"
                    )
                    if embeddings_result
                    else None,
                    "vector_count": embeddings_result.get("embeddings", {}).get(
                        "total_chunks", 0
                    )
                    if embeddings_result
                    else 0,
                    "vector_dimension": embeddings_result.get("embeddings", {}).get(
                        "vector_dimension"
                    )
                    if embeddings_result
                    else None,
                },
                "images_info": {
                    "images_saved": embeddings_result.get("images", {}).get(
                        "images_saved", 0
                    )
                    if embeddings_result
                    else 0,
                    "storage_path": embeddings_result.get("images", {}).get(
                        "storage_path"
                    )
                    if embeddings_result
                    else None,
                },
            }

        except Exception as processing_error:
            # Log detailed error information
            import traceback

            error_details = traceback.format_exc()
            logger.error(f"Real processing failed with detailed error: {error_details}")

            # Try to provide some basic file info at least
            try:
                import fitz

                doc = fitz.open(stream=file_content, filetype="pdf")
                page_count = doc.page_count
                doc.close()
            except Exception as pdf_error:
                logger.error(f"Even basic PDF reading failed: {pdf_error}")
                page_count = 0

            result = {
                "success": True,  # Still mark as success but with limited data
                "filename": filename,
                "message": f"PDF processing completed with limitations: {str(processing_error)}",
                "task_id": task_id,
                "processing_info": {
                    "ocr_applied": False,
                    "llm_analysis": False,
                    "processing_method": "celery_quick_analysis_fallback",
                    "error": str(processing_error),
                    "detailed_error": error_details,
                },
                "statistics": {
                    "total_pages": page_count,
                    "total_chapters": 0,
                    "total_lessons": 0,
                },
                "embeddings_created": False,
            }

        # Mark task as completed
        await mongodb_task_service.mark_task_completed(task_id, result)

        logger.info(f"PDF quick analysis task {task_id} completed successfully")
        return result

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing PDF quick analysis task {task_id}: {error_msg}")

        # Mark task as failed
        await mongodb_task_service.mark_task_failed(task_id, error_msg)
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
