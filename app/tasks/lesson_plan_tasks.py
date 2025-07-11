"""
Celery tasks cho lesson plan content generation
"""
import asyncio
import logging
from typing import Dict, Any

from app.core.celery_app import celery_app
from app.services.mongodb_task_service import MongoDBTaskService
from app.services.lesson_plan_content_service import LessonPlanContentService

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


@celery_app.task(name="app.tasks.lesson_plan_tasks.process_lesson_plan_content_generation", bind=True)
def process_lesson_plan_content_generation(self, task_id: str) -> Dict[str, Any]:
    """
    Celery task xử lý sinh nội dung giáo án với nội dung sách giáo khoa làm tài liệu tham khảo

    Args:
        task_id: ID của task trong MongoDB

    Returns:
        Dict kết quả xử lý với lesson_content_used trong statistics
    """
    logger.info(f"Starting lesson plan content generation task: {task_id}")
    
    try:
        # Update Celery state
        self.update_state(
            state="PROGRESS",
            meta={"progress": 0, "message": "Starting lesson plan content generation..."}
        )
        
        # Run async implementation
        logger.info(f"Creating coroutine for task {task_id}")
        coro = _process_lesson_plan_content_generation_async(task_id)
        logger.info(f"Coroutine created: {type(coro)} - {id(coro)}")
        result = run_async_task(coro)
        logger.info(f"Lesson plan content generation task {task_id} completed successfully")
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in lesson plan content generation task {task_id}: {error_msg}")
        
        # Mark task as failed in MongoDB
        try:
            mongodb_service = MongoDBTaskService()
            run_async_task(mongodb_service.mark_task_failed(task_id, error_msg))
        except Exception as mongo_error:
            logger.error(f"Failed to mark task as failed in MongoDB: {mongo_error}")
        
        # Update Celery state
        self.update_state(state="FAILURE", meta={"error": error_msg})
        
        raise


async def _process_lesson_plan_content_generation_async(task_id: str) -> Dict[str, Any]:
    """
    Async implementation của lesson plan content generation
    Sử dụng generate_lesson_plan_content để có nội dung sách giáo khoa làm tài liệu tham khảo
    """

    logger.info(f"🚀 Starting async processing for task {task_id}")

    # Create fresh MongoDB service instance for this task
    mongodb_task_service = MongoDBTaskService()
    await mongodb_task_service.initialize()

    # Get task from MongoDB
    task = await mongodb_task_service.get_task_status(task_id)
    if not task:
        raise Exception(f"Task {task_id} not found in MongoDB")
    
    task_data = task.get("data", {})
    lesson_plan_json = task_data.get("lesson_plan_json")
    lesson_id = task_data.get("lesson_id")
    print(f"DEBUG: lesson_plan_json: {lesson_id}")
    if not lesson_plan_json:
        raise Exception("lesson_plan_json is required in task data")
    
    # Mark task as processing
    await mongodb_task_service.mark_task_processing(task_id)
    
    try:
        # Update progress: Starting analysis
        await mongodb_task_service.update_task_progress(
            task_id, 10, "Analyzing lesson plan structure..."
        )
        
        # Count total nodes to process for progress tracking
        total_nodes = _count_nodes_recursive(lesson_plan_json)
        logger.info(f"Total nodes to process: {total_nodes}")
        
        # Update progress: Structure analyzed
        await mongodb_task_service.update_task_progress(
            task_id, 20, f"Found {total_nodes} nodes to process. Starting content generation..."
        )
        
        # Update progress: Starting content generation
        await mongodb_task_service.update_task_progress(
            task_id, 50, "Generating lesson plan content with textbook reference..."
        )

        # Generate lesson plan content (with lesson content from textbook)
        # Create fresh service instance to avoid coroutine reuse issues
        lesson_plan_content_service = LessonPlanContentService()
        result = await lesson_plan_content_service.generate_lesson_plan_content(
            lesson_plan_json=lesson_plan_json,
            lesson_id=lesson_id
        )

        # Update progress: Content generation completed
        await mongodb_task_service.update_task_progress(
            task_id, 90, "Content generation completed. Processing results..."
        )
        
        # Update progress: Finalizing
        await mongodb_task_service.update_task_progress(
            task_id, 95, "Finalizing lesson plan content..."
        )
        
        # Prepare final result
        final_result = {
            "success": result["success"],
            "lesson_plan": result.get("lesson_plan"),
            "statistics": result.get("statistics", {}),
            "task_id": task_id,
            "processing_info": {
                "total_nodes_processed": result.get("statistics", {}).get("content_nodes_processed", 0),
                "total_nodes_found": total_nodes,
                "processing_method": "celery_lesson_plan_content_generation",
                "lesson_content_used": result.get("statistics", {}).get("lesson_content_used", False)
            }
        }
        
        if not result["success"]:
            final_result["error"] = result.get("error", "Unknown error occurred")
        
        # Mark task as completed
        await mongodb_task_service.mark_task_completed(task_id, final_result)
        
        logger.info(f"Lesson plan content generation task {task_id} completed successfully")
        return final_result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing lesson plan content generation task {task_id}: {error_msg}")

        # Mark task as failed
        await mongodb_task_service.mark_task_failed(task_id, error_msg)

        raise Exception(error_msg)


def _count_nodes_recursive(node: Dict[str, Any]) -> int:
    """
    Đếm tổng số nodes trong cây lesson plan để tracking progress
    
    Args:
        node: Node gốc của lesson plan
        
    Returns:
        int: Tổng số nodes
    """
    count = 1  # Count current node
    
    children = node.get("children", [])
    if children:
        for child in children:
            count += _count_nodes_recursive(child)
    
    return count
