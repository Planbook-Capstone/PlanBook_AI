"""
Celery tasks cho lesson plan content generation
"""
import asyncio
import logging
from typing import Dict, Any

from app.core.celery_app import celery_app
from app.services.mongodb_task_service import get_mongodb_task_service

logger = logging.getLogger(__name__)


def safe_kafka_call(func, *args, **kwargs):
    """
    Simplified Kafka call without threading to avoid deadlocks
    """
    try:
        logger.info("📤 Attempting Kafka operation...")
        result = func(*args, **kwargs)
        logger.info("✅ Kafka operation completed")
        return result
    except Exception as kafka_error:
        logger.warning(f"⚠️ Kafka operation failed: {kafka_error}, continuing...")
        return False


def run_async_task(coro):
    """Helper để chạy async function trong Celery task"""
    loop = None
    try:
        # Create new event loop for each task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run with 30 minute timeout
        result = loop.run_until_complete(
            asyncio.wait_for(coro, timeout=1800)
        )
        return result

    except asyncio.TimeoutError:
        logger.error("Task timed out after 30 minutes")
        raise Exception("Task timed out after 30 minutes")

    except Exception as e:
        logger.error(f"Error in async task: {e}")
        raise

    finally:
        # Clean up event loop
        if loop:
            try:
                # Cancel pending tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()

                # Wait briefly for cancellation
                if pending:
                    try:
                        loop.run_until_complete(
                            asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True),
                                timeout=5.0
                            )
                        )
                    except asyncio.TimeoutError:
                        pass  # Ignore timeout during cleanup

                loop.close()
            except Exception:
                pass  # Ignore cleanup errors
            finally:
                asyncio.set_event_loop(None)


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
        coro = _process_lesson_plan_content_generation_async(task_id)
        result = run_async_task(coro)
        logger.info(f"Task {task_id} completed successfully")
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in lesson plan content generation task {task_id}: {error_msg}")

        # Mark task as failed in MongoDB and send Kafka notification
        try:
            mongodb_service = get_mongodb_task_service()
            run_async_task(mongodb_service.mark_task_failed(task_id, error_msg))
            logger.info(f"Task {task_id} marked as failed and Kafka notification sent")
        except Exception as mongo_error:
            logger.error(f"Failed to mark task as failed in MongoDB: {mongo_error}")

            # Fallback: Send Kafka notification directly if MongoDB fails
            # Use sync version for Celery compatibility
            try:
                from app.services.kafka_progress_service import sync_kafka_progress_service

                # Get task data to extract user_id
                mongodb_fallback = get_mongodb_task_service()
                task_data_coro = mongodb_fallback.get_task_status(task_id)
                task_data = run_async_task(task_data_coro)

                if task_data and task_data.get("data", {}).get("user_id"):
                    user_id = task_data["data"]["user_id"]
                    lesson_id = task_data["data"].get("lesson_id")

                    # Send error notification via sync Kafka
                    success = safe_kafka_call(
                        sync_kafka_progress_service.send_task_failed_sync,
                        task_id=task_id,
                        user_id=user_id,
                        error=error_msg,
                        lesson_id=lesson_id
                    )
                    if success:
                        logger.info(f"✅ Fallback Kafka error notification sent for task {task_id}")
                    else:
                        logger.warning(f"⚠️ Failed to send fallback Kafka error notification for task {task_id}, but error logged")

            except Exception as kafka_error:
                logger.error(f"Failed to send fallback Kafka notification: {kafka_error}")

        # Update Celery state
        self.update_state(state="FAILURE", meta={"error": error_msg})

        # Return error result instead of raising to ensure Kafka notification is sent
        return {
            "success": False,
            "error": error_msg,
            "task_id": task_id,
            "lesson_plan": None,
            "statistics": {}
        }


async def _process_lesson_plan_content_generation_async(task_id: str) -> Dict[str, Any]:
    """
    Async implementation của lesson plan content generation
    Sử dụng generate_lesson_plan_content để có nội dung sách giáo khoa làm tài liệu tham khảo
    """

    logger.info(f"Starting async processing for task {task_id}")

    mongodb_task_service = None
    lesson_plan_content_service = None

    try:
        # Initialize MongoDB service
        mongodb_task_service = get_mongodb_task_service()
        await mongodb_task_service.initialize()

        # Get task from MongoDB
        task = await mongodb_task_service.get_task_status(task_id)
        if not task:
            raise Exception(f"Task {task_id} not found in MongoDB")

        task_data = task.get("data", {})
        logger.info(f"data : {task_data}")
        lesson_plan_json = task_data.get("lesson_plan_json")
        lesson_id = task_data.get("lesson_id")
        user_id = task_data.get("user_id")
        book_id = task_data.get("book_id")  # Lấy book_id từ task data
        tool_log_id = lesson_plan_json.get("tool_log_id")
        logger.info(f"tool_log_id : {tool_log_id}")
        print(f"DEBUG: lesson_plan_json type: {type(lesson_plan_json)}, lesson_id: {lesson_id}")
        if not lesson_plan_json:
            raise Exception("lesson_plan_json is required in task data")

        # Mark task as processing
        await mongodb_task_service.mark_task_processing(task_id)

        # Initialize sync Kafka service for progress updates
        from app.services.kafka_progress_service import sync_kafka_progress_service

        # Update progress: Starting analysis
        await mongodb_task_service.update_task_progress(
            task_id, 10, "Analyzing lesson plan structure..."
        )
        # Send sync progress update to SpringBoot (with timeout protection)
        if user_id:
            logger.info(f"📤 Attempting to send Kafka progress update for task {task_id}")
            safe_kafka_call(
                sync_kafka_progress_service.send_progress_update_sync,
                tool_log_id=lesson_plan_json.get("tool_log_id"),task_id=task_id, user_id=user_id, progress=10,
                message="Analyzing lesson plan structure...", status="processing",
                additional_data={"lesson_id": lesson_id} if lesson_id else None
            )

        # Count total nodes to process for progress tracking
        total_nodes = _count_nodes_recursive(lesson_plan_json)
        logger.info(f"Total nodes to process: {total_nodes}")

        # Update progress: Structure analyzed
        await mongodb_task_service.update_task_progress(
            task_id, 20, f"Found {total_nodes} nodes to process. Starting content generation..."
        )
        # Send sync progress update to SpringBoot
        if user_id:
            safe_kafka_call(
                sync_kafka_progress_service.send_progress_update_sync,
                tool_log_id=tool_log_id,task_id=task_id, user_id=user_id, progress=20,
                message=f"Found {total_nodes} nodes to process. Starting content generation...",
                status="processing"
            )

        # Update progress: Starting content generation
        await mongodb_task_service.update_task_progress(
            task_id, 50, "Generating lesson plan content with textbook reference..."
        )
        # Send sync progress update to SpringBoot
        if user_id:
            safe_kafka_call(
                sync_kafka_progress_service.send_progress_update_sync,
                tool_log_id=tool_log_id,task_id=task_id, user_id=user_id, progress=50,
                message="Generating lesson plan content with textbook reference...",
                status="processing"
            )

        # Generate lesson plan content
        from app.services.lesson_plan_content_service import get_lesson_plan_content_service
        lesson_plan_content_service = get_lesson_plan_content_service()

        result = await lesson_plan_content_service.generate_lesson_plan_content(
            lesson_plan_json=lesson_plan_json,
            lesson_id=lesson_id,
            book_id=book_id
        )
        logger.info(f"Content generation completed for task {task_id}: success={result.get('success')}")

        # Update progress: Content generation completed
        await mongodb_task_service.update_task_progress(
            task_id, 90, "Content generation completed. Processing results..."
        )
        # Send sync progress update to SpringBoot
        if user_id:
            safe_kafka_call(
                sync_kafka_progress_service.send_progress_update_sync,
                tool_log_id=tool_log_id,task_id=task_id, user_id=user_id, progress=90,
                message="Content generation completed. Processing results...",
                status="processing"
            )

        # Update progress: Finalizing
        await mongodb_task_service.update_task_progress(
            task_id, 95, "Finalizing lesson plan content..."
        )
        # Send sync progress update to SpringBoot
        if user_id:
            safe_kafka_call(
                sync_kafka_progress_service.send_progress_update_sync,
                tool_log_id=tool_log_id,task_id=task_id, user_id=user_id, progress=95,
                message="Finalizing lesson plan content...", status="processing"
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
        
        # Always mark task as completed and send result to SpringBoot
        # Even if processing failed, the task itself completed successfully
        await mongodb_task_service.mark_task_completed(task_id, final_result)

        # Send final result to SpringBoot via Kafka (regardless of success/failure)
        # Use sync version for Celery compatibility
        try:
            task_data = await mongodb_task_service.get_task_status(task_id)
            if task_data and task_data.get("data", {}).get("user_id"):
                user_id = task_data["data"]["user_id"]
                lesson_id = task_data["data"].get("lesson_id")

                from app.services.kafka_progress_service import sync_kafka_progress_service
                success = safe_kafka_call(
                    sync_kafka_progress_service.send_final_result_sync,
                    task_id=task_id,
                    user_id=user_id,
                    result=final_result,
                    lesson_id=lesson_id
                )
                if success:
                    logger.info(f"Sent final result to SpringBoot for task {task_id}")
                else:
                    logger.warning(f"Failed to send final result to SpringBoot for task {task_id}")
        except Exception as kafka_error:
            logger.error(f"Failed to send final result to SpringBoot: {kafka_error}")

        logger.info(f"Task {task_id} completed successfully")
        return final_result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing lesson plan content generation task {task_id}: {error_msg}")

        # Mark task as failed
        await mongodb_task_service.mark_task_failed(task_id, error_msg)

        # Send error notification to SpringBoot via Kafka
        # Use sync version for Celery compatibility
        try:
            task_data = await mongodb_task_service.get_task_status(task_id)
            if task_data and task_data.get("data", {}).get("user_id"):
                user_id = task_data["data"]["user_id"]
                lesson_id = task_data["data"].get("lesson_id")

                from app.services.kafka_progress_service import sync_kafka_progress_service

                # Send final result with error
                error_result = {
                    "success": False,
                    "error": error_msg,
                    "task_id": task_id
                }

                success = safe_kafka_call(
                    sync_kafka_progress_service.send_final_result_sync,
                    task_id=task_id,
                    user_id=user_id,
                    result=error_result,
                    lesson_id=lesson_id
                )
                if success:
                    logger.info(f"✅ Sent error result to SpringBoot for task {task_id}")
                else:
                    logger.warning(f"⚠️ Failed to send error result to SpringBoot for task {task_id}, but error logged")
        except Exception as kafka_error:
            logger.error(f"Failed to send error result to SpringBoot: {kafka_error}")

        raise Exception(error_msg)

    finally:
        # Simple cleanup
        try:
            if lesson_plan_content_service:
                lesson_plan_content_service = None
            if mongodb_task_service and hasattr(mongodb_task_service, 'client'):
                mongodb_task_service.client.close()
                mongodb_task_service = None
        except Exception:
            pass  # Ignore cleanup errors


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
