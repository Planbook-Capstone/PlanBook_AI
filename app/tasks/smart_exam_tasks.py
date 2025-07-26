"""
Celery tasks cho Smart Exam Generation
Xử lý tạo đề thi thông minh theo chuẩn THPT 2025 bất đồng bộ với progress tracking
"""

import logging
import asyncio
import os
from typing import Dict, Any

from app.core.celery_app import celery_app
from app.services.mongodb_task_service import get_mongodb_task_service
from app.services.smart_exam_generation_service import get_smart_exam_generation_service
from app.services.textbook_retrieval_service import get_textbook_retrieval_service
from app.services.smart_exam_docx_service import smart_exam_docx_service
from app.services.google_drive_service import get_google_drive_service
from app.models.smart_exam_models import SmartExamRequest

logger = logging.getLogger(__name__)


async def _send_smart_exam_progress_notification(user_id: str, task_id: str, percentage: int, message: str):
    """Send progress notification to SpringBoot via Kafka"""
    try:
        from app.services.kafka_service import kafka_service
        from app.core.config import get_responses_topic
        from datetime import datetime

        response_message = {
            "type": "smart_exam_generation_response",
            "data": {
                "status": "processing",
                "user_id": user_id,
                "task_id": task_id,
                "progress": percentage,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
        }

        await kafka_service.send_message_async(response_message, topic=get_responses_topic(), key=user_id)
        logger.info(f"[KAFKA] 📊 Sent smart exam progress notification for user {user_id}, task {task_id}: {percentage}% - {message}")

    except Exception as e:
        logger.error(f"[KAFKA] ❌ Failed to send smart exam progress notification: {e}")


async def _send_smart_exam_completion_notification(user_id: str, task_id: str, result: Dict[str, Any]):
    """Send completion notification to SpringBoot via Kafka"""
    try:
        from app.services.kafka_service import kafka_service
        from app.core.config import get_responses_topic

        response_message = {
            "type": "smart_exam_generation_response",
            "data": {
                "status": "completed",
                "user_id": user_id,
                "task_id": task_id,
                "result": result,
                "message": "Smart exam generation completed successfully",
                "timestamp": result.get("timestamp", "")
            }
        }

        await kafka_service.send_message_async(response_message, topic=get_responses_topic(), key=user_id)
        logger.info(f"[KAFKA] ✅ Sent smart exam completion notification for user {user_id}, task {task_id}")

    except Exception as e:
        logger.error(f"[KAFKA] ❌ Failed to send smart exam completion notification: {e}")


async def _send_smart_exam_error_notification(user_id: str, task_id: str, error_result: Dict[str, Any]):
    """Send error notification to SpringBoot via Kafka"""
    try:
        from app.services.kafka_service import kafka_service
        from app.core.config import get_responses_topic

        response_message = {
            "type": "smart_exam_generation_response",
            "data": {
                "status": "error",
                "user_id": user_id,
                "task_id": task_id,
                "error": error_result.get("error", "Unknown error"),
                "error_details": error_result.get("error_details", {}),
                "message": "Smart exam generation failed",
                "timestamp": error_result.get("timestamp", "")
            }
        }

        await kafka_service.send_message_async(response_message, topic=get_responses_topic(), key=user_id)
        logger.info(f"[KAFKA] ✅ Sent smart exam error notification for user {user_id}, task {task_id}")

    except Exception as e:
        logger.error(f"[KAFKA] ❌ Failed to send smart exam error notification: {e}")


def run_async_task(coro):
    """Helper function để chạy async code trong Celery worker"""
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


def create_error_result(task_id: str, error_msg: str, error_type: str, stage: str, **kwargs) -> Dict[str, Any]:
    """Tạo error result chuẩn"""
    return {
        "success": False,
        "error": error_msg,
        "task_id": task_id,
        "error_details": {
            "error_type": error_type,
            "error_message": error_msg,
            "task_stage": stage,
            **kwargs
        }
    }





@celery_app.task(name="app.tasks.smart_exam_tasks.process_smart_exam_generation")
def process_smart_exam_generation(task_id: str) -> Dict[str, Any]:
    """
    Celery task xử lý tạo đề thi thông minh với progress tracking bằng tiếng Việt

    Args:
        task_id: ID của task trong MongoDB

    Returns:
        Dict chứa kết quả xử lý
    """
    logger.info(f"Bắt đầu tạo đề thi thông minh task: {task_id}")

    # Tạo một event loop duy nhất cho toàn bộ task
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        return loop.run_until_complete(_process_smart_exam_generation_async(task_id))
    finally:
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


async def _process_smart_exam_generation_async(task_id: str) -> Dict[str, Any]:
    """
    Async implementation của smart exam generation
    """
    try:
        # Khởi tạo services
        task_service = get_mongodb_task_service()

        # Lấy thông tin task từ database
        task_info = await task_service.get_task_status(task_id)
        if not task_info:
            return create_error_result(task_id, f"Không tìm thấy task: {task_id}",
                                     "TaskNotFoundError", "initialization")

        # Parse request data
        task_data = task_info.get("data", {})
        request_data = task_data.get("request_data", {})
        exam_request = SmartExamRequest(**request_data)

        # Get user_id for Kafka notifications
        user_id = getattr(exam_request, 'user_id', None)
        logger.info(f"[DEBUG] User ID for Kafka notifications: {user_id}")
        logger.info(f"[DEBUG] Kafka notifications enabled: {user_id is not None}")

        # Progress callback function with Kafka notification
        async def progress_callback(percentage: int, message: str):
            try:
                await task_service.update_task_progress(task_id, percentage, message)
                logger.info(f"Task {task_id}: {percentage}% - {message}")

                # Send Kafka progress notification if user_id is present
                if user_id:
                    logger.info(f"[DEBUG] Sending Kafka progress notification: {percentage}% - {message}")
                    await _send_smart_exam_progress_notification(user_id, task_id, percentage, message)
                else:
                    logger.info(f"[DEBUG] Skipping Kafka notification - no user_id")

            except Exception as e:
                logger.error(f"Error updating task progress {task_id}: {e}")
                # Continue execution even if progress update fails
        
        # Bước 1: Phân tích ma trận đề thi
        await progress_callback(10, "Đang phân tích ma trận đề thi...")
        
        if not exam_request.matrix:
            error_result = create_error_result(task_id, "Ma trận đề thi không được rỗng",
                                             "EmptyMatrixError", "matrix_validation")
            await progress_callback(100, f"Lỗi: {error_result['error']}")
            await task_service.mark_task_completed(task_id=task_id, result=error_result)
            return error_result

        # Lấy lesson_ids và thống kê
        lesson_ids = [lesson.lessonId for lesson in exam_request.matrix]
        total_lessons = len(lesson_ids)

        # Tính tổng số câu hỏi từ parts
        total_questions = 0
        for lesson in exam_request.matrix:
            for part in lesson.parts:
                total_questions += part.objectives.Biết + part.objectives.Hiểu + part.objectives.Vận_dụng

        await progress_callback(15, f"Tìm thấy {total_lessons} bài học, tổng {total_questions} câu hỏi cần tạo")

        # Bước 2: Lấy nội dung bài học
        await progress_callback(20, "Đang tìm kiếm nội dung bài học từ cơ sở dữ liệu...")
        textbook_service = get_textbook_retrieval_service()

        # Lấy bookID từ request nếu có
        book_id = getattr(exam_request, 'bookID', None)
        if book_id:
            await progress_callback(22, f"Tìm kiếm trong sách: {book_id}")

        lesson_content = await textbook_service.get_multiple_lessons_content_for_exam(lesson_ids, book_id)

        if not lesson_content.get("success", False):
            error_result = create_error_result(task_id,
                f"Không thể lấy nội dung bài học: {lesson_content.get('error', 'Lỗi không xác định')}",
                "ContentRetrievalError", "content_retrieval")
            await progress_callback(100, f"Lỗi: {error_result['error']}")
            await task_service.mark_task_completed(task_id=task_id, result=error_result)
            return error_result
            
        # Kiểm tra nội dung bài học
        lessons_content_data = lesson_content.get("lessons_content", {})
        available_lessons = [lid for lid in lesson_ids if lid in lessons_content_data and lessons_content_data[lid]]
        missing_lessons = [lid for lid in lesson_ids if lid not in available_lessons]

        if missing_lessons:
            if len(missing_lessons) == len(lesson_ids):
                error_result = create_error_result(task_id, "Không tìm thấy nội dung cho bất kỳ bài học nào",
                                                 "NoLessonsFoundError", "content_validation",
                                                 missing_lessons=missing_lessons)
                await progress_callback(100, f"Lỗi: {error_result['error']}")
                await task_service.mark_task_completed(task_id=task_id, result=error_result)
                return error_result
            else:
                await progress_callback(25, f"Cảnh báo: Không tìm thấy {len(missing_lessons)} bài học. Tiếp tục với {len(available_lessons)} bài học có sẵn")
        else:
            await progress_callback(25, f"Đã tìm thấy nội dung cho tất cả {len(available_lessons)} bài học")
        
        # Bước 3: Tạo đề thi thông minh
        await progress_callback(30, "Đang tạo câu hỏi theo ma trận đề thi...")
        smart_exam_service = get_smart_exam_generation_service()
        exam_result = await smart_exam_service.generate_smart_exam(exam_request, lessons_content_data)

        if not exam_result.get("success", False):
            error_result = create_error_result(task_id,
                f"Không thể tạo đề thi: {exam_result.get('error', 'Lỗi không xác định')}",
                "ExamGenerationError", "exam_generation")
            await progress_callback(100, f"Lỗi: {error_result['error']}")
            await task_service.mark_task_completed(task_id=task_id, result=error_result)
            return error_result

        generated_questions = len(exam_result.get("questions", []))
        await progress_callback(60, f"Đã tạo thành công {generated_questions} câu hỏi")
        
        # Bước 4: Tạo file DOCX
        await progress_callback(65, "Đang tạo file Word (.docx)...")
        docx_result = await smart_exam_docx_service.create_smart_exam_docx(exam_result, exam_request.model_dump())

        if not docx_result.get("success", False):
            error_result = create_error_result(task_id,
                f"Không thể tạo file DOCX: {docx_result.get('error', 'Lỗi không xác định')}",
                "DocxCreationError", "docx_creation")
            await progress_callback(100, f"Lỗi: {error_result['error']}")
            await task_service.mark_task_completed(task_id=task_id, result=error_result)
            return error_result

        file_path = docx_result.get("file_path")
        filename = docx_result.get("filename")
        await progress_callback(75, f"Đã tạo file Word: {filename}")
        
        # Bước 5: Upload lên Google Drive
        await progress_callback(80, "Đang tải lên Google Drive...")
        google_drive_service = get_google_drive_service()
        upload_result = await google_drive_service.upload_docx_file(
            file_path, filename or "smart_exam.docx", convert_to_google_docs=True)

        if not upload_result.get("success", False):
            error_result = create_error_result(task_id,
                f"Không thể tải lên Google Drive: {upload_result.get('error', 'Lỗi không xác định')}",
                "GoogleDriveUploadError", "file_upload")
            await progress_callback(100, f"Lỗi: {error_result['error']}")
            await task_service.mark_task_completed(task_id=task_id, result=error_result)
            return error_result

        await progress_callback(90, "Đã tải lên Google Drive thành công")
        
        # Bước 6: Dọn dẹp file tạm
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Đã xóa file tạm: {file_path}")
        except Exception as e:
            logger.warning(f"Không thể xóa file tạm: {e}")

        # Bước 7: Hoàn thành
        await progress_callback(95, "Đang hoàn thiện kết quả...")
        statistics = exam_result.get("statistics", {})
        statistics_dict = statistics.model_dump() if hasattr(statistics, 'model_dump') else statistics

        result = {
            "success": True,
            "exam_id": exam_result.get("exam_id"),
            "message": "Đề thi thông minh đã được tạo thành công theo chuẩn THPT 2025",
            "online_links": upload_result.get("links", {}),
            "statistics": statistics_dict,
            "task_id": task_id,
            "processing_info": {
                "total_questions_generated": generated_questions,
                "total_lessons_used": len(available_lessons),
                "missing_lessons": missing_lessons,
                "processing_method": "celery_smart_exam_generation"
            }
        }

        await progress_callback(100, f"Hoàn thành! Đã tạo {generated_questions} câu hỏi từ {len(available_lessons)} bài học")
        await task_service.mark_task_completed(task_id=task_id, result=result)

        # Send Kafka completion notification if user_id is present
        if user_id:
            await _send_smart_exam_completion_notification(user_id, task_id, result)

        logger.info(f"Smart exam generation task {task_id} hoàn thành thành công")
        return result

    except Exception as e:
        logger.error(f"Lỗi trong smart exam generation task {task_id}: {e}")
        error_result = create_error_result(task_id, f"Lỗi hệ thống: {str(e)}",
                                         type(e).__name__, "validation_or_processing")

        try:
            task_service = get_mongodb_task_service()
            await task_service.update_task_progress(task_id, 100, f"Lỗi: {error_result['error']}")
            await task_service.mark_task_completed(task_id=task_id, result=error_result)

            # Send Kafka error notification if user_id is present
            try:
                task_info = await task_service.get_task_status(task_id)
                if task_info:
                    task_data = task_info.get("data", {})
                    request_data = task_data.get("request_data", {})
                    user_id = request_data.get("user_id")
                    if user_id:
                        await _send_smart_exam_error_notification(user_id, task_id, error_result)
            except Exception as kafka_error:
                logger.error(f"Error sending Kafka error notification for task {task_id}: {kafka_error}")

            logger.info(f"✅ Saved error result to database for task {task_id}")
        except Exception as save_error:
            logger.error(f"Error saving error result for task {task_id}: {save_error}")

        return error_result
