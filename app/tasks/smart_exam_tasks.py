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


def run_async_task(coro):
    """Helper function để chạy async code trong Celery worker"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


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


def save_task_result(task_service, task_id: str, result: Dict[str, Any]):
    """Lưu kết quả task vào database"""
    try:
        run_async_task(task_service.mark_task_completed(task_id=task_id, result=result))
        logger.info(f"✅ Saved result to database for task {task_id}")
    except Exception as e:
        logger.error(f"❌ Error saving result to database: {e}")


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

    try:
        # Khởi tạo services
        task_service = get_mongodb_task_service()

        # Lấy thông tin task từ database
        task_info = run_async_task(task_service.get_task_status(task_id))
        if not task_info:
            return create_error_result(task_id, f"Không tìm thấy task: {task_id}",
                                     "TaskNotFoundError", "initialization")

        # Parse request data
        task_data = task_info.get("data", {})
        request_data = task_data.get("request_data", {})
        exam_request = SmartExamRequest(**request_data)

        # Progress callback function
        def progress_callback(percentage: int, message: str):
            try:
                run_async_task(task_service.update_task_progress(task_id, percentage, message))
                logger.info(f"Task {task_id}: {percentage}% - {message}")
            except Exception as e:
                logger.error(f"Lỗi cập nhật progress: {e}")
        
        # Bước 1: Phân tích ma trận đề thi
        progress_callback(10, "Đang phân tích ma trận đề thi...")

        if not exam_request.matrix:
            error_result = create_error_result(task_id, "Ma trận đề thi không được rỗng",
                                             "EmptyMatrixError", "matrix_validation")
            progress_callback(100, f"Lỗi: {error_result['error']}")
            save_task_result(task_service, task_id, error_result)
            return error_result
        
        # Lấy lesson_ids và thống kê
        lesson_ids = [lesson.lessonId for lesson in exam_request.matrix]
        total_lessons = len(lesson_ids)
        total_questions = sum(lesson.totalQuestions for lesson in exam_request.matrix)
        progress_callback(15, f"Tìm thấy {total_lessons} bài học, tổng {total_questions} câu hỏi cần tạo")

        # Bước 2: Lấy nội dung bài học
        progress_callback(20, "Đang tìm kiếm nội dung bài học từ cơ sở dữ liệu...")
        textbook_service = get_textbook_retrieval_service()
        lesson_content = run_async_task(textbook_service.get_multiple_lessons_content_for_exam(lesson_ids))

        if not lesson_content.get("success", False):
            error_result = create_error_result(task_id,
                f"Không thể lấy nội dung bài học: {lesson_content.get('error', 'Lỗi không xác định')}",
                "ContentRetrievalError", "content_retrieval")
            progress_callback(100, f"Lỗi: {error_result['error']}")
            save_task_result(task_service, task_id, error_result)
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
                progress_callback(100, f"Lỗi: {error_result['error']}")
                save_task_result(task_service, task_id, error_result)
                return error_result
            else:
                progress_callback(25, f"Cảnh báo: Không tìm thấy {len(missing_lessons)} bài học. Tiếp tục với {len(available_lessons)} bài học có sẵn")
        else:
            progress_callback(25, f"Đã tìm thấy nội dung cho tất cả {len(available_lessons)} bài học")
        
        # Bước 3: Tạo đề thi thông minh
        progress_callback(30, "Đang tạo câu hỏi theo ma trận đề thi...")
        smart_exam_service = get_smart_exam_generation_service()
        exam_result = run_async_task(smart_exam_service.generate_smart_exam(exam_request, lessons_content_data))

        if not exam_result.get("success", False):
            error_result = create_error_result(task_id,
                f"Không thể tạo đề thi: {exam_result.get('error', 'Lỗi không xác định')}",
                "ExamGenerationError", "exam_generation")
            progress_callback(100, f"Lỗi: {error_result['error']}")
            save_task_result(task_service, task_id, error_result)
            return error_result

        generated_questions = len(exam_result.get("questions", []))
        progress_callback(60, f"Đã tạo thành công {generated_questions} câu hỏi")
        
        # Bước 4: Tạo file DOCX
        progress_callback(65, "Đang tạo file Word (.docx)...")
        docx_result = run_async_task(smart_exam_docx_service.create_smart_exam_docx(exam_result, exam_request.model_dump()))

        if not docx_result.get("success", False):
            error_result = create_error_result(task_id,
                f"Không thể tạo file DOCX: {docx_result.get('error', 'Lỗi không xác định')}",
                "DocxCreationError", "docx_creation")
            progress_callback(100, f"Lỗi: {error_result['error']}")
            save_task_result(task_service, task_id, error_result)
            return error_result

        file_path = docx_result.get("file_path")
        filename = docx_result.get("filename")
        progress_callback(75, f"Đã tạo file Word: {filename}")
        
        # Bước 5: Upload lên Google Drive
        progress_callback(80, "Đang tải lên Google Drive...")
        google_drive_service = get_google_drive_service()
        upload_result = run_async_task(google_drive_service.upload_docx_file(
            file_path, filename or "smart_exam.docx", convert_to_google_docs=True))

        if not upload_result.get("success", False):
            error_result = create_error_result(task_id,
                f"Không thể tải lên Google Drive: {upload_result.get('error', 'Lỗi không xác định')}",
                "GoogleDriveUploadError", "file_upload")
            progress_callback(100, f"Lỗi: {error_result['error']}")
            save_task_result(task_service, task_id, error_result)
            return error_result

        progress_callback(90, "Đã tải lên Google Drive thành công")
        
        # Bước 6: Dọn dẹp file tạm
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Đã xóa file tạm: {file_path}")
        except Exception as e:
            logger.warning(f"Không thể xóa file tạm: {e}")

        # Bước 7: Hoàn thành
        progress_callback(95, "Đang hoàn thiện kết quả...")
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

        progress_callback(100, f"Hoàn thành! Đã tạo {generated_questions} câu hỏi từ {len(available_lessons)} bài học")
        save_task_result(task_service, task_id, result)
        logger.info(f"Smart exam generation task {task_id} hoàn thành thành công")
        return result

    except Exception as e:
        logger.error(f"Lỗi trong smart exam generation task {task_id}: {e}")
        error_result = create_error_result(task_id, f"Lỗi hệ thống: {str(e)}",
                                         type(e).__name__, "validation_or_processing")

        try:
            task_service = get_mongodb_task_service()
            run_async_task(task_service.update_task_progress(task_id, 100, f"Lỗi: {error_result['error']}"))
            save_task_result(task_service, task_id, error_result)
        except Exception as save_error:
            logger.error(f"❌ Error saving error result to database: {save_error}")

        return error_result
