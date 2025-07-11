"""
Celery tasks cho Smart Exam Generation
Xử lý tạo đề thi thông minh theo chuẩn THPT 2025 bất đồng bộ với progress tracking
"""

import logging
import asyncio
from typing import Dict, Any
from datetime import datetime

from app.core.celery_app import celery_app
from app.services.mongodb_task_service import get_mongodb_task_service
from app.services.smart_exam_generation_service import get_smart_exam_generation_service
from app.services.exam_content_service import get_exam_content_service
from app.services.smart_exam_docx_service import smart_exam_docx_service
from app.services.google_drive_service import get_google_drive_service
from app.models.smart_exam_models import SmartExamRequest
import os

logger = logging.getLogger(__name__)


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


@celery_app.task(name="app.tasks.smart_exam_tasks.process_smart_exam_generation", bind=True)
def process_smart_exam_generation(self, task_id: str) -> Dict[str, Any]:
    """
    Celery task xử lý tạo đề thi thông minh với progress tracking bằng tiếng Việt
    
    Args:
        task_id: ID của task trong MongoDB
        
    Returns:
        Dict chứa kết quả xử lý
    """
    logger.info(f"Bắt đầu tạo đề thi thông minh task: {task_id}")
    
    try:
        # Lấy task service
        task_service = get_mongodb_task_service()
        
        # Lấy thông tin task từ database
        task_info = run_async_task(task_service.get_task_status(task_id))
        if not task_info:
            logger.error(f"Không tìm thấy task: {task_id}")
            error_result = {
                "success": False,
                "error": f"Không tìm thấy task: {task_id}",
                "task_id": task_id,
                "error_details": {
                    "error_type": "TaskNotFoundError",
                    "error_message": f"Không tìm thấy task: {task_id}",
                    "task_stage": "initialization"
                }
            }
            # Note: Cannot save to database since task doesn't exist
            return error_result

        # Parse request data từ data field
        task_data = task_info.get("data", {})
        request_data = task_data.get("request_data", {})
        exam_request = SmartExamRequest(**request_data)
        
        # Progress callback function với tiếng Việt
        def progress_callback(percentage: int, message: str):
            """Callback để cập nhật progress với message tiếng Việt"""
            try:
                run_async_task(task_service.update_task_progress(
                    task_id=task_id,
                    progress=percentage,
                    message=message
                ))
                logger.info(f"Task {task_id}: {percentage}% - {message}")
            except Exception as e:
                logger.error(f"Lỗi cập nhật progress: {e}")
        
        # Bước 1: Phân tích ma trận đề thi
        progress_callback(10, "Đang phân tích ma trận đề thi...")
        
        if not exam_request.matrix:
            error_msg = "Ma trận đề thi không được rỗng"
            progress_callback(100, f"Lỗi: {error_msg}")

            error_result = {
                "success": False,
                "error": error_msg,
                "task_id": task_id,
                "error_details": {
                    "error_type": "EmptyMatrixError",
                    "error_message": error_msg,
                    "task_stage": "matrix_validation"
                }
            }

            # Lưu error result vào database
            try:
                logger.info(f"Attempting to save empty matrix error result to database for task {task_id}")
                run_async_task(task_service.mark_task_completed(
                    task_id=task_id,
                    result=error_result
                ))
                logger.info(f"✅ Successfully saved empty matrix error result to database for task {task_id}")
            except Exception as save_error:
                logger.error(f"❌ Error saving empty matrix error result to database: {save_error}")

            return error_result
        
        # Lấy lesson_ids
        lesson_ids = [lesson.lessonId for lesson in exam_request.matrix]
        total_lessons = len(lesson_ids)
        total_questions = sum(lesson.totalQuestions for lesson in exam_request.matrix)
        
        progress_callback(15, f"Tìm thấy {total_lessons} bài học, tổng {total_questions} câu hỏi cần tạo")
        
        # Bước 2: Tìm kiếm nội dung bài học
        progress_callback(20, "Đang tìm kiếm nội dung bài học từ cơ sở dữ liệu...")
        
        exam_content_service = get_exam_content_service()
        lesson_content = run_async_task(
            exam_content_service.get_multiple_lessons_content_for_exam(lesson_ids=lesson_ids)
        )
        
        if not lesson_content.get("success", False):
            error_msg = f"Không thể lấy nội dung bài học: {lesson_content.get('error', 'Lỗi không xác định')}"
            progress_callback(100, f"Lỗi: {error_msg}")

            # Tạo error result
            error_result = {
                "success": False,
                "error": error_msg,
                "task_id": task_id,
                "error_details": {
                    "error_type": "ContentRetrievalError",
                    "error_message": error_msg,
                    "task_stage": "content_retrieval"
                }
            }

            # Lưu error result vào database
            try:
                task_service = get_mongodb_task_service()
                logger.info(f"Attempting to save content retrieval error result to database for task {task_id}")
                run_async_task(task_service.mark_task_completed(
                    task_id=task_id,
                    result=error_result
                ))
                logger.info(f"✅ Successfully saved content retrieval error result to database for task {task_id}")
            except Exception as save_error:
                logger.error(f"❌ Error saving content retrieval error result to database: {save_error}")

            return error_result
        
        # Kiểm tra nội dung bài học
        content_data = lesson_content.get("content", {})
        available_lessons = [lid for lid in lesson_ids if lid in content_data and content_data[lid]]
        missing_lessons = [lid for lid in lesson_ids if lid not in available_lessons]
        
        if missing_lessons:
            if len(missing_lessons) == len(lesson_ids):
                error_msg = "Không tìm thấy nội dung cho bất kỳ bài học nào"
                progress_callback(100, f"Lỗi: {error_msg}")

                error_result = {
                    "success": False,
                    "error": error_msg,
                    "task_id": task_id,
                    "error_details": {
                        "error_type": "NoLessonsFoundError",
                        "error_message": error_msg,
                        "task_stage": "content_validation",
                        "missing_lessons": missing_lessons
                    }
                }

                # Lưu error result vào database
                try:
                    logger.info(f"Attempting to save no lessons found error result to database for task {task_id}")
                    run_async_task(task_service.mark_task_completed(
                        task_id=task_id,
                        result=error_result
                    ))
                    logger.info(f"✅ Successfully saved no lessons found error result to database for task {task_id}")
                except Exception as save_error:
                    logger.error(f"❌ Error saving no lessons found error result to database: {save_error}")

                return error_result
            else:
                progress_callback(25, f"Cảnh báo: Không tìm thấy {len(missing_lessons)} bài học. Tiếp tục với {len(available_lessons)} bài học có sẵn")
        else:
            progress_callback(25, f"Đã tìm thấy nội dung cho tất cả {len(available_lessons)} bài học")
        
        # Bước 3: Tạo đề thi thông minh
        progress_callback(30, "Đang khởi tạo dịch vụ tạo đề thi thông minh...")
        
        smart_exam_service = get_smart_exam_generation_service()
        
        progress_callback(35, "Đang tạo câu hỏi theo ma trận đề thi...")
        
        exam_result = run_async_task(
            smart_exam_service.generate_smart_exam(
                exam_request=exam_request,
                lesson_content=content_data
            )
        )
        
        if not exam_result.get("success", False):
            error_msg = f"Không thể tạo đề thi: {exam_result.get('error', 'Lỗi không xác định')}"
            progress_callback(100, f"Lỗi: {error_msg}")

            error_result = {
                "success": False,
                "error": error_msg,
                "task_id": task_id,
                "error_details": {
                    "error_type": "ExamGenerationError",
                    "error_message": error_msg,
                    "task_stage": "exam_generation"
                }
            }

            # Lưu error result vào database
            try:
                logger.info(f"Attempting to save exam generation error result to database for task {task_id}")
                run_async_task(task_service.mark_task_completed(
                    task_id=task_id,
                    result=error_result
                ))
                logger.info(f"✅ Successfully saved exam generation error result to database for task {task_id}")
            except Exception as save_error:
                logger.error(f"❌ Error saving exam generation error result to database: {save_error}")

            return error_result
        
        generated_questions = len(exam_result.get("questions", []))
        progress_callback(60, f"Đã tạo thành công {generated_questions} câu hỏi")
        
        # Bước 4: Tạo file DOCX
        progress_callback(65, "Đang tạo file Word (.docx)...")
        
        docx_result = run_async_task(
            smart_exam_docx_service.create_smart_exam_docx(
                exam_data=exam_result,
                exam_request=exam_request.model_dump()
            )
        )
        
        if not docx_result.get("success", False):
            error_msg = f"Không thể tạo file DOCX: {docx_result.get('error', 'Lỗi không xác định')}"
            progress_callback(100, f"Lỗi: {error_msg}")

            error_result = {
                "success": False,
                "error": error_msg,
                "task_id": task_id,
                "error_details": {
                    "error_type": "DocxCreationError",
                    "error_message": error_msg,
                    "task_stage": "docx_creation"
                }
            }

            # Lưu error result vào database
            try:
                logger.info(f"Attempting to save DOCX creation error result to database for task {task_id}")
                run_async_task(task_service.mark_task_completed(
                    task_id=task_id,
                    result=error_result
                ))
                logger.info(f"✅ Successfully saved DOCX creation error result to database for task {task_id}")
            except Exception as save_error:
                logger.error(f"❌ Error saving DOCX creation error result to database: {save_error}")

            return error_result
        
        file_path = docx_result.get("file_path")
        filename = docx_result.get("filename")
        
        progress_callback(75, f"Đã tạo file Word: {filename}")
        
        # Bước 5: Upload lên Google Drive
        progress_callback(80, "Đang tải lên Google Drive...")
        
        google_drive_service = get_google_drive_service()
        upload_result = run_async_task(
            google_drive_service.upload_docx_file(
                file_path=file_path,
                filename=filename or "smart_exam.docx",
                convert_to_google_docs=True
            )
        )
        
        if not upload_result.get("success", False):
            error_msg = f"Không thể tải lên Google Drive: {upload_result.get('error', 'Lỗi không xác định')}"
            progress_callback(100, f"Lỗi: {error_msg}")

            error_result = {
                "success": False,
                "error": error_msg,
                "task_id": task_id,
                "error_details": {
                    "error_type": "GoogleDriveUploadError",
                    "error_message": error_msg,
                    "task_stage": "file_upload"
                }
            }

            # Lưu error result vào database
            try:
                logger.info(f"Attempting to save Google Drive upload error result to database for task {task_id}")
                run_async_task(task_service.mark_task_completed(
                    task_id=task_id,
                    result=error_result
                ))
                logger.info(f"✅ Successfully saved Google Drive upload error result to database for task {task_id}")
            except Exception as save_error:
                logger.error(f"❌ Error saving Google Drive upload error result to database: {save_error}")

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
        statistics = exam_result.get("statistics", {})
        if hasattr(statistics, 'model_dump'):
            statistics_dict = statistics.model_dump()
        else:
            statistics_dict = statistics
        
        progress_callback(95, "Đang hoàn thiện kết quả...")
        
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

        # Lưu result vào database
        try:
            logger.info(f"Attempting to save result to database for task {task_id}")
            logger.info(f"Result data: {result}")

            task_service = get_mongodb_task_service()
            run_async_task(task_service.mark_task_completed(
                task_id=task_id,
                result=result
            ))
            logger.info(f"✅ Successfully saved result to database for task {task_id}")
        except Exception as save_error:
            logger.error(f"❌ Error saving result to database: {save_error}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        logger.info(f"Smart exam generation task {task_id} hoàn thành thành công")
        return result
        
    except Exception as e:
        error_msg = f"Lỗi hệ thống: {str(e)}"
        logger.error(f"Lỗi trong smart exam generation task {task_id}: {e}")

        # Tạo error result
        error_result = {
            "success": False,
            "error": error_msg,
            "task_id": task_id,
            "error_details": {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "task_stage": "validation_or_processing"
            }
        }

        try:
            task_service = get_mongodb_task_service()
            # Update progress với error message
            run_async_task(task_service.update_task_progress(
                task_id=task_id,
                progress=100,
                message=f"Lỗi: {error_msg}"
            ))

            # Lưu error result vào database
            logger.info(f"Attempting to save error result to database for task {task_id}")
            run_async_task(task_service.mark_task_completed(
                task_id=task_id,
                result=error_result
            ))
            logger.info(f"✅ Successfully saved error result to database for task {task_id}")

        except Exception as save_error:
            logger.error(f"❌ Error saving error result to database: {save_error}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

        return error_result
