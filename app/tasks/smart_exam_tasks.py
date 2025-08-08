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
from app.services.kafka_service import kafka_service, safe_kafka_call
from app.models.smart_exam_models import SmartExamRequest

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


def create_error_result(task_id: str, error_msg: str, error_type: str, stage: str, **kwargs) -> Dict[str, Any]:
    """Tạo error result chuẩn với format output"""
    return {
        "success": False,
        "error": error_msg,
        "output": {
            "task_id": task_id,
            "error_details": {
                "error_type": error_type,
                "error_message": error_msg,
                "task_stage": stage,
                **kwargs
            }
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
    loop = None
    try:
        # Always create a completely fresh event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Run the async task
        result = loop.run_until_complete(_process_smart_exam_generation_async(task_id))
        logger.info(f"Task {task_id} completed successfully")
        return result

    except Exception as e:
        logger.error(f"Task {task_id} failed with error: {e}")
        raise
    finally:
        # Comprehensive cleanup
        if loop:
            try:
                # Cancel all pending tasks first
                pending = asyncio.all_tasks(loop)
                if pending:
                    logger.info(f"Cancelling {len(pending)} pending tasks for task {task_id}")
                    for task in pending:
                        task.cancel()

                    # Wait for all tasks to be cancelled with timeout
                    try:
                        loop.run_until_complete(
                            asyncio.wait_for(
                                asyncio.gather(*pending, return_exceptions=True),
                                timeout=5.0
                            )
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout waiting for tasks to cancel for task {task_id}")

                # Close the loop
                if not loop.is_closed():
                    loop.close()
                    logger.info(f"Event loop closed for task {task_id}")

            except Exception as cleanup_error:
                logger.warning(f"Error during loop cleanup for task {task_id}: {cleanup_error}")
            finally:
                # Clear the event loop reference
                asyncio.set_event_loop(None)


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

        # Extract metadata fields
        user_id = request_data.get("user_id")
        tool_log_id = request_data.get("tool_log_id")
        # Create exam request (remove metadata fields)
        exam_params = {k: v for k, v in request_data.items()
                      if k not in ["user_id", "tool_log_id", "lesson_id", "book_id"]}

        # Handle invalid examCode - if it's not a valid 4-digit number, set to None
        if "examCode" in exam_params:
            exam_code = exam_params["examCode"]
            if not isinstance(exam_code, str) or not exam_code.isdigit() or len(exam_code) != 4:
                logger.warning(f"Invalid examCode '{exam_code}', setting to None for auto-generation")
                exam_params["examCode"] = None

        # Handle invalid grade - ensure it's an integer between 1-12
        if "grade" in exam_params:
            try:
                grade = int(exam_params["grade"])
                if grade < 1 or grade > 12:
                    logger.warning(f"Invalid grade '{grade}', setting to 12")
                    exam_params["grade"] = 12
                else:
                    exam_params["grade"] = grade
            except (ValueError, TypeError):
                logger.warning(f"Invalid grade '{exam_params['grade']}', setting to 12")
                exam_params["grade"] = 12

        # Handle invalid duration - ensure it's an integer between 15-180
        if "duration" in exam_params:
            try:
                duration = int(exam_params["duration"])
                if duration < 15 or duration > 180:
                    logger.warning(f"Invalid duration '{duration}', setting to 45")
                    exam_params["duration"] = 45
                else:
                    exam_params["duration"] = duration
            except (ValueError, TypeError):
                logger.warning(f"Invalid duration '{exam_params['duration']}', setting to 45")
                exam_params["duration"] = 45

        try:
            exam_request = SmartExamRequest(**exam_params)
        except Exception as validation_error:
            error_msg = f"Lỗi validation: {str(validation_error)}"
            logger.error(f"SmartExamRequest validation failed: {validation_error}")

            error_result = create_error_result(task_id, error_msg, "ValidationError", "validation")

            # Send Kafka error notification if user_id is present
            if user_id:
                safe_kafka_call(
                    kafka_service.send_final_result_sync,
                    task_id=task_id,
                    user_id=user_id,
                    result=error_result,
                    tool_log_id=tool_log_id
                )

            return error_result

        logger.info(f"[DEBUG] User ID for Kafka notifications: {user_id}")
        logger.info(f"[DEBUG] Tool Log ID for Kafka notifications: {tool_log_id}")

        # Progress callback function with Kafka notification
        async def progress_callback(percentage: int, message: str):
            try:
                await task_service.update_task_progress(task_id, percentage, message)
                logger.info(f"Task {task_id}: {percentage}% - {message}")

                # Send Kafka progress notification if user_id is present
                if user_id:
                    logger.info(f"[DEBUG] Sending Kafka progress notification: {percentage}% - {message}")
                    safe_kafka_call(
                        kafka_service.send_progress_update_sync,
                        tool_log_id=tool_log_id,
                        task_id=task_id,
                        user_id=user_id,
                        progress=percentage,
                        message=message,
                        status="processing"
                    )
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

        # Debug logging
        logger.info(f"[DEBUG] Lesson IDs to search: {lesson_ids}")
        logger.info(f"[DEBUG] Book ID: {getattr(exam_request, 'bookID', None)}")

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

        # Debug logging cho kết quả
        logger.info(f"[DEBUG] Lesson content result: success={lesson_content.get('success')}")
        if not lesson_content.get('success'):
            logger.error(f"[DEBUG] Lesson content errors: {lesson_content.get('errors', [])}")
            logger.error(f"[DEBUG] Full lesson content response: {lesson_content}")

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

        # Import formatter service
        from app.services.smart_exam_json_formatter import get_smart_exam_json_formatter
        formatter = get_smart_exam_json_formatter()

        # Tính tổng số câu hỏi cần tạo từ ma trận
        total_questions_needed = 0
        for lesson_matrix in exam_request.matrix:
            for part in lesson_matrix.parts:
                objectives = part.objectives
                total_questions_needed += objectives.Biết + objectives.Hiểu + objectives.Vận_dụng

        logger.info(f"📊 Total questions needed: {total_questions_needed}")

        # Tạo cấu trúc exam_data tổng thể để append câu hỏi
        accumulated_questions = []
        final_formatted_exam = None  # Lưu format cuối cùng để tái sử dụng

        # Tạo callback function để gửi từng câu hỏi qua Kafka
        async def question_callback(question: Dict[str, Any]):
            """Callback để append từng câu hỏi vào danh sách và gửi qua Kafka"""
            nonlocal final_formatted_exam

            # Thêm câu hỏi vào danh sách tích lũy (luôn luôn, không chỉ khi có user_id)
            accumulated_questions.append(question)

            # Tính progress dựa trên số câu đã tạo
            questions_created = len(accumulated_questions)
            if total_questions_needed > 0:
                # Progress từ 30% (bắt đầu tạo) đến 90% (hoàn thành tạo)
                # 30% + (60% * questions_created / total_questions_needed)
                progress = 30 + int(60 * questions_created / total_questions_needed)
                progress = min(progress, 90)  # Không vượt quá 90%
            else:
                progress = 60  # Fallback nếu không tính được

            # Format toàn bộ danh sách câu hỏi hiện tại bằng formatter có sẵn
            exam_data = {"questions": accumulated_questions}
            formatted_exam = formatter.format_exam_to_json_response(exam_data)
            final_formatted_exam = formatted_exam  # Lưu lại để dùng cuối

            # Gửi qua Kafka nếu có user_id
            if user_id:
                try:
                    # Wrap trong cấu trúc exam_data
                    response_data = {"exam_data": formatted_exam}

                    # Gửi toàn bộ cấu trúc đã được cập nhật qua Kafka
                    safe_kafka_call(
                        kafka_service.send_progress_update_sync,
                        tool_log_id=tool_log_id,
                        task_id=task_id,
                        user_id=user_id,
                        progress=progress,
                        message=f"Đã tạo xong {questions_created}/{total_questions_needed} câu hỏi",
                        status="processing",
                        additional_data=response_data
                    )
                    logger.info(f"✅ Sent accumulated exam data via Kafka for task {task_id} - Progress: {progress}% - Questions: {questions_created}/{total_questions_needed}")
                except Exception as e:
                    logger.warning(f"⚠️ Error sending accumulated exam data via Kafka: {e}")

        exam_result = await smart_exam_service.generate_smart_exam(
            exam_request, lessons_content_data, question_callback
        )

        if not exam_result.get("success", False):
            error_result = create_error_result(task_id,
                f"Không thể tạo đề thi: {exam_result.get('error', 'Lỗi không xác định')}",
                "ExamGenerationError", "exam_generation")
            await progress_callback(100, f"Lỗi: {error_result['error']}")
            await task_service.mark_task_completed(task_id=task_id, result=error_result)
            return error_result

        generated_questions = len(exam_result.get("questions", []))
        # Không cần progress callback ở đây nữa vì callback cuối cùng đã là 90%

        # Kiểm tra isExportDocx để quyết định xử lý
        is_export_docx = exam_request.isExportDocx

        if is_export_docx:
            # Bước 4: Tạo file DOCX
            await progress_callback(92, "Đang tạo file Word (.docx)...")
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
            await progress_callback(94, f"Đã tạo file Word: {filename}")

            # Bước 5: Upload lên Google Drive
            await progress_callback(96, "Đang tải lên Google Drive...")
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

            await progress_callback(98, "Đã tải lên Google Drive thành công")

            # Bước 6: Dọn dẹp file tạm
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"Đã xóa file tạm: {file_path}")
            except Exception as e:
                logger.warning(f"Không thể xóa file tạm: {e}")

            # Bước 7: Hoàn thành với DOCX
            await progress_callback(95, "Đang hoàn thiện kết quả...")
            statistics = exam_result.get("statistics", {})
            statistics_dict = statistics.model_dump() if hasattr(statistics, 'model_dump') else statistics

            result = {
                "success": True,
                "output": {
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
            }
        else:
            # Bước 4: Hoàn thành với JSON format (đã được format trong callback)
            await progress_callback(90, "Đang hoàn thiện kết quả JSON...")

            # Sử dụng format đã được tạo trong callback, fallback nếu cần
            if final_formatted_exam:
                formatted_json = final_formatted_exam
            else:
                # Fallback nếu không có callback data
                formatted_json = formatter.format_exam_to_json_response(exam_result)

            await progress_callback(100, "Đã hoàn thành format JSON")

            statistics = exam_result.get("statistics", {})
            statistics_dict = statistics.model_dump() if hasattr(statistics, 'model_dump') else statistics

            result = {
                "success": True,
                "output": {
                    "exam_id": exam_result.get("exam_id"),
                    "message": "Đề thi thông minh đã được tạo thành công theo chuẩn THPT 2025",
                    "exam_data": formatted_json,
                    "statistics": statistics_dict,
                    "task_id": task_id,
                    "processing_info": {
                        "total_questions_generated": generated_questions,
                        "total_lessons_used": len(available_lessons),
                        "missing_lessons": missing_lessons,
                        "processing_method": "celery_smart_exam_generation"
                    }
                }
            }

        await progress_callback(100, f"Hoàn thành! Đã tạo {generated_questions}/{total_questions_needed} câu hỏi từ {len(available_lessons)} bài học")
        await task_service.mark_task_completed(task_id=task_id, result=result)

        # Send Kafka completion notification if user_id is present
        if user_id:
            safe_kafka_call(
                kafka_service.send_final_result_sync,
                task_id=task_id,
                user_id=user_id,
                result=result,
                tool_log_id=tool_log_id
            )

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
                    tool_log_id = request_data.get("tool_log_id")
                    if user_id:
                        safe_kafka_call(
                            kafka_service.send_final_result_sync,
                            task_id=task_id,
                            user_id=user_id,
                            result=error_result,
                            tool_log_id=tool_log_id
                        )
            except Exception as kafka_error:
                logger.error(f"Error sending Kafka error notification for task {task_id}: {kafka_error}")

            logger.info(f"✅ Saved error result to database for task {task_id}")
        except Exception as save_error:
            logger.error(f"Error saving error result for task {task_id}: {save_error}")

        return error_result
