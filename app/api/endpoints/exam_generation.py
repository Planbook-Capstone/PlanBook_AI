"""
API endpoints cho chức năng tạo bài kiểm tra từ ma trận đề thi
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Dict, Any, List, Union
import logging
import os
import re
from urllib.parse import quote
from datetime import datetime
import asyncio
from pathlib import Path
from app.services.textbook_retrieval_service import get_textbook_retrieval_service
from app.models.exam_models import (
    ExamMatrixRequest,
)
from app.models.online_document_models import (
    ExamOnlineResponse,
    OnlineDocumentLinks
)
from app.models.smart_exam_models import (
    SmartExamRequest,
    SmartExamResponse,
    SmartExamError
)
from app.services.exam_generation_service import get_exam_generation_service
from app.services.exam_docx_service import exam_docx_service
from app.services.google_drive_service import get_google_drive_service
from app.services.smart_exam_generation_service import get_smart_exam_generation_service
from app.services.smart_exam_docx_service import smart_exam_docx_service
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


async def _create_online_document_response(
    docx_file_path: str,
    filename: str,
    exam_result: Dict[str, Any],
    request: ExamMatrixRequest,
    search_quality: float = 0.0
) -> Dict[str, Any]:
    """
    Tạo response cho document online - upload thật lên Google Drive

    Args:
        docx_file_path: Đường dẫn file DOCX đã tạo
        filename: Tên file
        exam_result: Kết quả tạo đề thi
        request: Request gốc
        search_quality: Chất lượng tìm kiếm

    Returns:
        Dict chứa response data với link online thật

    Raises:
        HTTPException: Nếu không thể tạo online document
    """
    # Kiểm tra Google Drive có enabled và available không
    if not settings.ENABLE_GOOGLE_DRIVE:
        raise HTTPException(
            status_code=503,
            detail="Google Drive service is disabled. Cannot create online document."
        )

    google_drive_service = get_google_drive_service()
    if not google_drive_service.is_available():
        raise HTTPException(
            status_code=503,
            detail="Google Drive service is not available. Please check configuration."
        )

    try:
        # Upload lên Google Drive
        logger.info(f"Uploading file to Google Drive: {filename}")
        upload_result = await google_drive_service.upload_docx_file(
            file_path=docx_file_path,
            filename=filename,
            convert_to_google_docs=True  # Convert thành Google Docs để edit online
        )

        if not upload_result.get("success"):
            error_msg = upload_result.get('error', 'Unknown upload error')
            logger.error(f"Failed to upload to Google Drive: {error_msg}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to upload document to Google Drive: {error_msg}"
            )

        # Xóa file local sau khi upload thành công
        try:
            if os.path.exists(docx_file_path):
                os.remove(docx_file_path)
                logger.info(f"Deleted local file after upload: {docx_file_path}")
        except Exception as e:
            logger.warning(f"Failed to delete local file: {e}")

        # Tạo response online với link thật
        links = OnlineDocumentLinks(**upload_result["links"])

        return ExamOnlineResponse(
            success=True,
            message="Đề thi đã được tạo và upload lên Google Drive thành công",
            file_id=upload_result["file_id"],
            filename=upload_result["filename"],
            mime_type=upload_result["mime_type"],
            links=links,
            primary_link=links.edit or links.view or "",
            created_at=datetime.now().isoformat(),
            storage_provider="Google Drive",
            exam_id=str(exam_result.get("exam_id", "unknown")),
            lesson_id=request.cau_hinh_de[0].lesson_id if request.cau_hinh_de else "unknown",
            mon_hoc=request.mon_hoc,
            lop=request.lop,
            total_questions=len(exam_result.get("questions", [])),
            search_quality=search_quality,
            additional_info={
                "web_view_link": upload_result.get("web_view_link"),
                "web_content_link": upload_result.get("web_content_link")
            }
        ).model_dump()

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading to Google Drive: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error while creating online document: {str(e)}"
        )





class AutoDeleteFileResponse(FileResponse):
    """Custom FileResponse that deletes the file after sending"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def __call__(self, scope, receive, send):
        try:
            # Send the file response
            await super().__call__(scope, receive, send)
        finally:
            # Delete the file after sending
            try:
                if os.path.exists(self.path):
                    os.remove(self.path)
                    logger.info(f"Deleted temporary file: {self.path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {self.path}: {e}")


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize filename để tránh lỗi encoding với ký tự tiếng Việt
    """
    try:
        # Loại bỏ ký tự đặc biệt và dấu tiếng Việt
        sanitized = re.sub(r"[^\w\-_]", "_", filename)
        sanitized = re.sub(r"_+", "_", sanitized)
        sanitized = sanitized.strip("_")
        if not sanitized:
            sanitized = "exam"
        return sanitized
    except Exception as e:
        logger.warning(f"Error sanitizing filename '{filename}': {e}")
        return "exam"


@router.post("/generate-exam")
async def generate_exam_from_matrix(request: ExamMatrixRequest):
    print("=== GENERATE-EXAM ENDPOINT CALLED ===")
    """
    Tạo bài kiểm tra từ ma trận đề thi và trả về link doc online

    Endpoint này nhận ma trận đề thi và lesson_id, sau đó:
    1. Tìm kiếm nội dung bài học trong Qdrant
    2. Sử dụng Gemini LLM để tạo câu hỏi theo ma trận
    3. Xuất ra file DOCX chứa đề thi và đáp án
    4. Upload lên Google Drive và trả về link online để xem/chỉnh sửa

    Args:
        request: Ma trận đề thi với lesson_id và cấu hình chi tiết

    Returns:
        ExamOnlineResponse: Link online để xem/chỉnh sửa đề thi hoặc OnlineDocumentError nếu thất bại

    Example:
        POST /api/v1/exam/generate-exam
        {
            "exam_id": "exam_001",
            "ten_truong": "Trường THPT ABC",
            "mon_hoc": "Sinh học",
            "lop": 12,
            "tong_so_cau": 10,
            "bookID": "sinh12",
            "cau_hinh_de": [...]
        }

        Response: JSON với links online để truy cập đề thi
    """
    try:
        logger.info(f"=== EXAM GENERATION START ===")
        logger.info(f"Request received: exam_id={request.exam_id}, mon_hoc={request.mon_hoc}")
        logger.info(f"Request data: {request.model_dump()}")

        # 1. Validate request
        if not request.cau_hinh_de or len(request.cau_hinh_de) == 0:
            raise HTTPException(status_code=400, detail="cau_hinh_de is required and cannot be empty")

        # Lấy tất cả lesson_id từ cấu hình đề thi
        lesson_ids = [cau_hinh.lesson_id for cau_hinh in request.cau_hinh_de]
        logger.info(f"Extracting lesson_ids from cau_hinh_de: {lesson_ids}")

        # 2. Tìm kiếm nội dung cho tất cả bài học
        logger.info("Searching for multiple lesson contents...")
        textbookService = get_textbook_retrieval_service()

        # Lấy bookID từ request nếu có
        book_id = getattr(request, 'bookID', None)
        if book_id:
            logger.info(f"Searching lessons in specific book: {book_id}")

        lesson_content = await textbookService.get_multiple_lessons_content_for_exam(
            lesson_ids=lesson_ids,
            book_id=book_id
        )
        print("LessonContent" ,lesson_content)

        if not lesson_content.get("success", False):
            failed_lessons = lesson_content.get("failed_lessons", [])
            successful_lessons = lesson_content.get("successful_lessons", [])
            error_detail = f"Failed to retrieve content for lessons: {failed_lessons}. "
            if successful_lessons:
                error_detail += f"Successfully retrieved: {successful_lessons}. "
            error_detail += f"Error: {lesson_content.get('error', 'Unknown error')}"

            raise HTTPException(
                status_code=404,
                detail=error_detail,
            )

        # 3. Kiểm tra chất lượng nội dung và thông báo về lessons
        search_quality = lesson_content.get("search_quality", 0.0)
        successful_lessons = lesson_content.get("successful_lessons", [])
        failed_lessons = lesson_content.get("failed_lessons", [])

        logger.info(f"Content retrieval summary:")
        logger.info(f"  - Total lessons requested: {lesson_content.get('total_lessons', 0)}")
        logger.info(f"  - Successfully retrieved: {len(successful_lessons)} lessons: {successful_lessons}")
        logger.info(f"  - Failed to retrieve: {len(failed_lessons)} lessons: {failed_lessons}")
        logger.info(f"  - Average search quality: {search_quality:.2f}")

        if search_quality < 0.3:
            logger.warning(f"Low search quality: {search_quality}")

        if failed_lessons:
            logger.warning(f"Some lessons failed to retrieve content: {failed_lessons}")
            logger.warning("Exam generation will proceed with available content only")

        # 4. Tạo câu hỏi từ ma trận
        logger.info("Generating questions from matrix...")
        exam_generation_service = get_exam_generation_service()
        exam_result = await exam_generation_service.generate_questions_from_matrix(
            exam_request=request, lesson_content=lesson_content
        )

        if not exam_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate questions: {exam_result.get('error', 'Unknown error')}",
            )

        # 5. Tạo file DOCX
        logger.info("Creating DOCX file...")
        try:
            logger.info(f"Calling create_exam_docx with exam_data type: {type(exam_result)}")
            logger.info(f"Calling create_exam_docx with exam_request type: {type(request.model_dump())}")

            docx_result = await exam_docx_service.create_exam_docx(
                exam_data=exam_result, exam_request=request.model_dump()
            )

            logger.info(f"DOCX creation result: {docx_result}")

        except Exception as docx_error:
            logger.error(f"Exception during DOCX creation: {docx_error}")
            logger.error(f"Exception type: {type(docx_error)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"DOCX creation failed: {str(docx_error)}"
            )

        if not docx_result.get("success", False):
            logger.warning(f"Failed to create DOCX: {docx_result.get('error')}")
            docx_file_path = None
        else:
            docx_file_path = docx_result.get("filepath")

        # 6. Upload lên Google Drive và trả về link online
        if not docx_result.get("success", False) or not docx_file_path:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create DOCX file: {docx_result.get('error', 'Unknown error')}",
            )

        # Kiểm tra file có tồn tại không
        if not os.path.exists(docx_file_path):
            raise HTTPException(
                status_code=500, detail="DOCX file was created but not found on disk"
            )

        # Tạo filename với thông tin đề thi (sanitize để tránh lỗi encoding)
        mon_hoc_safe = _sanitize_filename(request.mon_hoc)
        exam_id_safe = _sanitize_filename(str(exam_result.get("exam_id", "unknown")))
        filename = f"De_thi_{mon_hoc_safe}_{request.lop}_{exam_id_safe}.docx"

        logger.info(
            f"Exam generation completed successfully. Generated {len(exam_result.get('questions', []))} questions."
        )

        # Tạo online document response
        response = await _create_online_document_response(
            docx_file_path=docx_file_path,
            filename=filename,
            exam_result=exam_result,
            request=request,
            search_quality=search_quality
        )

        return ExamOnlineResponse(**response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating exam: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



# @router.get("/download-exam/{filename}")
async def download_exam_file(filename: str):
    """
    Download file DOCX đề thi đã tạo

    Args:
        filename: Tên file cần download

    Returns:
        FileResponse với file DOCX
    """
    try:
        file_path = os.path.join("exports", filename)

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading exam file: {e}")
        raise HTTPException(status_code=500, detail=f"Error downloading file: {str(e)}")



@router.get("/exam-templates", response_model=Dict[str, Any])
async def get_exam_templates() -> Dict[str, Any]:
    """
    Lấy danh sách template ma trận đề thi mẫu

    Returns:
        Dict chứa các template có sẵn
    """
    try:
        templates = [
            {
                "id": "biology_grade12_basic",
                "name": "Sinh học 12 - Cơ bản",
                "description": "Ma trận đề thi Sinh học lớp 12 cơ bản với 10 câu hỏi",
                "mon_hoc": "Sinh học",
                "lop": 12,
                "tong_so_cau": 10,
                "phan_bo_muc_do": {"Nhận biết": 40, "Thông hiểu": 40, "Vận dụng": 20},
                "phan_bo_loai_cau": {"TN": 60, "DT": 20, "DS": 20, "TL": 0},
            },
            {
                "id": "chemistry_grade12_advanced",
                "name": "Hóa học 12 - Nâng cao",
                "description": "Ma trận đề thi Hóa học lớp 12 nâng cao với 15 câu hỏi",
                "mon_hoc": "Hóa học",
                "lop": 12,
                "tong_so_cau": 15,
                "phan_bo_muc_do": {"Nhận biết": 30, "Thông hiểu": 40, "Vận dụng": 30},
                "phan_bo_loai_cau": {"TN": 50, "DT": 20, "DS": 20, "TL": 10},
            },
        ]

        return {
            "success": True,
            "templates": templates,
            "total_templates": len(templates),
        }

    except Exception as e:
        logger.error(f"Error getting exam templates: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")




@router.post("/generate-smart-exam")
async def generate_smart_exam(request: SmartExamRequest):
    """
    Tạo đề thi thông minh theo chuẩn THPT 2025 (Async với Celery)

    Endpoint này nhận ma trận đề thi và trả về task_id để theo dõi progress.
    Sử dụng Celery để xử lý bất đồng bộ với progress tracking bằng tiếng Việt.
    Hỗ trợ Kafka integration khi có user_id.

    Args:
        request: SmartExamRequest chứa thông tin trường, môn học, ma trận đề thi, user_id (optional)

    Returns:
        Dict: {"task_id": "...", "message": "..."} để theo dõi qua /api/v1/tasks/{task_id}/status

    Example:
        POST /api/v1/exam/generate-smart-exam
        {
            "school": "Trường THPT Hong Thinh",
            "examCode": "1234",
            "grade": 12,
            "subject": "Hoa hoc",
            "examTitle": "Kiểm tra ne",
            "duration": 90,
            "outputFormat": "docx",
            "outputLink": "online",
            "bookID": "hoa12",
            "user_id": "user123",
            "matrix": [
                {
                    "lessonId": "hoa12_bai1",
                    "parts": [
                        {
                            "partName": "Phần 1: Trắc nghiệm",
                            "objectives": {
                                "Biết": 2,
                                "Hiểu": 2,
                                "Vận_dụng": 1
                            }
                        }
                    ]
                }
            ]
        }

        Response:
        {
            "task_id": "abc-123-def",
            "message": "Đã tạo task tạo đề thi thông minh. Sử dụng task_id để theo dõi tiến độ."
        }
    """
    print("=== SMART EXAM ENDPOINT CALLED ===")
    try:
        logger.info(f"=== SMART EXAM GENERATION START (ASYNC) ===")
        logger.info(f"Request: {request.school} - {request.subject} - Grade {request.grade}")
        if request.user_id:
            logger.info(f"User ID: {request.user_id} (Kafka integration enabled)")

        # 1. Validate request
        if not request.matrix:
            raise HTTPException(
                status_code=400,
                detail="Ma trận đề thi không được rỗng"
            )

        # 2. Tạo task bất đồng bộ với Celery
        from app.services.background_task_processor import get_background_task_processor

        background_processor = get_background_task_processor()
        task_result = await background_processor.create_smart_exam_task(
            request_data=request.model_dump()
        )

        if not task_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Không thể tạo task: {task_result.get('error', 'Lỗi không xác định')}"
            )

        task_id = task_result.get("task_id")
        logger.info(f"Đã tạo smart exam task: {task_id}")

        return {
            "task_id": task_id,
            "message": "Đã tạo task tạo đề thi thông minh. Sử dụng task_id để theo dõi tiến độ qua API /api/v1/tasks/{task_id}/status"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in smart exam generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi hệ thống: {str(e)}"
        )
