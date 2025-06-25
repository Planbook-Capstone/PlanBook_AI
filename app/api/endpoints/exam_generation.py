"""
API endpoints cho chức năng tạo bài kiểm tra từ ma trận đề thi
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from typing import Dict, Any, List
import logging
import os
import re
from urllib.parse import quote
from datetime import datetime
import asyncio
from pathlib import Path

from app.models.exam_models import (
    ExamMatrixRequest,
    ExamResponse,
    ExamStatistics,
    SearchContentRequest,
    LessonContentResponse,
    ExamGenerationError,
)
from app.services.exam_content_service import exam_content_service
from app.services.exam_generation_service import exam_generation_service
from app.services.exam_docx_service import exam_docx_service

logger = logging.getLogger(__name__)
router = APIRouter()


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
    """
    Tạo bài kiểm tra từ ma trận đề thi và trả về file DOCX

    Endpoint này nhận ma trận đề thi và lesson_id, sau đó:
    1. Tìm kiếm nội dung bài học trong Qdrant
    2. Sử dụng Gemini LLM để tạo câu hỏi theo ma trận
    3. Xuất ra file DOCX chứa đề thi và đáp án
    4. Trả về file DOCX để download trực tiếp

    Args:
        request: Ma trận đề thi với lesson_id và cấu hình chi tiết

    Returns:
        FileResponse: File DOCX chứa đề thi và đáp án để download

    Example:
        POST /api/v1/exam/generate-exam
        {
            "lesson_id": "lesson_01_01",
            "mon_hoc": "Sinh học",
            "lop": 12,
            "tong_so_cau": 10,
            "cau_hinh_de": [...]
        }

        Response: File DOCX download với tên "De_thi_Sinh_hoc_12_[exam_id].docx"
    """
    try:
        logger.info(f"Starting exam generation for lesson: {request.lesson_id}")

        # 1. Validate request
        if not request.lesson_id:
            raise HTTPException(status_code=400, detail="lesson_id is required")

        # 2. Tìm kiếm nội dung bài học
        logger.info("Searching for lesson content...")
        lesson_content = await exam_content_service.get_lesson_content_for_exam(
            lesson_id=request.lesson_id
        )

        if not lesson_content.get("success", False):
            raise HTTPException(
                status_code=404,
                detail=f"Lesson content not found: {lesson_content.get('error', 'Unknown error')}",
            )

        # 3. Kiểm tra chất lượng nội dung
        search_quality = lesson_content.get("search_quality", 0.0)
        if search_quality < 0.3:
            logger.warning(f"Low search quality: {search_quality}")

        # 4. Tạo câu hỏi từ ma trận
        logger.info("Generating questions from matrix...")
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
        docx_result = await exam_docx_service.create_exam_docx(
            exam_data=exam_result, exam_request=request.model_dump()
        )

        if not docx_result.get("success", False):
            logger.warning(f"Failed to create DOCX: {docx_result.get('error')}")
            docx_file_path = None
        else:
            docx_file_path = docx_result.get("filepath")

        # 6. Trả về file DOCX trực tiếp
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
            f"Exam generation completed successfully. Generated {len(exam_result.get('questions', []))} questions. Returning file: {docx_file_path}"
        )

        # Trả về file DOCX để download và tự động xóa sau khi gửi
        return AutoDeleteFileResponse(
            path=docx_file_path,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
                "X-Exam-Info": f"Generated from lesson {_sanitize_filename(request.lesson_id)}",
                "X-Total-Questions": str(len(exam_result.get("questions", []))),
                "X-Search-Quality": str(search_quality),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating exam: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/generate-exam-test")
async def generate_exam_test(request: ExamMatrixRequest):
    """
    Test endpoint để tạo đề thi với mock data (bypass lesson content check)

    Endpoint này dùng để test chức năng tạo DOCX mà không cần lesson data thực
    """
    try:
        logger.info(f"Starting test exam generation for lesson: {request.lesson_id}")

        # Mock lesson content
        mock_lesson_content = {
            "success": True,
            "search_quality": 0.8,
            "content": {
                "lesson_info": {
                    "lesson_id": request.lesson_id,
                    "title": "Mock Lesson for Testing",
                    "subject": request.mon_hoc,
                    "grade": request.lop,
                },
                "content_chunks": [
                    {
                        "chunk_id": "mock_chunk_1",
                        "content": "This is mock content for testing exam generation. DNA is the genetic material that contains hereditary information.",
                        "metadata": {"page": 1, "section": "Introduction"},
                    },
                    {
                        "chunk_id": "mock_chunk_2",
                        "content": "DNA structure consists of two complementary strands forming a double helix. Each strand contains nucleotides with bases A, T, G, C.",
                        "metadata": {"page": 2, "section": "DNA Structure"},
                    },
                ],
            },
        }

        # Generate questions using mock content
        logger.info("Generating questions from matrix with mock content...")
        exam_result = await exam_generation_service.generate_questions_from_matrix(
            exam_request=request, lesson_content=mock_lesson_content
        )

        if not exam_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate questions: {exam_result.get('error', 'Unknown error')}",
            )

        # Create DOCX file
        logger.info("Creating DOCX file...")
        docx_result = await exam_docx_service.create_exam_docx(
            exam_data=exam_result, exam_request=request.model_dump()
        )

        if not docx_result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"Failed to create DOCX file: {docx_result.get('error', 'Unknown error')}",
            )

        docx_file_path = docx_result.get("filepath")

        # Check file exists
        if not docx_file_path or not os.path.exists(docx_file_path):
            raise HTTPException(
                status_code=500, detail="DOCX file was created but not found on disk"
            )

        # Create filename (sanitize để tránh lỗi encoding)
        mon_hoc_safe = _sanitize_filename(request.mon_hoc)
        exam_id_safe = _sanitize_filename(str(exam_result.get("exam_id", "unknown")))
        filename = f"Test_De_thi_{mon_hoc_safe}_{request.lop}_{exam_id_safe}.docx"

        logger.info(
            f"Test exam generation completed successfully. Returning file: {docx_file_path}"
        )

        # Return DOCX file và tự động xóa sau khi gửi
        return AutoDeleteFileResponse(
            path=docx_file_path,
            filename=filename,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}",
                "X-Exam-Info": f"Test exam from lesson {_sanitize_filename(request.lesson_id)}",
                "X-Total-Questions": str(len(exam_result.get("questions", []))),
                "X-Search-Quality": "0.8",
                "X-Test-Mode": "true",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in test exam generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/download-exam/{filename}")
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


@router.get("/lesson-content/{lesson_id}", response_model=Dict[str, Any])
async def get_lesson_content_for_exam(
    lesson_id: str,
    search_terms: List[str] = Query(None, description="Additional search terms"),
) -> Dict[str, Any]:
    """
    Lấy nội dung bài học để preview trước khi tạo đề thi

    Args:
        lesson_id: ID của bài học
        search_terms: Các từ khóa tìm kiếm bổ sung

    Returns:
        Dict chứa nội dung bài học và thông tin chất lượng
    """
    try:
        lesson_content = await exam_content_service.get_lesson_content_for_exam(
            lesson_id=lesson_id, search_terms=search_terms
        )

        if not lesson_content.get("success", False):
            raise HTTPException(
                status_code=404,
                detail=f"Lesson content not found: {lesson_content.get('error', 'Unknown error')}",
            )

        return {
            "success": True,
            "lesson_id": lesson_id,
            "lesson_info": lesson_content.get("content", {}).get("lesson_info", {}),
            "content_summary": {
                "total_words": lesson_content.get("content", {}).get("total_words", 0),
                "total_chunks": lesson_content.get("content", {}).get(
                    "total_chunks", 0
                ),
                "available_sections": lesson_content.get("content", {}).get(
                    "available_sections", []
                ),
                "search_quality": lesson_content.get("search_quality", 0.0),
            },
            "content_preview": lesson_content.get("content", {}).get(
                "main_content", ""
            )[:500]
            + "...",
            "recommendations": _get_content_recommendations(lesson_content),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting lesson content: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/search-content", response_model=Dict[str, Any])
async def search_lesson_content(request: SearchContentRequest) -> Dict[str, Any]:
    """
    Tìm kiếm nội dung trong bài học theo từ khóa

    Args:
        request: Thông tin tìm kiếm

    Returns:
        Dict chứa kết quả tìm kiếm
    """
    try:
        search_result = await exam_content_service.search_content_by_keywords(
            lesson_id=request.lesson_id,
            keywords=request.search_terms,
            limit=request.limit,
        )

        if not search_result.get("success", False):
            raise HTTPException(
                status_code=404,
                detail=f"Search failed: {search_result.get('error', 'Unknown error')}",
            )

        return {
            "success": True,
            "lesson_id": request.lesson_id,
            "search_terms": request.search_terms,
            "results": search_result.get("results", []),
            "total_found": search_result.get("total_found", 0),
            "search_quality": _calculate_search_result_quality(
                search_result.get("results", [])
            ),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching lesson content: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


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


def _get_content_recommendations(lesson_content: Dict[str, Any]) -> List[str]:
    """Tạo recommendations dựa trên chất lượng nội dung"""
    recommendations = []

    search_quality = lesson_content.get("search_quality", 0.0)
    content_info = lesson_content.get("content", {})

    if search_quality < 0.5:
        recommendations.append(
            "Chất lượng nội dung thấp. Nên bổ sung thêm từ khóa tìm kiếm."
        )

    if content_info.get("total_words", 0) < 500:
        recommendations.append(
            "Nội dung bài học ít. Có thể cần tìm thêm tài liệu bổ sung."
        )

    if len(content_info.get("available_sections", [])) < 3:
        recommendations.append(
            "Cấu trúc bài học đơn giản. Nên tạo câu hỏi đa dạng hơn."
        )

    if not recommendations:
        recommendations.append("Nội dung bài học đủ chất lượng để tạo đề thi.")

    return recommendations


def _calculate_search_result_quality(results: List[Dict[str, Any]]) -> float:
    """Tính chất lượng kết quả tìm kiếm"""
    if not results:
        return 0.0

    avg_score = sum(r.get("score", 0) for r in results) / len(results)
    return round(avg_score, 2)
