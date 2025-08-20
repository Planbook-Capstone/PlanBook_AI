"""
API endpoints cho chức năng tạo bài kiểm tra từ ma trận đề thi
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging
from app.models.smart_exam_models import SmartExamRequest

logger = logging.getLogger(__name__)
router = APIRouter()


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
        request: SmartExamRequest chứa thông tin trường, môn học, ma trận đề thi, user_id (optional), isExportDocx

    Returns:
        Dict: {"task_id": "...", "message": "..."} để theo dõi qua /api/v1/tasks/{task_id}/status
        - Nếu isExportDocx = true: tạo file DOCX và upload Google Drive
        - Nếu isExportDocx = false: trả về JSON format với cấu trúc parts

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
            "isExportDocx": false,
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

        Kết quả task (khi isExportDocx = false):
        {
            "success": true,
            "output": {
                "exam_data": {
                    "parts": [...]  // JSON format theo cấu trúc mới
                }
            }
        }
    """
    print("=== SMART EXAM ENDPOINT CALLED ===")
    try:
        logger.info(f"=== SMART EXAM GENERATION START (ASYNC) ===")
        logger.info(f"Request: {request.school} - {request.subject} - Grade {request.grade}")
        logger.info(f"Export mode: {'DOCX' if request.isExportDocx else 'JSON'}")
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
