"""
API endpoints cho chức năng import đề thi từ file DOCX
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import logging
import time

from app.models.exam_import_models import (
    ExamImportResponse,
    ExamImportError,
    ImportedExamData
)
from app.services.exam_import_service import get_exam_import_service
from app.constants.difficulty_levels import DifficultyLevel

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/import-docx", response_model=Dict[str, Any])
async def import_exam_from_docx(
    file: UploadFile = File(..., description="File DOCX chứa đề thi"),
    additional_instructions: Optional[str] = Form(None, description="Hướng dẫn bổ sung cho LLM"),
    staff_import: bool = Form(False, description="True nếu import cho staff")
):
    """
    Import đề thi từ file DOCX và chuyển đổi thành JSON

    Endpoint này nhận file DOCX chứa đề thi và:
    1. Trích xuất nội dung text từ file DOCX
    2. Sử dụng LLM để phân tích và chuyển đổi thành JSON
    3. Trả về dữ liệu đề thi theo định dạng chuẩn

    Args:
        file: File DOCX upload
        additional_instructions: Hướng dẫn bổ sung cho LLM (optional)
        staff_import: True để trả về format SpringBoot cho staff, False cho frontend (default: False)

    Returns:
        - Nếu staff_import=False: Dữ liệu template đề thi cho frontend
        - Nếu staff_import=True: Danh sách câu hỏi theo format SpringBoot QuestionBank với lessonId=null và suggest field

    Example:
        POST /api/v1/exam/import-docx
        Content-Type: multipart/form-data

        file: [DOCX file]
        additional_instructions: "Chú ý đặc biệt đến phần đáp án"
        staff_import: true

        Response: JSON với dữ liệu đề thi đã phân tích theo format tương ứng
    """
    start_time = time.time()
    
    try:
        logger.info(f"=== EXAM IMPORT START ===")
        logger.info(f"File: {file.filename}, Content-Type: {file.content_type}")
        
        # 1. Validate file
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Tên file không hợp lệ"
            )
        
        if not file.filename.lower().endswith('.docx'):
            raise HTTPException(
                status_code=400,
                detail="Chỉ hỗ trợ file DOCX. Vui lòng upload file có đuôi .docx"
            )
        
        # Kiểm tra content type
        if file.content_type and not file.content_type.startswith('application/'):
            logger.warning(f"Unexpected content type: {file.content_type}")
        
        # 2. Đọc nội dung file
        logger.info("Reading file content...")
        try:
            file_content = await file.read()
            file_size = len(file_content)
            logger.info(f"File size: {file_size} bytes")
            
            if file_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="File rỗng hoặc không thể đọc"
                )
            
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                raise HTTPException(
                    status_code=400,
                    detail="File quá lớn. Giới hạn 10MB"
                )
                
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Không thể đọc file: {str(e)}"
            )
        
        # 3. Import và xử lý file
        logger.info(f"Processing DOCX file... (staff_import={staff_import})")
        try:
            import_service = get_exam_import_service()
            result = await import_service.import_exam_from_docx_content(
                file_content=file_content,
                filename=file.filename,
                staff_import=staff_import
            )
            
            logger.info(f"Import result: statusCode={result.get('statusCode', 500)}")

            # Kiểm tra status code thay vì success field
            if result.get("statusCode", 500) != 200:
                # Trả về lỗi từ service
                return JSONResponse(
                    status_code=result.get("statusCode", 500),
                    content=result
                )

            # 4. Thành công - trả về dữ liệu
            processing_time = time.time() - start_time
            logger.info(f"Import completed successfully in {processing_time:.2f}s")

            # Thêm thông tin bổ sung vào response data
            if "data" in result:
                # Kiểm tra xem data có phải dict không (FE format) hay list (staff format)
                if isinstance(result["data"], dict):
                    # FE format - thêm thông tin vào data object
                    result["data"]["processing_time"] = processing_time
                    result["data"]["file_info"] = {
                        "filename": file.filename,
                        "file_size": file_size,
                        "content_type": file.content_type
                    }
                else:
                    # Staff format - thêm thông tin vào root level
                    result["processing_time"] = processing_time
                    result["file_info"] = {
                        "filename": file.filename,
                        "file_size": file_size,
                        "content_type": file.content_type
                    }

            if additional_instructions:
                result["additional_instructions"] = additional_instructions
            
            return JSONResponse(
                status_code=200,
                content=result
            )
            
        except Exception as e:
            logger.error(f"Error processing DOCX: {e}")
            processing_time = time.time() - start_time
            
            error_response = ExamImportError(
                message="Processing failed",
                error=f"Lỗi xử lý file: {str(e)}",
                error_code="PROCESSING_ERROR",
                details={
                    "filename": file.filename,
                    "file_size": file_size,
                    "processing_time": processing_time,
                    "error_type": type(e).__name__
                }
            )
            
            return JSONResponse(
                status_code=500,
                content=error_response.model_dump()
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in import endpoint: {e}")
        processing_time = time.time() - start_time
        
        error_response = ExamImportError(
            message="System error",
            error=f"Lỗi hệ thống: {str(e)}",
            error_code="SYSTEM_ERROR",
            details={
                "processing_time": processing_time,
                "error_type": type(e).__name__
            }
        )
        
        return JSONResponse(
            status_code=500,
            content=error_response.model_dump()
        )




@router.get("/import-status")
async def get_import_status():
    """
    Lấy thông tin trạng thái service import

    Returns:
        Dict: Thông tin trạng thái
    """
    try:
        return {
            "success": True,
            "service_status": "active",
            "supported_formats": ["docx"],
            "max_file_size": "10MB",
            "features": [
                "DOCX text extraction",
                "LLM-based analysis",
                "Multi-part exam support",
                "Answer key extraction",
                "Chemistry atomic masses support",
                "Dual format output (Frontend/SpringBoot)",
                "Staff import with difficulty analysis"
            ],
            "model_info": {
                "llm_model": "google/gemini-2.0-flash-001",
                "provider": "OpenRouter"
            },
            "output_formats": {
                "frontend": "Template format for exam creation UI",
                "staff": "QuestionBank format for SpringBoot backend"
            },
            "difficulty_levels": [
                f"{DifficultyLevel.KNOWLEDGE.value} - Recall of facts, terms, basic concepts",
                f"{DifficultyLevel.COMPREHENSION.value} - Understanding meaning, interpretation",
                f"{DifficultyLevel.APPLICATION.value} - Applying knowledge to new situations"
            ]
        }

    except Exception as e:
        logger.error(f"Error getting import status: {e}")
        return {
            "success": False,
            "error": str(e)
        }
