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

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/import-docx", response_model=Dict[str, Any])
async def import_exam_from_docx(
    file: UploadFile = File(..., description="File DOCX chứa đề thi"),
    additional_instructions: Optional[str] = Form(None, description="Hướng dẫn bổ sung cho LLM")
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

    Returns:
        ExamImportResponse: Dữ liệu đề thi đã chuyển đổi hoặc lỗi nếu thất bại

    Example:
        POST /api/v1/exam/import-docx
        Content-Type: multipart/form-data
        
        file: [DOCX file]
        additional_instructions: "Chú ý đặc biệt đến phần đáp án"

        Response: JSON với dữ liệu đề thi đã phân tích
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
        logger.info("Processing DOCX file...")
        try:
            import_service = get_exam_import_service()
            result = await import_service.import_exam_from_docx_content(
                file_content=file_content,
                filename=file.filename
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
                result["data"]["processing_time"] = processing_time
                result["data"]["file_info"] = {
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


# @router.post("/import-docx-test")
async def test_import_exam():
    """
    Test endpoint để kiểm tra chức năng import với dữ liệu mẫu

    Returns:
        Dict: Kết quả test import
    """
    try:
        logger.info("=== TEST IMPORT START ===")
        
        # Dữ liệu mẫu để test
        sample_exam_text = """
BỘ GIÁO DỤC VÀ ĐÀO TẠO
Trường THPT Hong Thinh
ĐỀ KIỂM TRA LỚP 12
Môn: HOA HOC
Thời gian làm bài: 90 phút, không kể thời gian phát đề

Họ, tên thí sinh: ..................................................
Mã đề: 1234
Số báo danh: .......................................................

BẢNG NGUYÊN TỬ KHỐI CỦA CÁC NGUYÊN TỐ HÓA HỌC
H = 1; C = 12; N = 14; O = 16; S = 32

PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn. Thí sinh trả lời từ câu 1 đến câu 2.
(Mỗi câu trả lời đúng thí sinh được 0,25 điểm)

Câu 1. Hạt nào sau đây không cấu tạo nên hạt nhân nguyên tử?
A. Proton
B. Neutron
C. Electron
D. Cả proton và neutron

Câu 2. Đơn vị khối lượng nguyên tử (amu) tương đương với bao nhiêu kg?
A. 1,6605 x 10^-27 kg
B. 1,6605 x 10^-24 kg
C. 1,602 x 10^-19 kg
D. 9,109 x 10^-31 kg

PHẦN II. Câu trắc nghiệm đúng sai. Thí sinh trả lời từ câu 1 đến câu 1.
Câu 1. Xét về cấu tạo nguyên tử Hydrogen (H), cho các phát biểu sau:
a) Nguyên tử Hydrogen chỉ chứa một proton trong hạt nhân.
b) Nguyên tử Hydrogen luôn chứa một neutron trong hạt nhân.
c) Nguyên tử Hydrogen trung hòa về điện vì số proton bằng số electron.
d) Khối lượng của nguyên tử Hydrogen chủ yếu tập trung ở lớp vỏ electron.

PHẦN III. Câu trắc nghiệm trả lời ngắn. Thí sinh trả lời từ câu 1 đến câu 1
Câu 1. Một nguyên tử X có cấu tạo gồm 1 proton, 1 electron và không có neutron. Tính khối lượng gần đúng của nguyên tử X theo đơn vị amu.

--- Hết ---

ĐÁP ÁN

PHẦN I. Câu trắc nghiệm nhiều phương án lựa chọn.
Câu    1    2
Chọn   C    A

PHẦN II. Câu trắc nghiệm đúng sai.
Câu 1: a) Đúng, b) Sai, c) Đúng, d) Sai

PHẦN III. Câu trắc nghiệm trả lời ngắn.
Câu 1: 1
"""
        
        # Gọi service để phân tích
        import_service = get_exam_import_service()
        result = await import_service._analyze_exam_with_llm(
            exam_text=sample_exam_text,
            filename="test_exam.docx"
        )
        
        return {
            "success": True,
            "message": "Test import completed",
            "test_data": sample_exam_text[:500] + "...",
            "analysis_result": result,
            "test_mode": True
        }

    except Exception as e:
        logger.error(f"Error in test import: {e}")
        return {
            "success": False,
            "error": str(e),
            "test_mode": True
        }


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
                "Chemistry atomic masses support"
            ],
            "model_info": {
                "llm_model": "google/gemini-2.0-flash-001",
                "provider": "OpenRouter"
            }
        }

    except Exception as e:
        logger.error(f"Error getting import status: {e}")
        return {
            "success": False,
            "error": str(e)
        }
