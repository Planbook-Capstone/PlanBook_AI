from fastapi import APIRouter, UploadFile, File
from typing import List
from app.services.grading_service import batch_grade_all

router = APIRouter(
    prefix="/auto_grading",
    tags=["Auto Grading"]
)

@router.post("/auto")
async def auto_grading_endpoint(
    image_files: List[UploadFile] = File(..., description="List of answer sheet images"),
    excel_file: UploadFile = File(..., description="Excel file containing answer keys"),
):
    """
    API để chấm điểm tự động.
    Nhận danh sách ảnh bài làm và file Excel đáp án.
    """
    results = await batch_grade_all(image_files, excel_file)
    return {
        "message": "Grading completed successfully",
        "results": results
    }
