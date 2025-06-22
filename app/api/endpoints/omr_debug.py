from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import Dict, List
import os
from pathlib import Path
import logging

from app.services.omr_debug_processor import OMRDebugProcessor

logger = logging.getLogger(__name__)

router = APIRouter(tags=["OMR Debug"])


@router.post("/process_test_image")
async def process_test_image():
    """
    Xử lý ảnh test với debug chi tiết
    """
    try:
        # Đường dẫn ảnh test
        image_path = "data/grading/test_images/1.jpeg"

        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=404, detail=f"Image not found: {image_path}"
            )

        # Tạo processor và xử lý
        processor = OMRDebugProcessor()
        result = processor.process_answer_sheet(image_path)

        if not result["success"]:
            raise HTTPException(
                status_code=500, detail=f"Processing failed: {result['error']}"
            )

        # Lấy danh sách debug images
        debug_dir = Path(result["debug_dir"])
        debug_files = []
        if debug_dir.exists():
            debug_files = [f.name for f in sorted(debug_dir.glob("*.jpg"))]

        return {
            "success": True,
            "student_id": result["student_id"],
            "test_code": result["test_code"],
            "total_answers": len(result["answers"]),
            "answers": result["answers"],
            "debug_dir": str(debug_dir),
            "debug_files": debug_files,
            "message": f"Processed successfully. {len(debug_files)} debug images created.",
        }

    except Exception as e:
        logger.error(f"Error in process_test_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug_images")
async def list_debug_images():
    """
    Lấy danh sách debug images
    """
    try:
        debug_dir = Path("data/grading/debug")

        if not debug_dir.exists():
            return {
                "debug_files": [],
                "message": "Debug directory not found. Run process_test_image first.",
            }

        debug_files = []
        for file in sorted(debug_dir.glob("*.jpg")):
            debug_files.append(
                {
                    "filename": file.name,
                    "size": file.stat().st_size,
                    "url": f"/omr_debug/debug_image/{file.name}",
                }
            )

        return {
            "debug_files": debug_files,
            "total_files": len(debug_files),
            "debug_dir": str(debug_dir),
        }

    except Exception as e:
        logger.error(f"Error listing debug images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/debug_image/{filename}")
async def get_debug_image(filename: str):
    """
    Lấy debug image theo tên file
    """
    try:
        debug_dir = Path("data/grading/debug")
        file_path = debug_dir / filename

        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Debug image not found: {filename}"
            )

        return FileResponse(
            path=str(file_path), media_type="image/jpeg", filename=filename
        )

    except Exception as e:
        logger.error(f"Error getting debug image {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/clear_debug")
async def clear_debug_images():
    """
    Xóa tất cả debug images
    """
    try:
        debug_dir = Path("data/grading/debug")

        if not debug_dir.exists():
            return {"message": "Debug directory not found"}

        deleted_count = 0
        for file in debug_dir.glob("*.jpg"):
            file.unlink()
            deleted_count += 1

        return {
            "message": f"Cleared {deleted_count} debug images",
            "deleted_count": deleted_count,
        }

    except Exception as e:
        logger.error(f"Error clearing debug images: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing_steps")
async def get_processing_steps():
    """
    Lấy thông tin các bước xử lý
    """
    return {
        "processing_steps": [
            {"step": 1, "name": "01_original", "description": "Ảnh gốc đầu vào"},
            {
                "step": 2,
                "name": "02_preprocessed",
                "description": "Ảnh sau tiền xử lý (grayscale, denoised, enhanced, binary)",
            },
            {
                "step": 3,
                "name": "03_corners_detected",
                "description": "Phát hiện 4 góc markers (hình vuông đen)",
            },
            {
                "step": 4,
                "name": "04_aligned",
                "description": "Ảnh đã căn chỉnh bằng perspective transform",
            },
            {
                "step": 5,
                "name": "05_region_student_id",
                "description": "Vùng Student ID (8 cột số)",
            },
            {
                "step": 6,
                "name": "06_region_test_code",
                "description": "Vùng Test Code (3 cột số)",
            },
            {
                "step": 7,
                "name": "07-10_region_answers",
                "description": "4 vùng câu trả lời (01-15, 16-30, 31-45, 46-60)",
            },
            {
                "step": 8,
                "name": "11_region_answers_full",
                "description": "Tổng hợp tất cả câu trả lời",
            },
            {
                "step": 9,
                "name": "12-13_binary",
                "description": "Ảnh nhị phân của Student ID và Test Code",
            },
            {
                "step": 10,
                "name": "14_grid",
                "description": "Grid detection với bubbles được đánh dấu",
            },
            {
                "step": 11,
                "name": "15_answers",
                "description": "Phát hiện câu trả lời với đáp án được chọn",
            },
            {
                "step": 12,
                "name": "99_final_result",
                "description": "Kết quả cuối cùng với thông tin tổng hợp",
            },
        ],
        "total_steps": 12,
        "description": "Quy trình xử lý OMR với debug chi tiết từng bước",
    }
