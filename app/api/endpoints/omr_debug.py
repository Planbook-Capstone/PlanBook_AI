from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from typing import Dict, List
import os
from pathlib import Path
import logging
import cv2
import numpy as np

from app.services.omr_debug_processor import OMRDebugProcessor
from app.services.omr_pipeline_processor import omr_pipeline_processor
from app.services.all_marker_scanner import all_marker_scanner
from app.services.marker_based_block_divider import marker_based_block_divider
from app.services.contour_block_cutter import contour_block_cutter

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


@router.post("/process_pipeline")
async def process_omr_pipeline():
    """
    Xử lý OMR theo pipeline AI Trắc Nghiệm mới

    Pipeline 7 bước:
    1. Đọc ảnh đầu vào (grayscale + blur)
    2. Phát hiện marker đen lớn (vuông) + perspective transform
    3. Cắt ảnh thành 2 vùng (top: thông tin, bottom: câu trả lời)
    4. Phân tích vùng trả lời:
       - Section I: 40 câu ABCD (4 cột x 10 hàng)
       - Section II: 8 câu đúng/sai với sub-questions
       - Section III: 6 câu điền số 0-9
    5. Tổng hợp kết quả
    6. Tạo ảnh kết quả đánh dấu
    """
    try:
        # Đường dẫn ảnh test
        image_path = "data/grading/sample.jpg"

        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=404, detail=f"Image not found: {image_path}"
            )

        # Xử lý với pipeline processor
        result = omr_pipeline_processor.process_omr_sheet(image_path)

        if not result["success"]:
            raise HTTPException(
                status_code=500, detail=f"Pipeline processing failed: {result['error']}"
            )

        # Lấy danh sách debug images
        debug_dir = Path(result["debug_dir"])
        debug_files = []
        if debug_dir.exists():
            debug_files = [f.name for f in sorted(debug_dir.glob("*.jpg"))]

        # Tính toán thống kê
        results_data = result["results"]
        section1_count = len(results_data.get("Section I", {}))
        section2_count = len(results_data.get("Section II", {}))
        section3_count = len(results_data.get("Section III", {}))

        return {
            "success": True,
            "pipeline_version": "AI Trắc Nghiệm v1.0",
            "processing_steps": result.get("processing_steps", 7),
            "total_markers": result.get("total_markers", 0),
            "results": {
                "Section I": {
                    "description": "40 câu Multiple Choice (A/B/C/D)",
                    "total_questions": section1_count,
                    "answers": results_data.get("Section I", {})
                },
                "Section II": {
                    "description": "8 câu True/False với sub-questions",
                    "total_questions": section2_count,
                    "answers": results_data.get("Section II", {})
                },
                "Section III": {
                    "description": "6 câu digit selection (0-9)",
                    "total_questions": section3_count,
                    "answers": results_data.get("Section III", {})
                }
            },
            "summary": results_data.get("summary", {}),
            "debug_info": {
                "debug_dir": str(debug_dir),
                "debug_files": debug_files,
                "total_debug_images": len(debug_files)
            },
            "message": f"Pipeline processing completed. {len(debug_files)} debug images created."
        }

    except Exception as e:
        logger.error(f"Error in process_omr_pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/pipeline_steps")
async def get_pipeline_steps():
    """
    Lấy thông tin các bước xử lý trong pipeline AI Trắc Nghiệm
    """
    return {
        "pipeline_name": "AI Trắc Nghiệm Pipeline",
        "version": "1.0",
        "processing_steps": [
            {
                "step": 1,
                "name": "01_original",
                "description": "Đọc ảnh đầu vào",
                "details": "Sử dụng cv2.imread, convert grayscale + GaussianBlur"
            },
            {
                "step": 2,
                "name": "02_preprocessed",
                "description": "Ảnh sau tiền xử lý",
                "details": "Grayscale conversion và làm mờ nhẹ"
            },
            {
                "step": 3,
                "name": "03_markers_detected",
                "description": "Phát hiện marker đen lớn (vuông)",
                "details": "Tìm contour, lọc theo diện tích lớn và tỉ lệ ≈ 1.0"
            },
            {
                "step": 4,
                "name": "04_aligned",
                "description": "Perspective transform",
                "details": "Xác định 4 marker ngoài cùng → chuẩn hóa ảnh"
            },
            {
                "step": 5,
                "name": "05_top_region",
                "description": "Vùng thông tin (top)",
                "details": "Cắt vùng chứa mã SBD, đề thi, thông tin"
            },
            {
                "step": 6,
                "name": "06_bottom_region",
                "description": "Vùng câu trả lời (bottom)",
                "details": "Cắt vùng chứa các phần trả lời"
            },
            {
                "step": 7,
                "name": "07_section1_abcd",
                "description": "Section I - 40 câu ABCD",
                "details": "4 cột x 10 hàng, mỗi hàng 4 ô tròn A,B,C,D"
            },
            {
                "step": 8,
                "name": "08_section2_true_false",
                "description": "Section II - 8 câu đúng/sai",
                "details": "8 khối, mỗi khối có 2 lựa chọn (Đúng/Sai) và 2-3 dòng a/b/c"
            },
            {
                "step": 9,
                "name": "09_section3_digits",
                "description": "Section III - 6 câu điền số",
                "details": "6 cột, mỗi cột 10 bubble theo hàng dọc (số 0-9)"
            },
            {
                "step": 10,
                "name": "99_final_result",
                "description": "Kết quả cuối cùng",
                "details": "Tổng hợp và hiển thị kết quả với đánh dấu"
            }
        ],
        "total_steps": 10,
        "sections": {
            "Section I": {
                "description": "40 câu Multiple Choice (A/B/C/D)",
                "layout": "4 cột x 10 hàng",
                "bubble_detection": "Contour tròn, diện tích nhỏ, aspect ≈ 1.0"
            },
            "Section II": {
                "description": "8 câu True/False với sub-questions",
                "layout": "8 khối, mỗi khối có 2 lựa chọn và 2-3 dòng",
                "bubble_detection": "Pixel ratio > threshold"
            },
            "Section III": {
                "description": "6 câu digit selection (0-9)",
                "layout": "6 cột x 10 hàng dọc",
                "bubble_detection": "Xác định dòng được tô → số tương ứng"
            }
        }
    }


@router.post("/scan_all_markers")
async def scan_all_markers():
    """
    Quét và đánh dấu tất cả marker (lớn và nhỏ) trên ảnh

    Features:
    - Phát hiện marker vuông lớn (large markers) - màu đỏ
    - Phát hiện marker vuông nhỏ (small markers) - màu xanh lá
    - Tạo ảnh tổng hợp với tất cả marker được label
    - Thống kê số lượng và vị trí marker
    """
    try:
        # Đường dẫn ảnh test
        image_path = "data/grading/sample.jpg"

        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=404, detail=f"Image not found: {image_path}"
            )

        # Quét tất cả marker
        result = all_marker_scanner.scan_all_markers(image_path)

        if not result["success"]:
            raise HTTPException(
                status_code=500, detail=f"Marker scanning failed: {result['error']}"
            )

        # Lấy thông tin marker
        large_markers = result["large_markers"]
        small_markers = result["small_markers"]
        statistics = result["statistics"]

        # Lấy danh sách debug images
        debug_dir = Path(result["debug_dir"])
        debug_files = []
        if debug_dir.exists():
            debug_files = [f.name for f in sorted(debug_dir.glob("*.jpg"))]

        return {
            "success": True,
            "scanner_version": "All Marker Scanner v1.0",
            "image_path": image_path,
            "markers": {
                "large_markers": {
                    "count": len(large_markers),
                    "color": "Red (đỏ)",
                    "description": "Corner/alignment markers",
                    "area_range": "150-500 pixels (19x19 ≈ 361px, thực tế ~200px)",
                    "aspect_ratio_range": "0.85-1.15",
                    "markers": [
                        {
                            "id": m["id"],
                            "center": m["center"],
                            "area": m["area"],
                            "aspect_ratio": round(m["aspect_ratio"], 2)
                        } for m in large_markers
                    ]
                },
                "small_markers": {
                    "count": len(small_markers),
                    "color": "Green (xanh lá)",
                    "description": "Section division markers",
                    "area_range": "25-150 pixels (9x9 ≈ 81px)",
                    "aspect_ratio_range": "0.8-1.25",
                    "markers": [
                        {
                            "id": m["id"],
                            "center": m["center"],
                            "area": m["area"],
                            "aspect_ratio": round(m["aspect_ratio"], 2)
                        } for m in small_markers[:20]  # Limit to first 20 for API response
                    ]
                }
            },
            "statistics": statistics,
            "visualization": {
                "all_markers_image": result.get("all_markers_image_path"),
                "features": [
                    "Color-coded markers (Red for large, Green for small)",
                    "Unique ID labels (L1, L2... for large, S1, S2... for small)",
                    "Bounding boxes and center points",
                    "Legend with counts and descriptions",
                    "Comprehensive statistics"
                ]
            },
            "debug_info": {
                "debug_dir": str(debug_dir),
                "debug_files": debug_files,
                "total_debug_images": len(debug_files)
            },
            "message": f"Enhanced marker scanning completed. Found {len(large_markers)} large + {len(small_markers)} small SQUARE markers (filtered out round bubbles). {len(debug_files)} debug images created."
        }

    except Exception as e:
        logger.error(f"Error in scan_all_markers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/all_markers_image/{filename}")
async def get_all_markers_image(filename: str):
    """
    Lấy ảnh marker từ all markers debug directory
    """
    try:
        debug_dir = Path("data/grading/all_markers_debug")
        file_path = debug_dir / filename

        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"All markers image not found: {filename}"
            )

        return FileResponse(
            path=str(file_path), media_type="image/jpeg", filename=filename
        )

    except Exception as e:
        logger.error(f"Error getting all markers image {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/divide_blocks")
async def divide_blocks(file: UploadFile = File(None)):
    """
    Chia form OMR thành các blocks dựa trên markers đã phát hiện

    Features:
    - Upload ảnh OMR để phân tích hoặc sử dụng sample.jpg mặc định
    - Phát hiện tất cả markers (large và small)
    - Chia form thành các regions chính
    - Chia regions thành blocks chi tiết
    - Tạo visualization với color-coding
    - Xuất kết quả JSON chi tiết
    """
    try:
        # Load image - either from upload or default sample
        if file and file.filename:
            # Validate file
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                raise HTTPException(status_code=400, detail="Only PNG, JPG, JPEG files are allowed")

            # Read uploaded image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

            filename = file.filename
        else:
            # Use default sample image
            image_path = "data/grading/sample.jpg"
            if not os.path.exists(image_path):
                raise HTTPException(status_code=404, detail=f"Sample image not found: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                raise HTTPException(status_code=400, detail="Cannot load sample image")

            filename = "sample.jpg"

        # Step 1: Scan all markers first
        marker_result = all_marker_scanner.scan_all_markers(image)

        if not marker_result['success']:
            raise HTTPException(status_code=400, detail="Failed to detect markers")

        # Step 2: Divide into blocks using detected markers
        large_markers = marker_result['large_markers']
        small_markers = marker_result['small_markers']

        block_result = marker_based_block_divider.divide_form_into_blocks(
            image, large_markers, small_markers
        )

        # Get debug files
        debug_dir = Path(block_result.get("debug_dir", "data/grading/block_division_debug"))
        debug_files = []
        if debug_dir.exists():
            debug_files = [f.name for f in sorted(debug_dir.glob("*.jpg"))]

        # Combine results
        combined_result = {
            "success": True,
            "filename": filename,
            "image_size": f"{image.shape[1]}x{image.shape[0]}",
            "marker_detection": {
                "large_markers_count": len(large_markers),
                "small_markers_count": len(small_markers),
                "total_markers": len(large_markers) + len(small_markers),
                "large_markers": [
                    {
                        "id": m["id"],
                        "center": m["center"],
                        "area": m["area"]
                    } for m in large_markers
                ],
                "small_markers": [
                    {
                        "id": m["id"],
                        "center": m["center"],
                        "area": m["area"]
                    } for m in small_markers[:15]  # Limit for API response
                ]
            },
            "block_division": {
                "total_blocks": block_result.get("total_blocks", 0),
                "main_regions": list(block_result.get("main_regions", {}).keys()),
                "block_summary": block_result.get("block_summary", {}),
                "layout_analysis": block_result.get("layout_analysis", {}),
                "detailed_blocks": [
                    {
                        "id": block["id"],
                        "region": block["region"],
                        "type": block["type"],
                        "bbox": block["bbox"],
                        "marker_count": block.get("marker_count", 0)
                    } for block in block_result.get("detailed_blocks", [])
                ]
            },
            "visualization": {
                "block_division_image": str(debug_dir / "block_division_result.jpg"),
                "features": [
                    "Color-coded markers (Red for large, Green for small)",
                    "Block boundaries with labels",
                    "Region separation lines",
                    "Comprehensive legend with statistics"
                ]
            },
            "debug_info": {
                "debug_dir": str(debug_dir),
                "debug_files": debug_files,
                "total_debug_images": len(debug_files)
            },
            "processing_summary": {
                "step_1": "Marker detection completed",
                "step_2": "Layout analysis completed",
                "step_3": "Region division completed",
                "step_4": "Block creation completed",
                "step_5": "Visualization generated"
            },
            "message": f"Block division completed successfully. Created {block_result.get('total_blocks', 0)} blocks from {len(large_markers)} large + {len(small_markers)} small markers."
        }

        return combined_result

    except Exception as e:
        logger.error(f"Error in divide_blocks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/block_division_image/{filename}")
async def get_block_division_image(filename: str):
    """
    Lấy ảnh block division từ debug directory
    """
    try:
        debug_dir = Path("data/grading/block_division_debug")
        file_path = debug_dir / filename

        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Block division image not found: {filename}"
            )

        return FileResponse(
            path=str(file_path), media_type="image/jpeg", filename=filename
        )

    except Exception as e:
        logger.error(f"Error getting block division image {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cut_contour_blocks")
async def cut_contour_blocks(file: UploadFile = File(None)):
    """
    Cắt các blocks dựa trên contours được phát hiện trong enhanced detection

    Features:
    - Phát hiện tất cả contours giống như 01_enhanced_all_contours.jpg
    - Phân loại contours thành large_regions, medium_blocks, small_markers
    - Cắt từng vùng thành file ảnh riêng biệt
    - Tạo visualization với bounding boxes
    - Lưu tất cả blocks vào thư mục individual_blocks
    """
    try:
        # Load image - either from upload or default sample
        if file and file.filename:
            # Validate file
            if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                raise HTTPException(status_code=400, detail="Only PNG, JPG, JPEG files are allowed")

            # Read uploaded image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise HTTPException(status_code=400, detail="Invalid image file")

            filename = file.filename
        else:
            # Use default sample image
            image_path = "data/grading/sample.jpg"
            if not os.path.exists(image_path):
                raise HTTPException(status_code=404, detail=f"Sample image not found: {image_path}")

            image = cv2.imread(image_path)
            if image is None:
                raise HTTPException(status_code=400, detail="Cannot load sample image")

            filename = "sample.jpg"

        # Process with contour block cutter
        result = contour_block_cutter.cut_contour_blocks(image)

        if not result['success']:
            raise HTTPException(status_code=400, detail="Failed to cut contour blocks")

        # Get debug files
        debug_dir = Path(result.get("visualization_path", "")).parent
        debug_files = []
        if debug_dir.exists():
            debug_files = [f.name for f in sorted(debug_dir.glob("*.jpg"))]

        # Get individual block files
        blocks_dir = Path(result.get("block_images_dir", ""))
        block_files = []
        if blocks_dir.exists():
            block_files = [f.name for f in sorted(blocks_dir.glob("*.jpg"))]

        # Create response
        response_result = {
            "success": True,
            "filename": filename,
            "image_size": f"{image.shape[1]}x{image.shape[0]}",
            "contour_detection": {
                "total_contours": result.get("total_contours", 0),
                "classified_contours": {
                    "large_regions": len(result.get("classified_contours", {}).get("large_regions", [])),
                    "medium_blocks": len(result.get("classified_contours", {}).get("medium_blocks", [])),
                    "small_markers": len(result.get("classified_contours", {}).get("small_markers", [])),
                    "noise": len(result.get("classified_contours", {}).get("noise", []))
                }
            },
            "block_cutting": {
                "total_blocks_cut": result.get("summary", {}).get("total_blocks_cut", 0),
                "blocks_by_type": result.get("summary", {}).get("blocks_by_type", {}),
                "average_block_size": result.get("summary", {}).get("average_block_size", 0),
                "largest_block": result.get("summary", {}).get("largest_block", ""),
                "smallest_block": result.get("summary", {}).get("smallest_block", "")
            },
            "output_files": {
                "visualization_image": str(debug_dir / "contour_blocks_result.jpg"),
                "individual_blocks_dir": str(blocks_dir),
                "total_block_images": len(block_files),
                "block_files": block_files[:10]  # Show first 10 files
            },
            "debug_info": {
                "debug_dir": str(debug_dir),
                "debug_files": debug_files,
                "total_debug_images": len(debug_files)
            },
            "processing_summary": {
                "step_1": "Contour detection completed",
                "step_2": "Contour classification completed",
                "step_3": "Block cutting completed",
                "step_4": "Individual block images saved",
                "step_5": "Visualization generated"
            },
            "message": f"Contour block cutting completed successfully. Cut {result.get('summary', {}).get('total_blocks_cut', 0)} blocks from {result.get('total_contours', 0)} contours."
        }

        return response_result

    except Exception as e:
        logger.error(f"Error in cut_contour_blocks: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/contour_blocks_image/{filename}")
async def get_contour_blocks_image(filename: str):
    """
    Lấy ảnh contour blocks từ debug directory
    """
    try:
        debug_dir = Path("data/grading/contour_blocks_debug")
        file_path = debug_dir / filename

        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Contour blocks image not found: {filename}"
            )

        return FileResponse(
            path=str(file_path), media_type="image/jpeg", filename=filename
        )

    except Exception as e:
        logger.error(f"Error getting contour blocks image {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/individual_block_image/{filename}")
async def get_individual_block_image(filename: str):
    """
    Lấy ảnh individual block từ individual_blocks directory
    """
    try:
        blocks_dir = Path("data/grading/contour_blocks_debug/individual_blocks")
        file_path = blocks_dir / filename

        if not file_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Individual block image not found: {filename}"
            )

        return FileResponse(
            path=str(file_path), media_type="image/jpeg", filename=filename
        )

    except Exception as e:
        logger.error(f"Error getting individual block image {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/viewer", response_class=HTMLResponse)
async def omr_debug_viewer():
    """
    Trang web hiển thị debug images của OMR processing với AI Pipeline và Block Division
    """

    # Lấy danh sách debug images từ nhiều thư mục
    debug_dirs = {
        "OMR Processing": Path("data/grading/debug"),
        "All Markers": Path("data/grading/all_markers_debug"),
        "Block Division": Path("data/grading/block_division_debug"),
        "Contour Blocks": Path("data/grading/contour_blocks_debug"),
        "Individual Blocks": Path("data/grading/contour_blocks_debug/individual_blocks")
    }

    all_debug_files = []
    debug_sections = {}

    for section_name, debug_dir in debug_dirs.items():
        section_files = []
        if debug_dir.exists():
            section_files = [f.name for f in sorted(debug_dir.glob("*.jpg"))]
        debug_sections[section_name] = {
            "files": section_files,
            "count": len(section_files),
            "dir": str(debug_dir)
        }
        all_debug_files.extend(section_files)

    def get_image_url(section_name, filename):
        """Get the correct URL for different types of debug images"""
        if section_name == "OMR Processing":
            return f"/api/v1/omr_debug/debug_image/{filename}"
        elif section_name == "All Markers":
            return f"/api/v1/omr_debug/all_markers_image/{filename}"
        elif section_name == "Block Division":
            return f"/api/v1/omr_debug/block_division_image/{filename}"
        elif section_name == "Contour Blocks":
            return f"/api/v1/omr_debug/contour_blocks_image/{filename}"
        elif section_name == "Individual Blocks":
            return f"/api/v1/omr_debug/individual_block_image/{filename}"
        else:
            return f"/api/v1/omr_debug/debug_image/{filename}"

    # Tạo HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OMR Debug Viewer - PlanBook AI</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                padding: 30px;
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 3px solid #3498db;
                padding-bottom: 15px;
            }}
            .controls {{
                text-align: center;
                margin-bottom: 30px;
            }}
            .btn {{
                background: #3498db;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
                margin: 0 10px;
                transition: background 0.3s;
            }}
            .btn:hover {{
                background: #2980b9;
            }}
            .btn.success {{
                background: #27ae60;
            }}
            .btn.success:hover {{
                background: #229954;
            }}
            .btn.info {{
                background: #17a2b8;
            }}
            .btn.info:hover {{
                background: #138496;
            }}
            .btn.accent {{
                background: #6f42c1;
            }}
            .btn.accent:hover {{
                background: #5a32a3;
            }}
            .btn.warning {{
                background: #f39c12;
            }}
            .btn.warning:hover {{
                background: #e67e22;
            }}
            .btn.danger {{
                background: #e74c3c;
            }}
            .btn.danger:hover {{
                background: #c0392b;
            }}
            .status {{
                text-align: center;
                margin: 20px 0;
                padding: 15px;
                border-radius: 5px;
                font-weight: bold;
            }}
            .status.success {{
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
            }}
            .status.info {{
                background: #d1ecf1;
                color: #0c5460;
                border: 1px solid #bee5eb;
            }}
            .status.error {{
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .stat-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                text-align: center;
                border: 1px solid #ddd;
            }}
            .stat-number {{
                font-size: 24px;
                font-weight: bold;
                color: #007bff;
            }}
            .stat-label {{
                color: #666;
                margin-top: 5px;
            }}
            .image-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }}
            .image-card {{
                background: white;
                border: 1px solid #ddd;
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                transition: transform 0.3s;
            }}
            .image-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }}
            .image-card img {{
                width: 100%;
                height: 200px;
                object-fit: cover;
                cursor: pointer;
            }}
            .image-card .title {{
                padding: 15px;
                font-weight: bold;
                color: #2c3e50;
                background: #ecf0f1;
                text-align: center;
            }}
            .modal {{
                display: none;
                position: fixed;
                z-index: 1000;
                left: 0;
                top: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.9);
            }}
            .modal-content {{
                margin: auto;
                display: block;
                width: 90%;
                max-width: 1000px;
                max-height: 90%;
                object-fit: contain;
            }}
            .close {{
                position: absolute;
                top: 15px;
                right: 35px;
                color: #f1f1f1;
                font-size: 40px;
                font-weight: bold;
                cursor: pointer;
            }}
            .close:hover {{
                color: #bbb;
            }}
            .loading {{
                text-align: center;
                color: #7f8c8d;
                font-style: italic;
                padding: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔍 OMR Debug Viewer + 🤖 AI Pipeline</h1>

            <div class="controls">
                <button class="btn success" onclick="processImage()">🚀 Xử lý ảnh test</button>
                <button class="btn success" onclick="processPipeline()">🤖 AI Pipeline</button>
                <button class="btn info" onclick="scanAllMarkers()">🔍 Scan All Markers</button>
                <button class="btn accent" onclick="divideBlocks()">📦 Divide Blocks</button>
                <button class="btn warning" onclick="cutContourBlocks()">✂️ Cut Contour Blocks</button>
                <button class="btn" onclick="refreshImages()">🔄 Làm mới</button>
                <button class="btn danger" onclick="clearImages()">🗑️ Xóa debug images</button>
            </div>

            <div id="status" class="status info">
                📊 Tìm thấy {len(all_debug_files)} debug images tổng cộng
            </div>

            <!-- Stats for different sections -->
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-number">{debug_sections["OMR Processing"]["count"]}</div>
                    <div class="stat-label">OMR Processing</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{debug_sections["All Markers"]["count"]}</div>
                    <div class="stat-label">Marker Detection</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{debug_sections["Block Division"]["count"]}</div>
                    <div class="stat-label">Block Division</div>
                </div>
                <div class="stat-card">
                    <div class="stat-number">{len(all_debug_files)}</div>
                    <div class="stat-label">Total Images</div>
                </div>
            </div>

            <!-- Sections for different types of debug images -->
            {"".join([f'''
            <div class="section" style="margin: 30px 0; padding: 20px; background: #f8f9fa; border-radius: 10px;">
                <h2 style="color: #333; margin-bottom: 20px;">📁 {section_name} ({section_info["count"]} images)</h2>
                <div class="image-grid">
                    {"".join([f"""
                    <div class="image-card">
                        <img src="{get_image_url(section_name, filename)}"
                             alt="{filename}"
                             onclick="openModal(this.src, '{filename}')">
                        <div class="title">{filename}</div>
                    </div>
                    """ for filename in section_info["files"]])}
                </div>
                {f'<div class="loading">Chưa có {section_name.lower()} images.</div>' if not section_info["files"] else ''}
            </div>
            ''' for section_name, section_info in debug_sections.items() if section_info["count"] > 0])}

            {f'<div class="loading">Chưa có debug images. Nhấn các nút xử lý để bắt đầu.</div>' if not all_debug_files else ''}
        </div>

        <!-- Modal để hiển thị ảnh lớn -->
        <div id="imageModal" class="modal">
            <span class="close" onclick="closeModal()">&times;</span>
            <img class="modal-content" id="modalImage">
        </div>

        <script>
            function openModal(src, filename) {{
                document.getElementById('imageModal').style.display = 'block';
                document.getElementById('modalImage').src = src;
                document.getElementById('modalImage').alt = filename;
            }}

            function closeModal() {{
                document.getElementById('imageModal').style.display = 'none';
            }}

            async function processImage() {{
                const statusDiv = document.getElementById('status');
                statusDiv.innerHTML = '⏳ Đang xử lý ảnh...';
                statusDiv.className = 'status info';

                try {{
                    const response = await fetch('/api/v1/omr_debug/process_test_image', {{
                        method: 'POST'
                    }});

                    if (response.ok) {{
                        const result = await response.json();
                        statusDiv.innerHTML = `✅ Xử lý thành công! Student ID: ${{result.student_id}}, Test Code: ${{result.test_code}}, Answers: ${{result.total_answers}}`;
                        statusDiv.className = 'status success';

                        // Làm mới trang sau 2 giây
                        setTimeout(() => {{
                            window.location.reload();
                        }}, 2000);
                    }} else {{
                        throw new Error('Processing failed');
                    }}
                }} catch (error) {{
                    statusDiv.innerHTML = '❌ Lỗi xử lý ảnh: ' + error.message;
                    statusDiv.className = 'status error';
                }}
            }}

            async function processPipeline() {{
                const statusDiv = document.getElementById('status');
                statusDiv.innerHTML = '🤖 Đang xử lý với AI Pipeline...';
                statusDiv.className = 'status info';

                try {{
                    const response = await fetch('/api/v1/omr_debug/process_pipeline', {{
                        method: 'POST'
                    }});

                    if (response.ok) {{
                        const result = await response.json();
                        const section1 = result.results['Section I'];
                        const section2 = result.results['Section II'];
                        const section3 = result.results['Section III'];

                        statusDiv.innerHTML = `
                            ✅ AI Pipeline thành công!
                            📊 Section I: ${{section1.total_questions}} câu ABCD |
                            ✅ Section II: ${{section2.total_questions}} câu T/F |
                            🔢 Section III: ${{section3.total_questions}} câu Digits |
                            🖼️ Debug: ${{result.debug_info.total_debug_images}} images
                        `;
                        statusDiv.className = 'status success';

                        // Làm mới trang sau 3 giây
                        setTimeout(() => {{
                            window.location.reload();
                        }}, 3000);
                    }} else {{
                        throw new Error('Pipeline processing failed');
                    }}
                }} catch (error) {{
                    statusDiv.innerHTML = '❌ Lỗi AI Pipeline: ' + error.message;
                    statusDiv.className = 'status error';
                }}
            }}

            async function scanAllMarkers() {{
                const statusDiv = document.getElementById('status');
                statusDiv.innerHTML = '🔍 Đang quét tất cả marker...';
                statusDiv.className = 'status info';

                try {{
                    const response = await fetch('/api/v1/omr_debug/scan_all_markers', {{
                        method: 'POST'
                    }});

                    if (response.ok) {{
                        const result = await response.json();
                        const largeMarkers = result.markers.large_markers;
                        const smallMarkers = result.markers.small_markers;

                        statusDiv.innerHTML = `
                            ✅ Marker scanning thành công!
                            🔴 Large markers: ${{largeMarkers.count}} |
                            🟢 Small markers: ${{smallMarkers.count}} |
                            📊 Total: ${{largeMarkers.count + smallMarkers.count}} markers |
                            🖼️ Debug: ${{result.debug_info.total_debug_images}} images
                        `;
                        statusDiv.className = 'status success';

                        // Refresh images để hiển thị marker debug images
                        setTimeout(() => {{
                            refreshImages();
                        }}, 1000);

                        // Auto clear status after 5 seconds
                        setTimeout(() => {{
                            statusDiv.innerHTML = 'Sẵn sàng xử lý';
                            statusDiv.className = 'status info';
                        }}, 5000);
                    }} else {{
                        throw new Error('Marker scanning failed');
                    }}
                }} catch (error) {{
                    statusDiv.innerHTML = '❌ Lỗi marker scanning: ' + error.message;
                    statusDiv.className = 'status error';
                }}
            }}

            async function divideBlocks() {{
                const statusDiv = document.getElementById('status');
                statusDiv.innerHTML = '📦 Đang chia blocks dựa trên markers...';
                statusDiv.className = 'status info';

                try {{
                    // Call divide_blocks without file - it will use sample.jpg by default
                    const formData = new FormData();
                    // Don't append any file - server will use default sample.jpg

                    const divideResponse = await fetch('/api/v1/omr_debug/divide_blocks', {{
                        method: 'POST',
                        body: formData
                    }});

                    if (divideResponse.ok) {{
                        const result = await divideResponse.json();
                        const markerDetection = result.marker_detection;
                        const blockDivision = result.block_division;

                        statusDiv.innerHTML = `
                            ✅ Block division thành công!
                            🔴 Large markers: ${{markerDetection.large_markers_count}} |
                            🟢 Small markers: ${{markerDetection.small_markers_count}} |
                            📦 Total blocks: ${{blockDivision.total_blocks}} |
                            🏷️ Regions: ${{blockDivision.main_regions.length}} |
                            🖼️ Debug: ${{result.debug_info.total_debug_images}} images
                        `;
                        statusDiv.className = 'status success';

                        // Refresh images để hiển thị block division debug images
                        setTimeout(() => {{
                            refreshImages();
                        }}, 1000);

                        // Auto clear status after 5 seconds
                        setTimeout(() => {{
                            statusDiv.innerHTML = 'Sẵn sàng xử lý';
                            statusDiv.className = 'status info';
                        }}, 5000);
                    }} else {{
                        throw new Error('Block division failed');
                    }}
                }} catch (error) {{
                    statusDiv.innerHTML = '❌ Lỗi block division: ' + error.message;
                    statusDiv.className = 'status error';
                }}
            }}

            async function cutContourBlocks() {{
                const statusDiv = document.getElementById('status');
                statusDiv.innerHTML = '✂️ Đang cắt contour blocks...';
                statusDiv.className = 'status info';

                try {{
                    // Call cut_contour_blocks without file - it will use sample.jpg by default
                    const formData = new FormData();
                    // Don't append any file - server will use default sample.jpg

                    const cutResponse = await fetch('/api/v1/omr_debug/cut_contour_blocks', {{
                        method: 'POST',
                        body: formData
                    }});

                    if (cutResponse.ok) {{
                        const result = await cutResponse.json();
                        const contourDetection = result.contour_detection;
                        const blockCutting = result.block_cutting;

                        statusDiv.innerHTML = `
                            ✅ Contour block cutting thành công!
                            🔍 Total contours: ${{contourDetection.total_contours}} |
                            📦 Blocks cut: ${{blockCutting.total_blocks_cut}} |
                            🔲 Individual images: ${{result.output_files.total_block_images}} |
                            🖼️ Debug: ${{result.debug_info.total_debug_images}} images
                        `;
                        statusDiv.className = 'status success';

                        // Refresh images để hiển thị contour blocks debug images
                        setTimeout(() => {{
                            refreshImages();
                        }}, 1000);

                        // Auto clear status after 5 seconds
                        setTimeout(() => {{
                            statusDiv.innerHTML = 'Sẵn sàng xử lý';
                            statusDiv.className = 'status info';
                        }}, 5000);
                    }} else {{
                        throw new Error('Contour block cutting failed');
                    }}
                }} catch (error) {{
                    statusDiv.innerHTML = '❌ Lỗi contour block cutting: ' + error.message;
                    statusDiv.className = 'status error';
                }}
            }}

            function refreshImages() {{
                window.location.reload();
            }}

            async function clearImages() {{
                if (confirm('Bạn có chắc muốn xóa tất cả debug images?')) {{
                    try {{
                        const response = await fetch('/api/v1/omr_debug/clear_debug', {{
                            method: 'DELETE'
                        }});

                        if (response.ok) {{
                            window.location.reload();
                        }}
                    }} catch (error) {{
                        alert('Lỗi xóa images: ' + error.message);
                    }}
                }}
            }}

            // Đóng modal khi click bên ngoài
            window.onclick = function(event) {{
                const modal = document.getElementById('imageModal');
                if (event.target == modal) {{
                    closeModal();
                }}
            }}
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)
