#!/usr/bin/env python3
"""
Test script để kiểm tra tọa độ ROI với ảnh thật
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import os

# Thêm đường dẫn để import module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.omr_debug_processor import OMRDebugProcessor


def test_with_real_image():
    """Test với ảnh thật nếu có"""
    
    # Tìm ảnh test trong thư mục data/grading
    test_dirs = [
        "data/grading",
        "data/grading/test_images", 
        "data/grading/samples"
    ]
    
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    test_image = None
    
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            for ext in image_extensions:
                images = list(test_path.glob(f"*{ext}"))
                if images:
                    test_image = images[0]
                    break
            if test_image:
                break
    
    if not test_image:
        print("❌ Không tìm thấy ảnh test, tạo ảnh giả...")
        # Tạo ảnh test với kích thước chuẩn
        test_dir = Path("data/grading/test_images")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo ảnh trắng với kích thước A4 (tỷ lệ 1:1.4)
        width, height = 1200, 1680
        test_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Vẽ một số hình vuông đen ở 4 góc để mô phỏng markers
        marker_size = 50
        # Top-left
        test_img[20:20+marker_size, 20:20+marker_size] = 0
        # Top-right  
        test_img[20:20+marker_size, width-20-marker_size:width-20] = 0
        # Bottom-left
        test_img[height-20-marker_size:height-20, 20:20+marker_size] = 0
        # Bottom-right
        test_img[height-20-marker_size:height-20, width-20-marker_size:width-20] = 0
        
        test_image = test_dir / "synthetic_answer_sheet.jpg"
        cv2.imwrite(str(test_image), test_img)
        print(f"✅ Đã tạo ảnh test: {test_image}")
    
    print(f"🔍 Testing với ảnh: {test_image}")
    
    # Đọc ảnh
    image = cv2.imread(str(test_image))
    if image is None:
        print(f"❌ Không thể đọc ảnh: {test_image}")
        return
    
    print(f"📐 Kích thước ảnh gốc: {image.shape}")
    
    # Khởi tạo processor
    processor = OMRDebugProcessor()
    
    # Test trích xuất regions
    print("\n🔍 Testing extract_regions...")
    regions = processor.extract_regions(image)
    
    print(f"\n📊 Kết quả trích xuất regions:")
    for region_name, region in regions.items():
        print(f"  ✅ {region_name}: {region.shape}")
    
    # Test xử lý toàn bộ
    print(f"\n🔍 Testing full OMR processing...")
    result = processor.process_image(image)
    
    print(f"\n📋 Kết quả OMR:")
    print(f"  Student ID: {result['student_id']}")
    print(f"  Test Code: {result['test_code']}")
    print(f"  Total answers: {result['total_questions']}")
    print(f"  Status: {result['processing_status']}")
    
    # Hiển thị một số câu trả lời
    if result['answers']:
        print(f"  Sample answers:")
        for i, (q_num, answer) in enumerate(list(result['answers'].items())[:10]):
            print(f"    Q{q_num}: {answer}")
        if len(result['answers']) > 10:
            print(f"    ... và {len(result['answers']) - 10} câu khác")
    
    print(f"\n📁 Debug images saved to: data/grading/debug/")
    debug_dir = Path("data/grading/debug")
    if debug_dir.exists():
        debug_files = list(debug_dir.glob("*.jpg"))
        print(f"  Tổng cộng {len(debug_files)} files debug")
        
        # Hiển thị file overview
        overview_file = debug_dir / "04b_roi_overview.jpg"
        if overview_file.exists():
            print(f"  📸 Xem file overview: {overview_file}")


def test_coordinate_scaling():
    """Test tỷ lệ tọa độ với các kích thước ảnh khác nhau"""
    print(f"\n🔍 Testing coordinate scaling...")
    
    processor = OMRDebugProcessor()
    
    # Test với các kích thước khác nhau
    test_sizes = [
        (800, 1120),   # Nhỏ
        (1000, 1400),  # Trung bình  
        (1200, 1680),  # Lớn
        (1500, 2100),  # Rất lớn
    ]
    
    for width, height in test_sizes:
        print(f"\n📏 Testing size: {width}x{height}")
        
        # Tạo ảnh test
        test_image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Trích xuất regions
        regions = processor.extract_regions(test_image)
        
        # Tính tỷ lệ so với kích thước tham chiếu
        max_weight, max_height = 1726, 2470
        scale_x = width / max_weight
        scale_y = height / max_height
        
        print(f"  Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
        
        # Kiểm tra một số regions chính
        key_regions = ["student_id", "test_code", "answers_01_15", "answers_16_30"]
        for region_name in key_regions:
            if region_name in regions:
                region = regions[region_name]
                print(f"    {region_name}: {region.shape[1]}x{region.shape[0]}")


if __name__ == "__main__":
    print("🚀 Testing OMR coordinates with percentage-based layout...")
    
    # Test 1: Với ảnh thật hoặc ảnh giả
    test_with_real_image()
    
    # Test 2: Kiểm tra tỷ lệ tọa độ
    test_coordinate_scaling()
    
    print("\n✅ All coordinate tests completed!")
    print("📁 Check debug images in data/grading/debug/ folder")
