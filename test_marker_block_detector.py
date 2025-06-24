#!/usr/bin/env python3
"""
Test script for Marker Block Detector
Kiểm tra pipeline 7 bước detect marker nhỏ và chia block
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from services.marker_block_detector import marker_block_detector

def load_bottom_region():
    """Load bottom region từ debug images hoặc tạo mock"""
    
    # Thử load từ debug images có sẵn
    debug_paths = [
        "data/grading/debug/06_bottom_region.jpg",
        "data/grading/debug/04_aligned.jpg",
        "data/grading/sample.jpg"
    ]
    
    for path in debug_paths:
        if os.path.exists(path):
            print(f"✅ Loading bottom region from: {path}")
            image = cv2.imread(path)
            if image is not None:
                # Nếu là ảnh full, crop bottom 70%
                if "sample.jpg" in path or "aligned.jpg" in path:
                    height = image.shape[0]
                    bottom_region = image[int(height * 0.3):, :]
                    print(f"   Cropped to bottom region: {bottom_region.shape}")
                    return bottom_region
                else:
                    print(f"   Using as bottom region: {image.shape}")
                    return image
    
    print("❌ No bottom region found, creating mock image")
    return create_mock_bottom_region()

def create_mock_bottom_region():
    """Tạo mock bottom region với marker nhỏ giả"""
    
    # Tạo ảnh trắng 800x600
    mock_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Vẽ các marker nhỏ giả cho 3 phần
    
    # PHẦN I: 4 marker ở hàng đầu
    part1_y = 100
    for i in range(4):
        x = 150 + i * 150
        cv2.rectangle(mock_image, (x-10, part1_y-10), (x+10, part1_y+10), (0, 0, 0), -1)
    
    # PHẦN II: 8 marker ở hàng giữa
    part2_y = 300
    for i in range(8):
        x = 50 + i * 90
        cv2.rectangle(mock_image, (x-8, part2_y-8), (x+8, part2_y+8), (0, 0, 0), -1)
    
    # PHẦN III: 6 marker ở hàng cuối
    part3_y = 500
    for i in range(6):
        x = 100 + i * 100
        cv2.rectangle(mock_image, (x-8, part3_y-8), (x+8, part3_y+8), (0, 0, 0), -1)
    
    # Thêm một số bubble giả để test filtering
    for i in range(10):
        x = np.random.randint(50, 750)
        y = np.random.randint(50, 550)
        cv2.circle(mock_image, (x, y), 5, (128, 128, 128), -1)
    
    print("✅ Created mock bottom region with markers")
    return mock_image

def test_marker_block_detector():
    """Test marker block detector pipeline"""
    
    print("🎯 Marker Block Detector Test")
    print("=" * 60)
    print("📋 Pipeline 7 bước:")
    print("   1. Đọc ảnh bottom_region (đã chuẩn hóa)")
    print("   2. Phát hiện tất cả contour marker nhỏ")
    print("   3. Sắp xếp marker nhỏ theo tọa độ y và x")
    print("   4. Phân cụm marker nhỏ thành nhóm theo phần")
    print("   5. Tính bounding box và crop ảnh cho mỗi nhóm")
    print("   6. Trả kết quả dict với các block ảnh")
    print("   7. Vẽ marker + bounding box để xác minh")
    print("=" * 60)
    
    # Load bottom region
    bottom_region = load_bottom_region()
    if bottom_region is None:
        print("❌ Cannot load bottom region")
        return False
    
    print(f"✅ Bottom region loaded: {bottom_region.shape}")
    
    try:
        # Chạy pipeline detect marker và chia block
        print("\n🔄 Running marker block detection pipeline...")
        result = marker_block_detector.detect_and_divide_blocks(bottom_region)
        
        if "error" in result:
            print(f"❌ Pipeline failed: {result['error']}")
            return False
        
        print("✅ Pipeline completed successfully!")
        
        # Phân tích kết quả
        print(f"\n📊 Block Detection Results:")
        
        part1_blocks = result.get("part1_blocks", [])
        part2_blocks = result.get("part2_blocks", [])
        part3_blocks = result.get("part3_blocks", [])
        
        print(f"   📝 PHẦN I (Multiple Choice): {len(part1_blocks)} blocks")
        for i, block in enumerate(part1_blocks):
            print(f"      Block {i+1}: {block.shape} (10 câu ABCD)")
        
        print(f"   ✅ PHẦN II (True/False): {len(part2_blocks)} blocks")
        for i, block in enumerate(part2_blocks):
            print(f"      Block {i+1}: {block.shape} (1 câu T/F)")
        
        print(f"   🔢 PHẦN III (Digits): {len(part3_blocks)} blocks")
        for i, block in enumerate(part3_blocks):
            print(f"      Block {i+1}: {block.shape} (1 câu số)")
        
        # Tổng kết
        total_blocks = len(part1_blocks) + len(part2_blocks) + len(part3_blocks)
        expected_blocks = 4 + 8 + 6  # 18 blocks tổng cộng
        
        print(f"\n📈 Summary:")
        print(f"   Total blocks detected: {total_blocks}")
        print(f"   Expected blocks: {expected_blocks}")
        print(f"   Success rate: {total_blocks/expected_blocks*100:.1f}%")
        
        # Kiểm tra debug images
        debug_dir = Path("data/grading/marker_debug")
        if debug_dir.exists():
            debug_files = list(debug_dir.glob("*.jpg"))
            print(f"\n🖼️ Debug Images: {len(debug_files)}")
            
            # Phân loại debug images
            categories = {
                "preprocessing": [],
                "marker_detection": [],
                "clustering": [],
                "block_division": [],
                "visualization": []
            }
            
            for img_file in debug_files:
                name = img_file.name.lower()
                if any(x in name for x in ["gray", "binary", "cleaned"]):
                    categories["preprocessing"].append(img_file.name)
                elif "marker" in name and "detected" in name:
                    categories["marker_detection"].append(img_file.name)
                elif "cluster" in name or "sorted" in name:
                    categories["clustering"].append(img_file.name)
                elif "part" in name and "block" in name:
                    categories["block_division"].append(img_file.name)
                elif "visualization" in name or "final" in name:
                    categories["visualization"].append(img_file.name)
            
            for category, files in categories.items():
                if files:
                    print(f"   📁 {category.replace('_', ' ').title()}: {len(files)} images")
                    for file in files[:3]:  # Show first 3
                        print(f"      • {file}")
                    if len(files) > 3:
                        print(f"      ... and {len(files) - 3} more")
        
        # Validation
        print(f"\n🎯 Validation:")
        
        # Check expected block counts
        validations = [
            ("PHẦN I blocks", len(part1_blocks), 4, "4 cột 10 câu ABCD"),
            ("PHẦN II blocks", len(part2_blocks), 8, "8 câu True/False"),
            ("PHẦN III blocks", len(part3_blocks), 6, "6 câu chọn số")
        ]
        
        all_valid = True
        for name, actual, expected, description in validations:
            if actual >= expected:
                print(f"   ✅ {name}: {actual} (≥{expected} expected) - {description}")
            else:
                print(f"   ⚠️ {name}: {actual} ({expected} expected) - {description}")
                all_valid = False
        
        # Check block dimensions
        print(f"\n📐 Block Dimensions Check:")
        
        if part1_blocks:
            avg_height = np.mean([block.shape[0] for block in part1_blocks])
            avg_width = np.mean([block.shape[1] for block in part1_blocks])
            print(f"   📝 PHẦN I average: {avg_height:.0f}h x {avg_width:.0f}w")
        
        if part2_blocks:
            avg_height = np.mean([block.shape[0] for block in part2_blocks])
            avg_width = np.mean([block.shape[1] for block in part2_blocks])
            print(f"   ✅ PHẦN II average: {avg_height:.0f}h x {avg_width:.0f}w")
        
        if part3_blocks:
            avg_height = np.mean([block.shape[0] for block in part3_blocks])
            avg_width = np.mean([block.shape[1] for block in part3_blocks])
            print(f"   🔢 PHẦN III average: {avg_height:.0f}h x {avg_width:.0f}w")
        
        print(f"\n🎉 Marker Block Detector test completed!")
        print(f"📁 Check debug outputs:")
        print(f"   Debug images: {debug_dir}")
        print(f"   Summary file: {debug_dir}/block_summary.json")
        
        return all_valid
        
    except Exception as e:
        print(f"❌ Error during marker block detection: {e}")
        import traceback
        traceback.print_exc()
        return False

def show_pipeline_description():
    """Hiển thị mô tả chi tiết pipeline"""
    print("\n📋 Marker Block Detection Pipeline:")
    print("=" * 50)
    print("INPUT: bottom_region.jpg (ảnh đã chuẩn hóa từ marker lớn)")
    print()
    print("PIPELINE 7 BƯỚC:")
    print("1️⃣ Preprocess:")
    print("   • Convert grayscale và threshold")
    print("   • Morphological operations để làm sạch")
    print()
    print("2️⃣ Detect Small Markers:")
    print("   • cv2.findContours để tìm contour")
    print("   • Lọc theo diện tích (100-800 pixels)")
    print("   • Lọc theo aspect ratio (0.85-1.15)")
    print("   • Loại bỏ marker ở vùng quá sâu")
    print()
    print("3️⃣ Sort Markers:")
    print("   • Sắp xếp theo y (chia hàng PHẦN I, II, III)")
    print("   • Sau đó theo x (chia cột trong mỗi phần)")
    print()
    print("4️⃣ Cluster by Sections:")
    print("   • DBSCAN clustering theo y-coordinate")
    print("   • Phân nhóm thành 3 phần chính")
    print()
    print("5️⃣ Crop Blocks:")
    print("   • PHẦN I: 4 block dọc (10 câu ABCD mỗi block)")
    print("   • PHẦN II: 8 block nhỏ (1 câu T/F mỗi block)")
    print("   • PHẦN III: 6 block số (1 câu số mỗi block)")
    print()
    print("6️⃣ Format Result:")
    print("   • Return dict với part1_blocks, part2_blocks, part3_blocks")
    print("   • Mỗi block là np.ndarray image")
    print()
    print("7️⃣ Visualize:")
    print("   • Vẽ marker + bounding box để xác minh")
    print("   • Save debug images cho từng bước")
    print()
    print("OUTPUT: Dict[str, List[np.ndarray]]")
    print("=" * 50)

if __name__ == "__main__":
    print("🎯 Marker Block Detector Test Suite")
    print("=" * 60)
    
    # Show pipeline description
    show_pipeline_description()
    
    # Run test
    success = test_marker_block_detector()
    
    if success:
        print("\n✅ Marker Block Detector test completed successfully!")
        print("💡 The marker-based block division pipeline is ready!")
        sys.exit(0)
    else:
        print("\n❌ Marker Block Detector test failed!")
        print("💡 Check the debug images for marker detection issues.")
        sys.exit(1)
