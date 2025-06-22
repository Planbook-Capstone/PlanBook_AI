import pytest
import os
import cv2
import numpy as np
from pathlib import Path
from app.services.omr_debug_processor import OMRDebugProcessor


@pytest.fixture
def omr_processor():
    """Tạo OMR processor instance"""
    return OMRDebugProcessor()


@pytest.fixture
def sample_image_path():
    """Đường dẫn đến ảnh test"""
    test_image_path = Path("data/grading/test_images/sample_answer_sheet.jpg")
    if test_image_path.exists():
        return str(test_image_path)

    # Nếu không có ảnh test, tạo ảnh giả
    test_dir = Path("data/grading/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)

    # Tạo ảnh trắng 1000x1400 để test
    fake_image = np.ones((1400, 1000, 3), dtype=np.uint8) * 255
    fake_path = test_dir / "fake_answer_sheet.jpg"
    cv2.imwrite(str(fake_path), fake_image)
    return str(fake_path)


def test_omr_debug_processing(omr_processor, sample_image_path):
    """Test xử lý OMR với debug output"""
    print(f"\n🔍 Testing OMR processing with image: {sample_image_path}")

    # Kiểm tra file tồn tại
    assert os.path.exists(
        sample_image_path
    ), f"Image file not found: {sample_image_path}"

    # Đọc ảnh
    image = cv2.imread(sample_image_path)
    assert image is not None, "Failed to load image"

    print(f"📐 Original image size: {image.shape}")

    # Xử lý OMR
    result = omr_processor.process_image(image)

    # Kiểm tra kết quả
    assert result is not None, "OMR processing failed"
    assert "student_id" in result, "Student ID not found in result"
    assert "test_code" in result, "Test code not found in result"
    assert "answers" in result, "Answers not found in result"

    print(f"✅ Student ID: {result['student_id']}")
    print(f"✅ Test Code: {result['test_code']}")
    print(f"✅ Total answers: {len(result['answers'])}")

    # Kiểm tra debug images được tạo
    debug_dir = Path("data/grading/debug")
    assert debug_dir.exists(), "Debug directory not created"

    debug_files = list(debug_dir.glob("*.jpg"))
    print(f"📁 Debug files created: {len(debug_files)}")
    for file in debug_files:
        print(f"  - {file.name}")

    assert len(debug_files) > 0, "No debug images created"


def test_extract_regions_coordinates(omr_processor):
    """Test tọa độ vùng ROI với kích thước ảnh khác nhau"""
    print(f"\n📏 Testing ROI coordinates with different image sizes")

    # Test với kích thước ảnh khác nhau
    test_sizes = [(1000, 1400), (800, 1200), (1200, 1600)]

    for width, height in test_sizes:
        print(f"\n🔍 Testing with size: {width}x{height}")

        # Tạo ảnh test
        test_image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Trích xuất regions
        regions = omr_processor.extract_regions(test_image)

        # Kiểm tra tất cả regions được tạo
        expected_regions = [
            "student_id",
            "test_code",
            "answers_01_15",
            "answers_16_30",
            "answers_31_45",
            "answers_46_60",
            "answers_full",
        ]

        for region_name in expected_regions:
            assert region_name in regions, f"Region {region_name} not found"
            region = regions[region_name]
            assert region.size > 0, f"Region {region_name} is empty"
            print(f"  ✅ {region_name}: {region.shape}")


if __name__ == "__main__":
    # Chạy test trực tiếp
    processor = OMRDebugProcessor()

    # Tạo ảnh test nếu cần
    test_dir = Path("data/grading/test_images")
    test_dir.mkdir(parents=True, exist_ok=True)

    fake_image = np.ones((1400, 1000, 3), dtype=np.uint8) * 255
    fake_path = test_dir / "test_sheet.jpg"
    cv2.imwrite(str(fake_path), fake_image)

    print("🚀 Running OMR debug test...")
    test_extract_regions_coordinates(processor)
    test_omr_debug_processing(processor, str(fake_path))
    print("✅ All tests passed!")
