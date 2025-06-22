#!/usr/bin/env python3
"""
Script test OMR Debug Processor
Chạy xử lý phiếu trắc nghiệm với debug chi tiết
"""

import sys
import os
import asyncio
from pathlib import Path

# Thêm thư mục gốc vào Python path
sys.path.append(str(Path(__file__).parent))

from app.services.omr_debug_processor import OMRDebugProcessor
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def main():
    """Hàm chính test OMR processor"""

    # Đường dẫn ảnh test
    image_path = "data/grading/test_images/1.jpeg"

    # Kiểm tra file tồn tại
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy file ảnh: {image_path}")
        return

    print(f"🔍 Bắt đầu xử lý ảnh: {image_path}")

    # Tạo processor
    processor = OMRDebugProcessor()

    # Xử lý ảnh
    result = processor.process_answer_sheet(image_path)

    # In kết quả
    print("\n" + "=" * 50)
    print("📊 KẾT QUẢ XỬ LÝ")
    print("=" * 50)

    if result["success"]:
        print(f"✅ Xử lý thành công!")
        print(f"📝 Student ID: {result['student_id']}")
        print(f"📋 Test Code: {result['test_code']}")
        print(f"📚 Số câu trả lời: {len(result['answers'])}")

        # In một số câu trả lời mẫu
        print("\n📖 Câu trả lời (10 câu đầu):")
        answers = result["answers"]
        for i, (q_num, answer) in enumerate(list(answers.items())[:10]):
            print(f"   Câu {q_num}: {answer}")

        if len(answers) > 10:
            print(f"   ... và {len(answers) - 10} câu khác")

        print(f"\n🖼️  Debug images đã lưu tại: {result['debug_dir']}")

        # Liệt kê các file debug
        debug_dir = Path(result["debug_dir"])
        if debug_dir.exists():
            debug_files = list(debug_dir.glob("*.jpg"))
            debug_files.sort()

            print(f"\n📁 Danh sách {len(debug_files)} debug images:")
            for file in debug_files:
                print(f"   - {file.name}")

    else:
        print(f"❌ Xử lý thất bại: {result['error']}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
