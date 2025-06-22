#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys

sys.path.append(".")

from app.services.omr_debug_processor import OMRDebugProcessor


def main():
    # Test image path
    image_path = "data/grading/1.jpeg"

    if not os.path.exists(image_path):
        print(f"Khong tim thay file anh: {image_path}")
        return

    print(f"Bat dau xu ly anh: {image_path}")

    # Tao processor
    processor = OMRDebugProcessor()

    # Xu ly anh
    result = processor.process_answer_sheet(image_path)

    # In ket qua
    print("\n" + "=" * 50)
    print("KET QUA XU LY")
    print("=" * 50)

    if result["success"]:
        print(f"Xu ly thanh cong!")
        print(f"Student ID: {result['student_id']}")
        print(f"Test Code: {result['test_code']}")
        print(f"So cau tra loi: {len(result['answers'])}")

        # In mot so cau tra loi mau
        print("\nCau tra loi (10 cau dau):")
        answers = result["answers"]
        for i, (q_num, answer) in enumerate(list(answers.items())[:10]):
            print(f"   Cau {q_num}: {answer}")

        # Hien thi debug images
        debug_dir = "data/grading/debug"
        print(f"\nDebug images saved to: {debug_dir}")

        # Liet ke cac file debug
        if os.path.exists(debug_dir):
            debug_files = [f for f in os.listdir(debug_dir) if f.endswith(".png")]
            for file in sorted(debug_files):
                print(f"   {file}")
    else:
        print(f"Loi xu ly: {result.get('error', 'Unknown error')}")

    print(f"\nHoan thanh xu ly anh: {image_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
