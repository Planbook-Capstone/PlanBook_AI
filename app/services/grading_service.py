# services/grading_service.py

from fastapi import UploadFile
from typing import List

from app.services.excel_processor import ExcelProcessor
from app.services.omr_processor import OMRProcessor
import logging
logger = logging.getLogger(__name__)
from typing import List, Dict, Any
async def batch_grade_all(image_files: List[UploadFile], excel_file: UploadFile) -> List[Dict[str, Any]]:
    """
    Xử lý chấm điểm hàng loạt cho tất cả các bài làm
    """
    try:
        # Khởi tạo processors
        omr_processor = OMRProcessor()
        excel_processor = ExcelProcessor()

        # Đọc đáp án từ file Excel
        answer_keys = await excel_processor.load_answer_keys(excel_file)
        logger.info(f"Loaded answer keys for {len(answer_keys)} test codes")

        results = []

        # Xử lý từng ảnh
        for i, image_file in enumerate(image_files):
            try:
                logger.info(f"Processing image {i + 1}/{len(image_files)}: {image_file.filename}")

                # Đọc nội dung ảnh
                image_content = await image_file.read()

                # Xử lý OMR để trích xuất thông tin
                student_info, student_answers = await omr_processor.process_image(image_content)

                # Tìm đáp án tương ứng với mã đề
                test_code = student_info.get('test_code', '')
                if test_code not in answer_keys:
                    results.append({
                        "filename": image_file.filename,
                        "student_info": student_info,
                        "error": f"Test code {test_code} not found in answer keys",
                        "status": "failed"
                    })
                    continue

                correct_answers = answer_keys[test_code]

                # Chấm điểm
                score, detailed_results = calculate_score(student_answers, correct_answers)

                results.append({
                    "filename": image_file.filename,
                    "student_info": student_info,
                    "student_answers": student_answers,
                    "correct_answers": correct_answers,
                    "score": score,
                    "total_questions": len(correct_answers),
                    "correct_count": detailed_results['correct_count'],
                    "incorrect_count": detailed_results['incorrect_count'],
                    "blank_count": detailed_results['blank_count'],
                    "detailed_results": detailed_results['question_results'],
                    "status": "success"
                })

            except Exception as e:
                logger.error(f"Error processing {image_file.filename}: {str(e)}")
                results.append({
                    "filename": image_file.filename,
                    "error": str(e),
                    "status": "failed"
                })

        return results

    except Exception as e:
        logger.error(f"Batch grading failed: {str(e)}")
        raise e


def calculate_score(student_answers: Dict[int, str], correct_answers: Dict[int, str]) -> tuple:
    """
    Tính điểm cho bài làm
    """
    correct_count = 0
    incorrect_count = 0
    blank_count = 0
    question_results = {}

    total_questions = max(len(student_answers), len(correct_answers))

    for question_num in range(1, total_questions + 1):
        student_answer = student_answers.get(question_num, '')
        correct_answer = correct_answers.get(question_num, '')

        if not student_answer:  # Câu bỏ trống
            blank_count += 1
            result = "blank"
        elif student_answer == correct_answer:  # Câu đúng
            correct_count += 1
            result = "correct"
        else:  # Câu sai
            incorrect_count += 1
            result = "incorrect"

        question_results[question_num] = {
            "student_answer": student_answer,
            "correct_answer": correct_answer,
            "result": result
        }

    # Tính điểm (thang điểm 10)
    score = (correct_count / total_questions) * 10 if total_questions > 0 else 0

    detailed_results = {
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "blank_count": blank_count,
        "question_results": question_results
    }

    return round(score, 2), detailed_results