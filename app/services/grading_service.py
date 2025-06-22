from fastapi import UploadFile
from typing import List, Dict, Any
import pandas as pd
import io
import logging

from app.services.omr_debug_processor import OMRDebugProcessor

logger = logging.getLogger(__name__)


async def batch_grade_all(
    image_files: List[UploadFile], excel_file: UploadFile
) -> List[Dict[str, Any]]:
    """
    Chấm điểm tự động cho nhiều ảnh phiếu trả lời

    Args:
        image_files: Danh sách file ảnh phiếu trả lời
        excel_file: File Excel chứa đáp án

    Returns:
        List kết quả chấm điểm
    """
    try:
        # Đọc file Excel đáp án
        excel_content = await excel_file.read()
        answer_keys_df = pd.read_excel(io.BytesIO(excel_content))

        # Tạo processor
        processor = OMRDebugProcessor()

        results = []

        for image_file in image_files:
            try:
                # Lưu file tạm để xử lý
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".jpg"
                ) as temp_file:
                    image_content = await image_file.read()
                    temp_file.write(image_content)
                    temp_path = temp_file.name

                # Xử lý ảnh
                result = processor.process_answer_sheet(temp_path)

                # Xóa file tạm
                os.unlink(temp_path)

                if not result["success"]:
                    raise Exception(result.get("error", "Processing failed"))

                student_info = {
                    "student_id": result["student_id"],
                    "test_code": result["test_code"],
                }
                answers = result["answers"]

                # Tìm đáp án đúng từ Excel
                test_code = student_info.get("test_code", "")
                correct_answers = get_correct_answers(answer_keys_df, test_code)

                # Chấm điểm
                score, details = calculate_score(answers, correct_answers)

                result = {
                    "filename": image_file.filename,
                    "student_info": student_info,
                    "answers": answers,
                    "correct_answers": correct_answers,
                    "score": score,
                    "details": details,
                    "success": True,
                }

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing {image_file.filename}: {e}")
                results.append(
                    {"filename": image_file.filename, "error": str(e), "success": False}
                )

        return results

    except Exception as e:
        logger.error(f"Error in batch grading: {e}")
        raise Exception(f"Batch grading failed: {str(e)}")


def get_correct_answers(answer_keys_df: pd.DataFrame, test_code: str) -> Dict[int, str]:
    """
    Lấy đáp án đúng từ Excel theo mã đề

    Args:
        answer_keys_df: DataFrame chứa đáp án
        test_code: Mã đề thi

    Returns:
        Dict mapping câu hỏi -> đáp án đúng
    """
    try:
        # Tìm dòng có mã đề tương ứng
        if "Mã Đề" in answer_keys_df.columns:
            test_row = answer_keys_df[answer_keys_df["Mã Đề"] == test_code]
        else:
            # Fallback: sử dụng dòng đầu tiên
            test_row = answer_keys_df.iloc[0:1]

        if test_row.empty:
            logger.warning(f"No answer key found for test code: {test_code}")
            return {}

        correct_answers = {}

        # Duyệt qua các cột để tìm đáp án
        for col in answer_keys_df.columns:
            if col.startswith("Q") or col.startswith("Câu"):
                try:
                    # Trích xuất số câu hỏi
                    question_num = int("".join(filter(str.isdigit, col)))
                    answer = test_row[col].iloc[0]

                    if pd.notna(answer):
                        correct_answers[question_num] = str(answer).strip().upper()

                except (ValueError, IndexError):
                    continue

        return correct_answers

    except Exception as e:
        logger.error(f"Error getting correct answers: {e}")
        return {}


def calculate_score(
    student_answers: Dict[int, str], correct_answers: Dict[int, str]
) -> tuple:
    """
    Tính điểm dựa trên câu trả lời của học sinh và đáp án đúng

    Args:
        student_answers: Câu trả lời của học sinh
        correct_answers: Đáp án đúng

    Returns:
        Tuple (điểm, chi tiết)
    """
    try:
        total_questions = len(correct_answers)
        correct_count = 0
        details = []

        for question_num, correct_answer in correct_answers.items():
            student_answer = student_answers.get(question_num, "")

            is_correct = student_answer.upper() == correct_answer.upper()
            if is_correct:
                correct_count += 1

            details.append(
                {
                    "question": question_num,
                    "student_answer": student_answer,
                    "correct_answer": correct_answer,
                    "is_correct": is_correct,
                }
            )

        # Tính điểm theo thang 10
        score = (correct_count / total_questions * 10) if total_questions > 0 else 0

        return round(score, 2), {
            "total_questions": total_questions,
            "correct_count": correct_count,
            "wrong_count": total_questions - correct_count,
            "percentage": round(correct_count / total_questions * 100, 2)
            if total_questions > 0
            else 0,
            "question_details": details,
        }

    except Exception as e:
        logger.error(f"Error calculating score: {e}")
        return 0, {"error": str(e)}
