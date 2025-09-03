"""
Service chuyển đổi format JSON cho smart exam từ format internal sang format API response
"""

import uuid
import logging
from typing import Dict, Any, List

from app.constants.difficulty_levels import DifficultyLevel
from app.services.smart_exam_docx_service import SmartExamDocxService

logger = logging.getLogger(__name__)


class SmartExamJsonFormatter:
    """Service chuyển đổi format JSON cho smart exam"""

    def __init__(self):
        # Tạo instance của SmartExamDocxService để sử dụng hàm _normalize_chemistry_format
        self._docx_service = SmartExamDocxService()

    def _map_cognitive_level_to_difficulty(self, cognitive_level: str) -> str:
        """
        Map cognitive_level từ smart exam sang DifficultyLevel enum

        Args:
            cognitive_level: Mức độ nhận thức từ smart exam (DifficultyLevel values)

        Returns:
            str: DifficultyLevel value ("KNOWLEDGE", "COMPREHENSION", "APPLICATION")
        """
        # Nếu đã là DifficultyLevel value thì trả về luôn
        if cognitive_level in [DifficultyLevel.KNOWLEDGE.value, DifficultyLevel.COMPREHENSION.value, DifficultyLevel.APPLICATION.value]:
            return cognitive_level

        # Mapping cho backward compatibility với tiếng Việt cũ
        mapping = {
            "Biết": DifficultyLevel.KNOWLEDGE.value,
            "Hiểu": DifficultyLevel.COMPREHENSION.value,
            "Vận_dụng": DifficultyLevel.APPLICATION.value
        }
        return mapping.get(cognitive_level, DifficultyLevel.KNOWLEDGE.value)

    def format_exam_to_json_response(self, exam_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Chuyển đổi từ format internal sang format JSON response theo yêu cầu

        Args:
            exam_data: Dữ liệu đề thi từ smart_exam_generation_service

        Returns:
            Dict: JSON format theo cấu trúc parts mới
        """
        try:
            questions = exam_data.get("questions", [])
            logger.info(f"🔍 Total questions to format: {len(questions)}")

            # Phân loại câu hỏi theo phần
            part_1_questions = [q for q in questions if q.get("part") == 1]
            part_2_questions = [q for q in questions if q.get("part") == 2]
            part_3_questions = [q for q in questions if q.get("part") == 3]

            logger.info(f"📊 Question distribution: Part 1: {len(part_1_questions)}, Part 2: {len(part_2_questions)}, Part 3: {len(part_3_questions)}")

            # Debug: Log first question structure
            if questions:
                logger.info(f"🔍 Sample question structure: {questions[0]}")

            parts = []

            # PHẦN I: Trắc nghiệm nhiều phương án (luôn có, kể cả khi rỗng)
            part_1 = self._format_part_1(part_1_questions)
            parts.append(part_1)

            # PHẦN II: Câu hỏi Đúng/Sai (luôn có, kể cả khi rỗng)
            part_2 = self._format_part_2(part_2_questions)
            parts.append(part_2)

            # PHẦN III: Câu hỏi tự luận (luôn có, kể cả khi rỗng)
            part_3 = self._format_part_3(part_3_questions)
            parts.append(part_3)

            return {
                "parts": parts
            }

        except Exception as e:
            logger.error(f"Error formatting exam to JSON response: {e}")
            # Trả về cấu trúc đầy đủ 3 phần rỗng khi có lỗi
            return {
                "parts": [
                    {
                        "part": "PHẦN I",
                        "title": "Câu trắc nghiệm nhiều phương án lựa chọn",
                        "questions": []
                    },
                    {
                        "part": "PHẦN II",
                        "title": "Câu hỏi Đúng/Sai",
                        "questions": []
                    },
                    {
                        "part": "PHẦN III",
                        "title": "Câu hỏi tự luận",
                        "questions": []
                    }
                ]
            }

    def _format_part_1(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format PHẦN I: Trắc nghiệm nhiều phương án"""
        formatted_questions = []

        # Xử lý từng câu hỏi nếu có
        for i, question in enumerate(questions, 1):
            answer_data = question.get("answer", {})
            logger.info(f"🔍 Formatting Part 1 Question {i}: answer_data = {answer_data}")

            # Tạo options từ A, B, C, D với chemistry format
            options = {}
            for option in ["A", "B", "C", "D"]:
                if option in answer_data:
                    options[option] = self._docx_service._normalize_chemistry_format(str(answer_data[option]))

            logger.info(f"🔍 Part 1 Question {i}: options = {options}")

            # Lấy đáp án đúng
            correct_answer = answer_data.get("correct_answer", "A")

            formatted_question = {
                "id": str(uuid.uuid4()),
                "questionNumber": i,
                "question": self._docx_service._normalize_chemistry_format(question.get("question", "")),
                "options": options,
                "answer": correct_answer,
                "explanation": self._docx_service._normalize_chemistry_format(question.get("explanation", "")),
                "difficultyLevel": self._map_cognitive_level_to_difficulty(
                    question.get("cognitive_level", DifficultyLevel.KNOWLEDGE.value)
                )
            }

            # Thêm illustrationImage nếu có
            if question.get("illustrationImage"):
                formatted_question["illustrationImage"] = question["illustrationImage"]

            # Chỉ thêm câu hỏi nếu có ít nhất một option
            if options:
                formatted_questions.append(formatted_question)
                logger.info(f"✅ Part 1 Question {i}: Added to formatted_questions")
            else:
                logger.warning(f"❌ Part 1 Question {i}: Skipped - no options found")

        logger.info(f"📊 Part 1 formatting result: {len(formatted_questions)} questions formatted")

        # Luôn trả về cấu trúc phần, kể cả khi không có câu hỏi
        return {
            "part": "PHẦN I",
            "title": "Câu trắc nghiệm nhiều phương án lựa chọn",
            "questions": formatted_questions  # Sẽ là mảng rỗng nếu không có câu hỏi
        }

    def _format_part_2(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format PHẦN II: Câu hỏi Đúng/Sai"""
        formatted_questions = []

        # Xử lý từng câu hỏi nếu có
        for i, question in enumerate(questions, 1):
            answer_data = question.get("answer", {})

            # Tạo statements từ a, b, c, d
            statements = {}
            for option in ["a", "b", "c", "d"]:
                if option in answer_data:
                    option_data = answer_data[option]

                    if isinstance(option_data, dict):
                        # Format mới với content và evaluation
                        text = self._docx_service._normalize_chemistry_format(option_data.get("content", ""))
                        evaluation = option_data.get("evaluation", "Đúng")
                        answer = evaluation.lower() == "đúng"
                    else:
                        # Format cũ - fallback
                        text = self._docx_service._normalize_chemistry_format(str(option_data))
                        answer = True  # Default

                    statements[option] = {
                        "text": text,
                        "answer": answer
                    }

            formatted_question = {
                "id": str(uuid.uuid4()),
                "questionNumber": i,
                "question": self._docx_service._normalize_chemistry_format(question.get("question", "")),
                "statements": statements,
                "explanation": self._docx_service._normalize_chemistry_format(question.get("explanation", "")),
                "difficultyLevel": self._map_cognitive_level_to_difficulty(
                    question.get("cognitive_level", "Chưa thể xác định")
                )
            }

            formatted_questions.append(formatted_question)

        # Luôn trả về cấu trúc phần, kể cả khi không có câu hỏi
        return {
            "part": "PHẦN II",
            "title": "Câu hỏi Đúng/Sai",
            "questions": formatted_questions  # Sẽ là mảng rỗng nếu không có câu hỏi
        }

    def _format_part_3(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format PHẦN III: Câu hỏi tự luận"""
        formatted_questions = []

        # Xử lý từng câu hỏi nếu có
        for i, question in enumerate(questions, 1):
            answer_data = question.get("answer", {})

            # Lấy đáp án từ các field có thể có
            answer = ""
            if isinstance(answer_data, dict):
                answer = str(answer_data.get("answer", answer_data.get("dap_an", "")))
            else:
                answer = str(answer_data)

            formatted_question = {
                "id": str(uuid.uuid4()),
                "questionNumber": i,
                "question": self._docx_service._normalize_chemistry_format(question.get("question", "")),
                "answer": answer,  # Đáp án số không cần format chemistry
                "explanation": self._docx_service._normalize_chemistry_format(question.get("explanation", "")),
                "difficultyLevel": self._map_cognitive_level_to_difficulty(
                    question.get("cognitive_level", DifficultyLevel.KNOWLEDGE.value)
                )
            }

            formatted_questions.append(formatted_question)

        # Luôn trả về cấu trúc phần, kể cả khi không có câu hỏi
        return {
            "part": "PHẦN III",
            "title": "Câu hỏi tự luận",
            "questions": formatted_questions  # Sẽ là mảng rỗng nếu không có câu hỏi
        }


def get_smart_exam_json_formatter() -> SmartExamJsonFormatter:
    """Get singleton instance của SmartExamJsonFormatter"""
    return SmartExamJsonFormatter()
