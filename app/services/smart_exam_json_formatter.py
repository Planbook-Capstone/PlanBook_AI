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
            cognitive_level: Mức độ nhận thức từ smart exam ("Biết", "Hiểu", "Vận_dụng")

        Returns:
            str: DifficultyLevel value ("KNOWLEDGE", "COMPREHENSION", "APPLICATION")
        """
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
            
            # Phân loại câu hỏi theo phần
            part_1_questions = [q for q in questions if q.get("part") == 1]
            part_2_questions = [q for q in questions if q.get("part") == 2]
            part_3_questions = [q for q in questions if q.get("part") == 3]

            parts = []

            # PHẦN I: Trắc nghiệm nhiều phương án
            if part_1_questions:
                part_1 = self._format_part_1(part_1_questions)
                parts.append(part_1)

            # PHẦN II: Câu hỏi Đúng/Sai
            if part_2_questions:
                part_2 = self._format_part_2(part_2_questions)
                parts.append(part_2)

            # PHẦN III: Câu hỏi tự luận
            if part_3_questions:
                part_3 = self._format_part_3(part_3_questions)
                parts.append(part_3)

            return {
                "parts": parts
            }

        except Exception as e:
            logger.error(f"Error formatting exam to JSON response: {e}")
            return {
                "parts": []
            }

    def _format_part_1(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format PHẦN I: Trắc nghiệm nhiều phương án"""
        formatted_questions = []
        
        for i, question in enumerate(questions, 1):
            answer_data = question.get("answer", {})
            
            # Tạo options từ A, B, C, D với chemistry format
            options = {}
            for option in ["A", "B", "C", "D"]:
                if option in answer_data:
                    options[option] = self._docx_service._normalize_chemistry_format(str(answer_data[option]))

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
                    question.get("cognitive_level", "Biết")
                )
            }

            # Thêm illustrationImage nếu có
            if question.get("illustrationImage"):
                formatted_question["illustrationImage"] = question["illustrationImage"]

            formatted_questions.append(formatted_question)

        return {
            "part": "PHẦN I",
            "title": "Câu trắc nghiệm nhiều phương án lựa chọn",
            "questions": formatted_questions
        }

    def _format_part_2(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format PHẦN II: Câu hỏi Đúng/Sai"""
        formatted_questions = []
        
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

        return {
            "part": "PHẦN II",
            "title": "Câu hỏi Đúng/Sai",
            "questions": formatted_questions
        }

    def _format_part_3(self, questions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format PHẦN III: Câu hỏi tự luận"""
        formatted_questions = []
        
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
                    question.get("cognitive_level", "Biết")
                )
            }

            formatted_questions.append(formatted_question)

        return {
            "part": "PHẦN III",
            "title": "Câu hỏi tự luận",
            "questions": formatted_questions
        }


def get_smart_exam_json_formatter() -> SmartExamJsonFormatter:
    """Get singleton instance của SmartExamJsonFormatter"""
    return SmartExamJsonFormatter()
