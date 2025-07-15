"""
Service cho việc tạo đề thi thông minh theo chuẩn THPT 2025
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.models.smart_exam_models import SmartExamRequest, ExamStatistics
from app.services.openrouter_service import get_openrouter_service
from app.core.config import settings

logger = logging.getLogger(__name__)


class SmartExamGenerationService:
    """Service tạo đề thi thông minh theo chuẩn THPT 2025"""

    def __init__(self):
        self.llm_service = get_openrouter_service()
        # Đảm bảo service được khởi tạo đầy đủ
        self.llm_service._ensure_service_initialized()
        logger.info("🔄 SmartExamGenerationService: First-time initialization triggered")

    async def generate_smart_exam(
        self, exam_request: SmartExamRequest, lesson_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Tạo đề thi thông minh theo chuẩn THPT 2025

        Args:
            exam_request: Request chứa ma trận đề thi
            lesson_content: Nội dung bài học từ Qdrant

        Returns:
            Dict chứa đề thi đã được tạo
        """
        try:
            start_time = datetime.now()
            print(f"Starting smart exam generation for subject: {exam_request.subject}")

            # Ensure LLM service is initialized
            self.llm_service._ensure_service_initialized()

            if not self.llm_service.is_available():
                return {
                    "success": False,
                    "error": "LLM service not available. Please check OpenRouter API configuration."
                }

            # Tạo câu hỏi cho từng phần theo chuẩn THPT 2025
            all_questions = []
            part_statistics = {"part_1": 0, "part_2": 0, "part_3": 0}

            for lesson_matrix in exam_request.matrix:
                lesson_questions = await self._generate_questions_for_lesson(
                    lesson_matrix, lesson_content, exam_request.subject
                )
                
                # Phân loại câu hỏi theo phần
                for question in lesson_questions:
                    part_num = question.get("part", 1)
                    part_statistics[f"part_{part_num}"] += 1
                
                all_questions.extend(lesson_questions)

            # Sắp xếp câu hỏi theo phần và đánh số lại
            sorted_questions = self._sort_and_renumber_questions(all_questions)

            # Tính toán thống kê
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            
            statistics = self._calculate_statistics(
                sorted_questions, exam_request, generation_time
            )

            return {
                "success": True,
                "exam_id": f"smart_exam_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "questions": sorted_questions,
                "statistics": statistics,
                "total_generated": len(sorted_questions),
                "exam_request": exam_request.model_dump()
            }

        except Exception as e:
            logger.error(f"Error generating smart exam: {e}")
            return {
                "success": False,
                "error": f"Failed to generate smart exam: {str(e)}"
            }

    async def _generate_questions_for_lesson(
        self, lesson_matrix, lesson_content: Dict[str, Any], subject: str
    ) -> List[Dict[str, Any]]:
        """Tạo câu hỏi cho một bài học theo ma trận"""
        try:
            lesson_id = lesson_matrix.lessonId
            print(f"Generating questions for lesson: {lesson_id}")

            # Debug logging cho cấu trúc dữ liệu
            print(f"DEBUG: lesson_content keys: {list(lesson_content.keys()) if isinstance(lesson_content, dict) else 'Not a dict'}")

            # Lấy nội dung bài học
            lesson_data = lesson_content.get(lesson_id, {})
            print(f"DEBUG: lesson_data type: {type(lesson_data)}")
            print(f"DEBUG: lesson_data keys: {list(lesson_data.keys()) if isinstance(lesson_data, dict) else 'Not a dict'}")

            if not lesson_data:
                logger.warning(f"No content found for lesson: {lesson_id}")
                return []

            # Kiểm tra nếu lesson_data có cấu trúc {"success": True, "content": {...}}
            if isinstance(lesson_data, dict) and "content" in lesson_data:
                actual_content = lesson_data.get("content", {})
                print(f"DEBUG: Found nested content structure, extracting actual content")
                print(f"DEBUG: actual_content keys: {list(actual_content.keys()) if isinstance(actual_content, dict) else 'Not a dict'}")
            else:
                actual_content = lesson_data
                print(f"DEBUG: Using lesson_data directly as content")

            if not actual_content:
                logger.warning(f"No actual content found for lesson: {lesson_id}")
                return []

            all_lesson_questions = []

            # Tạo câu hỏi cho từng phần
            for part in lesson_matrix.parts:
                part_questions = await self._generate_questions_for_part(
                    part, actual_content, subject, lesson_id
                )
                all_lesson_questions.extend(part_questions)

            return all_lesson_questions

        except Exception as e:
            logger.error(f"Error generating questions for lesson {lesson_matrix.lessonId}: {e}")
            return []

    async def _generate_questions_for_part(
        self, part, lesson_data: Dict[str, Any], subject: str, lesson_id: str
    ) -> List[Dict[str, Any]]:
        """Tạo câu hỏi cho một phần cụ thể"""
        try:
            part_questions = []
            part_num = part.part
            objectives = part.objectives

            # Tạo câu hỏi theo ma trận user input cho tất cả các phần
            for level, count in [("Biết", objectives.Biết), ("Hiểu", objectives.Hiểu), ("Vận_dụng", objectives.Vận_dụng)]:
                if count > 0:
                    level_questions = await self._generate_questions_for_level(
                        part_num, level, count, lesson_data, subject, lesson_id
                    )
                    part_questions.extend(level_questions)

            return part_questions

        except Exception as e:
            logger.error(f"Error generating questions for part {part.part}: {e}")
            return []

    async def _generate_questions_for_level(
        self, part_num: int, level: str, count: int, lesson_data: Dict[str, Any],
        subject: str, lesson_id: str
    ) -> List[Dict[str, Any]]:
        """Tạo câu hỏi cho một mức độ nhận thức cụ thể"""
        try:
            print(f"DEBUG: _generate_questions_for_level - Part {part_num}, Level {level}, Count {count}")

            # Xác định loại câu hỏi theo phần
            question_type = self._get_question_type_by_part(part_num)
            print(f"DEBUG: Question type for part {part_num}: {question_type}")

            # Tạo prompt cho LLM
            prompt = self._create_prompt_for_level(
                part_num, level, count, question_type, lesson_data, subject, lesson_id
            )
            print(f"DEBUG: Prompt created, length: {len(prompt)}")
            print(f"DEBUG: Prompt preview: {prompt[:200]}...")

            # Gọi LLM để tạo câu hỏi
            print(f"DEBUG: Calling LLM service...")
            response = await self.llm_service.generate_content(
                prompt=prompt,
                temperature=0.3,
                max_tokens=4000
            )
            print(f"DEBUG: LLM response received, length: {len(str(response))}")
            print(f"DEBUG: LLM response preview: {str(response)[:200]}...")

            # Debug logging for LLM response
            print(f"DEBUG: LLM response success: {response.get('success', False)}")
            if response.get("success", False):
                response_text = response.get("text", "")
                print(f"DEBUG: LLM response length: {len(response_text)}")
                print(f"DEBUG: LLM response preview: {response_text[:300]}...")
            else:
                logger.error(f"DEBUG: LLM error: {response.get('error', 'Unknown error')}")

            if not response.get("success", False):
                logger.error(f"LLM failed for part {part_num}, level {level}: {response.get('error')}")
                return []

            # Parse response JSON
            questions = self._parse_llm_response(response.get("text", ""), part_num, level, lesson_id)

            # Debug logging for parsed questions
            print(f"DEBUG: Parsed {len(questions)} questions from LLM response")
            
            # Giới hạn số câu hỏi theo yêu cầu
            return questions[:count]

        except Exception as e:
            logger.error(f"Error generating questions for level {level}: {e}")
            return []

    def _get_question_type_by_part(self, part_num: int) -> str:
        """Xác định loại câu hỏi theo phần"""
        if part_num == 1:
            return "TN"  # Trắc nghiệm nhiều phương án
        elif part_num == 2:
            return "DS"  # Đúng/Sai
        elif part_num == 3:
            return "TL"  # Tự luận
        else:
            return "TN"  # Default

    def _create_prompt_for_level(
        self, part_num: int, level: str, count: int, question_type: str,
        lesson_data: Dict[str, Any], subject: str, lesson_id: str
    ) -> str:
        """Create prompt for LLM according to THPT 2025 standards"""

        # Debug logging cho cấu trúc dữ liệu
        print(f"DEBUG: _create_prompt_for_level - lesson_data keys: {list(lesson_data.keys()) if isinstance(lesson_data, dict) else 'Not a dict'}")

        # Lấy nội dung bài học - lesson_data đã được xử lý ở _generate_questions_for_lesson
        main_content = lesson_data.get("main_content", "")
        lesson_info = lesson_data.get("lesson_info", {})

        # Debug logging
        print(f"DEBUG: main_content type: {type(main_content)}")
        print(f"DEBUG: main_content length: {len(str(main_content))}")
        print(f"DEBUG: lesson_info: {lesson_info}")

        # Ensure main_content is string and limit length
        if isinstance(main_content, str):
            content_preview = main_content[:2000] if len(main_content) > 2000 else main_content
        elif isinstance(main_content, list):
            # If it's a list, join the items
            content_preview = " ".join(str(item) for item in main_content)[:2000]
        else:
            content_preview = str(main_content)[:2000] if main_content else ""

        if not content_preview.strip():
            logger.warning(f"Empty content for lesson {lesson_id}, using fallback")
            content_preview = f"Nội dung bài học {lesson_id} - {lesson_info.get('lesson_title', 'Chưa có tiêu đề')}"

        # Part descriptions
        part_descriptions = {
            1: "PART I: Multiple choice questions (A, B, C, D)",
            2: "PART II: True/False evaluation questions (each question has a main scenario and 4 independent statements a, b, c, d to evaluate)",
            3: "PART III: Essay questions requiring calculations, chemical equations, and logical reasoning (6 questions total)"
        }

        prompt = f"""
You are an expert in creating {subject} exams according to THPT 2025 standards.

{part_descriptions.get(part_num, "")}

LESSON INFORMATION:
- Lesson: {lesson_id}
- Chapter: {lesson_info.get('chapter_title', '')}
- Content: {content_preview}...

REQUIREMENTS:
- Create {count} questions at "{level}" cognitive level
- Part {part_num} - {self._get_part_description(part_num)}
- Questions must be based on lesson content
- Ensure scientific accuracy

{self._get_specific_instructions_by_part(part_num, level)}

JSON RESPONSE FORMAT:
[
    {{
        "question": "Question content",
        "answer": {self._get_answer_format_by_part(part_num)},
        "explanation": "Answer explanation",
        "cognitive_level": "{level}",
        "part": {part_num}
    }}
]

Return only JSON, no additional text.
"""
        return prompt

    def _get_part_description(self, part_num: int) -> str:
        """Get detailed description for each part"""
        descriptions = {
            1: "Multiple choice questions",
            2: "True/False questions",
            3: "Short answer questions"
        }
        return descriptions.get(part_num, "")

    def _get_specific_instructions_by_part(self, part_num: int, level: str) -> str:
        """Hướng dẫn cụ thể cho từng phần"""
        if part_num == 1:
            return """
HƯỚNG DẪN PHẦN I:
- Mỗi câu có 4 phương án A, B, C, D
- Chỉ có 1 đáp án đúng
- Câu hỏi rõ ràng, không gây nhầm lẫn
"""
        elif part_num == 2:
            return """
HƯỚNG DẪN PHẦN II - QUAN TRỌNG:
- Tạo câu hỏi chính về một tình huống thí nghiệm hoặc phản ứng hóa học cụ thể
- Sau câu hỏi chính, tạo 4 phát biểu độc lập a), b), c), d)
- Mỗi phát biểu là một khẳng định cụ thể về tình huống đó
- Trong trường "answer", đặt cả nội dung phát biểu VÀ đánh giá đúng/sai:
  {"a": {"content": "Phát biểu a cụ thể", "evaluation": "Đúng"}, "b": {"content": "Phát biểu b cụ thể", "evaluation": "Sai"}, ...}
- Các phát biểu phải liên quan đến cùng một chủ đề/tình huống
- Ví dụ format:
  "question": "Xét thí nghiệm hòa tan kim loại X trong dung dịch HCl. Cho biết các phát biểu sau đúng hay sai:",
  "answer": {
    "a": {"content": "Kim loại X tác dụng với HCl tạo ra khí H2", "evaluation": "Đúng"},
    "b": {"content": "Phản ứng này là phản ứng oxi hóa khử", "evaluation": "Đúng"},
    "c": {"content": "Dung dịch sau phản ứng có pH > 7", "evaluation": "Sai"},
    "d": {"content": "Kim loại X bị oxi hóa trong phản ứng này", "evaluation": "Đúng"}
  }
"""
        elif part_num == 3:
            return """
HƯỚNG DẪN PHẦN III - TỰ LUẬN HÓA HỌC:
- Tạo 6 câu hỏi tự luận có tính toán, viết phương trình hóa học
- Các dạng câu phổ biến: tính hiệu suất/khối lượng, viết chuỗi phản ứng, tính ΔH, chuẩn độ, xác định công thức
- Câu hỏi phải có số liệu cụ thể và yêu cầu lập luận logic
- Đáp án chỉ là số (không có chữ A, B, C, D)
- Một số câu có thể yêu cầu sắp xếp hoặc suy luận từ lý thuyết
"""
        return ""

    def _get_answer_format_by_part(self, part_num: int) -> str:
        """Format đáp án theo từng phần"""
        if part_num == 1:
            return '{"A": "Phương án A", "B": "Phương án B", "C": "Phương án C", "D": "Phương án D", "dung": "A"}'
        elif part_num == 2:
            return '{"a": {"content": "Phát biểu a cụ thể", "evaluation": "Đúng"}, "b": {"content": "Phát biểu b cụ thể", "evaluation": "Sai"}, "c": {"content": "Phát biểu c cụ thể", "evaluation": "Đúng"}, "d": {"content": "Phát biểu d cụ thể", "evaluation": "Sai"}}'
        elif part_num == 3:
            return '{"dap_an": "Số hoặc giá trị cụ thể (VD: 2.5, 0.1M, 25%)"}'
        return '{"dung": "A"}'

    def _parse_llm_response(self, response_text: str, part_num: int, level: str, lesson_id: str) -> List[Dict[str, Any]]:
        """Parse response từ LLM"""
        try:
            # Tìm JSON trong response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON array found in LLM response")
                return []

            json_str = response_text[start_idx:end_idx]
            questions = json.loads(json_str)

            # Validate và bổ sung thông tin
            validated_questions = []
            for q in questions:
                if isinstance(q, dict) and "question" in q:
                    q["part"] = part_num
                    q["cognitive_level"] = level
                    q["lesson_id"] = lesson_id
                    q["question_type"] = self._get_question_type_by_part(part_num)
                    validated_questions.append(q)

            print(f"DEBUG: Validated {len(validated_questions)} questions out of {len(questions)} raw questions")
            return validated_questions

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from LLM response: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return []

    def _sort_and_renumber_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sắp xếp câu hỏi theo phần và đánh số lại"""
        try:
            # Sắp xếp theo phần
            sorted_questions = sorted(questions, key=lambda x: x.get("part", 1))
            
            # Đánh số lại theo từng phần
            part_counters = {1: 1, 2: 1, 3: 1}
            
            for question in sorted_questions:
                part = question.get("part", 1)
                question["stt"] = part_counters[part]
                question["stt_global"] = len([q for q in sorted_questions[:sorted_questions.index(question)+1]])
                part_counters[part] += 1

            return sorted_questions

        except Exception as e:
            logger.error(f"Error sorting and renumbering questions: {e}")
            return questions

    def _calculate_statistics(
        self, questions: List[Dict[str, Any]], exam_request: SmartExamRequest, generation_time: float
    ) -> ExamStatistics:
        """Tính toán thống kê đề thi"""
        try:
            # Đếm câu hỏi theo phần
            part_counts = {1: 0, 2: 0, 3: 0}
            difficulty_counts = {"Biết": 0, "Hiểu": 0, "Vận_dụng": 0}
            
            for question in questions:
                part = question.get("part", 1)
                part_counts[part] += 1

                level = question.get("cognitive_level", "Biết")
                if level in difficulty_counts:
                    difficulty_counts[level] += 1

            return ExamStatistics(
                total_questions=len(questions),
                part_1_questions=part_counts[1],
                part_2_questions=part_counts[2],
                part_3_questions=part_counts[3],
                lessons_used=len(exam_request.matrix),
                difficulty_distribution=difficulty_counts,
                generation_time=generation_time,
                created_at=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return ExamStatistics(
                total_questions=len(questions),
                part_1_questions=0,
                part_2_questions=0,
                part_3_questions=0,
                lessons_used=0,
                difficulty_distribution={"Biết": 0, "Hiểu": 0, "Vận_dụng": 0},
                generation_time=generation_time,
                created_at=datetime.now().isoformat()
            )


# Factory function để tạo SmartExamGenerationService instance
def get_smart_exam_generation_service() -> SmartExamGenerationService:
    """
    Tạo SmartExamGenerationService instance mới

    Returns:
        SmartExamGenerationService: Fresh instance
    """
    return SmartExamGenerationService()

# Backward compatibility - deprecated, sử dụng get_smart_exam_generation_service() thay thế
# Lazy loading để tránh khởi tạo ngay khi import
def _get_smart_exam_generation_service_lazy():
    """Lazy loading cho backward compatibility"""
    return get_smart_exam_generation_service()


