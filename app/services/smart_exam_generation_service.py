"""
Service cho việc tạo đề thi thông minh theo chuẩn THPT 2025
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Callable
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

    async def generate_smart_exam(
        self, exam_request: SmartExamRequest, lesson_content: Dict[str, Any],
        question_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Tạo đề thi thông minh theo chuẩn THPT 2025

        Args:
            exam_request: Request chứa ma trận đề thi
            lesson_content: Nội dung bài học từ Qdrant
            question_callback: Callback function để trả về từng câu hỏi ngay khi tạo xong

        Returns:
            Dict chứa đề thi đã được tạo
        """
        try:
            start_time = datetime.now()

            # Ensure LLM service is initialized
            self.llm_service._ensure_service_initialized()

            if not self.llm_service.is_available():
                return {
                    "success": False,
                    "error": "LLM service not available. Please check OpenRouter API configuration."
                }

            # Tạo câu hỏi cho từng phần theo chuẩn THPT 2025
            all_questions = []
            for lesson_matrix in exam_request.matrix:
                lesson_questions = await self._generate_questions_for_lesson(
                    lesson_matrix, lesson_content, exam_request.subject, question_callback
                )
                all_questions.extend(lesson_questions)

            # Sắp xếp câu hỏi theo phần và đánh số lại
            sorted_questions = self._sort_and_renumber_questions(all_questions)

            # Final validation: Loại bỏ câu hỏi có đáp án quá dài
            validated_questions = self._final_answer_validation(sorted_questions)

            # Tính toán thống kê
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()

            statistics = self._calculate_statistics(
                validated_questions, exam_request, generation_time
            )

            return {
                "success": True,
                "exam_id": f"smart_exam_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "questions": validated_questions,
                "statistics": statistics,
                "total_generated": len(validated_questions),
                "exam_request": exam_request.model_dump()
            }

        except Exception as e:
            logger.error(f"Error generating smart exam: {e}")
            return {
                "success": False,
                "error": f"Failed to generate smart exam: {str(e)}"
            }

    async def _generate_questions_for_lesson(
        self, lesson_matrix, lesson_content: Dict[str, Any], subject: str,
        question_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Tạo câu hỏi cho một bài học theo ma trận"""
        try:
            lesson_id = lesson_matrix.lessonId

            # Lấy nội dung bài học từ textbook_retrieval_service format
            lesson_data = lesson_content.get(lesson_id, {})

            if not lesson_data:
                error_msg = f"Không tìm thấy nội dung cho bài học {lesson_id} trong lesson_content"
                logger.error(error_msg)
                raise ValueError(error_msg)

            # Kiểm tra xem lesson_data có lesson_content không (từ textbook_retrieval_service)
            if not lesson_data.get("lesson_content"):
                error_msg = f"Lesson {lesson_id} không có nội dung lesson_content. Available keys: {list(lesson_data.keys())}"
                logger.error(error_msg)
                raise ValueError(error_msg)

            all_lesson_questions = []

            # Tạo câu hỏi cho từng phần
            for part in lesson_matrix.parts:
                part_questions = await self._generate_questions_for_part(
                    part, lesson_data, subject, lesson_id, question_callback
                )
                all_lesson_questions.extend(part_questions)

            return all_lesson_questions

        except Exception as e:
            logger.error(f"Error generating questions for lesson {lesson_matrix.lessonId}: {e}")
            return []

    async def _generate_questions_for_part(
        self, part, lesson_data: Dict[str, Any], subject: str, lesson_id: str,
        question_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Tạo câu hỏi cho một phần cụ thể"""
        try:
            part_questions = []
            part_num = part.part
            objectives = part.objectives

            # Tạo câu hỏi theo ma trận đa dạng THPT 2025 - hỗ trợ tất cả mức độ cho Phần 1 và 2
            if part_num == 1:
                # Phần I: Trắc nghiệm nhiều lựa chọn - hỗ trợ Biết, Hiểu, Vận dụng
                for level, count in [("Biết", objectives.Biết), ("Hiểu", objectives.Hiểu), ("Vận_dụng", objectives.Vận_dụng)]:
                    if count > 0:
                        level_questions = await self._generate_questions_for_level(
                            part_num, level, count, lesson_data, subject, lesson_id, question_callback
                        )
                        part_questions.extend(level_questions)
            elif part_num == 2:
                # Phần II: Trắc nghiệm Đúng/Sai - hỗ trợ Biết, Hiểu, Vận dụng
                for level, count in [("Biết", objectives.Biết), ("Hiểu", objectives.Hiểu), ("Vận_dụng", objectives.Vận_dụng)]:
                    if count > 0:
                        level_questions = await self._generate_questions_for_level(
                            part_num, level, count, lesson_data, subject, lesson_id, question_callback
                        )
                        part_questions.extend(level_questions)
            elif part_num == 3:
                # Phần III: Tự luận tính toán - hỗ trợ Biết, Hiểu, Vận dụng
                for level, count in [("Biết", objectives.Biết), ("Hiểu", objectives.Hiểu), ("Vận_dụng", objectives.Vận_dụng)]:
                    if count > 0:
                        level_questions = await self._generate_questions_for_level(
                            part_num, level, count, lesson_data, subject, lesson_id, question_callback
                        )
                        part_questions.extend(level_questions)

            return part_questions

        except Exception as e:
            logger.error(f"Error generating questions for part {part.part}: {e}")
            return []

    async def _generate_questions_for_level(
        self, part_num: int, level: str, count: int, lesson_data: Dict[str, Any],
        subject: str, lesson_id: str, question_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """Tạo câu hỏi cho một mức độ nhận thức cụ thể"""
        try:
            # Phần 3 sử dụng quy trình tư duy ngược với validation loop
            if part_num == 3:
                return await self._generate_part3_questions_with_reverse_thinking(
                    level, count, lesson_data, subject, lesson_id, question_callback
                )

            # Phần 1 và 2 sử dụng quy trình cũ
            prompt = self._create_prompt_for_level(
                part_num, level, count, lesson_data, subject, lesson_id
            )

            # Gọi LLM để tạo câu hỏi - tăng max_tokens cho nhiều câu hỏi
            max_tokens = 6000 if count > 3 else 4000  # Tăng token limit cho nhiều câu
            response = await self.llm_service.generate_content(
                prompt=prompt,
                temperature=0.3,
                max_tokens=max_tokens
            )

            if not response.get("success", False):
                logger.error(f"LLM failed for part {part_num}, level {level}: {response.get('error')}")
                return []

            # Parse response JSON
            questions = self._parse_llm_response(response.get("text", ""), part_num, level, lesson_id)

            # Giới hạn số câu hỏi theo yêu cầu
            limited_questions = questions[:count]

            # Gọi callback cho từng câu hỏi nếu có
            if question_callback and limited_questions:
                for question in limited_questions:
                    try:
                        await question_callback(question)
                    except Exception as e:
                        logger.warning(f"Error calling question callback: {e}")

            return limited_questions

        except Exception as e:
            logger.error(f"Error generating questions for level {level}: {e}")
            return []

    async def _generate_part3_questions_with_reverse_thinking(
        self, level: str, count: int, lesson_data: Dict[str, Any],
        subject: str, lesson_id: str, question_callback: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        Tạo câu hỏi phần 3 theo quy trình tư duy ngược với validation loop

        Quy trình:
        1. Tạo đáp án trước (4 chữ số phù hợp THPT 2025)
        2. Xây dựng ngược câu hỏi từ đáp án
        3. Validation loop với 2 LLM roles khác nhau
        """
        try:
            validated_questions = []
            max_retries = 5  # Tăng số lần retry để đảm bảo tạo đủ câu hỏi

            logger.info(f"🎯 Starting generation of {count} questions for level '{level}'")

            for i in range(count):
                question_created = False
                logger.info(f"📝 Generating question {i+1}/{count} for level '{level}'")

                # Retry logic để đảm bảo tạo đủ câu hỏi
                for retry in range(max_retries + 1):
                    try:
                        logger.info(f"🔄 Attempt {retry+1}/{max_retries+1} for question {i+1}/{count}")

                        # Bước 1: Tạo đáp án và câu hỏi ban đầu
                        initial_question = await self._create_initial_part3_question(
                            level, lesson_data, subject, lesson_id
                        )

                        if not initial_question:
                            logger.warning(f"❌ Failed to create initial question {i+1}/{count}, retry {retry+1}/{max_retries+1}")
                            continue

                        logger.info(f"✅ Created initial question {i+1}/{count}, proceeding to validation")

                        # Bước 2: Validation loop (với timeout ngắn hơn cho retry)
                        max_validation_iterations = 2 if retry > 0 else 3  # Giảm validation cho retry
                        final_question = await self._validate_and_improve_question(
                            initial_question, level, lesson_data, subject, lesson_id, max_validation_iterations
                        )

                        if final_question:
                            validated_questions.append(final_question)
                            question_created = True
                            logger.info(f"🎉 Successfully created question {i+1}/{count} for level '{level}' after {retry+1} attempts")

                            # Gọi callback cho câu hỏi vừa tạo xong nếu có
                            if question_callback:
                                try:
                                    await question_callback(final_question)
                                except Exception as e:
                                    logger.warning(f"Error calling question callback for Part 3: {e}")

                            break
                        else:
                            logger.warning(f"❌ Validation failed for question {i+1}/{count}, retry {retry+1}/{max_retries+1}")

                    except Exception as e:
                        logger.error(f"💥 Error creating question {i+1}/{count}, retry {retry+1}/{max_retries+1}: {e}")
                        continue

                if not question_created:
                    logger.error(f"🚫 FAILED to create question {i+1}/{count} after {max_retries+1} attempts")

            logger.info(f"📊 Final result: Generated {len(validated_questions)}/{count} questions for level '{level}'")
            return validated_questions

        except Exception as e:
            logger.error(f"Error in reverse thinking generation: {e}")
            return []

    async def _create_initial_part3_question(
        self, level: str, lesson_data: Dict[str, Any],
        subject: str, lesson_id: str
    ) -> Optional[Dict[str, Any]]:
        """Tạo câu hỏi ban đầu với đáp án được sinh trước"""
        try:
            # Lấy nội dung bài học
            main_content = self._extract_lesson_content(lesson_data)
            if not main_content.strip():
                return None

            # Tạo prompt cho việc sinh đáp án trước
            prompt = self._create_reverse_thinking_prompt(level, main_content, lesson_id)

            response = await self.llm_service.generate_content(
                prompt=prompt,
                temperature=0.4,
                max_tokens=3000
            )

            if not response.get("success", False):
                logger.error(f"Failed to create initial question: {response.get('error')}")
                return None

            # Parse response
            question_data = self._parse_reverse_thinking_response(
                response.get("text", ""), level, lesson_id
            )

            return question_data

        except Exception as e:
            logger.error(f"Error creating initial question: {e}")
            return None

    async def _validate_and_improve_question(
        self, question: Dict[str, Any], level: str, lesson_data: Dict[str, Any],
        subject: str, lesson_id: str, max_iterations: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        Validation loop với 2 LLM roles:
        - Role 1: Chuyên gia hóa học (giải và xác minh)
        - Role 2: Chuyên gia ra đề (cải thiện câu hỏi)
        """
        try:
            current_question = question.copy()

            for iteration in range(max_iterations):
                # Bước 3a: Gọi LLM với role chuyên gia hóa học
                validation_result = await self._validate_with_chemistry_expert(
                    current_question, lesson_data
                )

                if not validation_result:
                    continue

                # Kiểm tra xem câu hỏi đã đạt yêu cầu chưa
                accuracy_score = validation_result.get("accuracy_score", 0)
                # Convert string to int if needed
                if isinstance(accuracy_score, str):
                    try:
                        accuracy_score = int(accuracy_score)
                    except ValueError:
                        accuracy_score = 0

                # Giảm tiêu chuẩn validation để tạo được nhiều câu hỏi hơn
                min_score = 7 if max_iterations <= 2 else 8  # Giảm tiêu chuẩn cho retry
                if validation_result.get("is_valid", False) and accuracy_score >= min_score:
                    logger.info(f"✅ Question validated successfully after {iteration + 1} iterations (score: {accuracy_score}/{min_score})")
                    return current_question

                # Bước 3b: Gọi LLM với role chuyên gia ra đề để cải thiện
                improved_question = await self._improve_with_exam_expert(
                    current_question, validation_result, level, lesson_data
                )

                if improved_question:
                    current_question = improved_question
                else:
                    break

            # Nếu sau max_iterations vẫn chưa đạt, trả về phiên bản tốt nhất
            logger.warning(f"Question validation completed with {max_iterations} iterations")
            return current_question

        except Exception as e:
            logger.error(f"Error in validation loop: {e}")
            return question

    async def _validate_with_chemistry_expert(
        self, question: Dict[str, Any], lesson_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Gọi LLM với role chuyên gia hóa học để xác minh câu hỏi"""
        try:
            prompt = self._create_chemistry_expert_prompt(question, lesson_data)

            response = await self.llm_service.generate_content(
                prompt=prompt,
                temperature=0.1,
                max_tokens=2000
            )

            if not response.get("success", False):
                return None

            return self._parse_validation_response(response.get("text", ""))

        except Exception as e:
            logger.error(f"Error in chemistry expert validation: {e}")
            return None

    async def _improve_with_exam_expert(
        self, question: Dict[str, Any], validation_result: Dict[str, Any],
        level: str, lesson_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Gọi LLM với role chuyên gia ra đề để cải thiện câu hỏi"""
        try:
            prompt = self._create_exam_expert_prompt(question, validation_result, level, lesson_data)

            response = await self.llm_service.generate_content(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2500
            )

            if not response.get("success", False):
                return None

            improved_question = self._parse_improved_question_response(
                response.get("text", ""), question
            )

            return improved_question

        except Exception as e:
            logger.error(f"Error in exam expert improvement: {e}")
            return None

    def _extract_lesson_content(self, lesson_data: Dict[str, Any]) -> str:
        """Trích xuất nội dung bài học từ lesson_data"""
        if "lesson_content" in lesson_data:
            content = lesson_data.get("lesson_content", "")
        else:
            content = lesson_data.get("main_content", "")

        if isinstance(content, str):
            return content[:2000] if len(content) > 2000 else content
        elif isinstance(content, list):
            return " ".join(str(item) for item in content)[:2000]
        else:
            return str(content)[:2000] if content else ""

    def _create_reverse_thinking_prompt(self, level: str, content: str, lesson_id: str) -> str:
        """Tạo prompt cho quy trình tư duy ngược"""
        return f"""
Bạn là chuyên gia tạo đề thi Hóa học THPT 2025. Hãy áp dụng phương pháp TƯ DUY NGƯỢC để tạo câu hỏi tự luận tính toán.

QUY TRÌNH TƯ DUY NGƯỢC:
1. SINH ĐÁP ÁN TRƯỚC: Tạo một đáp án số thực dương phù hợp với phiếu trắc nghiệm THPT 2025
2. XÂY DỰNG NGƯỢC: Từ đáp án đó, thiết kế bối cảnh và nội dung câu hỏi

YÊU CẦU ĐÁP ÁN NGHIÊM NGẶT CHO PHIẾU TRẮC NGHIỆM:
- Đáp án phải có ÍT HƠN 5 ký tự (tối đa 4 ký tự bao gồm dấu thập phân)
- Đáp án phải chính xác theo tính toán hóa học
- Ví dụ hợp lệ: "12.5", "0.25", "75", "2.4", "1000"
- Ví dụ KHÔNG hợp lệ: "125.6" (5 ký tự), "35.25" (5 ký tự), "1234.5" (6 ký tự)
- Nếu kết quả tính toán ≥5 ký tự, hãy điều chỉnh dữ kiện đề bài để có đáp án <5 ký tự

THÔNG TIN BÀI HỌC:
- Lesson ID: {lesson_id}
- Nội dung: {content}

YÊU CẦU MỨC ĐỘ "{level}":
{self._get_reverse_thinking_requirements(level)}

ĐỊNH DẠNG JSON TRẢ VỀ:
{{
    "target_answer": "Số thực dương <5 ký tự - Ví dụ: 12.5, 0.25, 75, 2.4, 1000",
    "question": "Nội dung câu hỏi được xây dựng từ đáp án",
    "solution_steps": [
        "Bước 1: Mô tả bước giải",
        "Bước 2: Tính toán cụ thể",
        "Bước 3: Kết luận"
    ],
    "explanation": "Giải thích chi tiết từng bước giải bài với tính toán cụ thể, công thức sử dụng, và lý do tại sao đáp án chính xác",
    "cognitive_level": "{level}",
    "part": 3
}}

LƯU Ý QUAN TRỌNG VỀ ĐÁP ÁN:
- target_answer phải có ÍT HƠN 5 ký tự để phù hợp với phiếu trắc nghiệm THPT 2025
- Điều chỉnh dữ kiện đề bài (khối lượng, thể tích, nồng độ) để đáp án <5 ký tự
- KHÔNG được sửa đáp án sau khi tính toán - phải điều chỉnh từ đầu

LƯU Ý QUAN TRỌNG VỀ EXPLANATION:
- Field "explanation" phải là hướng dẫn giải bài chi tiết, từng bước
- KHÔNG được viết mô tả về câu hỏi hoặc thông tin meta
- Phải giải thích tại sao đáp án chính xác và cách tính toán

LƯU Ý QUAN TRỌNG VỀ HÓA HỌC - NGUYÊN TẮC CHUNG:
1. ĐỊNH LUẬT BẢO TOÀN:
   - Bảo toàn khối lượng: tổng khối lượng chất tham gia = tổng khối lượng sản phẩm
   - Bảo toàn nguyên tố: số nguyên tử mỗi nguyên tố ở 2 vế phương trình bằng nhau
   - Bảo toàn điện tích: tổng điện tích 2 vế phương trình ion bằng nhau

2. PHƯƠNG TRÌNH HÓA HỌC:
   - Viết đúng công thức hóa học của các chất
   - Cân bằng phương trình với hệ số nguyên tối giản
   - Tỉ lệ mol theo hệ số cân bằng phải chính xác

3. TÍNH TOÁN HÓA HỌC:
   - Sử dụng đúng khối lượng nguyên tử/phân tử theo bảng tuần hoàn
   - Kiểm tra tính hợp lý của kết quả (không âm, trong khoảng thực tế)
   - Đơn vị phải nhất quán và chính xác
   - TÍNH THEO TỈ LỆ MOL, KHÔNG PHẢI TỈ LỆ KHỐI LƯỢNG

4. LOGIC VÀ NHẤT QUÁN:
   - Kết quả các bước tính toán phải nhất quán với nhau
   - Công thức phân tử phải khớp với dữ liệu đã tính
   - Kiểm tra lại từng bước để tránh sai sót
   - KIỂM TRA KỸ TÍNH TOÁN: thực hiện phép tính từng bước và xác minh kết quả

5. QUY TRÌNH KIỂM TRA TÍNH TOÁN:
   - Bước 1: Xác định dữ liệu đầu vào và đơn vị
   - Bước 2: Viết phương trình phản ứng cân bằng
   - Bước 3: Tính số mol chất tham gia
   - Bước 4: Áp dụng tỉ lệ mol và hiệu suất
   - Bước 5: Tính khối lượng/thể tích sản phẩm
   - Bước 6: Kiểm tra tính hợp lý của kết quả

Lưu ý: Chỉ trả về JSON, không có văn bản bổ sung. THỰC HIỆN TÍNH TOÁN CHÍNH XÁC!
"""

    def _get_reverse_thinking_requirements(self, level: str) -> str:
        """Yêu cầu cụ thể cho từng mức độ trong tư duy ngược"""
        requirements = {
            "Biết": """
- Đáp án: Số đơn giản <5 ký tự, chính xác theo tính toán hóa học
- Bối cảnh: Áp dụng trực tiếp công thức cơ bản (n=m/M, C=n/V, pH=-log[H⁺])
- Ví dụ đáp án hợp lệ: "2.24", "5.6", "12", "0.5", "22.4"
- Điều chỉnh dữ kiện để đáp án <5 ký tự
""",
            "Hiểu": """
- Đáp án: Số vừa phải <5 ký tự, chính xác theo tính toán hóa học
- Bối cảnh: Cần hiểu bản chất phản ứng, áp dụng 2-3 bước tính toán
- Ví dụ đáp án hợp lệ: "16.2", "1.25", "48.6", "3.75"
- Điều chỉnh dữ kiện để đáp án <5 ký tự
""",
            "Vận_dụng": """
- Đáp án: Số phức tạp <5 ký tự, chính xác theo tính toán hóa học
- Bối cảnh: Bài toán nhiều bước, hiệu suất, hỗn hợp, quy trình công nghiệp
- Ví dụ đáp án hợp lệ: "125", "87.5", "2450", "67.8"
- Điều chỉnh dữ kiện để đáp án <5 ký tự
"""
        }
        return requirements.get(level, requirements["Biết"])

    def _create_chemistry_expert_prompt(self, question: Dict[str, Any], lesson_data: Dict[str, Any]) -> str:
        """Tạo prompt cho chuyên gia hóa học xác minh câu hỏi"""
        return f"""
Bạn là CHUYÊN GIA HÓA HỌC với 20 năm kinh nghiệm giảng dạy THPT. Hãy GIẢI THỬ câu hỏi dưới đây và đánh giá tính chính xác.

CÂU HỎI CẦN ĐÁNH GIÁ:
{question.get('question', '')}

ĐÁP ÁN ĐƯỢC CHO:
{question.get('target_answer', '')}

NHIỆM VỤ CỦA BẠN:
1. Giải chi tiết câu hỏi từ đầu đến cuối với từng bước tính toán cụ thể
2. So sánh kết quả của bạn với đáp án được cho
3. Đánh giá tính chính xác về mặt khoa học
4. KIỂM TRA ĐặC BIỆT: Logic hóa học, phương trình phản ứng, tỉ lệ mol
5. Kiểm tra ngữ cảnh có phù hợp với chương trình THPT không
6. Đưa ra góp ý cải thiện nếu cần

NGUYÊN TẮC KIỂM TRA CHUNG:
- Áp dụng các định luật bảo toàn (khối lượng, nguyên tố, điện tích)
- Phương trình phản ứng phải cân bằng chính xác
- Tỉ lệ mol theo hệ số cân bằng (KHÔNG DÙNG TỈ LỆ KHỐI LƯỢNG)
- Khối lượng mol tính đúng theo bảng tuần hoàn
- Giá trị kết quả trong khoảng hợp lý và thực tế

KIỂM TRA TÍNH TOÁN CHI TIẾT:
- Thực hiện từng phép tính một cách cụ thể
- Kiểm tra đơn vị trong mỗi bước
- Xác minh tỉ lệ mol và hiệu suất
- So sánh kết quả tính được với đáp án cho trước
- Nếu khác biệt, chỉ ra chính xác lỗi ở đâu

KIỂM TRA TÍNH NHẤT QUÁN:
- Kết quả các bước tính toán phải logic và nhất quán
- Công thức phân tử phải khớp với dữ liệu đã tính
- Đơn vị và số liệu phải chính xác
- Không có mâu thuẫn giữa các phần của bài giải

ĐỊNH DẠNG JSON TRẢ VỀ:
{{
    "my_solution": "Lời giải chi tiết của bạn",
    "my_answer": "Đáp án bạn tính được",
    "is_valid": true/false,
    "accuracy_score": "Điểm từ 1-10",
    "feedback": "Góp ý cụ thể để cải thiện",
    "suggested_improvements": [
        "Cải thiện 1",
        "Cải thiện 2"
    ]
}}

Lưu ý: Hãy nghiêm túc và chính xác trong đánh giá.
"""

    def _create_exam_expert_prompt(
        self, question: Dict[str, Any], validation_result: Dict[str, Any],
        level: str, lesson_data: Dict[str, Any]
    ) -> str:
        """Tạo prompt cho chuyên gia ra đề cải thiện câu hỏi"""
        return f"""
Bạn là CHUYÊN GIA RA ĐỀ THI HÓA HỌC THPT 2025. Hãy cải thiện câu hỏi dựa trên feedback từ chuyên gia hóa học.

CÂU HỎI HIỆN TẠI:
{question.get('question', '')}

ĐÁP ÁN HIỆN TẠI:
{question.get('target_answer', '')}

GIẢI THÍCH HIỆN TẠI:
{question.get('explanation', '')}

FEEDBACK TỪ CHUYÊN GIA HÓA HỌC:
- Điểm đánh giá: {validation_result.get('accuracy_score', 0)}/10
- Tính hợp lệ: {validation_result.get('is_valid', False)}
- Góp ý: {validation_result.get('feedback', '')}
- Cải thiện đề xuất: {validation_result.get('suggested_improvements', [])}

NHIỆM VỤ CỦA BẠN:
1. Chỉnh sửa câu hỏi dựa trên feedback
2. Điều chỉnh các thông số để đảm bảo đáp án chính xác
3. Cải thiện ngữ cảnh và cách diễn đạt
4. Cải thiện giải thích để phù hợp với câu hỏi mới
5. Đảm bảo phù hợp với mức độ "{level}"

ĐỊNH DẠNG JSON TRẢ VỀ:
{{
    "target_answer": "Đáp án đã được điều chỉnh",
    "question": "Câu hỏi đã được cải thiện",
    "solution_steps": [
        "Bước giải đã được cập nhật"
    ],
    "explanation": "Giải thích chi tiết cách giải câu hỏi đã cải thiện",
    "cognitive_level": "{level}",
    "part": 3,
    "improvements_made": [
        "Mô tả những thay đổi đã thực hiện"
    ]
}}

Lưu ý: Chỉ trả về JSON, tập trung vào việc cải thiện chất lượng câu hỏi. Field "explanation" phải là giải thích cách giải bài, không phải mô tả cải thiện.
"""

    def _parse_reverse_thinking_response(self, response_text: str, level: str, lesson_id: str) -> Optional[Dict[str, Any]]:
        """Parse response từ quy trình tư duy ngược"""
        try:
            # Tìm JSON trong response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON object found in reverse thinking response")
                return None

            json_str = response_text[start_idx:end_idx]
            question_data = json.loads(json_str)

            # Validate và bổ sung thông tin
            if not all(key in question_data for key in ["target_answer", "question"]):
                logger.error("Missing required fields in reverse thinking response")
                return None

            # Validate đáp án là số hợp lệ và có độ dài phù hợp với phiếu trắc nghiệm
            target_answer = str(question_data["target_answer"]).strip()
            logger.info(f"🔍 Validating answer: '{target_answer}' (length: {len(target_answer)} chars)")

            try:
                # Kiểm tra đáp án có phải là số hợp lệ không
                float(target_answer)

                # Kiểm tra độ dài đáp án phù hợp với phiếu trắc nghiệm THPT 2025
                if len(target_answer) >= 5:
                    logger.warning(f"❌ REJECTING: Answer too long for answer sheet: '{target_answer}' ({len(target_answer)} chars >= 5)")
                    return None

                logger.info(f"✅ ACCEPTING: Valid answer format: '{target_answer}' ({len(target_answer)} chars < 5)")
            except ValueError:
                logger.error(f"❌ REJECTING: Invalid answer format: '{target_answer}' is not a valid number")
                return None

            question_data["part"] = 3
            question_data["cognitive_level"] = level
            question_data["lesson_id"] = lesson_id
            question_data["question_type"] = "TL"
            question_data["answer"] = {"answer": question_data["target_answer"]}

            return question_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from reverse thinking response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing reverse thinking response: {e}")
            return None

    def _parse_validation_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse response từ chuyên gia hóa học"""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON object found in validation response")
                return None

            json_str = response_text[start_idx:end_idx]
            validation_data = json.loads(json_str)

            # Ensure required fields exist
            required_fields = ["is_valid", "accuracy_score", "feedback"]
            for field in required_fields:
                if field not in validation_data:
                    validation_data[field] = False if field == "is_valid" else ""

            return validation_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from validation response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing validation response: {e}")
            return None

    def _parse_improved_question_response(self, response_text: str, original_question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse response từ chuyên gia ra đề"""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON object found in improved question response")
                return original_question

            json_str = response_text[start_idx:end_idx]
            improved_data = json.loads(json_str)

            # Merge với câu hỏi gốc, ưu tiên dữ liệu mới
            result = original_question.copy()

            # Cập nhật từng field một cách có kiểm soát
            if "target_answer" in improved_data:
                improved_answer = str(improved_data["target_answer"]).strip()
                logger.info(f"🔍 Validating improved answer: '{improved_answer}' (length: {len(improved_answer)} chars)")

                # Validate độ dài đáp án cải thiện
                if len(improved_answer) >= 5:
                    logger.warning(f"❌ REJECTING IMPROVED: Answer too long: '{improved_answer}' ({len(improved_answer)} chars >= 5). Keeping original.")
                    # Giữ nguyên đáp án gốc nếu đáp án cải thiện quá dài
                    pass
                else:
                    logger.info(f"✅ ACCEPTING IMPROVED: Valid answer: '{improved_answer}' ({len(improved_answer)} chars < 5)")
                    result["answer"] = {"answer": improved_answer}
                    result["target_answer"] = improved_answer

            # Cập nhật các field khác nếu có
            for field in ["question", "solution_steps", "explanation"]:
                if field in improved_data:
                    result[field] = improved_data[field]
                    logger.info(f"✅ Updated field '{field}' from improved response")

            # Đảm bảo các field bắt buộc
            if "cognitive_level" in improved_data:
                result["cognitive_level"] = improved_data["cognitive_level"]
            if "part" in improved_data:
                result["part"] = improved_data["part"]

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from improved question response: {e}")
            return original_question
        except Exception as e:
            logger.error(f"Error parsing improved question response: {e}")
            return original_question



    def _final_answer_validation(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Final validation để loại bỏ câu hỏi có đáp án quá dài"""
        validated_questions = []

        for question in questions:
            # Lấy đáp án từ question
            answer_data = question.get("answer", {})
            if isinstance(answer_data, dict):
                answer = str(answer_data.get("answer", "")).strip()
            else:
                answer = str(answer_data).strip()

            # Validate độ dài đáp án
            if len(answer) >= 5:
                logger.warning(f"🚫 FINAL REJECT: Question with long answer '{answer}' ({len(answer)} chars) removed from final result")
                continue
            else:
                logger.info(f"✅ FINAL ACCEPT: Question with answer '{answer}' ({len(answer)} chars) included in final result")
                validated_questions.append(question)

        logger.info(f"📊 Final validation: {len(validated_questions)}/{len(questions)} questions passed")
        return validated_questions

    def _create_prompt_for_level(
        self, part_num: int, level: str, count: int,
        lesson_data: Dict[str, Any], subject: str, lesson_id: str
    ) -> str:
        """Create prompt for LLM according to THPT 2025 standards"""

        # Lấy nội dung bài học từ textbook_retrieval_service format
        main_content = ""

        if "lesson_content" in lesson_data:
            # Từ textbook_retrieval_service
            main_content = lesson_data.get("lesson_content", "")
        else:
            # Fallback cho format cũ
            main_content = lesson_data.get("main_content", "")

        # Ensure main_content is string and limit length
        if isinstance(main_content, str):
            content_preview = main_content[:2000] if len(main_content) > 2000 else main_content
        elif isinstance(main_content, list):
            # If it's a list, join the items
            content_preview = " ".join(str(item) for item in main_content)[:2000]
        else:
            content_preview = str(main_content)[:2000] if main_content else ""

        if not content_preview.strip():
            # Báo lỗi thay vì sử dụng fallback theo yêu cầu
            error_msg = f"Không tìm thấy nội dung cho bài học {lesson_id}. Lesson data keys: {list(lesson_data.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Part descriptions theo chuẩn THPT 2025 - đa dạng mức độ
        part_descriptions = {
            1: "PART I: Trắc nghiệm nhiều lựa chọn (A, B, C, D) - Hỗ trợ mức độ BIẾT, HIỂU, VẬN DỤNG: 18 câu đa dạng từ nhận biết đến tính toán đơn giản",
            2: "PART II: Trắc nghiệm Đúng/Sai - Hỗ trợ mức độ BIẾT, HIỂU, VẬN DỤNG: 4 câu lớn, mỗi câu có 4 phát biểu a,b,c,d để đánh giá",
            3: "PART III: Tự luận tính toán - Hỗ trợ mức độ BIẾT, HIỂU, VẬN DỤNG: Bài toán tính toán từ cơ bản đến phức tạp, đòi hỏi tư duy và tổng hợp kiến thức"
        }

        prompt = f"""
Bạn là chuyên gia tạo đề thi {subject} theo chuẩn THPT 2025, hãy dựa vào thông tin cung cấp bên dưới để tạo ra ma trận đề và trả về JSON tương ứng
{part_descriptions.get(part_num, "")}
THÔNG TIN BÀI HỌC:
- Nội dung: {content_preview}...

YÊU CẦU:
- Tạo {count} câu hỏi ở mức độ nhận thức "{level}"
- Phần {part_num} - {self._get_part_description(part_num)}
- Câu hỏi phải dựa trên nội dung bài học
- Ngữ liệu, dữ kiện trong câu phải khoa học, đúng thực tế.
- Tuân thủ nghiêm ngặt ma trận đề thi chuẩn THPT 2025
- Đảm bảo kiến thức chính xác, logic, không gây hiểu nhầm.
- KIỂM TRA KỸ LOGIC HÓA HỌC: phương trình phản ứng, tỉ lệ mol, bảo toàn nguyên tố, tính hợp lý
{self._get_specific_instructions_by_part(part_num, level)}

ĐỊNH DẠNG JSON TRẢ VỀ:
[
    {{
        "question": "Nội dung câu hỏi",
        "answer": {self._get_answer_format_by_part(part_num)},
        "explanation": "Giải thích chi tiết từng bước giải bài với công thức, tính toán cụ thể, và lý do tại sao đáp án đúng",
        "cognitive_level": "{level}",
        "part": {part_num}
    }}
]

LƯU Ý QUAN TRỌNG:
- Chỉ trả về JSON, không có văn bản bổ sung
- Field "explanation" phải là giải thích cách giải bài với tính toán chi tiết, không phải mô tả câu hỏi
- ÁP DỤNG NGUYÊN TẮC HÓA HỌC: bảo toàn, cân bằng, tỉ lệ mol (không phải tỉ lệ khối lượng)
- THỰC HIỆN TÍNH TOÁN CHÍNH XÁC: kiểm tra từng bước, đơn vị, công thức
- Đảm bảo tính chính xác khoa học và hợp lý thực tế

VALIDATION NGHIÊM NGẶT - PHẢI KIỂM TRA:
✓ Khối lượng mol chính xác: CaCO₃=100, NaCl=58.5, H₂SO₄=98...
✓ Công thức phân tử nhất quán: nếu n=17 thì C₁₇H₃₇N, không phải C₃H₉N
✓ Tỉ lệ mol theo phương trình cân bằng
✓ Bảo toàn nguyên tố trong mọi phản ứng
✓ Giá trị số học hợp lý và có thể tính được
"""
        return prompt

    def _get_part_description(self, part_num: int) -> str:
        """Get detailed description for each part theo chuẩn THPT 2025"""
        descriptions = {
            1: "Trắc nghiệm nhiều phương án (Hỗ trợ BIẾT, HIỂU, VẬN DỤNG)",
            2: "Trắc nghiệm đúng/sai (Hỗ trợ BIẾT, HIỂU, VẬN DỤNG)",
            3: "Tự luận tính toán (Hỗ trợ BIẾT, HIỂU, VẬN DỤNG)"
        }
        return descriptions.get(part_num, "")

    def _get_specific_instructions_by_part(self, part_num: int, level: str) -> str:
        """Hướng dẫn cụ thể cho từng phần theo chuẩn THPT 2025"""
        if part_num == 1:
            if level == "Biết":
                return """
HƯỚNG DẪN PHẦN I - MỨC ĐỘ BIẾT:
- Mỗi câu có 4 phương án A, B, C, D với chỉ 1 đáp án đúng
- Kiểm tra kiến thức lý thuyết nền tảng và khả năng nhận biết các khái niệm cơ bản
Có thể tham khảo các dạng bên dưới:
DẠNG 1: Nhận biết khái niệm, định nghĩa, tính chất
- Nhận biết công thức hóa học, tên gọi hợp chất
- Phân loại chất (axit, bazơ, muối, oxit)
- Nhận biết tính chất vật lý, hóa học cơ bản
- Ví dụ: "Chất nào sau đây là axit mạnh?" hoặc "Công thức phân tử của glucose là?"

DẠNG 2: Nhận biết phương trình phản ứng đơn giản
- Cân bằng phương trình hóa học cơ bản
- Nhận biết loại phản ứng (hóa hợp, phân hủy, thế, trao đổi)
- Ví dụ: "Phương trình nào sau đây được cân bằng đúng?"

DẠNG 3: Nhận biết ứng dụng, vai trò trong đời sống
- Ứng dụng của các chất trong công nghiệp, đời sống
- Tác hại và biện pháp phòng chống ô nhiễm
- Ví dụ: "Chất nào được dùng làm chất tẩy rửa?"
"""
            elif level == "Hiểu":
                return """
HƯỚNG DẪN PHẦN I - MỨC ĐỘ HIỂU (THÔNG HIỂU):
- Mỗi câu có 4 phương án A, B, C, D với chỉ 1 đáp án đúng
- Yêu cầu giải thích, so sánh, hoặc áp dụng trực tiếp một khái niệm đã học
Có thể tham khảo các dạng bên dưới:
DẠNG 1: So sánh tính chất hóa học/vật lý
- So sánh tính axit, tính bazơ, nhiệt độ sôi, tính tan, khả năng phản ứng
- Ví dụ: "Sắp xếp theo chiều tăng dần tính bazơ: anilin, metylamin, amoniac, đimetylamin"

DẠNG 2: Nhận biết hiện tượng thí nghiệm 🧪
- Mô tả thí nghiệm đơn giản và yêu cầu chỉ ra hiện tượng quan sát
- Màu sắc thay đổi, có kết tủa, sủi bọt khí, v.v.
- Ví dụ: "Cho dung dịch iot vào ống nghiệm chứa hồ tinh bột. Hiện tượng quan sát được là gì?"

DẠNG 3: Xác định phát biểu Đúng/Sai (dạng đơn giản)
- Đưa ra 4 phát biểu về một chủ đề cụ thể (polime, kim loại, đại cương hữu cơ)
- Ví dụ: "Phát biểu nào sau đây là đúng khi nói về tơ nilon-6,6?"

DẠNG 4: Danh pháp và Cấu tạo
- Cho công thức cấu tạo và yêu cầu gọi tên hợp chất hoặc ngược lại
- Ví dụ: "Hợp chất CH₃-CH(CH₃)-COOH có tên gọi là gì?"
"""
            elif level == "Vận_dụng":
                return """
HƯỚNG DẪN PHẦN I - MỨC ĐỘ VẬN DỤNG:
- Mỗi câu có 4 phương án A, B, C, D với chỉ 1 đáp án đúng
- Yêu cầu tính toán đơn giản hoặc giải quyết bài toán một hoặc hai bước
Có thể tham khảo các dạng bên dưới:
DẠNG 1: Bài toán Stoichiometry (tính theo phương trình hóa học)
- Cho phương trình phản ứng với lượng chất ở một vế, tính lượng chất ở vế còn lại
- Có thể kết hợp hiệu suất phản ứng ở mức cơ bản
- Ví dụ: "Đốt cháy hoàn toàn 6,4 gam đồng (Cu) trong oxi dư, thu được m gam đồng(II) oxit (CuO). Tính giá trị của m."

DẠNG 2: Bài toán về Nồng độ dung dịch
- Tính toán nồng độ mol, nồng độ phần trăm
- Bài toán pha loãng, trộn lẫn dung dịch không xảy ra phản ứng
- Ví dụ: "Hòa tan 20 gam NaOH vào 180 gam nước thu được dung dịch A. Tính nồng độ phần trăm của dung dịch A."

DẠNG 3: Xác định công thức phân tử đơn giản
- Dựa vào phần trăm khối lượng các nguyên tố hoặc kết quả đốt cháy (chỉ cho CO₂ và H₂O)
- Tìm công thức đơn giản nhất hoặc công thức phân tử
- Ví dụ: "Đốt cháy hoàn toàn một hiđrocacbon X thu được 4,48 lít CO₂ (đktc) và 3,6 gam H₂O. Tìm công thức phân tử của X."
"""
        elif part_num == 2:
            if level == "Biết":
                return """
HƯỚNG DẪN PHẦN II - MỨC ĐỘ BIẾT:
- Tạo câu hỏi chính về một chất hoặc khái niệm cơ bản
- Sau đó có 4 phát biểu a), b), c), d) để đánh giá đúng/sai
- Kiểm tra kiến thức lý thuyết nền tảng dưới dạng đúng/sai
Có thể tham khảo các dạng bên dưới:
DẠNG 1: Chùm phát biểu về định nghĩa và tính chất cơ bản
- Các nhận định về định nghĩa, công thức, tính chất vật lý cơ bản của một chất
- Ví dụ: "Cho các phát biểu về Glucose (C₆H₁₂O₆):"
  a) "Glucose là monosaccarit có 6 nguyên tử cacbon"
  b) "Glucose có công thức phân tử C₆H₁₂O₆"
  c) "Glucose tan tốt trong nước"
  d) "Glucose có vị ngọt"

Format answer: {"a": {"content": "Nội dung phát biểu a", "evaluation": "Đúng/Sai"}, ...}
"""
            elif level == "Hiểu":
                return """
HƯỚNG DẪN PHẦN II - MỨC ĐỘ HIỂU (THÔNG HIỂU):

- Tạo câu hỏi chính về một chất hoặc tình huống cụ thể
- Sau đó có 4 phát biểu a), b), c), d) để đánh giá đúng/sai
- Kiểm tra khả năng hiểu và giải thích các hiện tượng, quá trình hóa học
Có thể tham khảo các dạng bên dưới:
DẠNG 1: Chùm phát biểu về một chất cụ thể
- Cả 4 nhận định đều xoay quanh một chất duy nhất (sắt, nhôm, glucozơ, saccarozơ, etyl axetat)
- Các phát biểu kiểm tra về cấu trúc, tính chất vật lý, tính chất hóa học đặc trưng và ứng dụng
- Ví dụ: "Cho các phát biểu về Sắt (Fe):"
  a) "Sắt là kim loại có tính khử trung bình"
  b) "Trong tự nhiên, sắt chỉ tồn tại ở dạng hợp chất"
  c) "Hợp chất Sắt(II) vừa có tính khử vừa có tính oxi hóa"
  d) "Gang là hợp kim của sắt với cacbon, có hàm lượng cacbon từ 2-5%"

Format answer: {"a": {"content": "Nội dung phát biểu a", "evaluation": "Đúng/Sai"}, ...}
"""
            elif level == "Vận_dụng":
                return """
HƯỚNG DẪN PHẦN II - MỨC ĐỘ VẬN DỤNG:
- Tạo câu hỏi chính về một tình huống thực tiễn hoặc thí nghiệm
- Sau đó có 4 phát biểu a), b), c), d) để đánh giá đúng/sai
- Yêu cầu khả năng liên kết kiến thức với thực tiễn hoặc phân tích các bước trong quy trình
Có thể tham khảo các dạng bên dưới:
DẠNG 1: Chùm phát biểu mô tả một thí nghiệm hóa học 🔬
- Các nhận định mô tả về mục đích, các bước tiến hành, vai trò hóa chất, hiện tượng và giải thích kết quả
- Thí nghiệm cụ thể: tráng bạc, xà phòng hóa, điều chế este, ăn mòn điện hóa
- Ví dụ: "Cho các phát biểu về thí nghiệm điều chế Etyl axetat:"
  a) "H₂SO₄ đặc được dùng làm chất xúc tác và tăng hiệu suất phản ứng"
  b) "Có thể thay thế CH₃COOH bằng CH₃COONa để thực hiện phản ứng"
  c) "Sau phản ứng, este tạo thành nổi lên trên và có mùi thơm"
  d) "Mục đích của việc chưng cất là để tinh chế este"

DẠNG 2: Chùm phát biểu về ứng dụng thực tiễn và hóa học đời sống
- Các nhận định liên quan đến vấn đề thực tế: polime và vật liệu, phân bón hóa học, hóa học và môi trường, gang-thép, ăn mòn kim loại
- Ví dụ: "Cho các phát biểu về Polime:"
  a) "Cao su buna-S được điều chế bằng phản ứng trùng ngưng"
  b) "Tơ olon (nitron) được dùng để dệt vải may quần áo ấm"
  c) "Nhựa PVC có tính cách điện tốt, được dùng làm vật liệu cách điện"
  d) "Thủy tinh hữu cơ (plexiglas) có thể cho ánh sáng truyền qua tốt"

DẠNG 3: Chùm phát biểu kết hợp tính toán nhỏ
- Trong 4 nhận định, có 1-2 nhận định yêu cầu phép tính nhẩm hoặc tính toán nhanh
- Ví dụ: "Cho các phát biểu về dung dịch axit axetic 0,1M:"
  a) "Dung dịch này làm quỳ tím hóa đỏ"
  b) "Nồng độ ion H⁺ trong dung dịch nhỏ hơn 0,1M"
  c) "Để trung hòa 10ml dung dịch này cần dùng 10ml dung dịch NaOH 0,1M"
  d) "Giá trị pH của dung dịch này bằng 1"

Format answer: {"a": {"content": "Nội dung phát biểu a", "evaluation": "Đúng/Sai"}, ...}
"""
        elif part_num == 3:
            # PHẦN III - TỰ LUẬN TÍNH TOÁN - HỖ TRỢ TẤT CẢ MỨC ĐỘ
            if level == "Biết":
                return """
HƯỚNG DẪN PHẦN III - MỨC ĐỘ BIẾT:
- Câu hỏi tự luận đơn giản, áp dụng trực tiếp công thức cơ bản
- Đáp án là số thực dương, thường có giá trị đơn giản
Có thể tham khảo các dạng bên dưới:
DẠNG 1: Tính toán cơ bản theo công thức
- Áp dụng trực tiếp công thức n = m/M, C = n/V, pH = -log[H⁺]
- Ví dụ: "Tính số mol của 8g CuO" hoặc "Tính nồng độ mol của dung dịch chứa 0,1 mol NaCl trong 500ml"
DẠNG 2: Tính toán theo phương trình hóa học đơn giản
- Phản ứng 1 bước, tỉ lệ mol đơn giản 1:1 hoặc 1:2
- Ví dụ: "Cho 0,1 mol Zn tác dụng với HCl dư. Tính thể tích H₂ thu được ở đktc"

Yêu cầu: Đáp án phải là số cụ thể, sử dụng công thức cơ bản.
"""
            elif level == "Hiểu":
                return """
HƯỚNG DẪN PHẦN III - MỨC ĐỘ HIỂU:
- Câu hỏi tự luận yêu cầu hiểu bản chất phản ứng và áp dụng công thức phù hợp
- Đáp án là số thực dương, có thể cần 2-3 bước tính toán
Có thể tham khảo các dạng bên dưới:
DẠNG 1: Tính toán theo chuỗi phản ứng
- Phản ứng 2-3 bước liên tiếp, cần hiểu mối liên hệ giữa các chất
- Ví dụ: "Từ 11,2g Fe tạo thành FeCl₃ qua 2 giai đoạn. Tính khối lượng FeCl₃ thu được"
DẠNG 2: Bài toán dung dịch cơ bản
- Pha loãng, cô cạn, trộn dung dịch với tỉ lệ đơn giản
- Ví dụ: "Trộn 100ml dung dịch NaCl 0,2M với 200ml dung dịch NaCl 0,1M. Tính nồng độ dung dịch sau trộn"

Yêu cầu: Đáp án phải là số cụ thể, cần hiểu bản chất để chọn công thức đúng.
"""
            elif level == "Vận_dụng":
                return """
HƯỚNG DẪN PHẦN III - MỨC ĐỘ VẬN DỤNG:
- Câu hỏi yêu cầu áp dụng công thức và giải quyết bài toán nhiều bước trong bối cảnh quen thuộc
- Đáp án là số thực dương, thường có giá trị lớn (kg, tấn, %, mol)
Có thể tham khảo các dạng bên dưới:
DẠNG 1: Bài toán hiệu suất trong sản xuất công nghiệp
- Dựa trên quy trình sản xuất thực tế (điều chế NH₃, H₂SO₄, điện phân Al₂O₃, este hóa)
- Cho lượng nguyên liệu và hiệu suất → tính lượng sản phẩm (thuận)
- Cho lượng sản phẩm và hiệu suất → tính lượng nguyên liệu (nghịch)
- Ví dụ: "Sản xuất amoniac từ 10 tấn N₂ với hiệu suất 75%. Tính khối lượng NH₃ thu được."
DẠNG 2: Bài toán đốt cháy hợp chất hữu cơ
- Đốt cháy hoàn toàn hợp chất hữu cơ (este, amin, cacbohidrat)
- Dựa vào khối lượng/thể tích CO₂, H₂O, N₂ → tìm công thức phân tử, % khối lượng nguyên tố
- Ví dụ: "Đốt cháy 0,1 mol este X thu được 0,4 mol CO₂ và 0,3 mol H₂O. Tính phần trăm C trong X."

Yêu cầu: Đáp án phải là số cụ thể, sử dụng phương pháp bảo toàn nguyên tố và tỉ lệ mol.
"""
            else:  # Vận dụng cao
                return """
HƯỚNG DẪN PHẦN III - MỨC ĐỘ VẬN DỤNG CAO:
- Câu hỏi phức tạp, đòi hỏi tư duy sâu, tổng hợp nhiều mảng kiến thức
- Sử dụng phương pháp giải toán nâng cao (đồng đẳng hóa, quy đổi, dồn chất)
Có thể tham khảo các dạng bên dưới:
DẠNG 1: Bài toán biện luận hỗn hợp hữu cơ phức tạp
- Hỗn hợp nhiều chất có cấu trúc tương tự (este+axit, peptit+amino axit)
- Tham gia đồng thời nhiều phản ứng (thủy phân + đốt cháy)
- Ví dụ: "Hỗn hợp X gồm este và axit có cùng số C. Thủy phân X cần a mol NaOH, đốt cháy X thu được b mol CO₂. Tính % khối lượng este trong X."

DẠNG 2: Bài toán Vô cơ tổng hợp (Kim loại + Axit oxi hóa mạnh)
- Hỗn hợp kim loại và oxit tác dụng với HNO₃/H₂SO₄ đặc
- Tạo nhiều sản phẩm khử (NO, N₂O, SO₂, NH₄⁺)
- Ví dụ: "Hỗn hợp Fe, Al, FeO tác dụng với HNO₃ tạo NO và NH₄NO₃. Tính khối lượng muối khan."

DẠNG 3: Bài toán phân tích Đồ thị/Bảng biểu
- Phân tích dữ liệu quá trình hóa học (sục CO₂ vào kiềm, nhỏ axit vào muối, điện phân)
- Dựa vào điểm đặc biệt trên đồ thị → suy ra đại lượng ban đầu
- Ví dụ: "Cho đồ thị thể tích CO₂ theo thời gian khi sục vào Ba(OH)₂. Tính nồng độ Ba(OH)₂ ban đầu."

Yêu cầu: Sử dụng phương pháp bảo toàn electron, phân tích kỹ lưỡng các sản phẩm có thể tạo thành.
"""
        return ""

    def _get_answer_format_by_part(self, part_num: int) -> str:
        """Format đáp án theo từng phần"""
        if part_num == 1:
            return '{"A": "Phương án A", "B": "Phương án B", "C": "Phương án C", "D": "Phương án D", "correct_answer": "A"}'
        elif part_num == 2:
            return '{"a": {"content": "Phát biểu a cụ thể", "evaluation": "Đúng"}, "b": {"content": "Phát biểu b cụ thể", "evaluation": "Sai"}, "c": {"content": "Phát biểu c cụ thể", "evaluation": "Đúng"}, "d": {"content": "Phát biểu d cụ thể", "evaluation": "Sai"}}'
        elif part_num == 3:
            return '{"answer": "Số thực dương cụ thể (VD: 12.5, 0.25, 75, 2.4)"}'
        return '{"correct_answer": "A"}'

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
                    # Xác định loại câu hỏi theo phần
                    if part_num == 1:
                        q["question_type"] = "TN"  # Trắc nghiệm nhiều phương án
                    elif part_num == 2:
                        q["question_type"] = "DS"  # Đúng/Sai
                    elif part_num == 3:
                        q["question_type"] = "TL"  # Tự luận
                    else:
                        q["question_type"] = "TN"  # Default
                    validated_questions.append(q)

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
