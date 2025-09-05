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
from app.constants.difficulty_levels import DifficultyLevel

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
                # Phần I: Trắc nghiệm nhiều lựa chọn - hỗ trợ KNOWLEDGE, COMPREHENSION, APPLICATION
                for level, count in [(DifficultyLevel.KNOWLEDGE.value, objectives.KNOWLEDGE),
                                   (DifficultyLevel.COMPREHENSION.value, objectives.COMPREHENSION),
                                   (DifficultyLevel.APPLICATION.value, objectives.APPLICATION)]:
                    if count > 0:
                        level_questions = await self._generate_questions_for_level(
                            part_num, level, count, lesson_data, subject, lesson_id, question_callback
                        )
                        part_questions.extend(level_questions)
            elif part_num == 2:
                # Phần II: Trắc nghiệm Đúng/Sai - hỗ trợ KNOWLEDGE, COMPREHENSION, APPLICATION
                for level, count in [(DifficultyLevel.KNOWLEDGE.value, objectives.KNOWLEDGE),
                                   (DifficultyLevel.COMPREHENSION.value, objectives.COMPREHENSION),
                                   (DifficultyLevel.APPLICATION.value, objectives.APPLICATION)]:
                    if count > 0:
                        level_questions = await self._generate_questions_for_level(
                            part_num, level, count, lesson_data, subject, lesson_id, question_callback
                        )
                        part_questions.extend(level_questions)
            elif part_num == 3:
                # Phần III: Tự luận tính toán - hỗ trợ KNOWLEDGE, COMPREHENSION, APPLICATION
                for level, count in [(DifficultyLevel.KNOWLEDGE.value, objectives.KNOWLEDGE),
                                   (DifficultyLevel.COMPREHENSION.value, objectives.COMPREHENSION),
                                   (DifficultyLevel.APPLICATION.value, objectives.APPLICATION)]:
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
            logger.info(f"🎯 Starting generation: Part {part_num}, Level {level}, Count {count}")

            # Phần 3 sử dụng quy trình tư duy ngược với validation loop
            if part_num == 3:
                return await self._generate_part3_questions_with_reverse_thinking(
                    level, count, lesson_data, subject, lesson_id, question_callback
                )

            # Phần 1 và 2 sử dụng quy trình cải thiện
            prompt = self._create_prompt_for_level(
                part_num, level, count, lesson_data, subject, lesson_id
            )

            # Tăng max_tokens cho APPLICATION level và nhiều câu hỏi
            max_tokens = self._calculate_max_tokens(level, count)

            # Điều chỉnh temperature cho APPLICATION level
            temperature = 0.4 if level == "APPLICATION" else 0.3

            logger.info(f"📝 LLM params: max_tokens={max_tokens}, temperature={temperature}")

            response = await self.llm_service.generate_content(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            if not response.get("success", False):
                logger.error(f"❌ LLM failed for part {part_num}, level {level}: {response.get('error')}")
                return []

            # Log raw response để debug
            raw_response = response.get("text", "")
            logger.info(f"📥 Raw response length: {len(raw_response)} chars")
            logger.info(f"📥 Raw response preview: {raw_response[:200]}...")

            # Parse response JSON với improved parsing
            questions = self._parse_llm_response_improved(raw_response, part_num, level, lesson_id)

            logger.info(f"✅ Parsed {len(questions)} questions from LLM")

            # Nếu không đủ câu hỏi, thử retry một lần
            if len(questions) < count:
                logger.warning(f"⚠️ Only got {len(questions)}/{count} questions, attempting retry...")
                retry_questions = await self._retry_generation_if_needed(
                    part_num, level, count - len(questions), lesson_data, subject, lesson_id
                )
                questions.extend(retry_questions)
                logger.info(f"🔄 After retry: {len(questions)} total questions")

            # Giới hạn số câu hỏi theo yêu cầu
            limited_questions = questions[:count]

            logger.info(f"📊 Final result: {len(limited_questions)}/{count} questions for Part {part_num}, Level {level}")

            # Gọi callback cho từng câu hỏi nếu có
            if question_callback and limited_questions:
                for question in limited_questions:
                    try:
                        await question_callback(question)
                    except Exception as e:
                        logger.warning(f"Error calling question callback: {e}")

            return limited_questions

        except Exception as e:
            logger.error(f"💥 Error generating questions for level {level}: {e}")
            return []

    def _calculate_max_tokens(self, level: str, count: int) -> int:
        """Tính toán max_tokens dựa trên level và số lượng câu hỏi"""
        base_tokens = {
            "KNOWLEDGE": 3000,
            "COMPREHENSION": 4000,
            "APPLICATION": 5000  # Tăng cho APPLICATION level
        }

        base = base_tokens.get(level, 4000)

        # Tăng tokens cho nhiều câu hỏi
        if count > 5:
            return base + 2000
        elif count > 3:
            return base + 1000
        else:
            return base

    async def _retry_generation_if_needed(
        self, part_num: int, level: str, missing_count: int,
        lesson_data: Dict[str, Any], subject: str, lesson_id: str
    ) -> List[Dict[str, Any]]:
        """Retry generation nếu thiếu câu hỏi"""
        try:
            if missing_count <= 0:
                return []

            logger.info(f"🔄 Retrying generation for {missing_count} missing questions")

            # Tạo prompt đơn giản hơn cho retry
            retry_prompt = self._create_simple_retry_prompt(
                part_num, level, missing_count, lesson_data, subject, lesson_id
            )

            # Sử dụng params conservative hơn cho retry
            max_tokens = self._calculate_max_tokens(level, missing_count)

            response = await self.llm_service.generate_content(
                prompt=retry_prompt,
                temperature=0.5,  # Tăng creativity cho retry
                max_tokens=max_tokens
            )

            if not response.get("success", False):
                logger.error(f"❌ Retry failed: {response.get('error')}")
                return []

            retry_questions = self._parse_llm_response_improved(
                response.get("text", ""), part_num, level, lesson_id
            )

            logger.info(f"✅ Retry generated {len(retry_questions)} additional questions")
            return retry_questions[:missing_count]

        except Exception as e:
            logger.error(f"💥 Error in retry generation: {e}")
            return []

    def _create_simple_retry_prompt(
        self, part_num: int, level: str, count: int,
        lesson_data: Dict[str, Any], subject: str, lesson_id: str
    ) -> str:
        """Tạo prompt đơn giản cho retry generation"""

        main_content = self._extract_lesson_content(lesson_data)
        content_preview = main_content[:1000] if len(main_content) > 1000 else main_content

        return f"""
Bạn là chuyên gia tạo đề thi {subject}. Hãy tạo CHÍNH XÁC {count} câu hỏi cho:

PHẦN: {part_num} - {self._get_part_description(part_num)}
MỨC ĐỘ: {level}
NỘI DUNG: {content_preview}

YÊU CẦU:
- Tạo ĐÚNG {count} câu hỏi
- Format JSON array: [{{...}}, {{...}}]
- Mỗi câu hỏi phải có: question, answer, explanation

ĐỊNH DẠNG TRẢ VỀ:
[
    {{
        "question": "Nội dung câu hỏi",
        "answer": {self._get_answer_format_by_part(part_num)},
        "explanation": "Giải thích chi tiết",
        "cognitive_level": "{level}",
        "part": {part_num}
    }}
]

CHỈ TRẢ VỀ JSON ARRAY, KHÔNG CÓ TEXT KHÁC!
"""

    def _parse_llm_response_improved(self, response_text: str, part_num: int, level: str, lesson_id: str) -> List[Dict[str, Any]]:
        """Parse response từ LLM với improved logic"""
        try:
            logger.info(f"🔍 Parsing response for part {part_num}, level {level}")
            logger.info(f"📝 Response length: {len(response_text)} chars")

            # Method 1: Tìm JSON array
            questions = self._try_parse_json_array(response_text)
            if questions:
                logger.info(f"✅ Method 1 success: Found {len(questions)} questions in array")
                return self._validate_and_enrich_questions(questions, part_num, level, lesson_id)

            # Method 2: Tìm single JSON object
            questions = self._try_parse_single_object(response_text)
            if questions:
                logger.info(f"✅ Method 2 success: Found {len(questions)} questions from single object")
                return self._validate_and_enrich_questions(questions, part_num, level, lesson_id)

            # Method 3: Tìm multiple objects
            questions = self._try_parse_multiple_objects(response_text)
            if questions:
                logger.info(f"✅ Method 3 success: Found {len(questions)} questions from multiple objects")
                return self._validate_and_enrich_questions(questions, part_num, level, lesson_id)

            logger.error("❌ All parsing methods failed")
            return []

        except Exception as e:
            logger.error(f"💥 Error in improved parsing: {e}")
            return []

    def _try_parse_json_array(self, response_text: str) -> List[Dict[str, Any]]:
        """Thử parse JSON array từ response"""
        try:
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1

            if start_idx == -1 or end_idx == 0:
                return []

            json_str = response_text[start_idx:end_idx]
            questions = json.loads(json_str)

            if isinstance(questions, list):
                return questions
            else:
                return []

        except json.JSONDecodeError:
            return []
        except Exception:
            return []

    def _try_parse_single_object(self, response_text: str) -> List[Dict[str, Any]]:
        """Thử parse single JSON object và convert thành array"""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                return []

            json_str = response_text[start_idx:end_idx]
            question = json.loads(json_str)

            if isinstance(question, dict) and "question" in question:
                return [question]
            else:
                return []

        except json.JSONDecodeError:
            return []
        except Exception:
            return []

    def _try_parse_multiple_objects(self, response_text: str) -> List[Dict[str, Any]]:
        """Thử tìm multiple JSON objects trong text"""
        try:
            questions = []
            lines = response_text.split('\n')
            current_json = ""
            brace_count = 0

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                current_json += line + "\n"
                brace_count += line.count('{') - line.count('}')

                # Khi brace_count về 0, có thể là end của một object
                if brace_count == 0 and current_json.strip():
                    try:
                        obj = json.loads(current_json.strip())
                        if isinstance(obj, dict) and "question" in obj:
                            questions.append(obj)
                    except json.JSONDecodeError:
                        pass
                    current_json = ""

            return questions

        except Exception:
            return []

    def _validate_and_enrich_questions(self, questions: List[Dict[str, Any]], part_num: int, level: str, lesson_id: str) -> List[Dict[str, Any]]:
        """Validate và enrich questions với metadata"""
        validated_questions = []

        for i, q in enumerate(questions):
            if isinstance(q, dict) and "question" in q:
                # Enrich với metadata
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
                logger.info(f"✅ Question {i+1} validated and enriched")
            else:
                logger.warning(f"❌ Question {i+1} invalid: missing 'question' field or not dict")

        return validated_questions

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
                        attempt_msg = f"🔄 Attempt {retry+1}/{max_retries+1} for question {i+1}/{count}"
                        logger.info(attempt_msg)

                        # Bước 1: Tạo đáp án và câu hỏi ban đầu
                        initial_question = await self._create_initial_part3_question(
                            level, lesson_data, subject, lesson_id
                        )

                        if not initial_question:
                            fail_msg = f"❌ Failed to create initial question {i+1}/{count}, retry {retry+1}/{max_retries+1}"
                            logger.warning(fail_msg)
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
                            success_msg = f"🎉 Successfully created question {i+1}/{count} for level '{level}' after {retry+1} attempts"
                            logger.info(success_msg)

                            # Gọi callback cho câu hỏi vừa tạo xong nếu có
                            if question_callback:
                                try:
                                    await question_callback(final_question)
                                except Exception as e:
                                    logger.warning(f"Error calling question callback for Part 3: {e}")

                            break
                        else:
                            validation_fail_msg = f"❌ Validation failed for question {i+1}/{count}, retry {retry+1}/{max_retries+1}"
                            logger.warning(validation_fail_msg)

                    except Exception as e:
                        error_msg = f"💥 Error creating question {i+1}/{count}, retry {retry+1}/{max_retries+1}: {e}"
                        logger.error(error_msg)
                        continue

                if not question_created:
                    final_fail_msg = f"🚫 FAILED to create question {i+1}/{count} after {max_retries+1} attempts"
                    logger.error(final_fail_msg)

            logger.info(f"📊 Final result: Generated {len(validated_questions)}/{count} questions for level '{level}'")
            return validated_questions

        except Exception as e:
            logger.error(f"Error in reverse thinking generation: {e}")
            return []

    async def _create_initial_part3_question(
        self, level: str, lesson_data: Dict[str, Any],
        subject: str, lesson_id: str
    ) -> Optional[Dict[str, Any]]:
        """Tạo câu hỏi ban đầu với đáp án được sinh dựa trên context bài học"""
        try:
            # Lấy nội dung bài học
            main_content = self._extract_lesson_content(lesson_data)
            if not main_content.strip():
                return None

            # Phân tích context và sinh đáp án có cơ sở khoa học
            context_analysis = await self._analyze_lesson_context(main_content, level)

            # Tạo prompt dựa trên context analysis hoặc fallback
            if context_analysis:
                logger.info("✅ Using context-based approach for answer generation")
                prompt = self._create_context_based_prompt(level, main_content, context_analysis, lesson_id)
            else:
                logger.warning("⚠️ Context analysis failed, using traditional reverse thinking approach")
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

            # Nếu parse thành công nhưng có vấn đề với đáp án, thử auto-adjust
            if question_data:
                return question_data
            else:
                # Thử parse lại với auto-adjustment
                raw_question = self._parse_raw_response(response.get("text", ""))
                if raw_question:
                    adjusted_question = await self._auto_adjust_answer_if_needed(raw_question, level)
                    if adjusted_question:
                        logger.info("🔧 Successfully auto-adjusted question")
                        return self._finalize_question_data(adjusted_question, level, lesson_id)

            return None

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

                # Kiểm tra sai lệch đáp án
                answer_diff = validation_result.get("answer_difference_percent", 0)
                try:
                    answer_diff = float(str(answer_diff).replace("%", ""))
                except (ValueError, TypeError):
                    answer_diff = 100  # Nếu không parse được, coi như sai lệch lớn

                # Tiêu chuẩn validation nghiêm ngặt
                min_score = 7 if max_iterations <= 2 else 8
                max_answer_diff = 10  # Sai lệch tối đa 10%

                is_calculation_valid = answer_diff <= max_answer_diff
                is_score_valid = accuracy_score >= min_score
                is_overall_valid = validation_result.get("is_valid", False)

                # Kiểm tra xem có thể áp dụng làm tròn thông minh không (sai lệch nhỏ 2-5%)
                if (not is_calculation_valid and
                    2 <= answer_diff <= 5 and
                    is_score_valid and
                    validation_result.get("my_answer")):

                    smart_rounded_question = self._try_smart_rounding_from_validation(current_question, validation_result)
                    if smart_rounded_question:
                        logger.info(f"🎯 Applied smart rounding for small difference: {answer_diff}%")
                        return smart_rounded_question

                if is_overall_valid and is_score_valid and is_calculation_valid:
                    validation_success_msg = f"✅ Question validated successfully after {iteration + 1} iterations (score: {accuracy_score}/{min_score}, diff: {answer_diff}%)"
                    logger.info(validation_success_msg)
                    return current_question
                elif not is_calculation_valid:
                    logger.warning(f"❌ Answer difference too large: {answer_diff}% > {max_answer_diff}%")

                # Kiểm tra loại lỗi và xử lý tương ứng
                error_type = validation_result.get("error_type", "none")
                feedback = validation_result.get("feedback", "").lower()

                # Lỗi nghiêm trọng - cần tạo lại từ đầu
                critical_errors = [
                    "không thể giải", "đề bài sai", "mâu thuẫn", "không hợp lý",
                    "không tính được", "dữ kiện thiếu", "logic sai"
                ]

                if error_type == "data" or any(critical_error in feedback for critical_error in critical_errors):
                    logger.warning(f"🔄 Critical error detected (type: {error_type}), regenerating question from scratch")
                    return None  # Trigger retry từ đầu

                # Lỗi tính toán - có thể sửa được
                if error_type == "calculation" and answer_diff > max_answer_diff:
                    logger.info(f"🔧 Calculation error detected, attempting to fix answer")
                    # Thử điều chỉnh đáp án dựa trên kết quả validation
                    corrected_question = self._try_correct_answer(current_question, validation_result)
                    if corrected_question:
                        current_question = corrected_question
                        logger.info(f"✅ Answer corrected based on validation result")
                        continue

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

    async def _auto_adjust_answer_if_needed(self, question_data: Dict[str, Any], level: str) -> Optional[Dict[str, Any]]:
        """
        Tự động điều chỉnh đáp án và thêm yêu cầu làm tròn vào đề nếu cần
        """
        try:
            target_answer = str(question_data.get("target_answer", "")).strip()
            question_text = question_data.get("question", "")

            # Nếu đáp án quá dài, thử làm tròn và thêm yêu cầu vào đề
            if len(target_answer) >= 5:
                try:
                    answer_value = float(target_answer)

                    # Thử các cách làm tròn và tạo yêu cầu tương ứng
                    rounding_options = [
                        {
                            "rounded": str(round(answer_value, 1)),
                            "requirement": "(làm tròn đến 1 chữ số thập phân)",
                            "decimal_places": 1
                        },
                        {
                            "rounded": str(round(answer_value, 2)),
                            "requirement": "(làm tròn đến 2 chữ số thập phân)",
                            "decimal_places": 2
                        },
                        {
                            "rounded": str(int(round(answer_value))),
                            "requirement": "(làm tròn đến số nguyên)",
                            "decimal_places": 0
                        }
                    ]

                    for option in rounding_options:
                        rounded_answer = option["rounded"]
                        if len(rounded_answer) < 5 and float(rounded_answer) > 0:
                            logger.info(f"🔧 Auto-adjusted answer: {target_answer} → {rounded_answer}")

                            # Cập nhật đáp án
                            question_data["target_answer"] = rounded_answer
                            question_data["answer"] = {"answer": rounded_answer}

                            # Thêm yêu cầu làm tròn vào câu hỏi nếu chưa có
                            rounding_requirement = option["requirement"]
                            if rounding_requirement.replace("(", "").replace(")", "") not in question_text.lower():
                                # Thêm yêu cầu làm tròn vào cuối câu hỏi
                                if question_text.endswith("?"):
                                    updated_question = question_text[:-1] + f" {rounding_requirement}?"
                                else:
                                    updated_question = question_text + f" {rounding_requirement}"

                                question_data["question"] = updated_question
                                logger.info(f"📝 Added rounding requirement to question")

                            # Cập nhật explanation
                            original_explanation = question_data.get("explanation", "")
                            if option["decimal_places"] == 0:
                                question_data["explanation"] = f"Kết quả tính toán được làm tròn đến số nguyên: {rounded_answer}. {original_explanation}"
                            else:
                                decimal_places = option['decimal_places']
                                explanation_text = f"Kết quả tính toán được làm tròn đến {decimal_places} chữ số thập phân: {rounded_answer}. {original_explanation}"
                                question_data["explanation"] = explanation_text

                            return question_data

                except ValueError:
                    pass

            # Nếu không thể điều chỉnh, trả về None để trigger retry
            logger.warning(f"❌ Cannot auto-adjust answer: {target_answer}")
            return None

        except Exception as e:
            logger.error(f"Error in auto-adjustment: {e}")
            return None

    def _try_correct_answer(self, question: Dict[str, Any], validation_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Thử sửa đáp án dựa trên kết quả validation từ chuyên gia hóa học
        """
        try:
            my_answer = validation_result.get("my_answer", "").strip()
            if not my_answer:
                return None

            # Kiểm tra đáp án từ chuyên gia có hợp lệ không
            try:
                expert_answer_value = float(my_answer)
                if expert_answer_value <= 0 or expert_answer_value > 9999 or len(my_answer) >= 5:
                    return None
            except ValueError:
                return None

            # Kiểm tra xem có thể áp dụng logic làm tròn thông minh không
            original_answer = question.get("target_answer", "")
            try:
                original_value = float(original_answer)

                # Nếu sai lệch nhỏ (< 5%), thử áp dụng làm tròn thông minh
                difference_percent = abs(expert_answer_value - original_value) / expert_answer_value * 100
                if difference_percent < 5:
                    smart_rounded_question = self._apply_smart_rounding(question, expert_answer_value, original_value)
                    if smart_rounded_question:
                        smart_target = smart_rounded_question['target_answer']
                        smart_rounding_msg = f"🎯 Applied smart rounding: {original_answer} → {smart_target} (expert: {my_answer})"
                        logger.info(smart_rounding_msg)
                        return smart_rounded_question
            except ValueError:
                pass

            # Tạo câu hỏi mới với đáp án từ chuyên gia
            corrected_question = question.copy()
            corrected_question["target_answer"] = my_answer
            corrected_question["answer"] = {"answer": my_answer}

            # Cập nhật explanation với lời giải từ chuyên gia
            expert_solution = validation_result.get("my_solution", "")
            if expert_solution:
                corrected_question["explanation"] = f"{expert_solution}"

            logger.info(f"🔧 Corrected answer: {question.get('target_answer')} → {my_answer}")
            return corrected_question

        except Exception as e:
            logger.error(f"Error correcting answer: {e}")
            return None

    def _apply_smart_rounding(self, question: Dict[str, Any], expert_value: float, original_value: float) -> Optional[Dict[str, Any]]:
        """
        Áp dụng làm tròn thông minh khi có sai lệch nhỏ giữa đáp án gốc và đáp án chuyên gia
        """
        try:
            # Thử các cách làm tròn khác nhau để tìm cách phù hợp nhất
            rounding_options = [
                {
                    "rounded": round(expert_value),
                    "requirement": "làm tròn đến số nguyên",
                    "decimal_places": 0
                },
                {
                    "rounded": round(expert_value, 1),
                    "requirement": "làm tròn đến 1 chữ số thập phân",
                    "decimal_places": 1
                },
                {
                    "rounded": round(expert_value, 2),
                    "requirement": "làm tròn đến 2 chữ số thập phân",
                    "decimal_places": 2
                }
            ]

            # Tìm cách làm tròn phù hợp nhất với đáp án gốc
            best_option = None
            min_difference = float('inf')

            for option in rounding_options:
                rounded_value = option["rounded"]
                difference = abs(rounded_value - original_value)

                # Kiểm tra xem đáp án làm tròn có phù hợp không
                if (difference < min_difference and
                    len(str(rounded_value)) < 5 and
                    rounded_value > 0):
                    min_difference = difference
                    best_option = option

            if best_option and min_difference / original_value * 100 < 2:  # Sai lệch < 2%
                corrected_question = question.copy()
                rounded_answer = str(best_option["rounded"])

                # Cập nhật đáp án
                corrected_question["target_answer"] = rounded_answer
                corrected_question["answer"] = {"answer": rounded_answer}

                # Thêm yêu cầu làm tròn vào câu hỏi
                question_text = question.get("question", "")
                rounding_requirement = f"({best_option['requirement']})"

                if rounding_requirement.replace("(", "").replace(")", "") not in question_text.lower():
                    if question_text.endswith("?"):
                        updated_question = question_text[:-1] + f" {rounding_requirement}?"
                    else:
                        updated_question = question_text + f" {rounding_requirement}"

                    corrected_question["question"] = updated_question

                # Cập nhật explanation
                original_explanation = question.get("explanation", "")
                if best_option["decimal_places"] == 0:
                    corrected_question["explanation"] = f"Kết quả tính toán chính xác là {expert_value:.3f}, được làm tròn đến số nguyên: {rounded_answer}. {original_explanation}"
                else:
                    decimal_places = best_option['decimal_places']
                    explanation_text = f"Kết quả tính toán chính xác là {expert_value:.3f}, được làm tròn đến {decimal_places} chữ số thập phân: {rounded_answer}. {original_explanation}"
                    corrected_question["explanation"] = explanation_text

                return corrected_question

            return None

        except Exception as e:
            logger.error(f"Error in smart rounding: {e}")
            return None

    def _try_smart_rounding_from_validation(self, question: Dict[str, Any], validation_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Thử áp dụng làm tròn thông minh dựa trên kết quả validation khi có sai lệch nhỏ
        """
        try:
            expert_answer = validation_result.get("my_answer", "").strip()
            original_answer = question.get("target_answer", "")

            if not expert_answer or not original_answer:
                return None

            try:
                expert_value = float(expert_answer)
                original_value = float(original_answer)
            except ValueError:
                return None

            # Kiểm tra xem đáp án gốc có thể là kết quả làm tròn của đáp án chuyên gia không
            rounding_options = [
                {
                    "rounded": round(expert_value),
                    "requirement": "làm tròn đến số nguyên",
                    "decimal_places": 0
                },
                {
                    "rounded": round(expert_value, 1),
                    "requirement": "làm tròn đến 1 chữ số thập phân",
                    "decimal_places": 1
                },
                {
                    "rounded": round(expert_value, 2),
                    "requirement": "làm tròn đến 2 chữ số thập phân",
                    "decimal_places": 2
                }
            ]

            # Tìm cách làm tròn phù hợp với đáp án gốc
            for option in rounding_options:
                rounded_value = option["rounded"]

                # Kiểm tra khớp chính xác
                if abs(rounded_value - original_value) < 0.01:  # Gần như bằng nhau
                    corrected_question = question.copy()

                    # Giữ nguyên đáp án gốc nhưng thêm yêu cầu làm tròn vào câu hỏi
                    question_text = question.get("question", "")
                    rounding_requirement = f"({option['requirement']})"

                    if rounding_requirement.replace("(", "").replace(")", "") not in question_text.lower():
                        if question_text.endswith("?"):
                            updated_question = question_text[:-1] + f" {rounding_requirement}?"
                        else:
                            updated_question = question_text + f" {rounding_requirement}"

                        corrected_question["question"] = updated_question

                    # Cập nhật explanation để giải thích việc làm tròn
                    original_explanation = question.get("explanation", "")
                    if option["decimal_places"] == 0:
                        corrected_question["explanation"] = f"Kết quả tính toán chính xác là {expert_value:.3f}, được làm tròn đến số nguyên: {original_answer}. {original_explanation}"
                    else:
                        decimal_places = option['decimal_places']
                        explanation_text = f"Kết quả tính toán chính xác là {expert_value:.3f}, được làm tròn đến {decimal_places} chữ số thập phân: {original_answer}. {original_explanation}"
                        corrected_question["explanation"] = explanation_text

                    logger.info(f"🎯 Smart rounding applied: {expert_value:.3f} → {original_answer} ({option['requirement']})")
                    return corrected_question

            # Nếu không khớp chính xác, kiểm tra xem có thể là làm tròn với sai lệch nhỏ không
            for option in rounding_options:
                rounded_value = option["rounded"]
                difference_percent = abs(rounded_value - original_value) / max(rounded_value, original_value) * 100

                # Nếu sai lệch < 2% và có thể giải thích được bằng làm tròn
                if difference_percent < 2:
                    corrected_question = question.copy()

                    # Giữ nguyên đáp án gốc nhưng thêm yêu cầu làm tròn vào câu hỏi
                    question_text = question.get("question", "")
                    rounding_requirement = f"({option['requirement']})"

                    if rounding_requirement.replace("(", "").replace(")", "") not in question_text.lower():
                        if question_text.endswith("?"):
                            updated_question = question_text[:-1] + f" {rounding_requirement}?"
                        else:
                            updated_question = question_text + f" {rounding_requirement}"

                        corrected_question["question"] = updated_question

                    # Cập nhật explanation để giải thích việc làm tròn
                    original_explanation = question.get("explanation", "")
                    if option["decimal_places"] == 0:
                        corrected_question["explanation"] = f"Kết quả tính toán chính xác là {expert_value:.3f}, được làm tròn đến số nguyên: {original_answer}. {original_explanation}"
                    else:
                        decimal_places = option['decimal_places']
                        explanation_text = f"Kết quả tính toán chính xác là {expert_value:.3f}, được làm tròn đến {decimal_places} chữ số thập phân: {original_answer}. {original_explanation}"
                        corrected_question["explanation"] = explanation_text

                    requirement = option['requirement']
                    tolerance_msg = f"🎯 Smart rounding applied (with tolerance): {expert_value:.3f} → {original_answer} ({requirement}, diff: {difference_percent:.1f}%)"
                    logger.info(tolerance_msg)
                    return corrected_question

            return None

        except Exception as e:
            logger.error(f"Error in smart rounding from validation: {e}")
            return None

    def _parse_raw_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse raw response without validation để có thể auto-adjust"""
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                return None

            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)

        except json.JSONDecodeError:
            return None
        except Exception:
            return None

    def _finalize_question_data(self, question_data: Dict[str, Any], level: str, lesson_id: str) -> Dict[str, Any]:
        """Finalize question data với các field bắt buộc"""
        question_data["part"] = 3
        question_data["cognitive_level"] = level
        question_data["lesson_id"] = lesson_id
        question_data["question_type"] = "TL"

        if "target_answer" in question_data:
            question_data["answer"] = {"answer": question_data["target_answer"]}

        return question_data

    async def _analyze_lesson_context(self, content: str, level: str) -> Optional[Dict[str, Any]]:
        """
        Phân tích context bài học để xác định công thức, khái niệm và giá trị phù hợp
        """
        try:
            analysis_prompt = f"""
Bạn là chuyên gia phân tích nội dung hóa học THPT. Hãy phân tích nội dung bài học dưới đây để xác định:

NỘI DUNG BÀI HỌC:
{content}

YÊU CẦU PHÂN TÍCH:
1. Xác định các CÔNG THỨC HÓA HỌC chính trong bài học
2. Xác định các GIÁ TRỊ SỐ LIỆU thường gặp (khối lượng mol, thể tích, nồng độ, pH...)
3. Xác định các LOẠI BÀI TOÁN phù hợp với mức độ "{level}"
4. Đề xuất ĐÁNH SỐ CỤ THỂ cho đáp án dựa trên công thức và dữ liệu thực tế

ĐỊNH DẠNG JSON TRẢ VỀ:
{{
    "formulas": [
        {{"name": "Tên công thức", "formula": "Công thức", "variables": ["biến 1", "biến 2"]}},
        {{"name": "n = m/M", "formula": "n = m/M", "variables": ["n (mol)", "m (g)", "M (g/mol)"]}}
    ],
    "common_values": {{
        "molar_masses": [16, 18, 32, 44, 58.5, 98, 100],
        "volumes_stp": [22.4, 11.2, 44.8, 67.2],
        "concentrations": [0.1, 0.2, 0.5, 1.0, 2.0],
        "ph_values": [1, 2, 7, 12, 13]
    }},
    "problem_types": [
        "stoichiometry", "concentration", "gas_volume", "ph_calculation"
    ],
    "suggested_answers": [
        {{"value": "22.4", "context": "Thể tích 1 mol khí ở đktc", "formula_used": "V = n × 22.4"}},
        {{"value": "0.1", "context": "Số mol từ khối lượng", "formula_used": "n = m/M"}}
    ]
}}

Lưu ý: Chỉ trả về JSON, đáp án phải <5 ký tự và dựa trên tính toán thực tế từ nội dung bài học.
"""

            response = await self.llm_service.generate_content(
                prompt=analysis_prompt,
                temperature=0.2,
                max_tokens=2000
            )

            if not response.get("success", False):
                logger.error(f"Failed to analyze lesson context: {response.get('error')}")
                return None

            # Parse JSON response
            response_text = response.get("text", "")
            
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1

            if start_idx == -1 or end_idx == 0:
                logger.error("No JSON found in context analysis response")
                return None

            json_str = response_text[start_idx:end_idx]
            context_data = json.loads(json_str)

            logger.info(f"✅ Context analysis successful: {len(context_data.get('suggested_answers', []))} suggested answers")
            return context_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse context analysis JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error analyzing lesson context: {e}")
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

    def _create_context_based_prompt(self, level: str, content: str, context_analysis: Dict[str, Any], lesson_id: str) -> str:
        """Tạo prompt dựa trên phân tích context bài học"""

        # Lấy thông tin từ context analysis
        formulas = context_analysis.get("formulas", [])
        suggested_answers = context_analysis.get("suggested_answers", [])

        # Tạo danh sách công thức
        formulas_text = ""
        if formulas:
            formulas_text = "CÔNG THỨC CHÍNH TRONG BÀI HỌC:\n"
            for formula in formulas[:3]:  # Lấy tối đa 3 công thức
                formulas_text += f"- {formula.get('name', '')}: {formula.get('formula', '')}\n"

        # Tạo danh sách đáp án gợi ý
        suggested_answers_text = ""
        if suggested_answers:
            suggested_answers_text = "ĐÁP ÁN GỢI Ý DỰA TRÊN CONTEXT:\n"
            for answer in suggested_answers[:5]:  # Lấy tối đa 5 đáp án
                suggested_answers_text += f"- {answer.get('value', '')}: {answer.get('context', '')} ({answer.get('formula_used', '')})\n"

        # Tạo validation instructions động
        validation_instructions = self._generate_validation_instructions(level, formulas, suggested_answers)

        return f"""
Bạn là chuyên gia tạo đề thi Hóa học THPT 2025. Hãy áp dụng phương pháp TƯ DUY NGƯỢC DỰA TRÊN CONTEXT để tạo câu hỏi tự luận tính toán.

QUY TRÌNH TƯ DUY NGƯỢC DỰA TRÊN CONTEXT:
1. CHỌN ĐÁP ÁN TỪ CONTEXT: Chọn một đáp án từ danh sách gợi ý dựa trên nội dung bài học
2. XÂY DỰNG BÀI TOÁN: Từ đáp án và công thức, thiết kế bối cảnh và dữ kiện phù hợp
3. KIỂM TRA TÍNH TOÁN NGƯỢC: Tính toán từ dữ kiện về đáp án để đảm bảo chính xác
4. VALIDATION NGHIÊM NGẶT: Kiểm tra lại toàn bộ bài toán từ đầu đến cuối

THÔNG TIN BÀI HỌC:
- Lesson ID: {lesson_id}
- Nội dung: {content}...

{formulas_text}

{suggested_answers_text}

YÊU CẦU MỨC ĐỘ "{level}":
{self._get_reverse_thinking_requirements(level)}

YÊU CẦU ĐÁP ÁN NGHIÊM NGẶT:
- Đáp án phải có ÍT HƠN 5 ký tự (tối đa 4 ký tự bao gồm dấu thập phân)
- Ưu tiên chọn từ danh sách đáp án gợi ý ở trên
- Nếu không dùng đáp án gợi ý, phải đảm bảo tính chính xác khoa học

ĐỊNH DẠNG JSON TRẢ VỀ:
{{
    "target_answer": "Đáp án được chọn từ context hoặc tính toán chính xác <5 ký tự",
    "question": "Nội dung câu hỏi được xây dựng từ đáp án và context",
    "solution_steps": [
        "Bước 1: Xác định dữ liệu và công thức",
        "Bước 2: Thực hiện tính toán",
        "Bước 3: Kết luận đáp án"
    ],
    "explanation": "Giải thích chi tiết từng bước với công thức cụ thể từ context bài học",
    "formula_used": "Công thức chính được sử dụng",
    "cognitive_level": "{level}",
    "part": 3
}}

{validation_instructions}

Lưu ý: Chỉ trả về JSON, không có văn bản bổ sung. PHẢI TỰ VALIDATION TRƯỚC KHI TRẢ VỀ!
"""

    def _generate_validation_instructions(self, level: str, formulas: List[Dict], suggested_answers: List[Dict]) -> str:
        """Tạo validation instructions động dựa trên context trong format JSON"""

        # Tạo validation rules tổng quát dựa trên context
        validation_rules = {
            "general_rules": [
                "Ưu tiên sử dụng công thức và giá trị từ context analysis",
                "SAU KHI TẠO XONG: Hãy tự kiểm tra lại bài toán từ đầu đến cuối",
                "Tính toán ngược từ dữ kiện đề bài để xác minh đáp án",
                "Nếu phát hiện sai lệch, hãy điều chỉnh dữ kiện hoặc đáp án cho phù hợp",
                "Explanation phải là hướng dẫn giải bài với tính toán cụ thể"
            ],
            "context_warnings": [],
            "validation_steps": [
                "Đọc lại câu hỏi và xác định tất cả dữ kiện",
                "Áp dụng công thức và tính toán từng bước",
                "So sánh kết quả với target_answer",
                "Kiểm tra đơn vị và yêu cầu làm tròn",
                "Nếu sai lệch > 5%, điều chỉnh dữ kiện hoặc đáp án",
                "Đảm bảo tất cả số liệu hợp lý và thực tế"
            ],
            "universal_errors": [
                {
                    "error_type": "unit_mismatch",
                    "description": "Nhầm lẫn đơn vị hoặc đại lượng",
                    "prevention": "Luôn kiểm tra đề yêu cầu tính gì và trả về đúng đơn vị"
                },
                {
                    "error_type": "formula_application",
                    "description": "Áp dụng sai công thức hoặc thiếu bước",
                    "prevention": "Xác minh công thức phù hợp với dạng bài và áp dụng đầy đủ"
                },
                {
                    "error_type": "calculation_logic",
                    "description": "Sai logic tính toán hoặc tỉ lệ",
                    "prevention": "Kiểm tra tính hợp lý của kết quả (không âm, không quá lớn/nhỏ)"
                },
                {
                    "error_type": "data_interpretation",
                    "description": "Hiểu sai dữ kiện hoặc yêu cầu đề bài",
                    "prevention": "Đọc kỹ đề bài và xác định chính xác những gì cần tính"
                }
            ],
            "validation_examples": []
        }

        # Thêm warnings tổng quát dựa trên formulas có trong context
        if formulas:
            formula_types = set()
            for formula in formulas[:3]:
                formula_name = formula.get('name', '').lower()
                formula_content = formula.get('formula', '').lower()

                # Phát hiện các pattern tổng quát
                if any(keyword in formula_name + formula_content for keyword in ['tỉ lệ', 'ratio', 'proportion']):
                    formula_types.add("ratio_calculation")
                if any(keyword in formula_name + formula_content for keyword in ['nồng độ', 'concentration', 'molarity']):
                    formula_types.add("concentration_calculation")
                if any(keyword in formula_name + formula_content for keyword in ['thể tích', 'volume', 'v =']):
                    formula_types.add("volume_calculation")
                if any(keyword in formula_name + formula_content for keyword in ['khối lượng', 'mass', 'm =']):
                    formula_types.add("mass_calculation")
                if any(keyword in formula_name + formula_content for keyword in ['hiệu suất', 'efficiency', 'yield']):
                    formula_types.add("efficiency_calculation")

            # Thêm warnings dựa trên formula types
            for formula_type in formula_types:
                if formula_type == "ratio_calculation":
                    validation_rules["context_warnings"].append("KIỂM TRA tỉ lệ và đơn vị trong tính toán")
                elif formula_type == "concentration_calculation":
                    validation_rules["context_warnings"].append("CHÚ Ý đơn vị thể tích và nồng độ")
                elif formula_type == "volume_calculation":
                    validation_rules["context_warnings"].append("XÁC MINH đơn vị thể tích (L, mL, cm³)")
                elif formula_type == "mass_calculation":
                    validation_rules["context_warnings"].append("PHÂN BIỆT khối lượng thực tế và khối lượng mol")
                elif formula_type == "efficiency_calculation":
                    validation_rules["context_warnings"].append("KIỂM TRA hiệu suất phải ≤ 100%")

        # Thêm ví dụ validation tổng quát từ suggested answers
        if suggested_answers:
            for answer in suggested_answers[:2]:  # Lấy tối đa 2 ví dụ
                context = answer.get('context', '')
                value = answer.get('value', '')
                formula_used = answer.get('formula_used', '')

                if context and value:
                    # Tạo ví dụ tổng quát không hardcode
                    validation_rules["validation_examples"].append({
                        "scenario": f"Khi tính {context}",
                        "expected_answer": value,
                        "formula_reference": formula_used if formula_used else "Áp dụng công thức phù hợp",
                        "general_warning": "Đảm bảo đơn vị và công thức chính xác, tránh nhầm lẫn với các đại lượng khác"
                    })

        # Format thành JSON string dễ đọc
        import json
        validation_json = json.dumps(validation_rules, ensure_ascii=False, indent=2)

        return f"""
VALIDATION RULES (JSON FORMAT):
{validation_json}

LƯU Ý: Hãy tuân thủ nghiêm ngặt các rules trên khi tạo câu hỏi.
Đặc biệt chú ý đến context_warnings và validation_examples dựa trên nội dung bài học cụ thể.
"""

    def _create_reverse_thinking_prompt(self, level: str, content: str, lesson_id: str) -> str:
        """Tạo prompt cho quy trình tư duy ngược"""
        requirements = self._get_reverse_thinking_requirements(level)

        prompt = f"""
Bạn là chuyên gia tạo đề thi Hóa học THPT 2025. Hãy áp dụng phương pháp TƯ DUY NGƯỢC để tạo câu hỏi tự luận tính toán.

QUY TRÌNH TƯ DUY NGƯỢC VỚI VALIDATION:
1. SINH ĐÁP ÁN TRƯỚC: Tạo một đáp án số thực dương phù hợp với phiếu trắc nghiệm THPT 2025
2. XÂY DỰNG NGƯỢC: Từ đáp án đó, thiết kế bối cảnh và nội dung câu hỏi
3. TỰ KIỂM TRA: Tính toán ngược từ dữ kiện để xác minh đáp án
4. ĐIỀU CHỈNH: Nếu không khớp, sửa dữ kiện hoặc đáp án

THÔNG TIN BÀI HỌC:
- Lesson ID: {lesson_id}
- Nội dung: {content}

YÊU CẦU MỨC ĐỘ "{level}":
{requirements}

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

LƯU Ý QUAN TRỌNG VỀ ĐÁP ÁN VÀ LÀM TRÒN:
- target_answer phải có ÍT HƠN 5 ký tự để phù hợp với phiếu trắc nghiệm THPT 2025
- Điều chỉnh dữ kiện đề bài (khối lượng, thể tích, nồng độ) để đáp án <5 ký tự
- CHIẾN LƯỢC LÀM TRÒN THÔNG MINH:
  * Nếu kết quả tính toán chính xác là 307.45 nhưng muốn đáp án là 306:
    → Thêm "(làm tròn đến số nguyên)" vào câu hỏi
    → Giải thích trong explanation: "Kết quả chính xác là 307.45, làm tròn đến số nguyên: 306"
  * Nếu kết quả là 22.37 nhưng muốn đáp án là 22.4:
    → Thêm "(làm tròn đến 1 chữ số thập phân)" vào câu hỏi
  * Luôn giải thích rõ ràng việc làm tròn trong explanation
- KHÔNG được sửa đáp án sau khi tính toán - phải thêm yêu cầu làm tròn vào đề

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

Lưu ý: Chỉ trả về JSON sau khi đã VALIDATION HOÀN TOÀN. KHÔNG ĐƯỢC TRẢ VỀ CÂU HỎI SAI!
"""
        return prompt

    def _get_reverse_thinking_requirements(self, level: str) -> str:
        """Yêu cầu cụ thể cho từng mức độ trong tư duy ngược"""
        requirements = {
            DifficultyLevel.KNOWLEDGE.value: """
- Đáp án: Số đơn giản <5 ký tự, chính xác theo tính toán hóa học
- Bối cảnh: Áp dụng trực tiếp công thức cơ bản (n=m/M, C=n/V, pH=-log[H⁺])
- Ví dụ đáp án hợp lệ: "2.24", "5.6", "12", "0.5", "22.4"
- Điều chỉnh dữ kiện để đáp án <5 ký tự
""",
            DifficultyLevel.COMPREHENSION.value: """
- Đáp án: Số vừa phải <5 ký tự, chính xác theo tính toán hóa học
- Bối cảnh: Cần hiểu bản chất phản ứng, áp dụng 2-3 bước tính toán
- Ví dụ đáp án hợp lệ: "16.2", "1.25", "48.6", "3.75"
- Điều chỉnh dữ kiện để đáp án <5 ký tự
""",
            DifficultyLevel.APPLICATION.value: """
- Đáp án: Số phức tạp <5 ký tự, chính xác theo tính toán hóa học
- Bối cảnh: Bài toán nhiều bước, hiệu suất, hỗn hợp, quy trình công nghiệp
- Ví dụ đáp án hợp lệ: "125", "87.5", "2450", "67.8"
- Điều chỉnh dữ kiện để đáp án <5 ký tự
"""
        }
        return requirements.get(level, requirements[DifficultyLevel.KNOWLEDGE.value])

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

KIỂM TRA NGHIÊM NGẶT - CÁC LOẠI LỖI THƯỜNG GẶP:
1. LỖI TÍNH TOÁN:
   - Sai khối lượng mol (VD: CO₂ = 44, không phải 45)
   - Sai công thức hóa học (VD: amine CₙH₂ₙ₊₃N)
   - Sai tỉ lệ mol trong phương trình phản ứng
   - Sai đơn vị (L vs mL, g vs kg)

2. LỖI LOGIC HÓA HỌC:
   - Phương trình không cân bằng
   - Hiệu suất > 100% (không hợp lý)
   - Nồng độ âm hoặc quá lớn
   - Thể tích khí âm hoặc không hợp lý

3. LỖI DỮ KIỆN:
   - Thiếu thông tin cần thiết
   - Dữ kiện mâu thuẫn với nhau
   - Đáp án không khớp với tính toán

NGUYÊN TẮC KIỂM TRA:
- Áp dụng các định luật bảo toàn (khối lượng, nguyên tố, điện tích)
- Phương trình phản ứng phải cân bằng chính xác
- Tỉ lệ mol theo hệ số cân bằng (KHÔNG DÙNG TỈ LỆ KHỐI LƯỢNG)
- Khối lượng mol tính đúng theo bảng tuần hoàn
- Giá trị kết quả trong khoảng hợp lý và thực tế

KIỂM TRA TÍNH TOÁN CHI TIẾT:
- Thực hiện từng phép tính một cách cụ thể với số liệu chính xác
- Kiểm tra đơn vị trong mỗi bước
- Xác minh tỉ lệ mol và hiệu suất
- So sánh kết quả tính được với đáp án cho trước
- Nếu sai lệch >10%, đánh giá là KHÔNG HỢP LỆ

ĐỊNH DẠNG JSON TRẢ VỀ:
{{
    "my_solution": "Lời giải chi tiết của bạn với từng bước tính toán cụ thể",
    "my_answer": "Đáp án bạn tính được (số cụ thể)",
    "answer_difference_percent": "Phần trăm sai lệch so với đáp án cho trước",
    "is_valid": true/false,
    "accuracy_score": "Điểm từ 1-10",
    "error_type": "calculation/logic/data/none",
    "feedback": "Góp ý cụ thể về lỗi phát hiện",
    "suggested_improvements": [
        "Cải thiện cụ thể 1",
        "Cải thiện cụ thể 2"
    ]
}}

LƯU Ý QUAN TRỌNG:
- Nếu sai lệch >10% giữa đáp án tính được và đáp án cho trước → is_valid = false
- Nếu có lỗi logic hóa học nghiêm trọng → is_valid = false
- Nếu dữ kiện mâu thuẫn → is_valid = false
- Hãy nghiêm túc và chính xác trong đánh giá, không khoan dung với lỗi sai.
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

            # Kiểm tra explanation có chứa thông báo lỗi không
            explanation = question_data.get("explanation", "")
            if any(error_phrase in explanation.lower() for error_phrase in [
                "đề bài sai", "không thể tạo", "không hợp lệ", "cần có dữ kiện khác",
                "không thành công", "cố gắng chỉnh sửa", "thất bại"
            ]):
                logger.warning(f"❌ REJECTING: Question contains error message in explanation: {explanation[:100]}...")
                return None

            # Validate đáp án là số hợp lệ và có độ dài phù hợp với phiếu trắc nghiệm
            target_answer = str(question_data["target_answer"]).strip()
            logger.info(f"🔍 Validating answer: '{target_answer}' (length: {len(target_answer)} chars)")

            try:
                # Kiểm tra đáp án có phải là số hợp lệ không
                answer_value = float(target_answer)

                # Kiểm tra đáp án có hợp lý không (không âm, không quá lớn)
                if answer_value <= 0 or answer_value > 9999:
                    logger.warning(f"❌ REJECTING: Answer value out of reasonable range: {answer_value}")
                    return None

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

            # Log thông tin về công thức được sử dụng nếu có
            if "formula_used" in question_data:
                logger.info(f"📐 Formula used: {question_data['formula_used']}")

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
            1: "PART I: Trắc nghiệm nhiều lựa chọn (A, B, C, D) - Hỗ trợ mức độ KNOWLEDGE, COMPREHENSION, APPLICATION: 18 câu đa dạng từ nhận biết đến tính toán đơn giản",
            2: "PART II: Trắc nghiệm Đúng/Sai - Hỗ trợ mức độ KNOWLEDGE, COMPREHENSION, APPLICATION: 4 câu lớn, mỗi câu có 4 phát biểu a,b,c,d để đánh giá",
            3: "PART III: Tự luận tính toán - Hỗ trợ mức độ KNOWLEDGE, COMPREHENSION, APPLICATION: Bài toán tính toán từ cơ bản đến phức tạp, đòi hỏi tư duy và tổng hợp kiến thức"
        }

        prompt = f"""
Bạn là chuyên gia tạo đề thi {subject} theo chuẩn THPT 2025.

🎯 YÊU CẦU CHÍNH XÁC:
- Tạo ĐÚNG {count} câu hỏi (không nhiều hơn, không ít hơn)
- Mức độ nhận thức: "{level}"
- Phần {part_num}: {self._get_part_description(part_num)}

📚 THÔNG TIN BÀI HỌC:
{content_preview}

📋 HƯỚNG DẪN CHI TIẾT:
{self._get_specific_instructions_by_part(part_num, level)}

🔧 ĐỊNH DẠNG JSON BẮT BUỘC:
[
    {{
        "question": "Nội dung câu hỏi chi tiết",
        "answer": {self._get_answer_format_by_part(part_num)},
        "explanation": "Giải thích từng bước với công thức và tính toán cụ thể",
        "cognitive_level": "{level}",
        "part": {part_num}
    }},
    {{
        "question": "Câu hỏi thứ 2...",
        "answer": {self._get_answer_format_by_part(part_num)},
        "explanation": "Giải thích chi tiết...",
        "cognitive_level": "{level}",
        "part": {part_num}
    }}
]

⚠️ LƯU Ý QUAN TRỌNG:
1. PHẢI TRẢ VỀ ĐÚNG {count} CÂU HỎI trong JSON array
2. CHỈ trả về JSON array, KHÔNG có text khác
3. Mỗi câu hỏi phải có đầy đủ các field: question, answer, explanation, cognitive_level, part
4. Kiểm tra logic hóa học: phương trình, tỉ lệ mol, bảo toàn nguyên tố
5. Đảm bảo tính chính xác khoa học và phù hợp thực tế

✅ VALIDATION CHECKLIST:
- Khối lượng mol chính xác (CaCO₃=100, NaCl=58.5, H₂SO₄=98...)
- Công thức phân tử nhất quán
- Tỉ lệ mol theo phương trình cân bằng
- Bảo toàn nguyên tố
- Giá trị số học hợp lý

BẮT ĐẦU TẠO {count} CÂU HỎI:
"""
        return prompt

    def _get_part_description(self, part_num: int) -> str:
        """Get detailed description for each part theo chuẩn THPT 2025"""
        descriptions = {
            1: "Trắc nghiệm nhiều phương án (Hỗ trợ KNOWLEDGE, COMPREHENSION, APPLICATION)",
            2: "Trắc nghiệm đúng/sai (Hỗ trợ KNOWLEDGE, COMPREHENSION, APPLICATION)",
            3: "Tự luận tính toán (Hỗ trợ KNOWLEDGE, COMPREHENSION, APPLICATION)"
        }
        return descriptions.get(part_num, "")

    def _get_specific_instructions_by_part(self, part_num: int, level: str) -> str:
        """Hướng dẫn cụ thể cho từng phần theo chuẩn THPT 2025"""
        if part_num == 1:
            if level == DifficultyLevel.KNOWLEDGE.value:
                return """
HƯỚNG DẪN PHẦN I - MỨC ĐỘ KNOWLEDGE:
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
            elif level == DifficultyLevel.COMPREHENSION.value:
                return """
HƯỚNG DẪN PHẦN I - MỨC ĐỘ COMPREHENSION:
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
            elif level == DifficultyLevel.APPLICATION.value:
                return """
HƯỚNG DẪN PHẦN I - MỨC ĐỘ APPLICATION:
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
            if level == DifficultyLevel.KNOWLEDGE.value:
                return """
HƯỚNG DẪN PHẦN II - MỨC ĐỘ KNOWLEDGE:
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
            elif level == DifficultyLevel.COMPREHENSION.value:
                return """
HƯỚNG DẪN PHẦN II - MỨC ĐỘ COMPREHENSION:

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
            elif level == DifficultyLevel.APPLICATION.value:
                return """
HƯỚNG DẪN PHẦN II - MỨC ĐỘ APPLICATION:
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
            if level == DifficultyLevel.KNOWLEDGE.value:
                return """
HƯỚNG DẪN PHẦN III - MỨC ĐỘ KNOWLEDGE:
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
            elif level == DifficultyLevel.COMPREHENSION.value:
                return """
HƯỚNG DẪN PHẦN III - MỨC ĐỘ COMPREHENSION:
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
            elif level == DifficultyLevel.APPLICATION.value:
                return """
HƯỚNG DẪN PHẦN III - MỨC ĐỘ APPLICATION:
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
            difficulty_counts = {
                DifficultyLevel.KNOWLEDGE.value: 0,
                DifficultyLevel.COMPREHENSION.value: 0,
                DifficultyLevel.APPLICATION.value: 0
            }
            
            for question in questions:
                part = question.get("part", 1)
                part_counts[part] += 1

                level = question.get("cognitive_level", DifficultyLevel.KNOWLEDGE.value)
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
                difficulty_distribution={
                    DifficultyLevel.KNOWLEDGE.value: 0,
                    DifficultyLevel.COMPREHENSION.value: 0,
                    DifficultyLevel.APPLICATION.value: 0
                },
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
