"""
Service để tạo câu hỏi thi sử dụng Gemini LLM
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional
from app.services.llm_service import LLMService
from app.models.exam_models import (
    ExamMatrixRequest,
    CauHoiModel,
    ExamResponse,
    ExamStatistics,
    MucDoModel,
    NoiDungModel,
    CauHinhDeModel,
)
from datetime import datetime

logger = logging.getLogger(__name__)


class ExamGenerationService:
    """Service để tạo câu hỏi thi từ ma trận đề thi"""

    def __init__(self):
        self.llm_service = LLMService()

    async def generate_questions_from_matrix(
        self, exam_request: ExamMatrixRequest, lesson_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Tạo câu hỏi từ ma trận đề thi và nội dung bài học

        Args:
            exam_request: Ma trận đề thi
            lesson_content: Nội dung bài học từ Qdrant

        Returns:
            Dict chứa danh sách câu hỏi đã tạo
        """
        try:
            # Debug logging
            logger.info(f"=== EXAM GENERATION DEBUG ===")
            logger.info(f"Lesson ID: {exam_request.lesson_id}")
            logger.info(f"Subject: {exam_request.mon_hoc}")
            logger.info(f"Grade: {exam_request.lop}")
            logger.info(f"Total questions requested: {exam_request.tong_so_cau}")
            logger.info(f"Number of cau_hinh_de: {len(exam_request.cau_hinh_de)}")

            if not self.llm_service.model:
                logger.error(
                    "LLM service model is None - check Gemini API configuration"
                )
                return {"success": False, "error": "LLM service not available"}

            logger.info(f"LLM service is available: {self.llm_service.is_available()}")
            logger.info(f"Lesson content keys: {list(lesson_content.keys())}")

            # Tạo câu hỏi cho từng bài trong cấu hình đề
            all_questions = []
            question_counter = 1

            for i, cau_hinh in enumerate(exam_request.cau_hinh_de):
                logger.info(
                    f"Processing cau_hinh {i+1}/{len(exam_request.cau_hinh_de)}: {cau_hinh.bai}"
                )
                logger.info(
                    f"Number of noi_dung in this cau_hinh: {len(cau_hinh.noi_dung)}"
                )

                bai_questions = await self._generate_questions_for_bai(
                    cau_hinh, lesson_content, question_counter
                )
                logger.info(
                    f"Generated {len(bai_questions)} questions for bai: {cau_hinh.bai}"
                )
                all_questions.extend(bai_questions)
                question_counter += len(bai_questions)

            logger.info(f"Total questions generated: {len(all_questions)}")

            # Kiểm tra nếu không có câu hỏi nào được tạo
            if len(all_questions) == 0:
                logger.error("No questions were generated. This might indicate an API issue.")
                return {"success": False, "error": "Không thể tạo câu hỏi. Vui lòng kiểm tra API key hoặc thử lại sau."}

            # Tạo thống kê
            statistics = self._create_exam_statistics(all_questions, exam_request)

            # Tạo exam ID an toàn (không có ký tự đặc biệt)
            lesson_id_safe = self._sanitize_id(exam_request.lesson_id)
            exam_id = (
                f"exam_{lesson_id_safe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )

            logger.info(f"=== EXAM GENERATION COMPLETED ===")
            return {
                "success": True,
                "exam_id": exam_id,
                "questions": all_questions,
                "statistics": statistics,
                "total_generated": len(all_questions),
            }

        except Exception as e:
            logger.error(f"Error generating questions from matrix: {e}")
            return {"success": False, "error": str(e)}

    async def _generate_questions_for_bai(
        self,
        cau_hinh: CauHinhDeModel,
        lesson_content: Dict[str, Any],
        start_counter: int,
    ) -> List[Dict[str, Any]]:
        """Tạo câu hỏi cho một bài cụ thể"""
        try:
            logger.info(f"--- Generating questions for bai: {cau_hinh.bai} ---")
            bai_questions = []
            current_counter = start_counter

            for i, noi_dung in enumerate(cau_hinh.noi_dung):
                logger.info(
                    f"Processing noi_dung {i+1}/{len(cau_hinh.noi_dung)}: {noi_dung.ten_noi_dung}"
                )
                logger.info(
                    f"Number of muc_do in this noi_dung: {len(noi_dung.muc_do)}"
                )

                for j, muc_do in enumerate(noi_dung.muc_do):
                    logger.info(
                        f"Processing muc_do {j+1}/{len(noi_dung.muc_do)}: {muc_do.loai} ({muc_do.so_cau} questions)"
                    )
                    logger.info(f"Question types for this muc_do: {muc_do.loai_cau}")

                    # Tạo câu hỏi cho từng loại câu trong mức độ này
                    for k, loai_cau in enumerate(muc_do.loai_cau):
                        logger.info(
                            f"Generating {loai_cau} questions for {muc_do.loai} level..."
                        )
                        questions = await self._generate_questions_by_type(
                            noi_dung,
                            muc_do,
                            loai_cau,
                            lesson_content,
                            cau_hinh.bai,
                            current_counter,
                        )
                        logger.info(f"Generated {len(questions)} {loai_cau} questions")
                        bai_questions.extend(questions)
                        current_counter += len(questions)

            logger.info(
                f"Total questions generated for bai '{cau_hinh.bai}': {len(bai_questions)}"
            )
            return bai_questions

        except Exception as e:
            logger.error(f"Error generating questions for bai: {e}")
            return []

    async def _generate_questions_by_type(
        self,
        noi_dung: NoiDungModel,
        muc_do: MucDoModel,
        loai_cau: str,
        lesson_content: Dict[str, Any],
        bai_name: str,
        start_counter: int,
    ) -> List[Dict[str, Any]]:
        """Tạo câu hỏi theo loại cụ thể"""
        try:
            logger.info(
                f"*** Generating {loai_cau} questions for {muc_do.loai} level ***"
            )
            logger.info(f"Content topic: {noi_dung.ten_noi_dung}")
            logger.info(f"Number of questions to generate: {muc_do.so_cau}")

            # Kiểm tra LLM service
            if not self.llm_service.model:
                logger.error("LLM service model is None in _generate_questions_by_type")
                return []

            # Tạo prompt dựa trên loại câu hỏi và mức độ
            prompt = self._create_question_prompt(
                noi_dung, muc_do, loai_cau, lesson_content, bai_name
            )

            logger.info(f"Generated prompt length: {len(prompt)} characters")
            logger.debug(f"Prompt preview: {prompt[:500]}...")

            # Gọi Gemini để tạo câu hỏi
            logger.info("Calling Gemini API...")
            try:
                response = self.llm_service.model.generate_content(prompt)
                response_text = response.text.strip()

                logger.info(f"Gemini response length: {len(response_text)} characters")
                logger.debug(f"Gemini response preview: {response_text[:500]}...")

                # Parse JSON response
                logger.info("Parsing JSON response...")
                questions_data = self._parse_questions_response(response_text)
                logger.info(f"Parsed {len(questions_data)} questions from response")

            except Exception as api_error:
                logger.error(f"Gemini API error: {api_error}")
                error_msg = str(api_error).lower()

                # Kiểm tra các loại lỗi cụ thể
                if "quota" in error_msg or "resource_exhausted" in error_msg or "limit" in error_msg:
                    raise Exception("API quota đã hết. Vui lòng kiểm tra lại API key hoặc thử lại sau.")
                elif "api key" in error_msg or "authentication" in error_msg or "permission" in error_msg:
                    raise Exception("API key không hợp lệ hoặc không có quyền truy cập.")
                elif "invalid" in error_msg or "bad request" in error_msg:
                    raise Exception("Yêu cầu không hợp lệ. Vui lòng kiểm tra lại dữ liệu đầu vào.")
                else:
                    # Lỗi khác, trả về lỗi gốc
                    raise api_error

            # Format câu hỏi theo model
            formatted_questions = []
            for i, q_data in enumerate(questions_data):
                dap_an = q_data.get("dap_an", {})

                # Sửa lỗi thiếu trường "dung" cho câu trắc nghiệm
                if loai_cau == "TN" and isinstance(dap_an, dict) and "dung" not in dap_an:
                    logger.warning(f"Missing 'dung' field in question {i+1}, trying to extract from explanation")
                    correct_answer = self._extract_correct_answer_from_explanation(
                        q_data.get("giai_thich", ""), dap_an
                    )
                    if correct_answer:
                        dap_an["dung"] = correct_answer
                        logger.info(f"Extracted correct answer: {correct_answer}")
                    else:
                        logger.error(f"Could not extract correct answer for question {i+1}")

                question = {
                    "stt": start_counter + i,
                    "loai_cau": loai_cau,
                    "muc_do": muc_do.loai,
                    "noi_dung_cau_hoi": q_data.get("cau_hoi", ""),
                    "dap_an": dap_an,
                    "giai_thich": q_data.get("giai_thich", ""),
                    "bai_hoc": bai_name,
                    "noi_dung_kien_thuc": noi_dung.ten_noi_dung,
                }
                formatted_questions.append(question)
                logger.debug(
                    f"Formatted question {i+1}: {q_data.get('cau_hoi', '')[:100]}..."
                )

            logger.info(f"Successfully formatted {len(formatted_questions)} questions")
            return formatted_questions

        except Exception as e:
            logger.error(f"Error generating questions by type: {e}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            # Ném lại exception thay vì trả về danh sách rỗng
            raise e

    def _create_question_prompt(
        self,
        noi_dung: NoiDungModel,
        muc_do: MucDoModel,
        loai_cau: str,
        lesson_content: Dict[str, Any],
        bai_name: str,
    ) -> str:
        """Tạo prompt cho Gemini để tạo câu hỏi"""

        # Lấy nội dung bài học
        main_content = lesson_content.get("content", {}).get("main_content", "")
        lesson_info = lesson_content.get("content", {}).get("lesson_info", {})

        # Template prompt cơ bản
        base_prompt = f"""
Bạn là một chuyên gia giáo dục và ra đề thi chuyên nghiệp. Hãy tạo câu hỏi kiểm tra cho học sinh.

THÔNG TIN BÀI HỌC:
- Bài học: {bai_name}
- Chương: {lesson_info.get('chapter_title', '')}
- Nội dung kiến thức: {noi_dung.ten_noi_dung}
- Yêu cầu cần đạt: {noi_dung.yeu_cau_can_dat}

NỘI DUNG BÀI HỌC:
{main_content[:2000]}...

YÊU CẦU TẠO CÂU HỎI:
- Loại câu hỏi: {self._get_question_type_description(loai_cau)}
- Mức độ nhận thức: {muc_do.loai}
- Số lượng câu hỏi: {muc_do.so_cau}

{self._get_specific_prompt_by_type(loai_cau, muc_do.loai)}

ĐỊNH DẠNG TRาาẢ LỜI (JSON):
[
    {{
        "cau_hoi": "Nội dung câu hỏi",
        "dap_an": {self._get_answer_format_by_type(loai_cau)},
        "giai_thich": "Giải thích đáp án"
    }}
]

QUAN TRỌNG - ĐỊNH DẠNG BẮT BUỘC:
- Với câu trắc nghiệm (TN), BẮT BUỘC phải có trường "dung" trong dap_an để chỉ ra đáp án đúng (A, B, C hoặc D)
- Ví dụ: "dap_an": {{"A": "...", "B": "...", "C": "...", "D": "...", "dung": "A"}}
- Trường "dung" phải chứa chính xác một trong các giá trị: "A", "B", "C", "D"
- Đáp án trong trường "dung" phải khớp với nội dung giải thích
- KHÔNG BAO GIỜ để trống trường "dung" - luôn phải chỉ rõ đáp án đúng
- Trong phần "giai_thich", hãy bắt đầu bằng "Đáp án: [A/B/C/D]" để rõ ràng

Hãy tạo {muc_do.so_cau} câu hỏi chất lượng cao, phù hợp với mức độ {muc_do.loai}.
"""
        return base_prompt

    def _get_question_type_description(self, loai_cau: str) -> str:
        """Mô tả loại câu hỏi"""
        descriptions = {
            "TN": "Trắc nghiệm nhiều lựa chọn (4 đáp án A, B, C, D)",
            "DT": "Điền từ/cụm từ vào chỗ trống",
            "DS": "Đúng/Sai với 4 ý nhỏ",
            "TL": "Tự luận ngắn",
        }
        return descriptions.get(loai_cau, "Không xác định")

    def _get_specific_prompt_by_type(self, loai_cau: str, muc_do: str) -> str:
        """Tạo prompt cụ thể theo loại câu hỏi và mức độ"""

        if loai_cau == "TN":
            if muc_do == "Nhận biết":
                return """
HƯỚNG DẪN TẠO CÂU TRẮC NGHIỆM NHẬN BIẾT:
- Hỏi về định nghĩa, khái niệm cơ bản
- Nhận biết công thức, ký hiệu
- 4 đáp án rõ ràng, chỉ 1 đáp án đúng
- Tránh câu hỏi mơ hồ hoặc gây nhầm lẫn
- Đảm bảo đáp án đúng phản ánh chính xác kiến thức khoa học
- Các đáp án sai phải hợp lý nhưng rõ ràng là sai
"""
            elif muc_do == "Thông hiểu":
                return """
HƯỚNG DẪN TẠO CÂU TRẮC NGHIỆM THÔNG HIỂU:
- Hỏi về mối quan hệ giữa các khái niệm
- Giải thích hiện tượng, quá trình
- So sánh, phân loại
- Đáp án yêu cầu hiểu biết sâu hơn
"""
            else:  # Vận dụng
                return """
HƯỚNG DẪN TẠO CÂU TRẮC NGHIỆM VẬN DỤNG:
- Áp dụng kiến thức vào tình huống cụ thể
- Giải quyết bài tập, tính toán
- Phân tích, đánh giá
- Đáp án yêu cầu tư duy logic
"""

        elif loai_cau == "DT":
            return """
HƯỚNG DẪN TẠO CÂU ĐIỀN TỪ:
- Tạo câu có chỗ trống (...) hoặc _____
- Từ cần điền phải chính xác, không mơ hồ
- Có thể có nhiều từ đồng nghĩa được chấp nhận
- Độ dài từ cần điền phù hợp với mức độ
"""

        elif loai_cau == "DS":
            return """
HƯỚNG DẪN TẠO CÂU ĐÚNG/SAI:
- Tạo 4 ý nhỏ (a, b, c, d)
- Mỗi ý có thể đúng hoặc sai
- Các ý phải liên quan đến cùng chủ đề
- Tránh ý quá dễ hoặc quá khó
"""

        else:  # TL
            return """
HƯỚNG DẪN TẠO CÂU TỰ LUẬN:
- Câu hỏi mở, yêu cầu trình bày, giải thích
- Có thể chia thành nhiều ý nhỏ
- Đáp án có thể linh hoạt nhưng phải có điểm chính
- Phù hợp với thời gian làm bài
"""

    def _get_answer_format_by_type(self, loai_cau: str) -> str:
        """Định dạng đáp án theo loại câu hỏi"""
        formats = {
            "TN": '{"A": "Đáp án A", "B": "Đáp án B", "C": "Đáp án C", "D": "Đáp án D", "dung": "A"}',
            "DT": '{"dap_an_chinh": "từ cần điền", "dap_an_khac": ["từ đồng nghĩa 1", "từ đồng nghĩa 2"]}',
            "DS": '{"a": true, "b": false, "c": true, "d": false}',
            "TL": '{"y_chinh": ["Ý 1", "Ý 2", "Ý 3"], "diem_toi_da": 2}',
        }
        return formats.get(loai_cau, "{}")

    def _parse_questions_response(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse response từ Gemini thành list câu hỏi"""
        try:
            logger.info("Starting to parse Gemini response...")
            logger.debug(f"Original response text: {response_text}")

            original_text = response_text

            # Clean JSON text
            if response_text.startswith("```json"):
                response_text = response_text[7:]
                logger.debug("Removed ```json prefix")
            if response_text.startswith("```"):
                response_text = response_text[3:]
                logger.debug("Removed ``` prefix")
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                logger.debug("Removed ``` suffix")

            response_text = response_text.strip()
            logger.debug(f"Cleaned response text: {response_text}")

            # Tìm JSON array
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]")

            logger.info(f"JSON array bounds: start={start_idx}, end={end_idx}")

            if start_idx != -1 and end_idx != -1:
                json_text = response_text[start_idx : end_idx + 1]
                logger.info(f"Extracted JSON text length: {len(json_text)}")
                logger.debug(f"JSON text to parse: {json_text}")

                questions = json.loads(json_text)
                logger.info(f"Successfully parsed JSON. Type: {type(questions)}")

                if isinstance(questions, list):
                    logger.info(f"Found {len(questions)} questions in array")
                    return questions
                else:
                    logger.info("Single question object found, converting to list")
                    return [questions]
            else:
                logger.error("No JSON array found in response")
                logger.error(f"Full response text: {original_text}")
                return []

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Failed to parse JSON: {response_text}")
            return []
        except Exception as e:
            logger.error(f"Error parsing questions response: {e}")
            logger.error(f"Response text: {response_text}")
            return []

    def _create_exam_statistics(
        self, questions: List[Dict[str, Any]], exam_request: ExamMatrixRequest
    ) -> Dict[str, Any]:
        """Tạo thống kê cho đề thi"""
        try:
            # Thống kê theo loại câu hỏi
            loai_count = {}
            muc_do_count = {}
            bai_count = {}

            for q in questions:
                # Đếm theo loại câu
                loai = q.get("loai_cau", "")
                loai_count[loai] = loai_count.get(loai, 0) + 1

                # Đếm theo mức độ
                muc_do = q.get("muc_do", "")
                muc_do_count[muc_do] = muc_do_count.get(muc_do, 0) + 1

                # Đếm theo bài
                bai = q.get("bai_hoc", "")
                bai_count[bai] = bai_count.get(bai, 0) + 1

            return {
                "tong_so_cau": len(questions),
                "phan_bo_theo_loai": loai_count,
                "phan_bo_theo_muc_do": muc_do_count,
                "phan_bo_theo_bai": bai_count,
                "mon_hoc": exam_request.mon_hoc,
                "lop": exam_request.lop,
            }

        except Exception as e:
            logger.error(f"Error creating exam statistics: {e}")
            return {}



    def _sanitize_id(self, id_string: str) -> str:
        """
        Làm sạch ID để tránh lỗi encoding

        Args:
            id_string: ID gốc

        Returns:
            ID đã được làm sạch (chỉ chứa ASCII)
        """
        try:
            # Loại bỏ ký tự đặc biệt và dấu tiếng Việt
            # Chỉ giữ lại chữ cái, số, dấu gạch dưới và gạch ngang
            sanitized = re.sub(r"[^\w\-_]", "_", id_string)

            # Loại bỏ nhiều dấu gạch dưới liên tiếp
            sanitized = re.sub(r"_+", "_", sanitized)

            # Loại bỏ dấu gạch dưới ở đầu và cuối
            sanitized = sanitized.strip("_")

            # Nếu kết quả rỗng, dùng default
            if not sanitized:
                sanitized = "lesson"

            return sanitized

        except Exception as e:
            logger.warning(f"Error sanitizing ID '{id_string}': {e}")
            return "lesson"

    def _extract_correct_answer_from_explanation(self, explanation: str, dap_an: dict) -> str:
        """Trích xuất đáp án đúng từ giải thích"""
        try:
            if not explanation or not isinstance(dap_an, dict):
                return ""

            explanation_lower = explanation.lower()
            logger.debug(f"Analyzing explanation: {explanation[:100]}...")

            # Tìm các pattern rõ ràng nhất trước (pattern có từ "đúng")
            strong_patterns = [
                r"đáp án ([abcd]) đúng",
                r"đáp án đúng là ([abcd])",
                r"([abcd]) đúng vì",
                r"([abcd]) là đáp án đúng",
                r"([abcd]) đúng",
                r"chọn đáp án ([abcd])",
                r"đáp án:\s*([abcd])",
                r"đáp án\s+([abcd])"
            ]

            for pattern in strong_patterns:
                match = re.search(pattern, explanation_lower)
                if match:
                    answer = match.group(1).upper()
                    if answer in dap_an:
                        logger.info(f"Found correct answer '{answer}' using strong pattern: {pattern}")
                        return answer

            # Tìm pattern yếu hơn (chỉ đề cập đến đáp án)
            weak_patterns = [
                r"đáp án ([abcd])",
                r"chọn ([abcd])",
                r"([abcd])\s*[:\-\.]",
                r"^([abcd])\s",
                r"\b([abcd])\b.*chính xác",
                r"\b([abcd])\b.*đúng"
            ]

            for pattern in weak_patterns:
                match = re.search(pattern, explanation_lower)
                if match:
                    answer = match.group(1).upper()
                    if answer in dap_an:
                        logger.info(f"Found correct answer '{answer}' using weak pattern: {pattern}")
                        return answer

            # Phân tích ngữ cảnh thông minh hơn
            # Tìm các từ khóa chỉ ra đáp án đúng
            context_keywords = [
                'đúng', 'chính xác', 'phù hợp', 'là', 'vì', 'do', 'bởi vì',
                'nên', 'nó', 'điều này', 'vậy', 'như vậy'
            ]

            # Tách thành các câu và phân tích
            sentences = explanation_lower.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if any(keyword in sentence for keyword in context_keywords):
                    # Tìm đáp án được nhắc đến trong câu này
                    for option in ['A', 'B', 'C', 'D']:
                        if option.lower() in sentence and option in dap_an:
                            # Kiểm tra xem có phải đang nói về đáp án đúng không
                            if any(keyword in sentence for keyword in ['đúng', 'chính xác', 'phù hợp']):
                                logger.info(f"Found correct answer '{option}' by context analysis in sentence: {sentence[:50]}...")
                                return option

            # Nếu vẫn không tìm thấy, thử phân tích nội dung đáp án
            # Tìm đáp án có nội dung được nhắc đến nhiều nhất trong giải thích
            option_scores = {}
            for option, content in dap_an.items():
                if option in ['A', 'B', 'C', 'D'] and isinstance(content, str):
                    # Đếm số từ khóa từ nội dung đáp án xuất hiện trong giải thích
                    content_words = content.lower().split()
                    score = 0
                    for word in content_words:
                        if len(word) > 2 and word in explanation_lower:  # Chỉ đếm từ có ý nghĩa
                            score += 1
                    option_scores[option] = score

            if option_scores:
                best_option = max(option_scores.keys(), key=lambda x: option_scores[x])
                if option_scores[best_option] > 0:
                    logger.info(f"Found correct answer '{best_option}' by content analysis with score: {option_scores[best_option]}")
                    return best_option

            logger.warning("Could not extract correct answer from explanation")
            logger.debug(f"Full explanation: {explanation}")
            logger.debug(f"Available options: {list(dap_an.keys())}")
            return ""

        except Exception as e:
            logger.error(f"Error extracting correct answer: {e}")
            return ""


# Tạo instance global
exam_generation_service = ExamGenerationService()
