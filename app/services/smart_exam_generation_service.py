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

            # Tạo câu hỏi cho từng phần - truyền lesson_data thay vì actual_content
            for part in lesson_matrix.parts:
                # Debug logging
                total_expected = part.objectives.Biết + part.objectives.Hiểu + part.objectives.Vận_dụng
                logger.info(f"[DEBUG] Processing Part {part.part}: Expected {total_expected} questions (Biết:{part.objectives.Biết}, Hiểu:{part.objectives.Hiểu}, Vận_dụng:{part.objectives.Vận_dụng})")

                part_questions = await self._generate_questions_for_part(
                    part, lesson_data, subject, lesson_id
                )

                logger.info(f"[DEBUG] Part {part.part} generated {len(part_questions)} questions")
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

            # Tạo câu hỏi theo ma trận đa dạng THPT 2025 - hỗ trợ tất cả mức độ cho Phần 1 và 2
            if part_num == 1:
                # Phần I: Trắc nghiệm nhiều lựa chọn - hỗ trợ Biết, Hiểu, Vận dụng
                for level, count in [("Biết", objectives.Biết), ("Hiểu", objectives.Hiểu), ("Vận_dụng", objectives.Vận_dụng)]:
                    if count > 0:
                        level_questions = await self._generate_questions_for_level(
                            part_num, level, count, lesson_data, subject, lesson_id
                        )
                        part_questions.extend(level_questions)
            elif part_num == 2:
                # Phần II: Trắc nghiệm Đúng/Sai - hỗ trợ Biết, Hiểu, Vận dụng
                for level, count in [("Biết", objectives.Biết), ("Hiểu", objectives.Hiểu), ("Vận_dụng", objectives.Vận_dụng)]:
                    if count > 0:
                        level_questions = await self._generate_questions_for_level(
                            part_num, level, count, lesson_data, subject, lesson_id
                        )
                        part_questions.extend(level_questions)
            elif part_num == 3:
                # Phần III: Tự luận tính toán - hỗ trợ Biết, Hiểu, Vận dụng
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
            # Tạo prompt cho LLM
            prompt = self._create_prompt_for_level(
                part_num, level, count, lesson_data, subject, lesson_id
            )   
            print(f"Generated prompt: {prompt}")

            # Gọi LLM để tạo câu hỏi - tăng max_tokens cho nhiều câu hỏi
            max_tokens = 6000 if count > 3 else 4000  # Tăng token limit cho nhiều câu
            response = await self.llm_service.generate_content(
                prompt=prompt,
                temperature=0.3,
                max_tokens=max_tokens
            )
            print(f"LLM response: {response}")
            if not response.get("success", False):
                logger.error(f"LLM failed for part {part_num}, level {level}: {response.get('error')}")
                return []

            # Parse response JSON
            questions = self._parse_llm_response(response.get("text", ""), part_num, level, lesson_id)
            print(f"Parsed questions: {questions}")

            # Debug logging
            logger.info(f"[DEBUG] Part {part_num}, Level {level}: Requested {count} questions, LLM generated {len(questions)} questions")

            # Giới hạn số câu hỏi theo yêu cầu
            limited_questions = questions[:count]
            logger.info(f"[DEBUG] Part {part_num}, Level {level}: Returning {len(limited_questions)} questions after limit")

            return limited_questions

        except Exception as e:
            logger.error(f"Error generating questions for level {level}: {e}")
            return []



    def _create_prompt_for_level(
        self, part_num: int, level: str, count: int,
        lesson_data: Dict[str, Any], subject: str, lesson_id: str
    ) -> str:
        """Create prompt for LLM according to THPT 2025 standards"""

        # Lấy nội dung bài học từ textbook_retrieval_service format
        main_content = ""
        lesson_info = {}

        if "lesson_content" in lesson_data:
            # Từ textbook_retrieval_service
            main_content = lesson_data.get("lesson_content", "")
            lesson_info = {
                "lesson_id": lesson_data.get("lesson_id", lesson_id),
                "book_id": lesson_data.get("book_id", ""),
                "collection_name": lesson_data.get("collection_name", ""),
                "total_chunks": lesson_data.get("total_chunks", 0)
            }
        else:
            # Fallback cho format cũ
            main_content = lesson_data.get("main_content", "")
            lesson_info = lesson_data.get("lesson_info", {})

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
{self._get_specific_instructions_by_part(part_num, level)}

ĐỊNH DẠNG JSON TRẢ VỀ:
[
    {{
        "question": "Nội dung câu hỏi",
        "answer": {self._get_answer_format_by_part(part_num)},
        "explanation": "Giải thích đáp án",
        "cognitive_level": "{level}",
        "part": {part_num}
    }}
]

Lưu ý: chỉ trả về JSON, không có văn bản bổ sung.
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
                print(f"Question ne xt: {question}")
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




