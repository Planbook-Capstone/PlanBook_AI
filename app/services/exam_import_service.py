"""
Service để import đề thi từ file DOCX và chuyển đổi thành JSON
"""

import logging
import json
import re
import time
from typing import Dict, Any, Optional
from docx import Document
import io

from app.services.openrouter_service import get_openrouter_service
from app.models.exam_import_models import (
    ExamImportRequest,
    ExamImportResponse,
    ExamImportError,
    ImportedExamData,
    ExamImportStatistics
)

logger = logging.getLogger(__name__)


class ExamImportService:
    """Service để import và xử lý đề thi từ file DOCX"""

    def __init__(self):
        self.model_name = "google/gemini-2.0-flash-001"
        logger.info("🔄 ExamImportService: First-time initialization triggered")

    async def import_exam_from_docx_content(
        self, file_content: bytes, filename: str = "exam.docx"
    ) -> Dict[str, Any]:
        """
        Import đề thi từ nội dung file DOCX

        Args:
            file_content: Nội dung file DOCX dưới dạng bytes
            filename: Tên file gốc

        Returns:
            Dict chứa kết quả import
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting DOCX import for file: {filename}")
            
            # 1. Extract text từ DOCX
            extracted_text = self._extract_text_from_docx_bytes(file_content)
            
            if not extracted_text or len(extracted_text.strip()) < 100:
                return ExamImportError(
                    message="File extraction failed",
                    error="Không thể trích xuất nội dung từ file DOCX hoặc nội dung quá ngắn",
                    error_code="EXTRACTION_FAILED",
                    details={"filename": filename, "extracted_length": len(extracted_text)}
                ).model_dump()

            logger.info(f"Extracted {len(extracted_text)} characters from DOCX")

            # 2. Validate format đề thi trước khi gọi LLM
            logger.info("Validating exam format...")
            format_validation = self._validate_exam_format(extracted_text)

            if not format_validation["is_valid"]:
                return ExamImportError(
                    message="Invalid exam format",
                    error=f"Đề thi không đúng format chuẩn: {format_validation['error']}",
                    error_code="INVALID_FORMAT",
                    details={
                        "filename": filename,
                        "validation_details": format_validation["details"]
                    }
                ).model_dump()

            # Lưu thông tin warnings để trả về sau
            format_warnings = format_validation.get("warnings", [])
            missing_parts = format_validation.get("details", {}).get("missing_parts", [])

            logger.info(f"Exam format validation passed with warnings: {format_warnings}")

            # 3. Gửi cho LLM để phân tích và chuyển đổi
            logger.info("Sending content to LLM for analysis...")
            llm_result = await self._analyze_exam_with_llm(extracted_text, filename)

            if not llm_result.get("success", False):
                return ExamImportError(
                    message="LLM analysis failed",
                    error=f"Không thể phân tích đề thi: {llm_result.get('error', 'Unknown error')}",
                    error_code="LLM_ANALYSIS_FAILED",
                    details={"filename": filename}
                ).model_dump()

            # 3. Parse JSON response từ LLM
            exam_data = llm_result.get("data")
            if not exam_data:
                return ExamImportError(
                    message="No exam data returned",
                    error="LLM không trả về dữ liệu đề thi",
                    error_code="NO_EXAM_DATA",
                    details={"filename": filename}
                ).model_dump()

            # 4. Validate và clean dữ liệu từ LLM
            logger.info("Validating and cleaning LLM data...")
            validation_result = self._validate_and_clean_exam_data(exam_data)

            if not validation_result["is_valid"]:
                return ExamImportError(
                    message="Invalid exam data from LLM",
                    error=f"Dữ liệu từ LLM không hợp lệ: {validation_result['error']}",
                    error_code="INVALID_LLM_DATA",
                    details={
                        "filename": filename,
                        "validation_details": validation_result["details"]
                    }
                ).model_dump()

            # Sử dụng dữ liệu đã được clean
            exam_data = validation_result["cleaned_data"]

            # 4. Validate và tạo response
            processing_time = time.time() - start_time

            # Tính toán statistics
            statistics = self._calculate_import_statistics(exam_data)

            # Tạo message với thông tin về các phần thiếu
            success_message = "Đề thi đã được import thành công"
            if format_warnings:
                success_message += f" (Lưu ý: {'; '.join(format_warnings)})"

            # Tạo response với thông tin bổ sung
            response_data = ExamImportResponse(
                success=True,
                message=success_message,
                data=ImportedExamData(**exam_data),
                processing_time=processing_time
            ).model_dump()

            # Thêm thông tin về warnings và missing parts
            response_data["warnings"] = format_warnings
            response_data["missing_parts"] = missing_parts
            response_data["statistics"] = statistics.model_dump()

            return response_data

        except Exception as e:
            logger.error(f"Error importing exam from DOCX: {e}")
            processing_time = time.time() - start_time
            
            return ExamImportError(
                message="Import failed",
                error=f"Lỗi trong quá trình import: {str(e)}",
                error_code="IMPORT_ERROR",
                details={
                    "filename": filename,
                    "processing_time": processing_time,
                    "error_type": type(e).__name__
                }
            ).model_dump()

    def _extract_text_from_docx_bytes(self, file_content: bytes) -> str:
        """
        Trích xuất text từ file DOCX bytes

        Args:
            file_content: Nội dung file DOCX dưới dạng bytes

        Returns:
            Text đã trích xuất
        """
        try:
            # Tạo file-like object từ bytes
            file_stream = io.BytesIO(file_content)
            
            # Đọc document
            doc = Document(file_stream)
            
            # Trích xuất text từ tất cả paragraphs
            full_text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text.strip())
            
            # Trích xuất text từ tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        full_text.append(" | ".join(row_text))
            
            extracted_text = "\n".join(full_text)
            logger.info(f"Successfully extracted {len(extracted_text)} characters from DOCX")
            
            return extracted_text

        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            return ""

    def _validate_exam_format(self, exam_text: str) -> Dict[str, Any]:
        """
        Validate format đề thi cơ bản - kiểm tra có ít nhất 1 phần và đáp án

        Args:
            exam_text: Nội dung đề thi đã extract

        Returns:
            Dict chứa kết quả validation
        """
        try:
            # Normalize text để dễ kiểm tra
            normalized_text = exam_text.upper().replace('\n', ' ').replace('\r', '')

            validation_result = {
                "is_valid": True,
                "error": "",
                "details": {},
                "warnings": []
            }

            # 1. Kiểm tra cấu trúc các phần
            all_parts = ["PHẦN I", "PHẦN II", "PHẦN III"]
            found_parts = []
            missing_parts = []

            for part in all_parts:
                if part in normalized_text:
                    found_parts.append(part)
                else:
                    missing_parts.append(part)

            # Chỉ yêu cầu có ít nhất 1 phần
            if not found_parts:
                validation_result["is_valid"] = False
                validation_result["error"] = "Không tìm thấy cấu trúc phần nào (PHẦN I, II, III)"
                validation_result["details"]["missing_parts"] = missing_parts
                return validation_result

            # Ghi nhận các phần thiếu như warning
            if missing_parts:
                validation_result["warnings"].append(f"Thiếu các phần: {', '.join(missing_parts)}")
                validation_result["details"]["missing_parts"] = missing_parts
                validation_result["details"]["found_parts"] = found_parts

            # 2. Kiểm tra phần đáp án (không bắt buộc)
            if "ĐÁP ÁN" not in normalized_text:
                validation_result["warnings"].append("Không tìm thấy phần đáp án")
                validation_result["details"]["missing_answer_section"] = True

            logger.info(f"Exam format validation passed - Found parts: {found_parts}, Missing: {missing_parts}")
            return validation_result

        except Exception as e:
            logger.error(f"Error in format validation: {e}")
            return {
                "is_valid": False,
                "error": f"Lỗi trong quá trình validate format: {str(e)}",
                "details": {"exception": str(e)},
                "warnings": []
            }

    async def _analyze_exam_with_llm(self, exam_text: str, filename: str) -> Dict[str, Any]:
        """
        Sử dụng LLM để phân tích và chuyển đổi đề thi thành JSON

        Args:
            exam_text: Nội dung đề thi đã extract
            filename: Tên file gốc

        Returns:
            Dict chứa kết quả phân tích
        """
        try:
            # Tạo prompt cho LLM
            prompt = self._create_analysis_prompt(exam_text, filename)
            
            # Gọi LLM
            openrouter_service = get_openrouter_service()
            response = await openrouter_service.generate_content(
                prompt=prompt,
                temperature=0.1,
                max_tokens=8000
            )

            if not response.get("success", False):
                return {
                    "success": False,
                    "error": f"LLM request failed: {response.get('error', 'Unknown error')}"
                }

            # Parse JSON từ response
            llm_content = response.get("text", "") or response.get("content", "")
            logger.info(f"LLM response length: {len(llm_content)}")
            logger.info(f"LLM response preview: {llm_content[:500]}...")

            # Tìm JSON trong response
            json_data = self._extract_json_from_response(llm_content)
            
            if not json_data:
                return {
                    "success": False,
                    "error": "Không thể tìm thấy JSON hợp lệ trong response của LLM"
                }

            return {
                "success": True,
                "data": json_data
            }

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {
                "success": False,
                "error": f"LLM analysis error: {str(e)}"
            }

    def _create_analysis_prompt(self, exam_text: str, filename: str) -> str:
        """
        Tạo prompt cho LLM để phân tích đề thi

        Args:
            exam_text: Nội dung đề thi
            filename: Tên file

        Returns:
            Prompt string
        """
        prompt = f"""
Bạn là một chuyên gia phân tích đề thi. Hãy phân tích nội dung đề thi sau và chuyển đổi thành định dạng JSON chính xác.

NỘI DUNG ĐỀ THI:
{exam_text}

YÊU CẦU:
1. Phân tích và trích xuất thông tin đề thi thành JSON với cấu trúc chính xác như mẫu
2. Xác định môn học, lớp, thời gian làm bài, tên trường
3. Phân chia câu hỏi theo các phần có sẵn trong đề thi:
   - Phần I: Trắc nghiệm nhiều phương án lựa chọn (A, B, C, D) - nếu có
   - Phần II: Trắc nghiệm đúng/sai (a, b, c, d với true/false) - nếu có
   - Phần III: Trắc nghiệm trả lời ngắn (chỉ số) - nếu có
4. Chỉ xử lý các phần thực sự có trong đề thi, bỏ qua phần không có
5. Trích xuất đáp án chính xác từ phần đáp án (nếu có)
6. Nếu là môn Hóa học, trích xuất bảng nguyên tử khối (nếu có)

ĐỊNH DẠNG JSON MONG MUỐN:
{{
  "subject": "Hóa học",
  "grade": 12,
  "duration_minutes": 90,
  "school": "Trường THPT Hong Thinh",
  "exam_code": "1234",
  "atomic_masses": "H = 1; C = 12; N = 14; O = 16",
  "parts": [
    {{
      "part": "Phần I",
      "title": "Trắc nghiệm nhiều phương án lựa chọn",
      "description": "Mỗi câu trả lời đúng thí sinh được 0,25 điểm",
      "questions": [
        {{
          "id": 1,
          "question": "Nguyên tử carbon có bao nhiêu electron?",
          "options": {{
            "A": "4",
            "B": "6",
            "C": "8",
            "D": "12"
          }},
          "answer": "B"
        }}
      ]
    }},
    {{
      "part": "Phần II",
      "title": "Trắc nghiệm đúng/sai",
      "description": "Thí sinh trả lời từ câu 1 đến câu X. Mỗi câu có 4 phát biểu a), b), c), d)",
      "questions": [
        {{
          "id": 1,
          "question": "Cho các phát biểu về nguyên tử carbon:",
          "statements": {{
            "a": {{
              "text": "Nguyên tử carbon có 6 proton",
              "answer": true
            }},
            "b": {{
              "text": "Nguyên tử carbon có 8 neutron",
              "answer": false
            }},
            "c": {{
              "text": "Nguyên tử carbon có 6 electron",
              "answer": true
            }},
            "d": {{
              "text": "Nguyên tử carbon có khối lượng 14u",
              "answer": false
            }}
          }}
        }}
      ]
    }},
    {{
      "part": "Phần III",
      "title": "Trắc nghiệm trả lời ngắn",
      "description": "Chỉ ghi số, không ghi đơn vị. Tối đa 4 ký tự.",
      "questions": [
        {{
          "id": 1,
          "question": "Số electron trong nguyên tử carbon là bao nhiêu?",
          "answer": "6"
        }}
      ]
    }}
  ]
}}

LưU Ý QUAN TRỌNG VỀ CẤU TRÚC:
- Chỉ trả về JSON hợp lệ, không thêm text giải thích
- Đảm bảo tất cả câu hỏi và đáp án được trích xuất chính xác
- QUAN TRỌNG: Mỗi loại câu hỏi có cấu trúc khác nhau:

  * PHẦN I (Trắc nghiệm nhiều lựa chọn): PHẢI có "options" và "answer"
    {{
      "id": 1,
      "question": "Câu hỏi cụ thể...",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "answer": "A"
    }}

  * PHẦN II (Đúng/sai): PHẢI có "statements", KHÔNG có "answer" field
    {{
      "id": 1,
      "question": "Câu hỏi cụ thể...",
      "statements": {{
        "a": {{"text": "...", "answer": true}},
        "b": {{"text": "...", "answer": false}},
        "c": {{"text": "...", "answer": true}},
        "d": {{"text": "...", "answer": false}}
      }}
    }}

  * PHẦN III (Trả lời ngắn): PHẢI có "answer" string, KHÔNG có "options" hay "statements"
    {{
      "id": 1,
      "question": "Câu hỏi cụ thể...",
      "answer": "6"
    }}

- QUAN TRỌNG: Chỉ tạo parts cho những phần có NỘI DUNG CÂU HỎI thực tế trong đề thi
- Nếu phần đáp án có nhưng không có nội dung câu hỏi, KHÔNG tạo part cho phần đó
- Mảng "parts" có thể chứa 1, 2 hoặc 3 phần tùy theo nội dung đề thi thực tế
- Không tạo ra câu hỏi giả cho các phần không có nội dung
- Đảm bảo field "question" luôn là string không rỗng, không được null
- Ví dụ: Nếu đề thi chỉ có "PHẦN I" với nội dung câu hỏi, chỉ tạo 1 part cho Phần I, bỏ qua Phần II và III dù có trong đáp án

Hãy phân tích và trả về JSON:
"""
        return prompt

    def _extract_json_from_response(self, response_content: str) -> Optional[Dict[str, Any]]:
        """
        Trích xuất JSON từ response của LLM

        Args:
            response_content: Nội dung response từ LLM

        Returns:
            Dict chứa JSON data hoặc None nếu không tìm thấy
        """
        try:
            logger.info(f"Extracting JSON from response: {response_content[:200]}...")

            # 1. Tìm JSON trong markdown code blocks
            markdown_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            markdown_matches = re.findall(markdown_pattern, response_content, re.DOTALL)

            for match in markdown_matches:
                try:
                    json_data = json.loads(match.strip())
                    if self._validate_exam_json_structure(json_data):
                        logger.info("Successfully extracted JSON from markdown code block")
                        return json_data
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from markdown: {e}")
                    continue

            # 2. Tìm JSON trong response (có thể có text bao quanh)
            json_pattern = r'\{.*\}'
            matches = re.findall(json_pattern, response_content, re.DOTALL)

            for match in matches:
                try:
                    # Thử parse JSON
                    json_data = json.loads(match)

                    # Validate cấu trúc cơ bản
                    if self._validate_exam_json_structure(json_data):
                        logger.info("Successfully extracted and validated JSON from LLM response")
                        return json_data

                except json.JSONDecodeError:
                    continue

            # 3. Nếu không tìm thấy JSON hợp lệ, thử parse toàn bộ response
            try:
                json_data = json.loads(response_content.strip())
                if self._validate_exam_json_structure(json_data):
                    logger.info("Successfully parsed entire response as JSON")
                    return json_data
            except json.JSONDecodeError:
                pass

            logger.error("Could not extract valid JSON from LLM response")
            logger.error(f"Response content: {response_content}")
            return None

        except Exception as e:
            logger.error(f"Error extracting JSON from response: {e}")
            return None

    def _validate_exam_json_structure(self, json_data: Dict[str, Any]) -> bool:
        """
        Validate cấu trúc JSON của đề thi - linh hoạt với parts

        Args:
            json_data: JSON data cần validate

        Returns:
            True nếu cấu trúc hợp lệ
        """
        try:
            required_fields = ["subject", "grade", "duration_minutes", "school", "parts"]

            # Kiểm tra các field bắt buộc
            for field in required_fields:
                if field not in json_data:
                    logger.error(f"Missing required field: {field}")
                    return False

            # Kiểm tra parts - cho phép empty list
            parts = json_data.get("parts", [])
            if not isinstance(parts, list):
                logger.error("Parts must be a list")
                return False

            # Nếu có parts, kiểm tra cấu trúc của từng part
            for i, part in enumerate(parts):
                if not isinstance(part, dict):
                    logger.error(f"Part {i} must be a dictionary")
                    return False

                part_required = ["part", "title", "description", "questions"]
                for field in part_required:
                    if field not in part:
                        logger.error(f"Missing required field '{field}' in part {i}")
                        return False

                # Kiểm tra questions - cho phép empty list
                questions = part.get("questions", [])
                if not isinstance(questions, list):
                    logger.error(f"Questions in part {i} must be a list")
                    return False

            logger.info(f"JSON structure validation passed - {len(parts)} parts found")
            return True

        except Exception as e:
            logger.error(f"Error validating JSON structure: {e}")
            return False

    def _validate_and_clean_exam_data(self, exam_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate và clean dữ liệu đề thi từ LLM

        Args:
            exam_data: Dữ liệu thô từ LLM

        Returns:
            Dict chứa kết quả validation và dữ liệu đã clean
        """
        try:
            logger.info("Starting exam data validation and cleaning...")

            result = {
                "is_valid": True,
                "error": "",
                "details": {},
                "cleaned_data": {}
            }

            # 1. Validate basic fields
            required_fields = ["subject", "grade", "duration_minutes", "school", "parts"]
            for field in required_fields:
                if field not in exam_data:
                    result["is_valid"] = False
                    result["error"] = f"Missing required field: {field}"
                    return result

            # 2. Clean basic data
            cleaned_data = {
                "subject": str(exam_data.get("subject", "")).strip(),
                "grade": int(exam_data.get("grade", 12)),
                "duration_minutes": int(exam_data.get("duration_minutes", 90)),
                "school": str(exam_data.get("school", "")).strip(),
                "exam_code": str(exam_data.get("exam_code", "")).strip() if exam_data.get("exam_code") else None,
                "atomic_masses": str(exam_data.get("atomic_masses", "")).strip() if exam_data.get("atomic_masses") else None,
                "parts": []
            }

            # 3. Validate và clean parts
            parts = exam_data.get("parts", [])
            if not isinstance(parts, list):
                result["is_valid"] = False
                result["error"] = "Parts must be a list"
                return result

            cleaned_parts = []
            skipped_parts = []

            for i, part in enumerate(parts):
                cleaned_part = self._clean_exam_part(part, i)
                if cleaned_part["is_valid"]:
                    cleaned_parts.append(cleaned_part["data"])
                else:
                    # Log warning và skip part này thay vì fail toàn bộ
                    part_name = part.get("part", f"Part {i}")
                    logger.warning(f"Skipping invalid part '{part_name}': {cleaned_part['error']}")
                    skipped_parts.append({
                        "part_name": part_name,
                        "error": cleaned_part["error"]
                    })

            # Chỉ fail nếu không có part nào hợp lệ
            if not cleaned_parts:
                result["is_valid"] = False
                result["error"] = "No valid parts found in exam data"
                result["details"]["skipped_parts"] = skipped_parts
                return result

            cleaned_data["parts"] = cleaned_parts
            result["cleaned_data"] = cleaned_data

            # Thêm thông tin về parts bị skip
            if skipped_parts:
                result["details"]["skipped_parts"] = skipped_parts
                logger.info(f"Exam data validation completed - {len(cleaned_parts)} valid parts, {len(skipped_parts)} skipped parts")
            else:
                logger.info(f"Exam data validation completed - {len(cleaned_parts)} parts validated")

            return result

        except Exception as e:
            logger.error(f"Error in exam data validation: {e}")
            return {
                "is_valid": False,
                "error": f"Validation error: {str(e)}",
                "details": {"exception": str(e)},
                "cleaned_data": {}
            }

    def _clean_exam_part(self, part_data: Dict[str, Any], part_index: int) -> Dict[str, Any]:
        """
        Clean và validate một phần của đề thi

        Args:
            part_data: Dữ liệu phần thô từ LLM
            part_index: Index của phần

        Returns:
            Dict chứa kết quả validation và dữ liệu đã clean
        """
        try:
            result = {
                "is_valid": True,
                "error": "",
                "data": {}
            }

            # Validate basic part fields
            required_part_fields = ["part", "title", "description", "questions"]
            for field in required_part_fields:
                if field not in part_data:
                    result["is_valid"] = False
                    result["error"] = f"Missing field '{field}' in part"
                    return result

            # Clean part basic data
            cleaned_part = {
                "part": str(part_data.get("part", "")).strip(),
                "title": str(part_data.get("title", "")).strip(),
                "description": str(part_data.get("description", "")).strip(),
                "questions": []
            }

            # Validate và clean questions
            questions = part_data.get("questions", [])
            if not isinstance(questions, list):
                result["is_valid"] = False
                result["error"] = "Questions must be a list"
                return result

            cleaned_questions = []
            invalid_questions = []

            for j, question in enumerate(questions):
                cleaned_question = self._clean_question(question, cleaned_part["part"], j)
                if cleaned_question["is_valid"]:
                    cleaned_questions.append(cleaned_question["data"])
                else:
                    # Log invalid question nhưng không fail toàn bộ part
                    logger.warning(f"Skipping invalid question {j} in {cleaned_part['part']}: {cleaned_question['error']}")
                    invalid_questions.append({
                        "question_index": j,
                        "error": cleaned_question["error"]
                    })

            # Nếu không có câu hỏi hợp lệ nào, fail part này
            if not cleaned_questions:
                result["is_valid"] = False
                result["error"] = f"No valid questions found in part. Invalid questions: {len(invalid_questions)}"
                return result

            cleaned_part["questions"] = cleaned_questions
            result["data"] = cleaned_part

            return result

        except Exception as e:
            return {
                "is_valid": False,
                "error": f"Error cleaning part: {str(e)}",
                "data": {}
            }

    def _clean_question(self, question_data: Dict[str, Any], part_name: str, question_index: int) -> Dict[str, Any]:
        """
        Clean và validate một câu hỏi theo loại phần

        Args:
            question_data: Dữ liệu câu hỏi thô từ LLM
            part_name: Tên phần (để xác định loại câu hỏi)
            question_index: Index của câu hỏi

        Returns:
            Dict chứa kết quả validation và dữ liệu đã clean
        """
        try:
            result = {
                "is_valid": True,
                "error": "",
                "data": {}
            }

            # Validate basic question fields
            if "id" not in question_data or "question" not in question_data:
                result["is_valid"] = False
                result["error"] = "Missing 'id' or 'question' field"
                return result

            # Clean basic question data
            question_text = question_data.get("question")
            if not question_text or question_text is None or str(question_text).strip() == "":
                result["is_valid"] = False
                result["error"] = f"Question text cannot be null or empty. Got: {question_text}"
                return result

            cleaned_question = {
                "id": int(question_data.get("id", question_index + 1)),
                "question": str(question_text).strip()
            }

            # Clean theo loại phần
            if "Phần I" in part_name or "PHẦN I" in part_name:
                # MultipleChoiceQuestion
                if "options" not in question_data or "answer" not in question_data:
                    result["is_valid"] = False
                    result["error"] = "MultipleChoice question missing 'options' or 'answer'"
                    return result

                options = question_data.get("options", {})
                if not isinstance(options, dict) or not all(k in options for k in ["A", "B", "C", "D"]):
                    result["is_valid"] = False
                    result["error"] = "Options must contain A, B, C, D"
                    return result

                cleaned_question["options"] = {
                    "A": str(options.get("A", "")).strip(),
                    "B": str(options.get("B", "")).strip(),
                    "C": str(options.get("C", "")).strip(),
                    "D": str(options.get("D", "")).strip()
                }
                cleaned_question["answer"] = str(question_data.get("answer", "")).strip()

            elif "Phần II" in part_name or "PHẦN II" in part_name:
                # TrueFalseQuestion
                if "statements" not in question_data:
                    result["is_valid"] = False
                    result["error"] = "TrueFalse question missing 'statements'"
                    return result

                statements = question_data.get("statements", {})
                if not isinstance(statements, dict) or not all(k in statements for k in ["a", "b", "c", "d"]):
                    result["is_valid"] = False
                    result["error"] = "Statements must contain a, b, c, d"
                    return result

                cleaned_statements = {}
                for key in ["a", "b", "c", "d"]:
                    stmt = statements.get(key, {})
                    if not isinstance(stmt, dict) or "text" not in stmt or "answer" not in stmt:
                        result["is_valid"] = False
                        result["error"] = f"Statement {key} missing 'text' or 'answer'"
                        return result

                    cleaned_statements[key] = {
                        "text": str(stmt.get("text", "")).strip(),
                        "answer": bool(stmt.get("answer", False))
                    }

                cleaned_question["statements"] = cleaned_statements

            elif "Phần III" in part_name or "PHẦN III" in part_name:
                # ShortAnswerQuestion
                if "answer" not in question_data:
                    result["is_valid"] = False
                    result["error"] = "ShortAnswer question missing 'answer'"
                    return result

                cleaned_question["answer"] = str(question_data.get("answer", "")).strip()

            else:
                result["is_valid"] = False
                result["error"] = f"Unknown part type: {part_name}"
                return result

            result["data"] = cleaned_question
            return result

        except Exception as e:
            return {
                "is_valid": False,
                "error": f"Error cleaning question: {str(e)}",
                "data": {}
            }

    def _calculate_import_statistics(self, exam_data: Dict[str, Any]) -> ExamImportStatistics:
        """
        Tính toán thống kê cho đề thi đã import

        Args:
            exam_data: Dữ liệu đề thi

        Returns:
            ExamImportStatistics object
        """
        try:
            parts = exam_data.get("parts", [])
            
            total_questions = 0
            part_1_questions = 0
            part_2_questions = 0
            part_3_questions = 0
            
            for part in parts:
                questions = part.get("questions", [])
                part_name = part.get("part", "").lower()
                
                if "i" in part_name or "1" in part_name:
                    part_1_questions = len(questions)
                elif "ii" in part_name or "2" in part_name:
                    part_2_questions = len(questions)
                elif "iii" in part_name or "3" in part_name:
                    part_3_questions = len(questions)
                
                total_questions += len(questions)
            
            has_atomic_masses = bool(exam_data.get("atomic_masses"))
            
            # Tính chất lượng xử lý (dựa trên số câu hỏi và cấu trúc)
            processing_quality = min(1.0, (total_questions / 20) * 0.8 + 0.2)
            
            return ExamImportStatistics(
                total_questions=total_questions,
                part_1_questions=part_1_questions,
                part_2_questions=part_2_questions,
                part_3_questions=part_3_questions,
                has_atomic_masses=has_atomic_masses,
                processing_quality=processing_quality
            )

        except Exception as e:
            logger.error(f"Error calculating import statistics: {e}")
            return ExamImportStatistics(
                total_questions=0,
                part_1_questions=0,
                part_2_questions=0,
                part_3_questions=0,
                has_atomic_masses=False,
                processing_quality=0.0
            )


# Lazy loading global instance để tránh khởi tạo ngay khi import
_exam_import_service_instance = None

def get_exam_import_service() -> ExamImportService:
    """
    Lấy singleton instance của ExamImportService
    Lazy initialization

    Returns:
        ExamImportService: Service instance
    """
    global _exam_import_service_instance
    if _exam_import_service_instance is None:
        _exam_import_service_instance = ExamImportService()
    return _exam_import_service_instance

# Backward compatibility - deprecated, sử dụng get_exam_import_service() thay thế
# Lazy loading để tránh khởi tạo ngay khi import
def _get_exam_import_service_lazy():
    """Lazy loading cho backward compatibility"""
    return get_exam_import_service()

# Tạo proxy object để lazy loading
class _ExamImportServiceProxy:
    def __getattr__(self, name):
        return getattr(_get_exam_import_service_lazy(), name)

    def __call__(self, *args, **kwargs):
        return _get_exam_import_service_lazy()(*args, **kwargs)

exam_import_service = _ExamImportServiceProxy()
