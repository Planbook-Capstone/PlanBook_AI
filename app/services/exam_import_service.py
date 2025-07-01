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

from app.services.openrouter_service import openrouter_service
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

            logger.info("Exam format validation passed")

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

            # 4. Validate và tạo response
            processing_time = time.time() - start_time
            
            # Tính toán statistics
            statistics = self._calculate_import_statistics(exam_data)
            
            return ExamImportResponse(
                success=True,
                message="Đề thi đã được import thành công",
                data=ImportedExamData(**exam_data),
                processing_time=processing_time
            ).model_dump()

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
        Validate format đề thi cơ bản - chỉ kiểm tra cấu trúc 3 phần và đáp án

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
                "details": {}
            }

            # 1. Kiểm tra cấu trúc 3 phần
            required_parts = ["PHẦN I", "PHẦN II", "PHẦN III"]
            missing_parts = []

            for part in required_parts:
                if part not in normalized_text:
                    missing_parts.append(part)

            if missing_parts:
                validation_result["is_valid"] = False
                validation_result["error"] = f"Thiếu cấu trúc phần: {', '.join(missing_parts)}"
                validation_result["details"]["missing_parts"] = missing_parts
                return validation_result

            # 2. Kiểm tra phần đáp án
            if "ĐÁP ÁN" not in normalized_text:
                validation_result["is_valid"] = False
                validation_result["error"] = "Thiếu phần đáp án"
                validation_result["details"]["missing_answer_section"] = True
                return validation_result

            logger.info("Exam format validation passed successfully")
            return validation_result

        except Exception as e:
            logger.error(f"Error in format validation: {e}")
            return {
                "is_valid": False,
                "error": f"Lỗi trong quá trình validate format: {str(e)}",
                "details": {"exception": str(e)}
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
3. Phân chia câu hỏi thành 3 phần:
   - Phần I: Trắc nghiệm nhiều phương án lựa chọn (A, B, C, D)
   - Phần II: Trắc nghiệm đúng/sai (a, b, c, d với true/false)
   - Phần III: Trắc nghiệm trả lời ngắn (chỉ số)
4. Trích xuất đáp án chính xác từ phần đáp án
5. Nếu là môn Hóa học, trích xuất bảng nguyên tử khối

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
          "question": "Nội dung câu hỏi...",
          "options": {{
            "A": "Đáp án A",
            "B": "Đáp án B", 
            "C": "Đáp án C",
            "D": "Đáp án D"
          }},
          "answer": "C"
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
          "question": "Nội dung câu hỏi...",
          "statements": {{
            "a": {{
              "text": "Phát biểu a",
              "answer": true
            }},
            "b": {{
              "text": "Phát biểu b", 
              "answer": false
            }},
            "c": {{
              "text": "Phát biểu c",
              "answer": true
            }},
            "d": {{
              "text": "Phát biểu d",
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
          "question": "Nội dung câu hỏi...",
          "answer": "1"
        }}
      ]
    }}
  ]
}}

LưU Ý QUAN TRỌNG:
- Chỉ trả về JSON hợp lệ, không thêm text giải thích
- Đảm bảo tất cả câu hỏi và đáp án được trích xuất chính xác
- Với câu đúng/sai, xác định chính xác true/false cho từng phát biểu
- Với câu trả lời ngắn, chỉ lấy số, tối đa 4 ký tự
- Nếu không tìm thấy thông tin nào, sử dụng giá trị mặc định hợp lý

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
        Validate cấu trúc JSON của đề thi

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
            
            # Kiểm tra parts
            parts = json_data.get("parts", [])
            if not isinstance(parts, list) or len(parts) == 0:
                logger.error("Parts must be a non-empty list")
                return False
            
            # Kiểm tra cấu trúc của từng part
            for part in parts:
                if not isinstance(part, dict):
                    return False
                
                part_required = ["part", "title", "description", "questions"]
                for field in part_required:
                    if field not in part:
                        logger.error(f"Missing required field in part: {field}")
                        return False
                
                # Kiểm tra questions
                questions = part.get("questions", [])
                if not isinstance(questions, list):
                    return False
            
            logger.info("JSON structure validation passed")
            return True

        except Exception as e:
            logger.error(f"Error validating JSON structure: {e}")
            return False

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


# Tạo instance global
exam_import_service = ExamImportService()
