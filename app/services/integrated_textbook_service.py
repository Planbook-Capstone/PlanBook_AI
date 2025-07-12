"""
Service tích hợp: OCR → LLM Format + Metadata cùng lúc
"""

import json
import logging
import uuid
from typing import Dict, Any
from datetime import datetime

from app.services.simple_ocr_service import get_simple_ocr_service
from app.services.llm_service import get_llm_service

logger = logging.getLogger(__name__)


class IntegratedTextbookService:
    """Service xử lý PDF một lần: OCR + LLM Format + Metadata"""

    def __init__(self):
        self.llm_service = get_llm_service()
        self.ocr_service = get_simple_ocr_service()

    async def process_pdf_complete(
        self, pdf_content: bytes, filename: str
    ) -> Dict[str, Any]:
        """
        Xử lý PDF hoàn chỉnh: OCR + LLM Format + Metadata extraction

        Returns:
            {
                "success": bool,
                "extracted_metadata": {...},
                "formatted_structure": {...},
                "raw_text": str,
                "processing_info": {...}
            }
        """
        try:
            logger.info(f"Starting integrated processing for: {filename}")

            # Step 1: OCR extraction
            raw_text, ocr_metadata = await self.ocr_service.extract_text_from_pdf(
                pdf_content, filename
            )

            if not raw_text.strip():
                raise Exception("No text extracted from PDF")

            # Step 2: LLM analysis - Format + Extract Metadata cùng lúc
            llm_result = await self._llm_format_and_extract_metadata(
                raw_text, filename, ocr_metadata
            )

            # Step 3: Combine results
            result = {
                "success": True,
                "extracted_metadata": llm_result["metadata"],
                "formatted_structure": llm_result["structure"],
                "raw_text": raw_text,
                "processing_info": {
                    "filename": filename,
                    "ocr_info": ocr_metadata,
                    "llm_processing": True,
                    "processing_timestamp": datetime.now().isoformat(),
                    "text_length": len(raw_text),
                },
            }

            logger.info(f"Integrated processing completed for: {filename}")
            return result

        except Exception as e:
            logger.error(f"Integrated processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "extracted_metadata": self._generate_fallback_metadata(filename),
                "formatted_structure": None,
                "raw_text": raw_text if "raw_text" in locals() else "",
                "processing_info": {
                    "filename": filename,
                    "error": str(e),
                    "processing_timestamp": datetime.now().isoformat(),
                },
            }

    async def _llm_format_and_extract_metadata(
        self, raw_text: str, filename: str, ocr_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Sử dụng LLM để vừa format structure vừa extract metadata
        """

        # Truncate text nếu quá dài (để tránh token limit)
        max_chars = 8000
        text_sample = raw_text[:max_chars]
        if len(raw_text) > max_chars:
            text_sample += "\n...[text truncated]"

        prompt = f"""
Bạn là chuyên gia phân tích sách giáo khoa. Hãy phân tích văn bản sau và trả về JSON với 2 phần:

1. METADATA: Thông tin về sách
2. STRUCTURE: Cấu trúc nội dung đã format

Văn bản từ file "{filename}":
{text_sample}

Hãy trả về JSON theo format chính xác sau:

{{
    "metadata": {{
        "id": "book_<random_id>",
        "title": "tên sách được phát hiện",
        "subject": "môn học (toán/văn/anh/khoa_học/lịch_sử/địa_lý/gdcd/khác)",
        "grade": "lớp (1-12/mầm_non/chưa_xác_định)",
        "publisher": "nhà xuất bản nếu có",
        "language": "vi/en",
        "academic_year": "năm học nếu có",
        "curriculum": "chương trình (2018/cũ/khác)",
        "chapter_info": "thông tin chương hiện tại",
        "description": "mô tả ngắn gọn",
        "auto_detected": true,
        "confidence": 0.8
    }},
    "structure": {{
        "book_info": {{
            "id": "cùng id như metadata",
            "title": "cùng title như metadata",
            "subject": "cùng subject như metadata",
            "grade": "cùng grade như metadata"
        }},
        "chapters": [
            {{
                "chapter_id": "CHAPTER_01",
                "chapter_number": 1,
                "chapter_title": "tên chương được phát hiện",
                "lessons": [
                    {{
                        "lesson_id": "LESSON_01_01", 
                        "lesson_number": 1,
                        "lesson_title": "tên bài học",
                        "content": [
                            {{
                                "type": "text/example/exercise/image",
                                "text": "nội dung đã được format",
                                "page": 1,
                                "section": "warm_up/new_knowledge/practice/application"
                            }}
                        ]
                    }}
                ]
            }}
        ]
    }}
}}

LƯU Ý:
- Phân tích cẩn thận để detect đúng môn học và lớp
- Tách nội dung thành các lesson logic
- Classify content type (text/example/exercise)
- Giữ nguyên nội dung quan trọng, format lại cho dễ đọc
- Nếu không chắc chắn thì ghi "chưa_xác_định"
- Chỉ trả về JSON, không giải thích thêm
"""

        try:
            # Sử dụng model trực tiếp thay vì qua service
            if not self.llm_service.model:
                raise Exception("LLM service not available")

            response = self.llm_service.model.generate_content(prompt)
            response_text = response.text.strip()

            # Clean JSON - cải thiện việc xử lý
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.startswith("```"):
                response_text = response_text[3:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            # Tìm JSON hợp lệ trong response
            response_text = response_text.strip()

            # Tìm vị trí bắt đầu và kết thúc của JSON
            start_idx = response_text.find("{")
            if start_idx == -1:
                raise ValueError("No JSON object found in response")

            # Tìm vị trí kết thúc JSON bằng cách đếm dấu ngoặc
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(response_text[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            # Extract JSON hợp lệ
            clean_json = response_text[start_idx:end_idx]

            # Parse JSON response
            llm_result = json.loads(clean_json)

            # Validate and enhance
            llm_result = self._validate_and_enhance_llm_result(
                llm_result, filename, ocr_metadata
            )

            return llm_result

        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}")
            logger.error(
                f"LLM Response: {response_text if 'response_text' in locals() else 'No response'}"
            )
            return self._generate_fallback_result(filename, raw_text, ocr_metadata)

        except Exception as e:
            logger.error(f"LLM processing failed: {e}")
            return self._generate_fallback_result(filename, raw_text, ocr_metadata)

    def _validate_and_enhance_llm_result(
        self, llm_result: Dict[str, Any], filename: str, ocr_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate và enhance kết quả từ LLM"""

        # Ensure required fields in metadata
        metadata = llm_result.get("metadata", {})

        if not metadata.get("id"):
            metadata["id"] = f"book_{uuid.uuid4().hex[:8]}"

        if not metadata.get("title"):
            metadata["title"] = filename.replace(".pdf", "").replace("_", " ").title()

        # Add technical metadata
        metadata.update(
            {
                "filename": filename,
                "created_at": datetime.now().isoformat(),
                "file_size": ocr_metadata.get("total_pages", 0),
                "ocr_method": ocr_metadata.get("extraction_method", "unknown"),
                "processing_method": "integrated_ocr_llm",
            }
        )

        # Ensure structure has book_info
        structure = llm_result.get("structure", {})
        if not structure.get("book_info"):
            structure["book_info"] = {
                "id": metadata["id"],
                "title": metadata["title"],
                "subject": metadata.get("subject", "chưa xác định"),
                "grade": metadata.get("grade", "chưa xác định"),
            }

        return {"metadata": metadata, "structure": structure}

    def _generate_fallback_result(
        self, filename: str, raw_text: str, ocr_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Tạo kết quả fallback khi LLM thất bại"""

        book_id = f"book_{uuid.uuid4().hex[:8]}"
        title = filename.replace(".pdf", "").replace("_", " ").title()

        # Simple text analysis for fallback
        text_lower = raw_text.lower()

        # Detect subject (basic)
        subject = "chưa xác định"
        if any(word in text_lower for word in ["toán", "math", "số", "phép"]):
            subject = "toán"
        elif any(word in text_lower for word in ["văn", "ngữ văn", "literature"]):
            subject = "văn"
        elif any(word in text_lower for word in ["english", "tiếng anh"]):
            subject = "anh"

        # Detect grade (basic)
        import re

        grade_matches = re.findall(r"lớp\s*(\d+)", text_lower)
        grade = grade_matches[0] if grade_matches else "chưa xác định"

        metadata = {
            "id": book_id,
            "title": title,
            "subject": subject,
            "grade": grade,
            "publisher": "chưa xác định",
            "language": "vi",
            "auto_detected": True,
            "confidence": 0.3,
            "fallback_mode": True,
            "filename": filename,
            "created_at": datetime.now().isoformat(),
        }

        # Simple structure
        structure = {
            "book_info": {
                "id": book_id,
                "title": title,
                "subject": subject,
                "grade": grade,
            },
            "chapters": [
                {
                    "chapter_id": "CHAPTER_01",
                    "chapter_number": 1,
                    "chapter_title": "Nội dung chính",
                    "lessons": [
                        {
                            "lesson_id": "LESSON_01_01",
                            "lesson_number": 1,
                            "lesson_title": "Bài học từ OCR",
                            "content": [
                                {
                                    "type": "text",
                                    "text": raw_text[:1000] + "..."
                                    if len(raw_text) > 1000
                                    else raw_text,
                                    "page": 1,
                                    "section": "content",
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        return {"metadata": metadata, "structure": structure}

    def _generate_fallback_metadata(self, filename: str) -> Dict[str, Any]:
        """Generate basic fallback metadata"""
        return {
            "id": f"book_{uuid.uuid4().hex[:8]}",
            "title": filename.replace(".pdf", "").replace("_", " ").title(),
            "subject": "chưa xác định",
            "grade": "chưa xác định",
            "publisher": "chưa xác định",
            "language": "vi",
            "auto_detected": True,
            "confidence": 0.1,
            "fallback_mode": True,
            "filename": filename,
            "created_at": datetime.now().isoformat(),
        }


# Global instance
integrated_textbook_service = IntegratedTextbookService()
