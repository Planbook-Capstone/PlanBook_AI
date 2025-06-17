"""
Enhanced Textbook Service - Cải tiến xử lý sách giáo khoa với OCR và LLM
Trả về cấu trúc: Sách → Chương → Bài → Nội dung
"""

import logging
import asyncio
import json
import base64
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
from PIL import Image
import io

from app.services.simple_ocr_service import simple_ocr_service
from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)


class EnhancedTextbookService:
    """Service cải tiến để xử lý sách giáo khoa với OCR và LLM"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def process_textbook_to_structure(
        self,
        pdf_content: bytes,
        filename: str,
        book_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Xử lý PDF sách giáo khoa và trả về cấu trúc hoàn chỉnh

        Args:
            pdf_content: Nội dung PDF
            filename: Tên file
            book_metadata: Metadata sách (title, subject, grade, etc.)

        Returns:
            Dict với cấu trúc: book -> chapters -> lessons -> content
        """
        try:
            logger.info(f"🚀 Starting enhanced textbook processing: {filename}")

            # Step 1: Extract all pages with OCR
            logger.info("📄 Extracting pages with OCR...")
            pages_data = await self._extract_pages_with_ocr(pdf_content)
            logger.info(f"✅ Extracted {len(pages_data)} pages")

            # Step 2: Analyze book structure with LLM
            logger.info("🧠 Analyzing book structure...")
            book_structure = await self._analyze_book_structure_enhanced(
                pages_data, book_metadata
            )
            logger.info(
                f"📚 Detected {len(book_structure.get('chapters', []))} chapters"
            )

            # Step 3: Process content for each lesson
            logger.info("🔄 Processing lesson content...")
            processed_book = await self._process_lessons_content(
                book_structure, pages_data
            )

            logger.info("✅ Textbook processing completed successfully")

            return {
                "success": True,
                "book": processed_book,
                "total_pages": len(pages_data),
                "total_chapters": len(processed_book.get("chapters", [])),
                "total_lessons": sum(
                    len(ch.get("lessons", []))
                    for ch in processed_book.get("chapters", [])
                ),
                "message": "Textbook processed successfully",
            }

        except Exception as e:
            logger.error(f"❌ Error processing textbook: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to process textbook",
            }

    async def _extract_pages_with_ocr(self, pdf_content: bytes) -> List[Dict[str, Any]]:
        """Extract tất cả pages với OCR nếu cần"""

        def extract_page_data():
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            pages_data = []

            for page_num in range(doc.page_count):
                page = doc[page_num]

                # Extract text normally first
                text = page.get_text("text")  # type: ignore

                # Extract images
                images = []
                image_list = page.get_images()

                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)

                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_data = pix.tobytes("png")
                            img_base64 = base64.b64encode(img_data).decode()

                            images.append(
                                {
                                    "index": img_index,
                                    "data": img_base64,
                                    "format": "png",
                                    "page": page_num + 1,
                                }
                            )

                        pix = None
                    except Exception as e:
                        logger.warning(
                            f"Error extracting image {img_index} from page {page_num}: {e}"
                        )

                pages_data.append(
                    {
                        "page_number": page_num + 1,
                        "text": text,
                        "images": images,
                        "has_text": len(text.strip()) > 50,
                    }
                )

            doc.close()
            return pages_data

        # Extract pages in background thread
        pages_data = await asyncio.get_event_loop().run_in_executor(
            self.executor, extract_page_data
        )

        # Apply OCR to pages with insufficient text
        ocr_tasks = []
        for page in pages_data:
            if not page["has_text"]:
                ocr_tasks.append(self._apply_ocr_to_page(page, pdf_content))

        if ocr_tasks:
            logger.info(
                f"🔍 Applying OCR to {len(ocr_tasks)} pages with insufficient text"
            )
            ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)

            # Update pages with OCR results
            ocr_index = 0
            for page in pages_data:
                if not page["has_text"] and ocr_index < len(ocr_results):
                    if not isinstance(ocr_results[ocr_index], Exception):
                        page["text"] = ocr_results[ocr_index]
                        page["ocr_applied"] = True
                    ocr_index += 1

        return pages_data

    async def _apply_ocr_to_page(
        self, page_data: Dict[str, Any], pdf_content: bytes
    ) -> str:
        """Apply OCR to a specific page"""
        try:
            # Use existing OCR service for this page
            page_num = page_data["page_number"]

            # Extract just this page as bytes
            def extract_single_page():
                doc = fitz.open(stream=pdf_content, filetype="pdf")
                page = doc[page_num - 1]

                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                pix = None
                doc.close()

                return img_data

            img_data = await asyncio.get_event_loop().run_in_executor(
                self.executor, extract_single_page
            )

            # Apply OCR using PIL and simple_ocr_service logic
            image = Image.open(io.BytesIO(img_data))

            # Use simple OCR service's OCR logic
            if (
                hasattr(simple_ocr_service, "easyocr_reader")
                and simple_ocr_service.easyocr_reader
            ):
                import numpy as np

                results = simple_ocr_service.easyocr_reader.readtext(np.array(image))
                text_parts = [str(result[1]) for result in results if len(result) >= 2]
                return " ".join(text_parts)
            else:
                # Fallback to Tesseract
                import pytesseract

                return pytesseract.image_to_string(
                    image, config=simple_ocr_service.tesseract_config
                )

        except Exception as e:
            logger.error(f"OCR failed for page {page_data['page_number']}: {e}")
            return ""

    async def _analyze_book_structure_enhanced(
        self,
        pages_data: List[Dict[str, Any]],
        book_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Phân tích cấu trúc sách với LLM cải tiến"""

        if not llm_service.is_available():
            logger.warning("LLM not available, using pattern-based analysis")
            return await self._pattern_based_structure_analysis(
                pages_data, book_metadata
            )

        # Tạo text sample từ các trang để phân tích
        sample_text = ""
        for i, page in enumerate(pages_data[:20]):  # Lấy 20 trang đầu để phân tích
            if page["text"].strip():
                sample_text += f"\n--- Trang {page['page_number']} ---\n{page['text'][:500]}"  # 500 chars per page

        prompt = f"""
Bạn là chuyên gia phân tích sách giáo khoa Việt Nam. Phân tích nội dung và trả về cấu trúc chính xác.

THÔNG TIN SÁCH:
- Tổng số trang: {len(pages_data)}
- Metadata: {json.dumps(book_metadata or {}, ensure_ascii=False)}

NỘI DUNG SAMPLE:
{sample_text}

YÊU CẦU:
1. Xác định tiêu đề sách, môn học, lớp
2. Tìm tất cả CHƯƠNG (Chapter) trong sách
3. Tìm tất cả BÀI HỌC (Lesson) trong mỗi chương
4. Xác định trang bắt đầu và kết thúc cho mỗi chương/bài
5. Trả về JSON chuẩn

JSON FORMAT:
{{
  "book_info": {{
    "title": "Tên sách chính xác",
    "subject": "Môn học (Toán/Lý/Hóa/...)",
    "grade": "Lớp (10/11/12)",
    "total_pages": {len(pages_data)}
  }},
  "chapters": [
    {{
      "chapter_id": "chapter_01",
      "chapter_title": "Tên chương chính xác",
      "start_page": 1,
      "end_page": 20,
      "lessons": [
        {{
          "lesson_id": "lesson_01_01",
          "lesson_title": "Tên bài học chính xác",
          "start_page": 1,
          "end_page": 5
        }}
      ]
    }}
  ]
}}

Trả về JSON:"""

        try:
            if not llm_service.model:
                raise Exception("LLM model not available")

            response = llm_service.model.generate_content(prompt)
            json_text = response.text.strip()

            # Clean JSON - cải thiện việc xử lý
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]

            # Tìm JSON hợp lệ trong response
            json_text = json_text.strip()

            # Tìm vị trí bắt đầu và kết thúc của JSON
            start_idx = json_text.find("{")
            if start_idx == -1:
                raise ValueError("No JSON object found in response")

            # Tìm vị trí kết thúc JSON bằng cách đếm dấu ngoặc
            brace_count = 0
            end_idx = start_idx
            for i, char in enumerate(json_text[start_idx:], start_idx):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break

            # Extract JSON hợp lệ
            clean_json = json_text[start_idx:end_idx]

            structure = json.loads(clean_json)

            # Validate structure
            if "chapters" in structure and len(structure["chapters"]) > 0:
                logger.info(f"LLM detected {len(structure['chapters'])} chapters")
                return structure
            else:
                logger.warning("LLM returned invalid structure, using fallback")
                return await self._pattern_based_structure_analysis(
                    pages_data, book_metadata
                )

        except Exception as e:
            logger.error(f"LLM structure analysis failed: {e}")
            logger.debug(
                f"Raw LLM response: {response.text[:500] if 'response' in locals() else 'No response'}"
            )
            return await self._pattern_based_structure_analysis(
                pages_data, book_metadata
            )

    async def _pattern_based_structure_analysis(
        self,
        pages_data: List[Dict[str, Any]],
        book_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Phân tích cấu trúc dựa trên pattern matching"""

        total_pages = len(pages_data)

        # Extract book info from metadata or first pages
        book_info = {
            "title": book_metadata.get("title", "Sách giáo khoa")
            if book_metadata
            else "Sách giáo khoa",
            "subject": book_metadata.get("subject", "Chưa xác định")
            if book_metadata
            else "Chưa xác định",
            "grade": book_metadata.get("grade", "Chưa xác định")
            if book_metadata
            else "Chưa xác định",
            "total_pages": total_pages,
        }

        # Find chapters and lessons using pattern matching
        chapters = []
        current_chapter = None
        current_lesson = None

        for page in pages_data:
            lines = page["text"].split("\n")

            # Look for chapter patterns
            for line in lines:
                line_clean = line.strip()
                if len(line_clean) > 5 and len(line_clean) < 100:
                    # Chapter detection
                    if any(
                        pattern in line_clean.lower()
                        for pattern in ["chương", "chapter", "phần", "bài tập chương"]
                    ):
                        # Save previous chapter
                        if current_chapter:
                            chapters.append(current_chapter)

                        # Start new chapter
                        chapter_num = len(chapters) + 1
                        current_chapter = {
                            "chapter_id": f"chapter_{chapter_num:02d}",
                            "chapter_title": line_clean,
                            "start_page": page["page_number"],
                            "end_page": page["page_number"],
                            "lessons": [],
                        }
                        current_lesson = None

                    # Lesson detection
                    elif (
                        any(
                            pattern in line_clean.lower()
                            for pattern in ["bài", "lesson", "tiết"]
                        )
                        and current_chapter
                    ):
                        # Save previous lesson
                        if current_lesson:
                            current_chapter["lessons"].append(current_lesson)

                        # Start new lesson
                        lesson_num = len(current_chapter["lessons"]) + 1
                        current_lesson = {
                            "lesson_id": f"lesson_{len(chapters)+1:02d}_{lesson_num:02d}",
                            "lesson_title": line_clean,
                            "start_page": page["page_number"],
                            "end_page": page["page_number"],
                        }

            # Update end pages
            if current_chapter:
                current_chapter["end_page"] = page["page_number"]
            if current_lesson:
                current_lesson["end_page"] = page["page_number"]

        # Add final chapter and lesson
        if current_lesson and current_chapter:
            current_chapter["lessons"].append(current_lesson)
        if current_chapter:
            chapters.append(current_chapter)

        # If no chapters found, create default structure
        if not chapters:
            chapters = self._create_default_structure(total_pages)

        return {"book_info": book_info, "chapters": chapters}

    def _create_default_structure(self, total_pages: int) -> List[Dict[str, Any]]:
        """Tạo cấu trúc mặc định khi không detect được"""

        chapters = []
        pages_per_chapter = max(total_pages // 3, 10)  # Ít nhất 3 chương

        for chapter_num in range(1, 4):  # 3 chương
            start_page = (chapter_num - 1) * pages_per_chapter + 1
            end_page = min(chapter_num * pages_per_chapter, total_pages)

            if start_page > total_pages:
                break

            # Tạo 2-3 bài trong mỗi chương
            lessons = []
            pages_per_lesson = max((end_page - start_page + 1) // 3, 3)

            for lesson_num in range(1, 4):  # 3 bài mỗi chương
                lesson_start = start_page + (lesson_num - 1) * pages_per_lesson
                lesson_end = min(
                    start_page + lesson_num * pages_per_lesson - 1, end_page
                )

                if lesson_start > end_page:
                    break

                lessons.append(
                    {
                        "lesson_id": f"lesson_{chapter_num:02d}_{lesson_num:02d}",
                        "lesson_title": f"Bài {lesson_num}",
                        "start_page": lesson_start,
                        "end_page": lesson_end,
                    }
                )

            chapters.append(
                {
                    "chapter_id": f"chapter_{chapter_num:02d}",
                    "chapter_title": f"Chương {chapter_num}",
                    "start_page": start_page,
                    "end_page": end_page,
                    "lessons": lessons,
                }
            )

        return chapters

    async def _process_lessons_content(
        self, book_structure: Dict[str, Any], pages_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Xử lý nội dung chi tiết cho từng bài học"""

        processed_book = {
            "book_info": book_structure.get("book_info", {}),
            "chapters": [],
        }

        for chapter in book_structure.get("chapters", []):
            processed_chapter = {
                "chapter_id": chapter["chapter_id"],
                "chapter_title": chapter["chapter_title"],
                "start_page": chapter["start_page"],
                "end_page": chapter["end_page"],
                "lessons": [],
            }

            for lesson in chapter.get("lessons", []):
                logger.info(f"Processing lesson: {lesson['lesson_title']}")

                # Extract content for this lesson
                lesson_content = await self._extract_lesson_content(lesson, pages_data)

                processed_lesson = {
                    "lesson_id": lesson["lesson_id"],
                    "lesson_title": lesson["lesson_title"],
                    "start_page": lesson["start_page"],
                    "end_page": lesson["end_page"],
                    "content": lesson_content,
                }

                processed_chapter["lessons"].append(processed_lesson)

            processed_book["chapters"].append(processed_chapter)

        return processed_book

    async def _extract_lesson_content(
        self, lesson: Dict[str, Any], pages_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Extract nội dung chi tiết của một bài học"""

        start_page = lesson["start_page"]
        end_page = lesson["end_page"]

        # Collect all text and images for this lesson
        lesson_text = ""
        lesson_images = []
        lesson_pages = []

        for page_num in range(start_page, end_page + 1):
            # Find page data (pages_data is 0-indexed but page_number is 1-indexed)
            page_data = None
            for page in pages_data:
                if page["page_number"] == page_num:
                    page_data = page
                    break

            if not page_data:
                continue

            lesson_pages.append(page_num)

            # Add text content
            if page_data["text"].strip():
                # Clean text with LLM if available
                cleaned_text = await self._clean_text_with_llm(page_data["text"])
                lesson_text += f"\n--- Trang {page_num} ---\n{cleaned_text}\n"

            # Add images with LLM descriptions only
            for img in page_data.get("images", []):
                # Describe image with LLM using base64 data
                img_description = await self._describe_image_with_llm(img["data"])

                lesson_images.append(
                    {
                        "page": page_num,
                        "index": img["index"],
                        "format": img["format"],
                        "description": img_description,
                        # Note: Removed base64 data to reduce response size
                    }
                )

        return {
            "text": lesson_text.strip(),
            "images": lesson_images,
            "pages": lesson_pages,
            "total_pages": len(lesson_pages),
            "has_images": len(lesson_images) > 0,
        }

    async def _clean_text_with_llm(self, raw_text: str) -> str:
        """Clean và format text bằng LLM"""

        if not llm_service.is_available() or not raw_text.strip():
            return raw_text.strip()

        try:
            prompt = f"""
Bạn là chuyên gia xử lý text từ sách giáo khoa. Hãy làm sạch và format text sau:

YÊU CẦU:
1. Sửa lỗi OCR (ký tự nhận dạng sai)
2. Loại bỏ ký tự lạ, khoảng trắng thừa
3. Sắp xếp đoạn văn cho dễ đọc
4. Giữ nguyên ý nghĩa và cấu trúc
5. Trả về text tiếng Việt chuẩn

Text gốc:
{raw_text[:1000]}  # Limit to 1000 chars

Text đã làm sạch:"""

            if not llm_service.model:
                return raw_text.strip()

            response = llm_service.model.generate_content(prompt)
            cleaned_text = response.text.strip()

            return cleaned_text if cleaned_text else raw_text.strip()

        except Exception as e:
            logger.error(f"Text cleaning failed: {e}")
            return raw_text.strip()

    async def _describe_image_with_llm(self, image_base64: str) -> str:
        """Mô tả hình ảnh bằng LLM với Gemini Vision API"""

        if not llm_service.is_available():
            return "Hình ảnh minh họa trong sách giáo khoa"

        try:
            if not llm_service.model:
                return "Hình ảnh minh họa trong sách giáo khoa"

            # Sử dụng Gemini để mô tả hình ảnh
            prompt = """
Bạn là chuyên gia phân tích hình ảnh trong sách giáo khoa Việt Nam.
Hãy mô tả hình ảnh này một cách chi tiết và hữu ích cho việc tạo giáo án.

YÊU CẦU MÔ TẢ:
1. Xác định loại hình ảnh: biểu đồ, công thức, sơ đồ, hình minh họa, bảng biểu, thí nghiệm
2. Mô tả nội dung chính và các yếu tố quan trọng
3. Giải thích mục đích giáo dục và cách sử dụng trong giảng dạy
4. Đề xuất cách giải thích cho học sinh
5. Mô tả ngắn gọn, rõ ràng bằng tiếng Việt (tối đa 200 từ)

Ví dụ format mong muốn:
"Biểu đồ chu trình nước trong tự nhiên, minh họa quá trình bay hơi, ngưng tụ và mưa.
Hình ảnh này giúp học sinh hiểu rõ các giai đoạn của chu trình nước và vai trò của
mặt trời trong quá trình này. Có thể sử dụng để giải thích hiện tượng thời tiết và
tầm quan trọng của nước trong hệ sinh thái."

Hãy mô tả hình ảnh:"""

            # Tạo image part cho Gemini
            import base64
            from PIL import Image
            import io

            # Decode base64 để kiểm tra và resize nếu cần
            img_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(img_data))

            # Resize nếu ảnh quá lớn để tiết kiệm API cost
            max_size = (800, 800)
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Convert back to base64
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                img_data = buffer.getvalue()
                image_base64 = base64.b64encode(img_data).decode()

            # Tạo content với image để Gemini phân tích
            image_part = {
                "mime_type": "image/png",
                "data": base64.b64decode(image_base64),
            }

            response = llm_service.model.generate_content([prompt, image_part])
            description = response.text.strip()

            # Validate và clean description
            if description and len(description) > 10:
                # Giới hạn độ dài mô tả
                if len(description) > 500:
                    description = description[:500] + "..."
                return description
            else:
                return "Hình ảnh minh họa trong sách giáo khoa (không thể tạo mô tả chi tiết)"

        except Exception as e:
            logger.error(f"Image description with LLM failed: {e}")
            # Fallback descriptions based on context
            fallback_descriptions = [
                "Biểu đồ hoặc sơ đồ minh họa khái niệm trong bài học",
                "Hình ảnh thí nghiệm hoặc thực hành trong phòng lab",
                "Công thức toán học hoặc phương trình hóa học",
                "Bảng biểu thống kê hoặc dữ liệu khoa học",
                "Hình minh họa cấu trúc hoặc quy trình tự nhiên",
                "Sơ đồ tư duy hoặc bản đồ khái niệm",
            ]
            import random

            return random.choice(fallback_descriptions)

    async def _add_image_descriptions(self, pages_data: List[Dict[str, Any]]) -> None:
        """Add LLM-generated descriptions for all images in pages_data"""

        if not llm_service.is_available():
            logger.warning("LLM not available for image descriptions")
            return

        image_tasks = []
        for page in pages_data:
            for img in page.get("images", []):
                if img.get("data"):
                    image_tasks.append(self._describe_image_with_llm(img["data"]))

        if image_tasks:
            logger.info(f"🖼️ Generating descriptions for {len(image_tasks)} images...")
            descriptions = await asyncio.gather(*image_tasks, return_exceptions=True)
    
            # Apply descriptions back to images
            desc_index = 0
            for page in pages_data:
                for img in page.get("images", []):
                    if img.get("data") and desc_index < len(descriptions):
                        if not isinstance(descriptions[desc_index], Exception):
                            img["description"] = descriptions[desc_index]
                            img["description_method"] = "llm_generated"
                        else:
                            img["description"] = (
                                "Hình ảnh minh họa trong sách giáo khoa"
                            )
                            img["description_method"] = "fallback"
                        desc_index += 1

    async def _build_final_structure(
        self,
        analysis_result: Dict[str, Any],
        pages_data: List[Dict[str, Any]],
        book_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build final book structure from analysis result"""

        book_structure = {
            "title": analysis_result.get("book_info", {}).get(
                "title", book_metadata["title"]
            ),
            "subject": analysis_result.get("book_info", {}).get(
                "subject", book_metadata["subject"]
            ),
            "grade": analysis_result.get("book_info", {}).get(
                "grade", book_metadata["grade"]
            ),
            "chapters": [],
        }

        # Process chapters and lessons
        for chapter in analysis_result.get("chapters", []):
            chapter_obj = {
                "title": chapter.get("chapter_title", "Chương không xác định"),
                "lessons": [],
            }

            # Extract lessons
            for lesson in chapter.get("lessons", []):
                lesson_content = ""
                lesson_images = []
                start_page = lesson.get("start_page", 1)
                end_page = lesson.get("end_page", start_page)

                # Collect content and images from lesson pages
                for page in pages_data:
                    page_num = page.get("page_number", 0)
                    if start_page <= page_num <= end_page:
                        lesson_content += page.get("text", "") + "\n"
                        # Add images with descriptions
                        for img in page.get("images", []):
                            lesson_images.append(
                                {
                                    "page": page_num,
                                    "description": img.get(
                                        "description", "Hình ảnh minh họa"
                                    ),
                                    "format": img.get("format", "png"),
                                }
                            )

                lesson_obj = {
                    "title": lesson.get("lesson_title", "Bài học không xác định"),
                    "content": lesson_content.strip(),
                    "page_numbers": list(range(start_page, end_page + 1)),
                    "images": lesson_images,
                }
                chapter_obj["lessons"].append(lesson_obj)

            book_structure["chapters"].append(chapter_obj)

        return book_structure


# Global instance
enhanced_textbook_service = EnhancedTextbookService()
