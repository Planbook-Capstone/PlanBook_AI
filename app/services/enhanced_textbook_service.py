"""
Enhanced Textbook Service - Cải tiến xử lý sách giáo khoa với OCR và LLM
Trả về cấu trúc: Sách → Chương → Bài → Nội dung
"""

import logging
import asyncio
import json
import uuid
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
from PIL import Image
import io

from app.services.simple_ocr_service import simple_ocr_service

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
        lesson_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Xử lý PDF sách giáo khoa và trả về cấu trúc hoàn chỉnh

        Args:
            pdf_content: Nội dung PDF
            filename: Tên file
            book_metadata: Metadata sách (title, subject, grade, etc.)
            lesson_id: ID bài học tùy chọn để liên kết với lesson cụ thể

        Returns:
            Dict với cấu trúc: book -> chapters -> lessons -> content
        """
        try:
            logger.info(f"🚀 Starting enhanced textbook processing: {filename}")

            # Step 1: Extract all pages with OCR
            logger.info("📄 Extracting pages with OCR...")
            pages_data = await self._extract_pages_with_ocr(pdf_content)
            logger.info(f"✅ Extracted {len(pages_data)} pages")

            # Skip image analysis to improve speed
            logger.info("⚡ Skipping image analysis for faster processing")

            # Step 3: Analyze book structure with LLM
            logger.info("🧠 Analyzing book structure...")
            book_structure = await self._analyze_book_structure_enhanced(
                pages_data, book_metadata
            )
            logger.info(
                f"📚 Detected {len(book_structure.get('chapters', []))} chapters"
            )

            # Step 4: Build final structure with content and lesson IDs
            logger.info("🔄 Building final lesson structure...")
            processed_book = await self._build_final_structure(
                book_structure,
                pages_data,
                book_metadata or {},
                lesson_id,  # Pass lesson_id
            )

            logger.info("✅ Textbook processing completed successfully")

            # Skip image extraction for faster processing
            images_data = []
            logger.info("⚡ Skipping image extraction for faster processing")

            # Refine content with OpenRouter LLM before saving to Qdrant
            logger.info("🤖 Refining content with OpenRouter LLM...")
            refined_book_structure = await self.refine_content_with_llm(processed_book)
            logger.info("✅ Content refinement completed")

            # Prepare clean structure for Qdrant
            clean_book_structure = self.prepare_structure_for_qdrant(refined_book_structure)

            # Tính toán thống kê đơn giản cho 1 bài học
            total_lessons = sum(len(ch.get("lessons", [])) for ch in refined_book_structure.get("chapters", []))

            return {
                "success": True,
                "book": refined_book_structure,  # Full structure with refined content and image data
                "clean_book_structure": clean_book_structure,  # Structure without image data for Qdrant
                "images_data": images_data,  # Separate image data for external storage
                "total_pages": len(pages_data),
                "total_chapters": len(refined_book_structure.get("chapters", [])),
                "total_lessons": total_lessons,
                "total_images": len(images_data),
                "message": f"Textbook processed successfully with LLM content refinement ({total_lessons} lesson{'s' if total_lessons != 1 else ''})",
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

                # Skip image extraction for faster processing
                images = []

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
                text_parts = []
                for result in results:
                    if isinstance(result, (list, tuple)) and len(result) >= 2:
                        # EasyOCR returns [bbox, text, confidence]
                        text_parts.append(str(result[1]))
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
        """Tạo cấu trúc đơn giản: 1 PDF = 1 bài học"""

        logger.info("📚 Creating simple structure: 1 PDF = 1 lesson")

        # Tạo cấu trúc đơn giản cố định - không cần LLM phức tạp
        total_pages = len(pages_data)

        # Lấy tiêu đề từ metadata hoặc trang đầu
        title = "Bài học"
        if book_metadata and book_metadata.get("title"):
            title = book_metadata["title"]
        else:
            # Thử extract tiêu đề từ trang đầu
            if pages_data and pages_data[0]["text"].strip():
                first_lines = pages_data[0]["text"].strip().split('\n')[:5]
                for line in first_lines:
                    if len(line.strip()) > 5 and len(line.strip()) < 100:
                        title = line.strip()
                        break

        structure = {
            "book_info": {
                "title": title,
                "subject": book_metadata.get("subject", "Chưa xác định") if book_metadata else "Chưa xác định",
                "total_chapters": 1,
                "total_lessons": 1
            },
            "chapters": [
                {
                    "chapter_id": "chapter_01",
                    "chapter_title": "Nội dung chính",
                    "start_page": 1,
                    "end_page": total_pages,
                    "lessons": [
                        {
                            "lesson_id": "lesson_01",
                            "lesson_title": title,
                            "start_page": 1,
                            "end_page": total_pages
                        }
                    ]
                }
            ]
        }

        logger.info(f"✅ Created simple structure: 1 chapter, 1 lesson, {total_pages} pages")
        return structure











    async def _build_final_structure(
        self,
        analysis_result: Dict[str, Any],
        pages_data: List[Dict[str, Any]],
        book_metadata: Dict[str, Any],
        external_lesson_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Tạo cấu trúc đơn giản: 1 PDF = 1 bài học"""

        logger.info("🏗️ Building simple structure: 1 PDF = 1 lesson")

        # Tập hợp toàn bộ nội dung từ tất cả các trang
        full_content = ""
        all_page_numbers = []

        for page in pages_data:
            full_content += page.get("text", "") + "\n"
            all_page_numbers.append(page.get("page_number", 0))

        # Tạo lesson_id
        if external_lesson_id:
            lesson_id = external_lesson_id
            logger.info(f"Using provided lesson_id: {lesson_id}")
        else:
            lesson_id = str(uuid.uuid4())
            logger.info(f"Generated new lesson_id: {lesson_id}")

        # Lấy tiêu đề từ analysis_result hoặc metadata
        lesson_title = analysis_result.get("book_info", {}).get("title", book_metadata.get("title", "Bài học"))

        # Tạo cấu trúc đơn giản
        book_structure = {
            "title": lesson_title,
            "subject": analysis_result.get("book_info", {}).get("subject", book_metadata.get("subject", "Chưa xác định")),
            "grade": analysis_result.get("book_info", {}).get("grade", book_metadata.get("grade", "Chưa xác định")),
            "chapters": [
                {
                    "chapter_id": "chapter_01",
                    "title": "Nội dung chính",
                    "lessons": [
                        {
                            "lesson_id": lesson_id,
                            "title": lesson_title,
                            "content": full_content.strip(),
                            "page_numbers": all_page_numbers,
                        }
                    ]
                }
            ]
        }

        logger.info(f"✅ Created simple structure with lesson_id: {lesson_id}")
        return book_structure



    async def refine_content_with_llm(
        self, book_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Gửi nội dung đến OpenRouter LLM để lọc và chỉnh sửa nội dung bài giảng

        Tối ưu cho việc xử lý 1 bài học duy nhất trong mỗi PDF
        """
        try:
            from app.services.openrouter_service import OpenRouterService

            openrouter_service = OpenRouterService()
            if not openrouter_service.available:
                logger.warning("OpenRouter service not available, skipping content refinement")
                return book_structure

            import copy
            refined_structure = copy.deepcopy(book_structure)

            logger.info("🤖 Starting content refinement with OpenRouter LLM...")

            # Tìm bài học đầu tiên để xử lý (vì chỉ có 1 bài/PDF)
            first_lesson = None
            for chapter in refined_structure.get("chapters", []):
                for lesson in chapter.get("lessons", []):
                    if lesson.get("content"):
                        first_lesson = lesson
                        break
                if first_lesson:
                    break

            if not first_lesson:
                logger.warning("No lesson content found to refine")
                return book_structure

            # Tập hợp tất cả text content từ lesson
            all_text_content = []
            for content_item in first_lesson.get("content", []):
                if content_item.get("type") == "text" and content_item.get("text"):
                    all_text_content.append(content_item.get("text", ""))

            if not all_text_content:
                logger.warning("No text content found in lesson")
                return book_structure

            # Ghép nội dung lại
            combined_content = "\n\n".join(all_text_content)

            # Tạo prompt để LLM lọc nội dung
            prompt = f"""
Bạn là chuyên gia giáo dục, hãy lọc và chỉnh sửa nội dung bài giảng sau để chỉ giữ lại những thông tin quan trọng và chi tiết của bài giảng.

YÊU CẦU:
1. Loại bỏ thông tin không liên quan đến nội dung bài học (header, footer, số trang, thông tin xuất bản, etc.)
2. Giữ lại toàn bộ kiến thức chính, khái niệm, định nghĩa, công thức, ví dụ
3. Giữ lại các bài tập, câu hỏi, hoạt động thực hành
4. Sắp xếp nội dung theo logic rõ ràng, dễ hiểu
5. Đảm bảo nội dung đầy đủ và chính xác, không bỏ sót thông tin quan trọng
6. Trả về nội dung đã được chỉnh sửa bằng tiếng Việt

TIÊU ĐỀ BÀI HỌC: {first_lesson.get("lesson_title", "Không có tiêu đề")}

NỘI DUNG GỐC:
{combined_content[:3000]}  # Giới hạn 3000 ký tự để tránh vượt quá token limit

Hãy trả về nội dung đã được lọc và chỉnh sửa:
"""

            # Gọi OpenRouter API
            result = await openrouter_service.generate_content(
                prompt=prompt,
                temperature=0.1,
                max_tokens=2048
            )

            if result.get("success") and result.get("text"):
                refined_content = result.get("text", "").strip()

                # Cập nhật nội dung đã được chỉnh sửa
                first_lesson["content"] = [
                    {
                        "type": "text",
                        "text": refined_content,
                        "page": first_lesson["content"][0].get("page", 1) if first_lesson["content"] else 1,
                        "section": "refined_content",
                        "refined_by_llm": True
                    }
                ]

                logger.info(f"✅ Refined content for lesson: {first_lesson.get('lesson_title', 'Unknown')}")
            else:
                logger.warning(f"❌ Failed to refine content for lesson: {first_lesson.get('lesson_title', 'Unknown')}")

            logger.info("🎯 Content refinement completed")
            return refined_structure

        except Exception as e:
            logger.error(f"Error in content refinement: {e}")
            # Trả về cấu trúc gốc nếu có lỗi
            return book_structure

    def prepare_structure_for_qdrant(
        self, book_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare book structure for Qdrant storage (no image processing needed)"""
        # Since we skip image processing, just return the structure as-is
        return book_structure


# Global instance
enhanced_textbook_service = EnhancedTextbookService()
