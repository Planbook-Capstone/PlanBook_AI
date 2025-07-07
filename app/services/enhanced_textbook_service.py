"""
Enhanced Textbook Service - Cải tiến xử lý sách giáo khoa với OCR và LLM
Trả về cấu trúc: Sách → Chương → Bài → Nội dung
"""

import logging
import asyncio
import re
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

            # Step 2: Combine all text from pages
            logger.info("📝 Combining text from all pages...")
            full_text = ""
            all_page_numbers = []
            for page in pages_data:
                full_text += page.get("text", "") + "\n"
                all_page_numbers.append(page.get("page_number", 0))

            logger.info(f"📄 ALl Data {len(pages_data)} pages")
            logger.info(f"📄 Combined text from {len(pages_data)} pages")

            # Step 3: Refine content directly with OpenRouter LLM
            logger.info("🤖 Refining content with OpenRouter LLM...")
            refined_content = await self.refine_raw_content_with_llm(full_text)
            logger.info("✅ Content refinement completed")
            logger.info("✅ Content refinement {refined_content}")
            # Step 4: Return clean text content directly
            logger.info("✅ Content processing completed")

            # Skip image extraction for faster processing
            images_data = []
            logger.info("⚡ Skipping image extraction for faster processing")

            return {
                "success": True,
                "clean_book_structure": refined_content,  # Return clean text directly
                "images_data": images_data,  # Empty array
                "total_pages": len(pages_data),
                "message": f"Textbook processed successfully with LLM content refinement",
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

            # Apply OCR using SimpleOCRService
            image = Image.open(io.BytesIO(img_data))

            # Use SimpleOCRService's _ocr_image method which handles EasyOCR initialization
            text = await simple_ocr_service._ocr_image(image, page_data['page_number'])
            return text

        except Exception as e:
            logger.error(f"OCR failed for page {page_data['page_number']}: {e}")
            return ""





















    def clean_text_content(self, text: str) -> str:
        """Làm sạch nội dung text - loại bỏ ký tự đặc biệt, format không cần thiết"""
        if not text:
            return ""

        logger.info("🧹 Cleaning text content...")

        # Loại bỏ các ký tự đặc biệt và format markdown
        cleaned_text = text

        # Loại bỏ dấu * (markdown bold/italic)
        cleaned_text = re.sub(r'\*+', '', cleaned_text)

        # Loại bỏ dấu # (markdown headers)
        cleaned_text = re.sub(r'#+\s*', '', cleaned_text)

        # Loại bỏ dấu _ (markdown underline)
        cleaned_text = re.sub(r'_+', '', cleaned_text)

        # Loại bỏ dấu ` (markdown code)
        cleaned_text = re.sub(r'`+', '', cleaned_text)

        # Loại bỏ dấu [] và () (markdown links)
        cleaned_text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', cleaned_text)

        # Loại bỏ ký tự \b (backspace)
        cleaned_text = cleaned_text.replace('\b', '')

        # Loại bỏ ký tự \r
        cleaned_text = cleaned_text.replace('\r', '')

        # Thay thế nhiều dấu xuống dòng liên tiếp bằng 1 dấu xuống dòng
        cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)

        # Loại bỏ khoảng trắng thừa ở đầu và cuối mỗi dòng
        lines = cleaned_text.split('\n')
        cleaned_lines = [line.strip() for line in lines if line.strip()]

        # Ghép lại thành một đoạn text liên tục
        final_text = ' '.join(cleaned_lines)

        # Loại bỏ khoảng trắng thừa
        final_text = re.sub(r'\s+', ' ', final_text).strip()

        logger.info(f"🧹 Text cleaned: {len(text)} → {len(final_text)} chars")
        return final_text

    async def refine_raw_content_with_llm(self, raw_text: str) -> str:
        """Gửi text thô trực tiếp đến OpenRouter LLM để lọc và chỉnh sửa nội dung"""
        try:
            from app.services.openrouter_service import get_openrouter_service

            openrouter_service = get_openrouter_service()
            if not openrouter_service.available:
                logger.warning("OpenRouter service not available, returning original content")
                return self.clean_text_content(raw_text)

            logger.info("🤖 Sending raw content to OpenRouter for refinement...")

            prompt = f"""
Bạn là chuyên gia giáo dục, hãy lọc và chỉnh sửa nội dung sách giáo khoa sau đây.

YÊU CẦU:
1. Giữ lại toàn bộ nội dung giáo dục quan trọng (khái niệm, định nghĩa, công thức, ví dụ)
2. Loại bỏ các thông tin không liên quan (header, footer, số trang, watermark)
3. Sắp xếp lại nội dung theo logic rõ ràng
4. Giữ nguyên thuật ngữ chuyên môn
5. Đảm bảo nội dung đầy đủ và dễ hiểu cho học sinh
6. Trả về CHỈ NỘI DUNG THUẦN TÚY, không có format markdown, không có ký tự đặc biệt

NỘI DUNG CẦN CHỈNH SỬA:
{raw_text[:8000]}

Trả về nội dung đã được lọc và chỉnh sửa (chỉ text thuần túy):"""

            result = await openrouter_service.generate_content(
                prompt=prompt,
                temperature=0.1,
                max_tokens=2048
            )

            if result.get("success") and result.get("text"):
                refined_content = result.get("text", "").strip()
                # Làm sạch nội dung sau khi nhận từ OpenRouter
                clean_content = self.clean_text_content(refined_content)
                logger.info(f"🤖{prompt} ")
                logger.info(f"🤖{refined_content} ")
                logger.info(f"✅ Content refined and cleaned: {len(raw_text)} → {len(clean_content)} chars")
                return clean_content
            else:
                logger.warning("OpenRouter returned insufficient content, using original")
                return self.clean_text_content(raw_text)

        except Exception as e:
            logger.error(f"❌ Error refining content with OpenRouter: {e}")
            return self.clean_text_content(raw_text)




# Global instance
enhanced_textbook_service = EnhancedTextbookService()
