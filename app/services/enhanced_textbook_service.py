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

from app.services.simple_ocr_service import get_simple_ocr_service

logger = logging.getLogger(__name__)


class EnhancedTextbookService:
    """Service cải tiến để xử lý sách giáo khoa với OCR và LLM"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.ocr_service = get_simple_ocr_service()

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
            text = await self.ocr_service._ocr_image(image, page_data['page_number'])
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

        # Chuẩn hóa định dạng hóa học trước tiên để tránh xung đột với dấu -
        cleaned_text = self._normalize_chemistry_format(cleaned_text)

        # Bước 1: Chuẩn hóa format cơ bản
        # Thêm khoảng cách sau dấu : nếu thiếu
        cleaned_text = re.sub(r':([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ])', r': \1', cleaned_text)

        # Bước 2: Xử lý các tiêu đề phần học tập (in nghiêng)
        learning_sections = ['Nhận biết', 'Trình bày', 'Vận dụng', 'Phân tích', 'Đánh giá', 'Tổng hợp']
        for title in learning_sections:
            # Loại bỏ dấu - trước tiêu đề phần
            cleaned_text = re.sub(rf'- {title}:', f'{title}:', cleaned_text)
            # Format in nghiêng cho tiêu đề phần học tập
            cleaned_text = re.sub(rf'([^<br/>]){title}:', rf'\1<br/><em>{title}:</em>', cleaned_text)
            cleaned_text = re.sub(rf'^{title}:', f'<em>{title}:</em>', cleaned_text)

        # Bước 3: Xử lý các tiêu đề năng lực (in nghiêng)
        capacity_sections = ['Năng lực chung', 'Năng lực đặc thù', 'Năng lực giao tiếp và hợp tác', 'Năng lực giải quyết vấn đề và sáng tạo', 'Năng lực vận dụng kiến thức, kĩ năng đã học']
        for title in capacity_sections:
            # Format in nghiêng cho tiêu đề năng lực
            cleaned_text = re.sub(rf'([^<br/>]){title}:', rf'\1<br/><em>{title}:</em>', cleaned_text)
            cleaned_text = re.sub(rf'^{title}:', f'<em>{title}:</em>', cleaned_text)

        # Bước 4: Xử lý danh sách - thêm dấu - cho mục con (không thụt lùi)
        # Xử lý sau dấu chấm
        cleaned_text = re.sub(r'\.(\s*)([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ])', r'.<br/>- \2', cleaned_text)

        # Xử lý sau dấu hai chấm (không phải tiêu đề)
        cleaned_text = re.sub(r':(\s*)([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ])(?![^<]*</em>)', r':<br/>- \2', cleaned_text)

        # Xử lý sau <br/> (chưa có dấu -)
        cleaned_text = re.sub(r'<br/>(\s*)([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ])(?![^<]*</em>)', r'<br/>- \2', cleaned_text)

        # Bước 5: Xử lý các trường hợp đặc biệt
        # Thay thế ". - " bằng ".<br/>- "
        cleaned_text = re.sub(r'\.\s*-\s*', '.<br/>- ', cleaned_text)

        # Xử lý ký hiệu + thành -
        cleaned_text = re.sub(r'<br/>\+([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ])', r'<br/>- \1', cleaned_text)

        # Bước 6: Loại bỏ thụt lùi không nhất quán - chuẩn hóa tất cả về dạng không thụt lùi
        cleaned_text = re.sub(r'<br/>&nbsp;&nbsp;&nbsp;&nbsp;-\s*', '<br/>- ', cleaned_text)
        cleaned_text = re.sub(r'&nbsp;&nbsp;&nbsp;&nbsp;-\s*', '- ', cleaned_text)

        # Loại bỏ khoảng trắng thừa ở đầu và cuối mỗi dòng nhưng giữ lại cấu trúc xuống hàng
        lines = cleaned_text.split('\n')
        cleaned_lines = [line.strip() for line in lines]

        # Ghép lại với việc giữ nguyên các dòng trống (xuống hàng)
        final_text = '\n'.join(cleaned_lines)

        # Loại bỏ khoảng trắng thừa trong mỗi dòng nhưng giữ lại xuống hàng
        final_text = re.sub(r'[ \t]+', ' ', final_text).strip()

        # Thay thế ký tự xuống hàng bằng thẻ <br/> để frontend hiển thị đúng
        final_text = final_text.replace('\n', '<br/>')

        # Xử lý thêm trường hợp có <br/> liên tiếp
        # Thay thế 2 hoặc nhiều <br/> liên tiếp bằng chỉ 1 <br/>
        final_text = re.sub(r'(<br/>){2,}', '<br/>', final_text)

        # Xử lý trường hợp đặc biệt: sau dấu : cần có <br/> để tách phần
        final_text = re.sub(r':(\s*)<br/>', ':<br/>', final_text)

        logger.info(f"🧹 Text cleaned: {len(text)} → {len(final_text)} chars")
        return final_text

    def _normalize_chemistry_format(self, text: str) -> str:
        """
        Chuyển đổi định dạng hóa học từ HTML sang định dạng chuẩn
        VD: <sup>6</sup>Li -> ⁶Li, S<sub>8</sub> -> S₈, Fe<sup>2+</sup> -> Fe²⁺
        """
        if not text:
            return text

        # Chuyển đổi superscript với số và ký hiệu (chỉ số trên)
        # Chỉ xử lý dấu + và - khi chúng nằm trong thẻ <sup>
        sup_pattern = r'<sup>([^<]+)</sup>'

        def replace_sup(match):
            content = match.group(1)
            result = ''
            superscript_map = {
                '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
                '+': '⁺', '-': '⁻'
            }
            for char in content:
                result += superscript_map.get(char, char)
            return result

        text = re.sub(sup_pattern, replace_sup, text)

        # Chuyển đổi subscript (chỉ số dưới)
        sub_pattern = r'<sub>(\d+)</sub>'
        subscript_map = {
            '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
            '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'
        }

        def replace_sub(match):
            number = match.group(1)
            return ''.join(subscript_map.get(digit, digit) for digit in number)

        text = re.sub(sub_pattern, replace_sub, text)

        return text

    async def refine_raw_content_with_llm(self, raw_text: str) -> str:
        """Gửi text thô trực tiếp đến OpenRouter LLM để lọc và chỉnh sửa nội dung"""
        try:
            from app.services.openrouter_service import get_openrouter_service

            openrouter_service = get_openrouter_service()
            logger.info("🤖 Sending raw content to OpenRouter for refinement...")

            prompt = f"""
Bạn là chuyên gia giáo dục, hãy lọc và cấu trúc lại nội dung sách giáo khoa để tối ưu cho hệ thống chunking thông minh.

YÊU CẦU CẤU TRÚC:
1. ĐỊNH NGHĨA: Bắt đầu bằng "Định nghĩa:" hoặc sử dụng cấu trúc "X là..." rõ ràng
2. BÀI TẬP/VÍ DỤ: Đánh số rõ ràng "Bài 1.", "Ví dụ 1:", "Hãy cho biết..."
3. BẢNG BIỂU: Bắt đầu bằng "Bảng X:" và giữ nguyên cấu trúc bảng
4. TIỂU MỤC: Sử dụng "I.", "II.", "1.", "2." cho các phần chính

YÊU CẦU NỘI DUNG:
- Giữ lại toàn bộ khái niệm, định nghĩa, công thức, ví dụ quan trọng
- Loại bỏ header, footer, số trang, watermark không liên quan
- Giữ nguyên thuật ngữ khoa học và công thức (H2O, 1.672 x 10^-27, etc.)
- Đảm bảo mỗi bài tập/ví dụ có đầy đủ đề bài và lời giải
- Bảng phải hoàn chỉnh với tiêu đề và nội dung

ĐỊNH DẠNG XUẤT:
- Text thuần túy, không markdown
- Mỗi phần cách nhau bằng dòng trống
- Giữ nguyên ký hiệu khoa học (^, ², ³, →, ←, etc.)

NỘI DUNG CẦN CHỈNH SỬA:
{raw_text[:8000]}

Trả về nội dung đã được cấu trúc lại theo yêu cầu:"""

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




# Factory function để tạo EnhancedTextbookService instance
def get_enhanced_textbook_service() -> EnhancedTextbookService:
    """
    Tạo EnhancedTextbookService instance mới

    Returns:
        EnhancedTextbookService: Fresh instance
    """
    return EnhancedTextbookService()
