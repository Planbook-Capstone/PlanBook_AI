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
            logger.info("📄 Đang trích xuất trang với OCR...")
            pages_data = await self._extract_pages_with_ocr(pdf_content)
            logger.info(f"✅ Extracted {len(pages_data)} pages")

            # Skip image analysis to improve speed
            logger.info("⚡ Bỏ qua phân tích hình ảnh để xử lý nhanh hơn")

            # Step 2: Combine all text from pages
            logger.info("📝 Đang kết hợp văn bản từ tất cả các trang...")
            full_text = ""
            all_page_numbers = []
            for page in pages_data:
                full_text += page.get("text", "") + "\n"
                all_page_numbers.append(page.get("page_number", 0))

            logger.info(f"📄 ALl Data {len(pages_data)} pages")
            logger.info(f"📄 Combined text from {len(pages_data)} pages")

            # Step 3: Refine content directly with OpenRouter LLM
            logger.info("🤖 Đang tinh chỉnh nội dung với OpenRouter LLM...")
            refined_content = await self.refine_raw_content_with_llm(full_text)
            logger.info("✅ Hoàn thành tinh chỉnh nội dung")
            logger.info("✅ Content refinement {refined_content}")
            # Step 4: Return clean text content directly
            logger.info("✅ Hoàn thành xử lý nội dung")

            # Skip image extraction for faster processing
            images_data = []
            logger.info("⚡ Bỏ qua trích xuất hình ảnh để xử lý nhanh hơn")

            return {
                "success": True,
                "clean_book_structure": refined_content,  # Return clean text directly
                "images_data": images_data,  # Empty array
                "total_pages": len(pages_data),
                "message": f"Xử lý sách giáo khoa thành công với tinh chỉnh nội dung LLM",
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
                f"🔍 Đang áp dụng OCR cho {len(ocr_tasks)} trang có văn bản không đủ"
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
        """Làm sạch nội dung text - format đơn giản, dễ đọc"""
        if not text:
            return ""

        logger.info("🧹 Cleaning text content...")

        # Loại bỏ markdown và ký tự đặc biệt
        cleaned_text = text.strip()
        cleaned_text = re.sub(r'\*+', '', cleaned_text)
        cleaned_text = re.sub(r'#+\s*', '', cleaned_text)
        cleaned_text = re.sub(r'_+', '', cleaned_text)
        cleaned_text = re.sub(r'`+', '', cleaned_text)
        cleaned_text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', cleaned_text)
        cleaned_text = cleaned_text.replace('\b', '').replace('\r', '')

        # Loại bỏ các tham chiếu SGK và hướng dẫn tham khảo
        sgk_patterns = [
            r'SGK\s+trang\s+\d+',  # SGK trang X
            r'Xem\s+bảng\s+\d+\.\d+\s*\([^)]*SGK[^)]*\)',  # Xem bảng X.X (SGK trang Y)
            r'Quan\s+sát\s+hình\s+\d+\.\d+\s*\([^)]*SGK[^)]*\)',  # Quan sát hình X.X (SGK trang Y)
            r'Tìm\s+hiểu\s+về[^.]*\([^)]*SGK[^)]*\)',  # Tìm hiểu về... (SGK trang Y)
            r'Xem\s+bảng\s+\d+\.\d+[^.]*',  # Xem bảng X.X về...
            r'Quan\s+sát\s+hình\s+\d+\.\d+[^.]*',  # Quan sát hình X.X về...
            r'\([^)]*SGK\s+trang[^)]*\)',  # (SGK trang X)
            r'- Xem bảng[^.]*\.',  # - Xem bảng...
            r'- Quan sát[^.]*\.',  # - Quan sát...
            r'- Tìm hiểu[^.]*\.',  # - Tìm hiểu...
        ]

        for pattern in sgk_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)

        # Loại bỏ các dòng chỉ chứa tham chiếu hoặc hướng dẫn
        lines = cleaned_text.split('\n')
        filtered_lines = []
        for line in lines:
            line_stripped = line.strip()
            # Bỏ qua các dòng chỉ chứa tham chiếu
            if (line_stripped.lower().startswith(('xem ', 'quan sát ', 'tìm hiểu ')) and
                len(line_stripped) < 100):  # Các câu hướng dẫn thường ngắn
                continue
            if line_stripped:  # Chỉ giữ các dòng có nội dung
                filtered_lines.append(line)
        cleaned_text = '\n'.join(filtered_lines)

        # Loại bỏ bảng markdown (xử lý sau khi đã lọc tham chiếu SGK)
        lines = cleaned_text.split('\n')
        final_filtered_lines = []
        for line in lines:
            line_stripped = line.strip()
            if (line_stripped.count('|') >= 2 or
                (line_stripped.count('-') >= 3 and '|' in line_stripped)):
                continue
            final_filtered_lines.append(line)
        cleaned_text = '\n'.join(final_filtered_lines)

        # Chuẩn hóa định dạng hóa học
        cleaned_text = self._normalize_chemistry_format(cleaned_text)

        # Thay thế xuống dòng bằng <br/>
        cleaned_text = cleaned_text.replace('\n', '<br/>')

        # FORMAT MỚI: Đơn giản và dễ đọc
        # 1. Định nghĩa: Giữ nguyên trên 1 dòng
        cleaned_text = re.sub(r'Định nghĩa:\s*<br/>', 'Định nghĩa: ', cleaned_text)

        # 2. Các tiêu đề chính: In đậm, không thụt lề
        main_sections = ['Biểu hiện', 'Ví dụ', 'Ý nghĩa', 'Nhận biết', 'Trình bày', 'Vận dụng', 'Phân tích', 'Đánh giá']
        for section in main_sections:
            cleaned_text = cleaned_text.replace(f'<br/>{section}:', f'<br/><strong>{section}:</strong>')
            if cleaned_text.startswith(f'{section}:'):
                cleaned_text = cleaned_text.replace(f'{section}:', f'<strong>{section}:</strong>', 1)

        # 3. Các mục con: Dùng bullet point (•) thay vì thụt lề nhiều
        lines = cleaned_text.split('<br/>')
        result_lines = []

        for line in lines:
            line = line.strip()
            if line:
                # Chỉ thêm bullet cho các mục con (không phải tiêu đề)
                if (re.match(r'^[A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]', line)
                    and '<strong>' not in line
                    and not line.startswith('Định nghĩa:')):
                    result_lines.append('• ' + line)
                else:
                    result_lines.append(line)

        cleaned_text = '<br/>'.join(result_lines)

        # Loại bỏ khoảng trắng thừa và <br/> liên tiếp
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text).strip()
        cleaned_text = re.sub(r'(<br/>){2,}', '<br/>', cleaned_text)

        logger.info(f"🧹 Text cleaned: {len(text)} → {len(cleaned_text)} chars")
        return cleaned_text

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

        # Chuyển đổi các công thức hóa học thô (không có HTML tags)
        # Pattern: chuyển số thành subscript cho tất cả ký hiệu nguyên tố
        # VD: CH3, H2O, C6H12O6, Ca(OH)2, Al2(SO4)3, C2(H2O)2
        # Tất cả số sau dấu ngoặc đóng đều chuyển thành subscript

        # Không cần bảo vệ gì cả - tất cả số đều chuyển thành subscript
        protected_text = text

        # 2. Chuyển đổi subscript cho tất cả ký hiệu nguyên tố
        # Pattern 1: Số ngay sau ký hiệu nguyên tố (VD: H2, O2, Ca2)
        chemistry_pattern = r'([A-Z][a-z]?)(\d+)'

        # Pattern 2: Số sau dấu ngoặc đóng (VD: (OH)2, (SO4)3)
        parenthesis_pattern = r'\)(\d+)'

        def replace_chemistry(match):
            element = match.group(1)
            number = match.group(2)
            # Chuyển số thành subscript
            subscript_number = ''.join(subscript_map.get(digit, digit) for digit in number)
            return element + subscript_number

        def replace_parenthesis(match):
            number = match.group(1)
            # Chuyển số thành subscript
            subscript_number = ''.join(subscript_map.get(digit, digit) for digit in number)
            return ')' + subscript_number

        # Áp dụng cả hai pattern
        protected_text = re.sub(chemistry_pattern, replace_chemistry, protected_text)
        text = re.sub(parenthesis_pattern, replace_parenthesis, protected_text)

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
3. TIỂU MỤC: Sử dụng "I.", "II.", "1.", "2." cho các phần chính

YÊU CẦU NỘI DUNG:
- Giữ lại toàn bộ khái niệm, định nghĩa, công thức, ví dụ quan trọng
- Loại bỏ header, footer, số trang, watermark không liên quan
- LOẠI BỎ HOÀN TOÀN tất cả các bảng dạng markdown (| col1 | col2 |) vì khó hiển thị trên UI
- LOẠI BỎ HOÀN TOÀN các tham chiếu sách giáo khoa như "SGK trang X", "Xem bảng X.X (SGK trang Y)", "Quan sát hình X.X (SGK trang Y)" vì chúng ảnh hưởng đến chất lượng lesson plan
- LOẠI BỎ các câu chỉ dẫn tham chiếu như "Tìm hiểu về...", "Xem bảng...", "Quan sát hình..." mà không có nội dung cụ thể
- Giữ nguyên thuật ngữ khoa học và công thức (H2O, 1.672 x 10^-27, etc.)
- Đảm bảo mỗi bài tập/ví dụ có đầy đủ đề bài và lời giải
- Tập trung vào nội dung kiến thức cốt lõi, không cần các phần hướng dẫn tham khảo

ĐỊNH DẠNG XUẤT:
- Text thuần túy, không markdown, KHÔNG có bảng
- Mỗi phần cách nhau bằng dòng trống
- Giữ nguyên ký hiệu khoa học (^, ², ³, →, ←, etc.)
- Nội dung phải độc lập, không cần tham chiếu external

NỘI DUNG CẦN CHỈNH SỬA:
{raw_text[:8000]}

Trả về nội dung đã được cấu trúc lại theo yêu cầu (KHÔNG bao gồm bảng và tham chiếu SGK):"""

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
            logger.error(f"❌ Lỗi tinh chỉnh nội dung với OpenRouter: {e}")
            return self.clean_text_content(raw_text)




# Factory function để tạo EnhancedTextbookService instance
def get_enhanced_textbook_service() -> EnhancedTextbookService:
    """
    Tạo EnhancedTextbookService instance mới

    Returns:
        EnhancedTextbookService: Fresh instance
    """
    return EnhancedTextbookService()
