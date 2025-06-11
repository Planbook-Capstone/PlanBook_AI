"""
Textbook Parser Service - Xử lý sách giáo khoa thành cấu trúc dữ liệu cho giáo án
"""
import logging
import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
from concurrent.futures import ThreadPoolExecutor

from app.services.llm_service import llm_service

logger = logging.getLogger(__name__)

class TextbookParserService:
    """Service để parse sách giáo khoa thành cấu trúc dữ liệu"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.output_base_path = Path("data/processed_textbooks")
        self.output_base_path.mkdir(parents=True, exist_ok=True)
    
    async def process_textbook(
        self, 
        pdf_content: bytes, 
        filename: str,
        book_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Xử lý sách giáo khoa thành cấu trúc dữ liệu
        
        Args:
            pdf_content: Nội dung PDF
            filename: Tên file
            book_metadata: Metadata của sách
            
        Returns:
            Dict chứa kết quả xử lý
        """
        try:
            logger.info(f"🚀 Starting textbook processing: {filename}")

            # Tạo thư mục cho sách
            book_id = book_metadata.get('id', filename.replace('.pdf', ''))
            book_path = self.output_base_path / book_id
            book_path.mkdir(exist_ok=True)

            # Tạo thư mục con
            lessons_path = book_path / "lessons"
            images_path = book_path / "images"
            lessons_path.mkdir(exist_ok=True)
            images_path.mkdir(exist_ok=True)

            logger.info(f"📁 Created directories for book: {book_id}")

            # Lưu metadata
            await self._save_metadata(book_path, book_metadata, filename)
            logger.info(f"💾 Saved metadata for book: {book_id}")

            # Extract và phân tích PDF
            logger.info(f"📄 Starting PDF extraction...")
            pages_data = await self._extract_pdf_pages(pdf_content)
            logger.info(f"✅ Extracted {len(pages_data)} pages from PDF")

            # Phân tích cấu trúc sách bằng LLM
            logger.info(f"🧠 Analyzing book structure with LLM...")
            book_structure = await self._analyze_book_structure(pages_data)
            logger.info(f"📚 Detected {len(book_structure.get('chapters', []))} chapters")

            # Xử lý từng chương và bài học
            chapters_processed = 0
            lessons_processed = 0
            total_chapters = len(book_structure.get('chapters', []))

            logger.info(f"🔄 Starting to process {total_chapters} chapters...")

            for i, chapter_data in enumerate(book_structure.get('chapters', []), 1):
                try:
                    logger.info(f"📖 Processing chapter {i}/{total_chapters}: {chapter_data.get('chapter_title', 'Unknown')}")

                    chapter_result = await self._process_chapter_with_lessons(
                        chapter_data,
                        pages_data,
                        lessons_path,
                        images_path
                    )
                    chapters_processed += 1
                    lessons_processed += chapter_result.get('lessons_count', 0)

                    logger.info(f"✅ Completed chapter {chapters_processed}/{total_chapters} - {chapter_result.get('lessons_count', 0)} lessons processed")
                    logger.info(f"📊 Progress: {chapters_processed}/{total_chapters} chapters, {lessons_processed} total lessons")

                except Exception as e:
                    logger.error(f"❌ Error processing chapter {chapter_data.get('chapter_title', 'Unknown')}: {e}")

            # Cập nhật metadata với số liệu thực tế
            book_metadata['chapters_count'] = chapters_processed
            book_metadata['lessons_count'] = lessons_processed
            book_metadata['book_structure'] = book_structure.get('book_info', {})
            await self._save_metadata(book_path, book_metadata, filename)
            
            return {
                "success": True,
                "book_id": book_id,
                "book_path": str(book_path),
                "lessons_processed": lessons_processed,
                "total_pages": len(pages_data),
                "message": f"Successfully processed {lessons_processed} lessons from textbook"
            }
            
        except Exception as e:
            logger.error(f"Error processing textbook: {e}")
            return {
                "success": False,
                "error": str(e),
                "book_id": book_metadata.get('id', filename),
                "message": "Failed to process textbook"
            }
    
    async def _save_metadata(self, book_path: Path, metadata: Dict[str, Any], filename: str):
        """Lưu metadata của sách"""
        metadata_file = book_path / "metadata.json"
        
        # Thêm thông tin bổ sung
        metadata.update({
            "original_filename": filename,
            "processed_date": str(asyncio.get_event_loop().time()),
            "structure_version": "1.0"
        })
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    async def _extract_pdf_pages(self, pdf_content: bytes) -> List[Dict[str, Any]]:
        """Extract pages từ PDF với text và images"""
        
        def extract_pages():
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            pages_data = []
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text
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
                            
                            images.append({
                                "index": img_index,
                                "data": img_base64,
                                "format": "png"
                            })
                        
                        pix = None
                    except Exception as e:
                        logger.warning(f"Error extracting image {img_index} from page {page_num}: {e}")
                
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "images": images
                })
            
            doc.close()
            return pages_data
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, extract_pages
        )
    
    async def _analyze_book_structure(self, pages_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Phân tích cấu trúc sách bằng LLM để extract thực tế"""

        if not llm_service.is_available():
            logger.warning("LLM not available, using basic structure analysis")
            return await self._basic_structure_analysis(pages_data)

        # Tạo text tổng hợp từ TẤT CẢ các trang để phân tích đầy đủ
        full_text = ""
        for page in pages_data:
            if page['text'].strip():  # Chỉ lấy trang có text
                full_text += f"\n--- Trang {page['page_number']} ---\n{page['text']}"

        logger.info(f"Analyzing {len(pages_data)} pages with LLM for structure detection")

        prompt = f"""
Bạn là chuyên gia phân tích sách giáo khoa Việt Nam. Hãy phân tích TOÀN BỘ nội dung sách và trả về cấu trúc thực tế.

YÊU CẦU PHÂN TÍCH:
1. Đọc kỹ TOÀN BỘ text từ tất cả các trang
2. Xác định các CHƯƠNG (Chapter) - thường có tiêu đề lớn như "CHƯƠNG 1", "CHƯƠNG I", etc.
3. Trong mỗi chương, xác định các BÀI (Lesson) - thường có tiêu đề như "Bài 1", "Bài 2", etc.
4. Xác định nội dung thực tế của từng bài
5. Ghi chú trang bắt đầu và kết thúc cho mỗi phần

CẤU TRÚC MONG MUỐN: SÁCH → CHƯƠNG → BÀI → NỘI DUNG

FORMAT JSON CHÍNH XÁC:
{{
  "book_info": {{
    "title": "Tên sách thực tế từ text",
    "total_chapters": số_chương_thực_tế,
    "total_lessons": số_bài_thực_tế,
    "subject": "môn_học_từ_nội_dung"
  }},
  "chapters": [
    {{
      "chapter_id": "chapter_01",
      "chapter_title": "Tên chương thực tế",
      "start_page": trang_bắt_đầu,
      "end_page": trang_kết_thúc,
      "lessons": [
        {{
          "lesson_id": "lesson_01",
          "lesson_title": "Tên bài thực tế",
          "start_page": trang_bắt_đầu,
          "end_page": trang_kết_thúc,
          "content_summary": "Tóm tắt nội dung bài học"
        }}
      ]
    }}
  ]
}}

QUAN TRỌNG:
- Phải đọc và phân tích TOÀN BỘ text, không bỏ sót
- Tên chương/bài phải chính xác từ text gốc
- Số trang phải chính xác
- Nếu không có cấu trúc rõ ràng, hãy tự phân chia hợp lý

Text từ sách giáo khoa (TOÀN BỘ):
{full_text[:8000]}

Trả về JSON cấu trúc thực tế:
"""

        try:
            response = llm_service.model.generate_content(prompt)
            json_text = response.text.strip()

            # Clean JSON
            if json_text.startswith('```json'):
                json_text = json_text[7:]
            if json_text.startswith('```'):
                json_text = json_text[3:]
            if json_text.endswith('```'):
                json_text = json_text[:-3]

            structure = json.loads(json_text)
            logger.info(f"LLM detected {len(structure.get('chapters', []))} chapters")
            return structure

        except Exception as e:
            logger.error(f"LLM structure analysis failed: {e}")
            return await self._basic_structure_analysis(pages_data)
    
    async def _basic_structure_analysis(self, pages_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Phân tích cấu trúc cơ bản không dùng LLM - theo format CHƯƠNG → BÀI"""

        total_pages = len(pages_data)

        # Phân tích text để tìm cấu trúc cơ bản
        chapters = []

        # Chia thành 2-3 chương, mỗi chương có 2-3 bài
        pages_per_chapter = max(total_pages // 3, 1)  # Ít nhất 3 chương

        for chapter_num in range(1, 4):  # Tạo 3 chương
            chapter_start = (chapter_num - 1) * pages_per_chapter + 1
            chapter_end = min(chapter_num * pages_per_chapter, total_pages)

            if chapter_start > total_pages:
                break

            # Tạo 2-3 bài trong mỗi chương
            lessons_in_chapter = []
            pages_per_lesson = max((chapter_end - chapter_start + 1) // 2, 1)

            for lesson_num in range(1, 3):  # 2 bài mỗi chương
                lesson_start = chapter_start + (lesson_num - 1) * pages_per_lesson
                lesson_end = min(chapter_start + lesson_num * pages_per_lesson - 1, chapter_end)

                if lesson_start > chapter_end:
                    break

                # Thử extract tiêu đề từ text
                lesson_title = self._extract_title_from_pages(pages_data, lesson_start - 1, lesson_end - 1)
                if not lesson_title:
                    lesson_title = f"Bài {len(lessons_in_chapter) + 1}"

                lessons_in_chapter.append({
                    "lesson_id": f"lesson_{chapter_num:02d}_{lesson_num:02d}",
                    "lesson_title": lesson_title,
                    "start_page": lesson_start,
                    "end_page": lesson_end,
                    "content_summary": f"Nội dung bài học từ trang {lesson_start} đến {lesson_end}"
                })

            # Thử extract tiêu đề chương từ text
            chapter_title = self._extract_chapter_title_from_pages(pages_data, chapter_start - 1, chapter_end - 1)
            if not chapter_title:
                chapter_title = f"Chương {chapter_num}"

            chapters.append({
                "chapter_id": f"chapter_{chapter_num:02d}",
                "chapter_title": chapter_title,
                "start_page": chapter_start,
                "end_page": chapter_end,
                "lessons": lessons_in_chapter
            })

        return {
            "book_info": {
                "title": "Sách giáo khoa",
                "total_chapters": len(chapters),
                "total_lessons": sum(len(ch['lessons']) for ch in chapters),
                "subject": "Chưa xác định"
            },
            "chapters": chapters
        }

    def _extract_title_from_pages(self, pages_data: List[Dict[str, Any]], start_idx: int, end_idx: int) -> str:
        """Extract tiêu đề từ các trang"""
        for i in range(start_idx, min(end_idx + 1, len(pages_data))):
            if i < 0:
                continue
            text = pages_data[i]['text']
            lines = text.split('\n')
            for line in lines[:5]:  # Kiểm tra 5 dòng đầu
                line = line.strip()
                if len(line) > 5 and len(line) < 100:
                    if any(keyword in line.lower() for keyword in ['bài', 'lesson', 'chương']):
                        return line
        return ""

    def _extract_chapter_title_from_pages(self, pages_data: List[Dict[str, Any]], start_idx: int, end_idx: int) -> str:
        """Extract tiêu đề chương từ các trang"""
        for i in range(start_idx, min(end_idx + 1, len(pages_data))):
            if i < 0:
                continue
            text = pages_data[i]['text']
            lines = text.split('\n')
            for line in lines[:3]:  # Kiểm tra 3 dòng đầu
                line = line.strip()
                if len(line) > 5 and len(line) < 100:
                    if any(keyword in line.lower() for keyword in ['chương', 'chapter', 'phần']):
                        return line
        return ""
    
    async def _process_lesson(
        self, 
        lesson_data: Dict[str, Any], 
        pages_data: List[Dict[str, Any]],
        lessons_path: Path,
        images_path: Path
    ):
        """Xử lý một bài học"""
        
        lesson_id = lesson_data['lesson_id']
        start_page = lesson_data.get('start_page', 1)
        end_page = lesson_data.get('end_page', len(pages_data))
        
        # Tạo thư mục cho images của bài học
        lesson_images_path = images_path / lesson_id
        lesson_images_path.mkdir(exist_ok=True)
        
        # Xử lý các chương trong bài học
        processed_chapters = []
        
        for chapter in lesson_data.get('chapters', []):
            chapter_content = await self._process_chapter(
                chapter, 
                pages_data, 
                start_page, 
                end_page,
                lesson_images_path
            )
            processed_chapters.append(chapter_content)
        
        # Tạo lesson JSON
        lesson_json = {
            "lesson_id": lesson_id,
            "title": lesson_data['title'],
            "chapters": processed_chapters
        }
        
        # Lưu lesson file
        lesson_file = lessons_path / f"{lesson_id}.json"
        with open(lesson_file, 'w', encoding='utf-8') as f:
            json.dump(lesson_json, f, indent=2, ensure_ascii=False)

    async def _process_chapter_with_lessons(
        self,
        chapter_data: Dict[str, Any],
        pages_data: List[Dict[str, Any]],
        lessons_path: Path,
        images_path: Path
    ) -> Dict[str, Any]:
        """Xử lý một chương với các bài học bên trong"""

        chapter_id = chapter_data['chapter_id']
        chapter_title = chapter_data['chapter_title']
        chapter_start = chapter_data.get('start_page', 1)
        chapter_end = chapter_data.get('end_page', len(pages_data))

        logger.info(f"Processing chapter: {chapter_title} (pages {chapter_start}-{chapter_end})")

        # Tạo thư mục cho images của chương
        chapter_images_path = images_path / chapter_id
        chapter_images_path.mkdir(exist_ok=True)

        lessons_count = 0

        # Xử lý từng bài học trong chương
        for lesson_data in chapter_data.get('lessons', []):
            try:
                await self._process_lesson_in_chapter(
                    lesson_data,
                    chapter_data,
                    pages_data,
                    lessons_path,
                    chapter_images_path
                )
                lessons_count += 1
                logger.info(f"Processed lesson: {lesson_data.get('lesson_title', 'Unknown')}")
            except Exception as e:
                logger.error(f"Error processing lesson {lesson_data.get('lesson_title', 'Unknown')}: {e}")

        return {
            "chapter_id": chapter_id,
            "lessons_count": lessons_count
        }

    async def _process_lesson_in_chapter(
        self,
        lesson_data: Dict[str, Any],
        chapter_data: Dict[str, Any],
        pages_data: List[Dict[str, Any]],
        lessons_path: Path,
        images_path: Path
    ):
        """Xử lý một bài học trong chương"""

        lesson_id = lesson_data['lesson_id']
        lesson_title = lesson_data['lesson_title']
        lesson_start = lesson_data.get('start_page', 1)
        lesson_end = lesson_data.get('end_page', len(pages_data))

        # Extract nội dung thực tế từ các trang
        lesson_content = await self._extract_lesson_content(
            lesson_start,
            lesson_end,
            pages_data,
            images_path
        )

        # Tạo lesson JSON với cấu trúc mới
        lesson_json = {
            "lesson_id": lesson_id,
            "lesson_title": lesson_title,
            "chapter_id": chapter_data['chapter_id'],
            "chapter_title": chapter_data['chapter_title'],
            "pages": {
                "start": lesson_start,
                "end": lesson_end
            },
            "content": lesson_content,
            "summary": lesson_data.get('content_summary', '')
        }

        # Lưu lesson file
        lesson_file = lessons_path / f"{lesson_id}.json"
        with open(lesson_file, 'w', encoding='utf-8') as f:
            json.dump(lesson_json, f, indent=2, ensure_ascii=False)

    async def _extract_lesson_content(
        self,
        start_page: int,
        end_page: int,
        pages_data: List[Dict[str, Any]],
        images_path: Path
    ) -> List[Dict[str, Any]]:
        """Extract nội dung thực tế từ các trang của bài học"""

        content = []
        image_counter = 1

        for page_num in range(start_page - 1, min(end_page, len(pages_data))):
            if page_num < 0:
                continue

            page = pages_data[page_num]

            # Thêm text content nếu có
            if page['text'].strip():
                # Sử dụng LLM để clean và format text
                cleaned_text = await self._clean_text_with_llm(page['text'])

                content.append({
                    "type": "text",
                    "page": page['page_number'],
                    "data": cleaned_text
                })

            # Xử lý images trong trang
            for img_data in page['images']:
                try:
                    # Lưu image
                    img_filename = f"img{image_counter}.png"
                    img_path = images_path / img_filename

                    # Decode và lưu image
                    img_bytes = base64.b64decode(img_data['data'])
                    with open(img_path, 'wb') as f:
                        f.write(img_bytes)

                    # Tạo mô tả image bằng LLM
                    img_description = await self._describe_image_with_llm(img_bytes)

                    # Lưu description
                    desc_path = images_path / f"img{image_counter}_description.txt"
                    with open(desc_path, 'w', encoding='utf-8') as f:
                        f.write(img_description)

                    # Thêm vào content (chỉ mô tả, không lưu base64)
                    content.append({
                        "type": "image",
                        "page": page['page_number'],
                        "description": img_description,
                        "local_path": f"images/{images_path.name}/img{image_counter}.png"
                    })

                    image_counter += 1

                except Exception as e:
                    logger.error(f"Error processing image {image_counter}: {e}")

        return content

    async def _clean_text_with_llm(self, raw_text: str) -> str:
        """Sử dụng LLM để clean và format text từ PDF"""

        if not llm_service.is_available():
            return raw_text.strip()

        try:
            prompt = f"""
Bạn là chuyên gia xử lý text từ sách giáo khoa. Hãy làm sạch và format lại text sau:

YÊU CẦU:
1. Sửa lỗi OCR (ký tự bị nhận dạng sai)
2. Loại bỏ ký tự lạ, khoảng trắng thừa
3. Sắp xếp lại đoạn văn cho dễ đọc
4. Giữ nguyên ý nghĩa và cấu trúc gốc
5. Trả về text tiếng Việt chuẩn

Text gốc:
{raw_text}

Text đã được làm sạch:
"""

            response = llm_service.model.generate_content(prompt)
            cleaned_text = response.text.strip()

            return cleaned_text if cleaned_text else raw_text.strip()

        except Exception as e:
            logger.error(f"Error cleaning text with LLM: {e}")
            return raw_text.strip()

    async def _describe_image_with_llm(self, img_bytes: bytes) -> str:
        """Sử dụng LLM để mô tả hình ảnh"""

        if not llm_service.is_available():
            return "Hình ảnh trong sách giáo khoa (LLM không khả dụng để mô tả chi tiết)"

        try:
            # Tạo mô tả dựa trên context (không cần vision API)
            prompt = """
Bạn là chuyên gia phân tích sách giáo khoa. Hãy tạo mô tả cho hình ảnh trong sách giáo khoa.

YÊU CẦU MÔ TẢ:
1. Mô tả có thể là: biểu đồ, công thức, hình minh họa, bảng biểu
2. Nội dung liên quan đến giáo dục (khoa học, toán học, văn học, etc.)
3. Mục đích giáo dục và cách sử dụng trong giảng dạy
4. Mô tả ngắn gọn, rõ ràng bằng tiếng Việt

Trả về mô tả hình ảnh giáo dục:
"""

            response = llm_service.model.generate_content(prompt)
            description = response.text.strip()

            return description if description else "Hình ảnh minh họa trong sách giáo khoa"

        except Exception as e:
            logger.error(f"Error describing image with LLM: {e}")
            return "Hình ảnh trong sách giáo khoa (không thể tạo mô tả)"

    async def _process_chapter(
        self,
        chapter_data: Dict[str, Any],
        pages_data: List[Dict[str, Any]],
        lesson_start: int,
        lesson_end: int,
        images_path: Path
    ) -> Dict[str, Any]:
        """Xử lý một chương trong bài học"""

        chapter_start = max(chapter_data.get('start_page', lesson_start), lesson_start)
        chapter_end = min(chapter_data.get('end_page', lesson_end), lesson_end)

        # Lấy content từ các trang của chương
        chapter_content = []
        image_counter = 1

        for page_num in range(chapter_start - 1, chapter_end):  # -1 vì index từ 0
            if page_num >= len(pages_data):
                break

            page = pages_data[page_num]

            # Thêm text content
            if page['text'].strip():
                chapter_content.append({
                    "type": "text",
                    "data": page['text'].strip()
                })

            # Xử lý images trong trang
            for img_data in page['images']:
                try:
                    # Lưu image
                    img_filename = f"img{image_counter}.png"
                    img_path = images_path / img_filename

                    # Decode và lưu image
                    img_bytes = base64.b64decode(img_data['data'])
                    with open(img_path, 'wb') as f:
                        f.write(img_bytes)

                    # Tạo mô tả image bằng LLM
                    img_description = await self._describe_image(img_bytes)

                    # Lưu description
                    desc_path = images_path / f"img{image_counter}_description.txt"
                    with open(desc_path, 'w', encoding='utf-8') as f:
                        f.write(img_description)

                    # Thêm vào content (chỉ mô tả, không lưu base64)
                    chapter_content.append({
                        "type": "image",
                        "description": img_description,
                        "local_path": f"images/{images_path.name}/img{image_counter}.png"
                    })

                    image_counter += 1

                except Exception as e:
                    logger.error(f"Error processing image {image_counter}: {e}")

        return {
            "chapter_id": chapter_data['chapter_id'],
            "title": chapter_data['title'],
            "content": chapter_content
        }

    async def _describe_image(self, img_bytes: bytes) -> str:
        """Tạo mô tả cho hình ảnh bằng LLM"""

        if not llm_service.is_available():
            return "Hình ảnh trong sách giáo khoa (LLM không khả dụng để mô tả chi tiết)"

        try:
            # Convert image to base64 for LLM
            img_base64 = base64.b64encode(img_bytes).decode()

            prompt = """
Bạn là chuyên gia phân tích hình ảnh trong sách giáo khoa. Hãy mô tả hình ảnh này một cách chi tiết và có ích cho việc tạo giáo án.

YÊU CẦU MÔ TẢ:
1. Nội dung chính của hình ảnh
2. Các yếu tố quan trọng (biểu đồ, công thức, minh họa)
3. Mục đích giáo dục của hình ảnh
4. Cách sử dụng trong giảng dạy

Trả về mô tả ngắn gọn, rõ ràng bằng tiếng Việt:
"""

            # Note: Gemini vision API call would go here
            # For now, return a placeholder
            return "Hình ảnh minh họa trong sách giáo khoa - cần cập nhật mô tả chi tiết"

        except Exception as e:
            logger.error(f"Error describing image: {e}")
            return "Hình ảnh trong sách giáo khoa (không thể tạo mô tả)"

    async def get_book_structure(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Lấy cấu trúc của một cuốn sách đã xử lý"""

        book_path = self.output_base_path / book_id
        metadata_file = book_path / "metadata.json"

        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            # Lấy danh sách lessons
            lessons_path = book_path / "lessons"
            lessons = []

            if lessons_path.exists():
                for lesson_file in sorted(lessons_path.glob("*.json")):
                    with open(lesson_file, 'r', encoding='utf-8') as f:
                        lesson_data = json.load(f)
                        lessons.append({
                            "lesson_id": lesson_data['lesson_id'],
                            "title": lesson_data['title'],
                            "chapters_count": len(lesson_data.get('chapters', []))
                        })

            return {
                "metadata": metadata,
                "lessons": lessons,
                "book_path": str(book_path)
            }

        except Exception as e:
            logger.error(f"Error reading book structure: {e}")
            return None

    async def get_lesson_content(self, book_id: str, lesson_id: str) -> Optional[Dict[str, Any]]:
        """Lấy nội dung chi tiết của một bài học"""

        book_path = self.output_base_path / book_id
        lesson_file = book_path / "lessons" / f"{lesson_id}.json"

        if not lesson_file.exists():
            return None

        try:
            with open(lesson_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading lesson content: {e}")
            return None

# Global instance
textbook_parser_service = TextbookParserService()
