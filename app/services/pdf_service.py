import fitz  # PyMuPDF
import pdfplumber
import re
import os
import aiofiles
from typing import List, Dict, Any, Tuple
import logging
from datetime import datetime
from app.database.connection import get_database_sync, CHEMISTRY_TEXTBOOK_COLLECTION, CHEMISTRY_CHAPTERS_COLLECTION, CHEMISTRY_LESSONS_COLLECTION
from app.database.models import ChemistryTextbook, ChemistryChapter, ChemistryLesson
from app.services.embedding_service import embedding_service
from app.services.file_storage_service import file_storage

logger = logging.getLogger(__name__)

class PDFProcessingService:
    """
    Service xử lý PDF sách giáo khoa Hóa học
    - Extract text từ PDF
    - Phân tích cấu trúc chương/bài
    - Tạo embeddings cho nội dung
    """

    def __init__(self):
        # Không cần upload_dir nữa vì dùng GridFS
        pass

    async def process_textbook_pdf(
        self, 
        file_content: bytes, 
        filename: str,
        book_title: str, 
        grade: str, 
        publisher: str = "Unknown",
        academic_year: str = "2024-2025"
    ) -> Dict[str, Any]:
        """
        Xử lý PDF sách giáo khoa và lưu vào GridFS + MongoDB
        """
        try:
            # Store PDF to GridFS
            pdf_metadata = {
                "book_title": book_title,
                "grade": grade,
                "publisher": publisher,
                "academic_year": academic_year,
                "category": "textbook"
            }
            
            pdf_file_id = await file_storage.store_pdf_file(
                file_content=file_content,
                filename=filename,
                metadata=pdf_metadata
            )
            
            # Extract text from content (not file path)
            full_text, total_pages = self.extract_text_from_content(file_content)
            
            # Parse structure
            chapters_data = self.parse_chemistry_structure(full_text)
            
            # Save to database
            result = await self.save_textbook_to_database(
                book_title=book_title,
                grade=grade,
                publisher=publisher,
                filename=filename,
                file_size=len(file_content),
                total_pages=total_pages,
                pdf_file_id=str(pdf_file_id),
                chapters_data=chapters_data
            )
            
            return {
                "success": True,
                "textbook_id": result["textbook_id"],
                "pdf_file_id": str(pdf_file_id),
                "total_chapters": len(chapters_data),
                "total_pages": total_pages,
                "message": "PDF đã được xử lý và lưu vào GridFS thành công"
            }
            
        except Exception as e:
            logger.error(f"Error processing textbook PDF: {e}")
            raise

    def extract_text_from_content(self, file_content: bytes) -> Tuple[str, int]:
        """
        Extract text từ PDF content (bytes) hoặc text content cho testing
        """
        try:
            # Try to detect if it's a PDF file by checking the magic bytes
            if file_content.startswith(b'%PDF'):
                # It's a PDF file, use PyMuPDF
                doc = fitz.open(stream=file_content, filetype="pdf")
                full_text = ""
                
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text = page.get_text()
                    full_text += f"\n--- Page {page_num + 1} ---\n{text}"
                
                total_pages = doc.page_count
                doc.close()
                
                return full_text, total_pages
            else:
                # Treat as text file for testing
                try:
                    full_text = file_content.decode('utf-8')
                    # Estimate pages (assuming ~500 chars per page)
                    estimated_pages = max(1, len(full_text) // 500)
                    return full_text, estimated_pages
                except UnicodeDecodeError:
                    # Try other encodings
                    try:
                        full_text = file_content.decode('utf-8-sig')
                        estimated_pages = max(1, len(full_text) // 500)
                        return full_text, estimated_pages
                    except UnicodeDecodeError:
                        raise ValueError("Không thể đọc nội dung file")
            
        except Exception as e:
            logger.error(f"Error extracting text from content: {e}")
            raise

    def parse_chemistry_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Phân tích cấu trúc chương/bài từ text
        Tối ưu cho sách giáo khoa Hóa học Việt Nam
        """
        try:
            chapters = []

            # Patterns cho chương và bài
            chapter_pattern = r'CHƯƠNG\s+(\d+|[IVX]+)[\s\.:]*([^\n]+)'
            lesson_pattern = r'Bài\s+(\d+)[\s\.:]*([^\n]+)'

            # Tìm tất cả chương
            chapter_matches = list(re.finditer(chapter_pattern, text, re.IGNORECASE))

            for i, chapter_match in enumerate(chapter_matches):
                chapter_start = chapter_match.start()
                chapter_end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(text)

                chapter_text = text[chapter_start:chapter_end]
                chapter_number = chapter_match.group(1)
                chapter_title = chapter_match.group(2).strip()

                # Tìm các bài trong chương
                lessons = []
                lesson_matches = list(re.finditer(lesson_pattern, chapter_text, re.IGNORECASE))

                for j, lesson_match in enumerate(lesson_matches):
                    lesson_start = lesson_match.start()
                    lesson_end = lesson_matches[j + 1].start() if j + 1 < len(lesson_matches) else len(chapter_text)

                    lesson_text = chapter_text[lesson_start:lesson_end]
                    lesson_number = lesson_match.group(1)
                    lesson_title = lesson_match.group(2).strip()

                    # Extract thông tin bài học
                    lesson_info = self._extract_lesson_info(lesson_text)

                    lessons.append({
                        "lesson_number": int(lesson_number),
                        "title": lesson_title,
                        "content": lesson_text,
                        "objectives": lesson_info.get("objectives", []),
                        "key_concepts": lesson_info.get("key_concepts", []),
                        "formulas": lesson_info.get("formulas", []),
                        "experiments": lesson_info.get("experiments", []),
                        "exercises": lesson_info.get("exercises", [])
                    })

                chapters.append({
                    "chapter_number": self._convert_chapter_number(chapter_number),
                    "title": chapter_title,
                    "content": chapter_text,
                    "lessons": lessons
                })

            logger.info(f"Analyzed structure: {len(chapters)} chapters")
            return chapters

        except Exception as e:
            logger.error(f"Failed to analyze PDF structure: {e}")
            raise

    def _convert_chapter_number(self, chapter_num: str) -> int:
        """Convert chapter number từ Roman hoặc Arabic sang int"""
        roman_to_int = {
            'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
            'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10
        }

        if chapter_num.upper() in roman_to_int:
            return roman_to_int[chapter_num.upper()]

        try:
            return int(chapter_num)
        except ValueError:
            return 1

    def _extract_lesson_info(self, lesson_text: str) -> Dict[str, List[str]]:
        """Extract thông tin chi tiết từ bài học"""
        info = {
            "objectives": [],
            "key_concepts": [],
            "formulas": [],
            "experiments": [],
            "exercises": []
        }

        # Extract mục tiêu
        objectives_pattern = r'(?:Mục tiêu|Học xong bài này|Sau bài học)[\s\S]*?(?=\n\n|\n[A-Z]|$)'
        objectives_match = re.search(objectives_pattern, lesson_text, re.IGNORECASE)
        if objectives_match:
            objectives_text = objectives_match.group(0)
            # Tách thành các mục tiêu riêng
            objectives = re.findall(r'[-•]\s*([^\n]+)', objectives_text)
            info["objectives"] = [obj.strip() for obj in objectives]

        # Extract công thức hóa học
        formula_pattern = r'[A-Z][a-z]?(?:\d+)?(?:[+-]\d*)?(?:\([A-Za-z0-9]+\))?(?:\s*[+\-→←⇌]\s*[A-Z][a-z]?(?:\d+)?(?:[+-]\d*)?(?:\([A-Za-z0-9]+\))?)*'
        formulas = re.findall(formula_pattern, lesson_text)
        info["formulas"] = list(set(formulas))  # Remove duplicates

        # Extract khái niệm chính (từ in đậm hoặc in nghiêng)
        concept_pattern = r'\*\*([^*]+)\*\*|__([^_]+)__|(?:^|\n)([A-ZÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẽÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ\s]+)(?=:|\.)'
        concepts = re.findall(concept_pattern, lesson_text, re.MULTILINE)
        # Flatten và clean concepts
        flat_concepts = []
        for concept_tuple in concepts:
            for concept in concept_tuple:
                if concept and len(concept.strip()) > 3:
                    flat_concepts.append(concept.strip())
        info["key_concepts"] = flat_concepts[:10]  # Limit to 10 concepts

        # Extract thí nghiệm
        experiment_pattern = r'(?:Thí nghiệm|TN)\s*\d*[\s\.:]*([^\n]+(?:\n(?!(?:Thí nghiệm|TN|\d+\.))[^\n]*)*)'
        experiments = re.findall(experiment_pattern, lesson_text, re.IGNORECASE)
        info["experiments"] = [{"title": exp.strip(), "description": ""} for exp in experiments]

        # Extract bài tập
        exercise_pattern = r'(?:Bài tập|Câu hỏi)\s*\d*[\s\.:]*([^\n]+)'
        exercises = re.findall(exercise_pattern, lesson_text, re.IGNORECASE)
        info["exercises"] = [{"question": ex.strip(), "answer": ""} for ex in exercises]

        return info

    async def process_pdf(
        self,
        file_path: str,
        title: str,
        grade: str,
        publisher: str,
        year: int
    ) -> str:
        """
        Xử lý hoàn chỉnh một file PDF

        Returns:
            str: ID của textbook đã tạo
        """
        try:
            # 1. Extract text từ PDF
            full_text, total_pages = self.extract_text_from_pdf(file_path)
            file_size = os.path.getsize(file_path)

            # 2. Lưu textbook vào database
            db = get_database_sync()
            textbook_collection = db[CHEMISTRY_TEXTBOOK_COLLECTION]

            textbook = ChemistryTextbook(
                title=title,
                grade=grade,
                publisher=publisher,
                year=year,
                file_path=file_path,
                file_size=file_size,
                total_pages=total_pages,
                processed=False
            )

            textbook_result = textbook_collection.insert_one(textbook.model_dump(by_alias=True))
            textbook_id = str(textbook_result.inserted_id)

            logger.info(f"Created textbook record: {textbook_id}")

            # 3. Phân tích cấu trúc
            chapters_data = self.parse_chemistry_structure(full_text)

            # 4. Lưu chapters và lessons
            chapter_collection = db[CHEMISTRY_CHAPTERS_COLLECTION]
            lesson_collection = db[CHEMISTRY_LESSONS_COLLECTION]

            for chapter_data in chapters_data:
                # Lưu chapter
                chapter = ChemistryChapter(
                    textbook_id=textbook_id,
                    chapter_number=chapter_data["chapter_number"],
                    title=chapter_data["title"],
                    content=chapter_data["content"],
                    start_page=1,  # TODO: Calculate actual page numbers
                    end_page=1
                )

                chapter_result = chapter_collection.insert_one(chapter.model_dump(by_alias=True))
                chapter_id = str(chapter_result.inserted_id)

                # Tạo embeddings cho chapter
                embedding_service.store_embeddings(
                    content_id=chapter_id,
                    content_type="chapter",
                    text=chapter_data["content"],
                    metadata={
                        "textbook_id": textbook_id,
                        "chapter_number": chapter_data["chapter_number"],
                        "title": chapter_data["title"],
                        "grade": grade
                    }
                )

                # Lưu lessons
                for lesson_data in chapter_data["lessons"]:
                    lesson = ChemistryLesson(
                        chapter_id=chapter_id,
                        textbook_id=textbook_id,
                        lesson_number=lesson_data["lesson_number"],
                        title=lesson_data["title"],
                        content=lesson_data["content"],
                        objectives=lesson_data["objectives"],
                        key_concepts=lesson_data["key_concepts"],
                        formulas=lesson_data["formulas"],
                        experiments=lesson_data["experiments"],
                        exercises=lesson_data["exercises"],
                        start_page=1,  # TODO: Calculate actual page numbers
                        end_page=1
                    )

                    lesson_result = lesson_collection.insert_one(lesson.model_dump(by_alias=True))
                    lesson_id = str(lesson_result.inserted_id)

                    # Tạo embeddings cho lesson
                    embedding_service.store_embeddings(
                        content_id=lesson_id,
                        content_type="lesson",
                        text=lesson_data["content"],
                        metadata={
                            "textbook_id": textbook_id,
                            "chapter_id": chapter_id,
                            "lesson_number": lesson_data["lesson_number"],
                            "title": lesson_data["title"],
                            "grade": grade,
                            "key_concepts": lesson_data["key_concepts"],
                            "formulas": lesson_data["formulas"]
                        }
                    )

            # 5. Cập nhật trạng thái processed
            textbook_collection.update_one(
                {"_id": textbook_result.inserted_id},
                {"$set": {"processed": True}}
            )

            logger.info(f"Successfully processed PDF: {textbook_id}")
            return textbook_id

        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            raise

    async def save_textbook_to_database(
        self,
        book_title: str,
        grade: str,
        publisher: str,
        filename: str,
        file_size: int,
        total_pages: int,
        pdf_file_id: str,
        chapters_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Lưu thông tin textbook và chapters vào database
        """
        try:
            db = get_database_sync()
            textbook_collection = db[CHEMISTRY_TEXTBOOK_COLLECTION]
            chapter_collection = db[CHEMISTRY_CHAPTERS_COLLECTION]
            lesson_collection = db[CHEMISTRY_LESSONS_COLLECTION]

            # Create textbook record
            textbook = ChemistryTextbook(
                title=book_title,
                grade=grade,
                publisher=publisher,
                year=int(datetime.now().year),
                file_path=f"gridfs://{pdf_file_id}",  # GridFS reference
                file_size=file_size,
                total_pages=total_pages,
                processed=True
            )

            textbook_result = textbook_collection.insert_one(textbook.model_dump(by_alias=True))
            textbook_id = str(textbook_result.inserted_id)

            # Save chapters and lessons
            for chapter_data in chapters_data:
                # Save chapter
                chapter = ChemistryChapter(
                    textbook_id=textbook_id,
                    chapter_number=chapter_data["chapter_number"],
                    title=chapter_data["title"],
                    content=chapter_data["content"],
                    start_page=1,  # TODO: Calculate actual page numbers
                    end_page=1
                )

                chapter_result = chapter_collection.insert_one(chapter.model_dump(by_alias=True))
                chapter_id = str(chapter_result.inserted_id)

                # Create embeddings for chapter
                try:
                    embedding_service.store_embeddings(
                        content_id=chapter_id,
                        content_type="chapter",
                        text=chapter_data["content"],
                        metadata={
                            "textbook_id": textbook_id,
                            "chapter_number": chapter_data["chapter_number"],
                            "title": chapter_data["title"],
                            "grade": grade
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to create embeddings for chapter {chapter_id}: {e}")

                # Save lessons
                for lesson_data in chapter_data["lessons"]:
                    lesson = ChemistryLesson(
                        chapter_id=chapter_id,
                        textbook_id=textbook_id,
                        lesson_number=lesson_data["lesson_number"],
                        title=lesson_data["title"],
                        content=lesson_data["content"],
                        objectives=lesson_data["objectives"],
                        key_concepts=lesson_data["key_concepts"],
                        formulas=lesson_data["formulas"],
                        experiments=lesson_data["experiments"],
                        exercises=lesson_data["exercises"],
                        start_page=1,  # TODO: Calculate actual page numbers
                        end_page=1
                    )

                    lesson_result = lesson_collection.insert_one(lesson.model_dump(by_alias=True))
                    lesson_id = str(lesson_result.inserted_id)

                    # Create embeddings for lesson
                    try:
                        embedding_service.store_embeddings(
                            content_id=lesson_id,
                            content_type="lesson",
                            text=lesson_data["content"],
                            metadata={
                                "textbook_id": textbook_id,
                                "chapter_id": chapter_id,
                                "lesson_number": lesson_data["lesson_number"],
                                "title": lesson_data["title"],
                                "grade": grade,
                                "key_concepts": lesson_data["key_concepts"],
                                "formulas": lesson_data["formulas"]
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create embeddings for lesson {lesson_id}: {e}")

            logger.info(f"Successfully saved textbook to database: {textbook_id}")
            return {
                "textbook_id": textbook_id,
                "total_chapters": len(chapters_data)
            }

        except Exception as e:
            logger.error(f"Error saving textbook to database: {e}")
            raise

# Global instance
pdf_service = PDFProcessingService()
