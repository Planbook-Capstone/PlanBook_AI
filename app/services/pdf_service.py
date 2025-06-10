"""
PDF Service cho xử lý sách giáo khoa PDF
Tích hợp với OCR Service và File Storage Service
"""
import logging
from typing import Dict, Any, Tuple
from datetime import datetime
from bson import ObjectId

from app.services.ocr_service import OCRService
from app.services.file_storage_service import file_storage
from app.database.connection import get_database, CHEMISTRY_TEXTBOOK_COLLECTION

logger = logging.getLogger(__name__)

class PDFService:
    """Service xử lý PDF sách giáo khoa"""
    
    def __init__(self):
        self.ocr_service = OCRService()
    
    async def process_textbook_pdf(
        self,
        file_content: bytes,
        filename: str,
        book_title: str,
        grade: str,
        publisher: str = "NXB Giáo dục Việt Nam",
        academic_year: str = "2024-2025"
    ) -> Dict[str, Any]:
        """
        Xử lý PDF sách giáo khoa và lưu vào database
        
        Args:
            file_content: Nội dung file PDF
            filename: Tên file
            book_title: Tiêu đề sách
            grade: Lớp (10, 11, 12)
            publisher: Nhà xuất bản
            academic_year: Năm học
            
        Returns:
            Dict chứa thông tin kết quả xử lý
        """
        try:
            logger.info(f"Processing textbook PDF: {book_title} - Grade {grade}")
            
            # 1. Extract text using OCR service
            extracted_text, total_pages, ocr_metadata = await self.ocr_service.process_pdf_with_ocr(
                file_content=file_content,
                filename=filename,
                use_vietocr=True
            )
            
            logger.info(f"Extracted {len(extracted_text)} characters from {total_pages} pages")
            
            # 2. Store PDF file in GridFS
            pdf_metadata = {
                "book_title": book_title,
                "grade": grade,
                "publisher": publisher,
                "academic_year": academic_year,
                "total_pages": total_pages,
                "category": "textbook",
                "subject": "chemistry"
            }
            
            pdf_file_id = await file_storage.store_pdf_file(
                file_content=file_content,
                filename=filename,
                metadata=pdf_metadata
            )
            
            # 3. Process and structure content (basic chapter detection)
            chapters = await self._detect_chapters(extracted_text)
            
            # 4. Store textbook metadata in database
            textbook_data = {
                "book_title": book_title,
                "grade": grade,
                "publisher": publisher,
                "academic_year": academic_year,
                "filename": filename,
                "pdf_file_id": pdf_file_id,
                "total_pages": total_pages,
                "total_chapters": len(chapters),
                "chapters": chapters,
                "extracted_text": extracted_text[:5000],  # Store first 5000 chars as preview
                "ocr_metadata": ocr_metadata,
                "upload_date": datetime.utcnow(),
                "processing_status": "completed"
            }
            
            # Insert into database
            db = await get_database()
            collection = db[CHEMISTRY_TEXTBOOK_COLLECTION]
            result = await collection.insert_one(textbook_data)
            textbook_id = str(result.inserted_id)
            
            logger.info(f"Successfully processed textbook: {textbook_id}")
            
            return {
                "textbook_id": textbook_id,
                "pdf_file_id": str(pdf_file_id),
                "total_pages": total_pages,
                "total_chapters": len(chapters),
                "processing_status": "completed",
                "ocr_used": ocr_metadata.get("ocr_used", False)
            }
            
        except Exception as e:
            logger.error(f"Error processing textbook PDF: {e}")
            raise
    
    async def _detect_chapters(self, text: str) -> list:
        """
        Phát hiện các chương trong text (basic implementation)
        
        Args:
            text: Text đã extract từ PDF
            
        Returns:
            List các chương được phát hiện
        """
        try:
            chapters = []
            lines = text.split('\n')
            
            chapter_keywords = [
                'CHƯƠNG', 'Chương', 'chương',
                'BÀI', 'Bài', 'bài',
                'PHẦN', 'Phần', 'phần'
            ]
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line contains chapter keywords
                for keyword in chapter_keywords:
                    if keyword in line and len(line) < 200:  # Reasonable title length
                        # Try to extract chapter number and title
                        chapter_info = {
                            "title": line,
                            "line_number": i + 1,
                            "type": "chapter" if "CHƯƠNG" in line.upper() else "lesson"
                        }
                        chapters.append(chapter_info)
                        break
            
            logger.info(f"Detected {len(chapters)} chapters/lessons")
            return chapters
            
        except Exception as e:
            logger.error(f"Error detecting chapters: {e}")
            return []
    
    async def get_textbook_content(self, textbook_id: str) -> Dict[str, Any]:
        """
        Lấy nội dung sách giáo khoa từ database
        
        Args:
            textbook_id: ID của sách giáo khoa
            
        Returns:
            Dict chứa thông tin sách giáo khoa
        """
        try:
            db = await get_database()
            collection = db[CHEMISTRY_TEXTBOOK_COLLECTION]
            
            textbook = await collection.find_one({"_id": ObjectId(textbook_id)})
            
            if not textbook:
                raise ValueError(f"Textbook not found: {textbook_id}")
            
            # Convert ObjectId to string
            textbook["_id"] = str(textbook["_id"])
            textbook["pdf_file_id"] = str(textbook["pdf_file_id"])
            
            return textbook
            
        except Exception as e:
            logger.error(f"Error getting textbook content: {e}")
            raise

# Global instance
pdf_service = PDFService()
