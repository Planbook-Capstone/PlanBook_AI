import os
import json
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

# Các imports khác dựa trên cả 2 services hiện tại
import numpy as np
from PIL import Image
import pytesseract
import fitz  # PyMuPDF

from app.core.config import settings

logger = logging.getLogger(__name__)

class TextbookService:
    """Unified service for textbook processing, combining enhanced and legacy functionality"""

    def __init__(self):
        self.output_base_path = os.path.join(settings.DATA_DIR, "processed_textbooks")
        self.executor = ThreadPoolExecutor(max_workers=settings.CONCURRENT_WORKERS)
        os.makedirs(self.output_base_path, exist_ok=True)

    # MAIN PUBLIC METHODS

    async def process_textbook_to_structure(self, pdf_content: bytes, filename: str, book_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a textbook and return its complete structure (enhanced version)"""
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

    # Legacy method renamed for compatibility
    async def process_textbook(self, pdf_content: bytes, filename: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Legacy method for processing textbooks (to maintain compatibility)"""
        # This will delegate to enhanced version
        return await self.process_textbook_to_structure(pdf_content, filename, metadata)

    # Content retrieval methods
    async def get_book_structure(self, book_id: str) -> Optional[Dict[str, Any]]:
        """Get book structure from file system"""
        try:
            structure_file = os.path.join(self.output_base_path, book_id, "structure.json")
            if os.path.exists(structure_file):
                with open(structure_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error reading book structure: {e}")
            return None

    async def get_lesson_content(self, book_id: str, lesson_id: str) -> Optional[Dict[str, Any]]:
        """Get content of a specific lesson"""
        try:
            lesson_file = os.path.join(self.output_base_path, book_id, f"{lesson_id}.json")
            if os.path.exists(lesson_file):
                with open(lesson_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            return None
        except Exception as e:
            logger.error(f"Error reading lesson content: {e}")
            return None

    # INTERNAL IMPLEMENTATION METHODS
    # Private methods shared between enhanced and legacy functionality
    # These would include methods from both existing services...

    # For brevity, I'm not including all implementation details here
    # In the real implementation, include all methods from both services

    async def _extract_pages_with_ocr(self, pdf_content: bytes) -> List[Dict[str, Any]]:
        """Extract all pages from PDF with OCR processing"""
        # Implementation from enhanced_textbook_service
        pass

    async def _analyze_book_structure_enhanced(self, pages_data: List[Dict[str, Any]], book_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze book structure using advanced methods"""
        # Implementation from enhanced_textbook_service  
        pass

    async def _basic_structure_analysis(self, pages_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze structure using basic pattern recognition"""
        # Implementation from textbook_parser_service
        pass

    # Additional implementation methods would go here

# Create singleton instance
textbook_service = TextbookService()
