"""
PDF Service đơn giản cho xử lý PDF
"""
import logging
from typing import Dict, Any

from app.services.simple_ocr_service import simple_ocr_service

logger = logging.getLogger(__name__)


class SimplePDFService:
    """Service xử lý PDF đơn giản"""

    def __init__(self):
        pass

    async def extract_text_from_pdf(
        self,
        file_content: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """
        Extract text from PDF using OCR

        Args:
            file_content: Nội dung file PDF
            filename: Tên file

        Returns:
            Dict chứa text và metadata
        """
        try:
            logger.info(f"Processing PDF: {filename}")

            # Extract text using OCR service
            extracted_text, metadata = await simple_ocr_service.extract_text_from_pdf(
                file_content=file_content,
                filename=filename
            )

            logger.info(f"Extracted {len(extracted_text)} characters from {metadata.get('total_pages', 0)} pages")

            return {
                "filename": filename,
                "extracted_text": extracted_text,
                "metadata": metadata,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            # Vẫn trả về success=True với thông tin lỗi trong metadata
            return {
                "filename": filename,
                "extracted_text": f"[PDF_EXTRACTION_FAILED] Could not extract text from PDF: {str(e)}",
                "metadata": {
                    "total_pages": 0,
                    "ocr_used": False,
                    "extraction_method": "failed",
                    "error": str(e)
                },
                "success": True  # Vẫn trả về success để có thể format
            }


# Global instance
simple_pdf_service = SimplePDFService()
