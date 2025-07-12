"""
OCR Service đơn giản cho xử lý PDF
Sử dụng EasyOCR và Tesseract
"""
import io
import logging
import threading
from typing import Tuple, Dict, Any, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
import pytesseract
import numpy as np

# pdf2image import (requires Poppler)
try:
    from pdf2image import convert_from_bytes
    from PIL import Image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False
    logging.warning("pdf2image not available. Install poppler-utils for OCR support")

# EasyOCR lazy import - chỉ import khi cần thiết
EASYOCR_AVAILABLE = None  # Will be determined when first needed
_easyocr_module = None

def _get_easyocr():
    """Lazy import EasyOCR"""
    global EASYOCR_AVAILABLE, _easyocr_module
    if EASYOCR_AVAILABLE is None:
        try:
            import easyocr
            _easyocr_module = easyocr
            EASYOCR_AVAILABLE = True
        except ImportError:
            EASYOCR_AVAILABLE = False
            logging.warning("EasyOCR not available. Install with: pip install easyocr")
    return _easyocr_module if EASYOCR_AVAILABLE else None

logger = logging.getLogger(__name__)

class SimpleOCRService:
    """
    Service OCR đơn giản cho PDF
    """

    def __init__(self):
        """Initialize OCR service"""
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.easyocr_reader = None
        self.tesseract_config = '--oem 3 --psm 6 -l vie+eng'
        self.dpi = 200  # Moderate DPI for balance between quality and speed

        # EasyOCR sẽ được khởi tạo lazy khi cần thiết
        self._easyocr_initialized = False

    def _ensure_easyocr_initialized(self):
        """Ensure EasyOCR is initialized"""
        if not self._easyocr_initialized:
            # Check if EasyOCR is available (this will set EASYOCR_AVAILABLE)
            easyocr_module = _get_easyocr()
            if easyocr_module:
                self._init_easyocr()
                self._easyocr_initialized = True
    
    def _init_easyocr(self):
        """Initialize EasyOCR reader"""
        try:
            logger.info("🔄 SimpleOCRService: First-time EasyOCR initialization triggered")
            easyocr_module = _get_easyocr()
            if easyocr_module:
                self.easyocr_reader = easyocr_module.Reader(['vi', 'en'], gpu=False)
                logger.info("✅ SimpleOCRService: EasyOCR initialization completed")
            else:
                self.easyocr_reader = None
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_reader = None
    
    async def extract_text_from_pdf(
        self, 
        file_content: bytes, 
        filename: str
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF using OCR
        
        Returns:
            (extracted_text, metadata)
        """
        try:
            logger.info(f"Processing PDF: {filename}")
            
            # First, try to extract text normally
            normal_text, total_pages = await self._extract_text_normal(file_content)
            
            # Check if normal extraction yielded sufficient text
            if len(normal_text.strip()) > 50:  # Lowered threshold
                logger.info(f"PDF {filename} has extractable text, using normal extraction")
                return normal_text, {
                    "total_pages": total_pages,
                    "ocr_used": False,
                    "extraction_method": "normal"
                }

            # PDF seems to be scan/image-based, try alternative methods
            logger.info(f"PDF {filename} appears to be scanned or has minimal text, trying OCR methods")

            # Try PyMuPDF image extraction first (doesn't need Poppler)
            try:
                alternative_text = await self._extract_with_pymupdf_images(file_content)
                if len(alternative_text.strip()) > 20:  # Even lower threshold for OCR
                    return alternative_text, {
                        "total_pages": total_pages,
                        "ocr_used": True,
                        "extraction_method": "pymupdf_images"
                    }
            except Exception as e:
                logger.error(f"PyMuPDF image extraction failed: {e}")

            # If pdf2image is available, try that method
            if PDF2IMAGE_AVAILABLE:
                logger.info("Trying pdf2image OCR method")
                return await self._extract_with_ocr(file_content, total_pages)

            # Return whatever we got from normal extraction
            return normal_text, {
                "total_pages": total_pages,
                "ocr_used": False,
                "extraction_method": "normal",
                "warning": "Limited text extraction - PDF may be scanned. Install poppler-utils for better OCR support"
            }


            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
    
    async def _extract_text_normal(self, file_content: bytes) -> Tuple[str, int]:
        """Extract text normally from PDF"""
        def extract():
            doc = fitz.open(stream=file_content, filetype="pdf")
            full_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                # Use the correct method for text extraction
                text = page.get_text("text")  # type: ignore
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            total_pages = doc.page_count
            doc.close()
            return full_text, total_pages
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, extract
        )
    
    async def _extract_with_ocr(self, file_content: bytes, total_pages: int) -> Tuple[str, Dict[str, Any]]:
        """Extract text using OCR"""

        try:
            # Convert PDF to images
            images = await self._pdf_to_images(file_content)
            logger.info(f"Converting {len(images)} pages to images for OCR")
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            # Return empty result with error info
            return "", {
                "total_pages": total_pages,
                "successful_pages": 0,
                "failed_pages": total_pages,
                "ocr_used": False,
                "extraction_method": "failed",
                "error": "PDF to image conversion failed - Poppler may not be installed"
            }
        
        # Process images with OCR
        full_text = ""
        successful_pages = 0
        
        for i, image in enumerate(images):
            try:
                page_text = await self._ocr_image(image, i + 1)
                full_text += f"\n--- Page {i + 1} ---\n{page_text}"
                successful_pages += 1
            except Exception as e:
                logger.error(f"OCR failed for page {i + 1}: {e}")
                full_text += f"\n--- Page {i + 1} ---\n[OCR_ERROR: {str(e)}]\n"
        
        metadata = {
            "total_pages": total_pages,
            "successful_pages": successful_pages,
            "failed_pages": total_pages - successful_pages,
            "ocr_used": True,
            "extraction_method": "easyocr" if self.easyocr_reader else "tesseract"
        }
        
        logger.info(f"OCR completed: {successful_pages}/{total_pages} pages successful")
        return full_text, metadata
    
    async def _pdf_to_images(self, file_content: bytes) -> List[Image.Image]:
        """Convert PDF pages to images"""
        def convert():
            return convert_from_bytes(
                file_content,
                dpi=self.dpi,
                fmt='RGB'
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, convert
        )
    
    async def _ocr_image(self, image: Image.Image, page_num: int) -> str:
        """
        OCR a single image
        """
        def process_image():
            try:
                # Ensure EasyOCR is initialized if available
                self._ensure_easyocr_initialized()

                # Try EasyOCR first if available
                if self.easyocr_reader:
                    try:
                        results = self.easyocr_reader.readtext(np.array(image))
                        # EasyOCR returns list of [bbox, text, confidence]
                        text_parts = []
                        for result in results:
                            if isinstance(result, (list, tuple)) and len(result) >= 2:
                                text_parts.append(str(result[1]))
                        text = ' '.join(text_parts)
                        return text
                    except Exception as e:
                        logger.warning(f"EasyOCR failed for page {page_num}: {e}, falling back to Tesseract")
                
                # Fallback to Tesseract
                text = pytesseract.image_to_string(image, config=self.tesseract_config)
                return text
                
            except Exception as e:
                logger.error(f"OCR error for page {page_num}: {e}")
                return ""
        
        result = await asyncio.get_event_loop().run_in_executor(
            self.executor, process_image
        )
        return result
    
    async def _extract_with_pymupdf_images(self, file_content: bytes) -> str:
        """
        Extract text using PyMuPDF to get images from PDF and OCR them
        Alternative method when pdf2image is not available
        """
        def extract_and_ocr():
            try:
                # Ensure EasyOCR is initialized if available
                self._ensure_easyocr_initialized()

                doc = fitz.open(stream=file_content, filetype="pdf")
                full_text = ""

                for page_num in range(doc.page_count):
                    page = doc[page_num]

                    # First try to get text normally
                    page_text = page.get_text("text")  # type: ignore
                    if len(page_text.strip()) > 50:
                        full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                        continue

                    # If no text, try to extract images and OCR them
                    image_list = page.get_images()
                    page_ocr_text = ""

                    for img_index, img in enumerate(image_list):
                        try:
                            # Get image data
                            xref = img[0]
                            pix = fitz.Pixmap(doc, xref)

                            # Convert to PIL Image if possible
                            if pix.n - pix.alpha < 4:  # GRAY or RGB
                                img_data = pix.tobytes("ppm")
                                from PIL import Image
                                pil_image = Image.open(io.BytesIO(img_data))

                                # OCR the image
                                if self.easyocr_reader:
                                    try:
                                        results = self.easyocr_reader.readtext(np.array(pil_image))
                                        for result in results:
                                            if isinstance(result, (list, tuple)) and len(result) >= 2:
                                                page_ocr_text += str(result[1]) + " "
                                    except Exception as e:
                                        logger.warning(f"EasyOCR failed for image {img_index}: {e}")
                                        # Fallback to Tesseract
                                        try:
                                            ocr_text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
                                            page_ocr_text += ocr_text + " "
                                        except Exception as te:
                                            logger.error(f"Tesseract also failed: {te}")
                                else:
                                    # Use Tesseract only
                                    try:
                                        ocr_text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
                                        page_ocr_text += ocr_text + " "
                                    except Exception as te:
                                        logger.error(f"Tesseract failed: {te}")

                            pix = None  # Clean up

                        except Exception as e:
                            logger.error(f"Error processing image {img_index} on page {page_num + 1}: {e}")

                    if page_ocr_text.strip():
                        full_text += f"\n--- Page {page_num + 1} ---\n{page_ocr_text}"
                    else:
                        full_text += f"\n--- Page {page_num + 1} ---\n[No text extracted]\n"

                doc.close()
                return full_text

            except Exception as e:
                logger.error(f"PyMuPDF image extraction failed: {e}")
                return ""

        return await asyncio.get_event_loop().run_in_executor(
            self.executor, extract_and_ocr
        )

    def get_supported_languages(self) -> List[str]:
        """Get list of supported OCR languages"""
        try:
            # Ensure EasyOCR is initialized if available
            self._ensure_easyocr_initialized()

            supported = []

            # Check EasyOCR first (preferred)
            if self.easyocr_reader:
                supported.extend(['vietnamese', 'english', 'vietnamese_easyocr'])
                return supported

            # Fallback to Tesseract
            try:
                langs = pytesseract.get_languages()
                if 'vie' in langs:
                    supported.append('vietnamese')
                if 'eng' in langs:
                    supported.append('english')
            except Exception as tesseract_error:
                logger.warning(f"Tesseract not available: {tesseract_error}")
                # If both EasyOCR and Tesseract fail, return basic support
                supported = ['english']

            return supported if supported else ['english']

        except Exception as e:
            logger.error(f"Failed to get supported languages: {e}")
            return ['english']  # Fallback

# Factory function để tạo SimpleOCRService instance
def get_simple_ocr_service() -> SimpleOCRService:
    """
    Tạo SimpleOCRService instance mới

    Returns:
        SimpleOCRService: Fresh instance
    """
    return SimpleOCRService()
