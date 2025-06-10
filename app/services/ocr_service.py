"""
OCR Service cho xử lý PDF scan với VietOCR
Hỗ trợ tiếng Việt và xử lý song song nhiều trang
"""
import os
import io
import logging
from typing import List, Tuple, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes
from PIL import Image
import pytesseract
import numpy as np
import cv2

# VietOCR imports (install: pip install vietocr)
try:
    from vietocr.tool.predictor import Predictor
    from vietocr.tool.config import Cfg
    VIETOCR_AVAILABLE = True
except ImportError:
    VIETOCR_AVAILABLE = False
    logging.warning("VietOCR not available. Install with: pip install vietocr")

from app.core.config import settings

logger = logging.getLogger(__name__)

class OCRService:
    """
    Service xử lý OCR cho PDF scan với hỗ trợ tiếng Việt
    """
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.vietocr_detector = None
        self.tesseract_config = '--oem 3 --psm 6 -l vie+eng'
        self.dpi = 300  # High DPI for better OCR
        
        # Initialize VietOCR if available
        if VIETOCR_AVAILABLE:
            self._init_vietocr()
    
    def _init_vietocr(self):
        """Initialize VietOCR model"""
        try:
            config = Cfg.load_config_from_name('vgg_transformer')
            config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
            config['cnn']['pretrained'] = False
            config['device'] = 'cpu'  # Use CPU for compatibility
            
            self.vietocr_detector = Predictor(config)
            logger.info("VietOCR initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize VietOCR: {e}")
            self.vietocr_detector = None
    
    async def process_pdf_with_ocr(
        self, 
        file_content: bytes, 
        filename: str,
        use_vietocr: bool = True
    ) -> Tuple[str, int, Dict[str, Any]]:
        """
        Xử lý PDF scan với OCR
        
        Returns:
            (extracted_text, total_pages, ocr_metadata)
        """
        try:
            # First, try to extract text normally
            normal_text, total_pages = await self._extract_text_normal(file_content)
            
            # Check if normal extraction yielded sufficient text
            if len(normal_text.strip()) > 100:  # If we got decent text
                logger.info(f"PDF {filename} has extractable text, using normal extraction")
                return normal_text, total_pages, {
                    "ocr_used": False,
                    "extraction_method": "normal",
                    "confidence": 1.0
                }
            
            # PDF seems to be scan/image-based, use OCR
            logger.info(f"PDF {filename} appears to be scanned, using OCR")
            return await self._process_with_ocr(file_content, filename, use_vietocr)
            
        except Exception as e:
            logger.error(f"Error processing PDF with OCR: {e}")
            raise
    
    async def _extract_text_normal(self, file_content: bytes) -> Tuple[str, int]:
        """Extract text normally from PDF"""
        def extract():
            doc = fitz.open(stream=file_content, filetype="pdf")
            full_text = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            total_pages = doc.page_count
            doc.close()
            return full_text, total_pages
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, extract
        )
    
    async def _process_with_ocr(
        self, 
        file_content: bytes, 
        filename: str,
        use_vietocr: bool = True
    ) -> Tuple[str, int, Dict[str, Any]]:
        """Process PDF with OCR"""
        
        # Convert PDF to images
        images = await self._pdf_to_images(file_content)
        total_pages = len(images)
        
        logger.info(f"Converting {total_pages} pages to images for OCR")
        
        # Process pages in parallel
        ocr_tasks = []
        for i, image in enumerate(images):
            task = self._ocr_image(image, i + 1, use_vietocr)
            ocr_tasks.append(task)
        
        # Process in batches to avoid memory issues
        batch_size = 4
        all_results = []
        
        for i in range(0, len(ocr_tasks), batch_size):
            batch = ocr_tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            all_results.extend(batch_results)
        
        # Combine results
        full_text = ""
        total_confidence = 0
        successful_pages = 0
        
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                logger.error(f"OCR failed for page {i + 1}: {result}")
                full_text += f"\n--- Page {i + 1} ---\n[OCR_ERROR: {str(result)}]\n"
            else:
                text, confidence = result
                full_text += f"\n--- Page {i + 1} ---\n{text}"
                total_confidence += confidence
                successful_pages += 1
        
        average_confidence = total_confidence / successful_pages if successful_pages > 0 else 0
        
        ocr_metadata = {
            "ocr_used": True,
            "extraction_method": "vietocr" if use_vietocr and self.vietocr_detector else "tesseract",
            "total_pages": total_pages,
            "successful_pages": successful_pages,
            "failed_pages": total_pages - successful_pages,
            "average_confidence": average_confidence,
            "dpi": self.dpi
        }
        
        logger.info(f"OCR completed for {filename}: {successful_pages}/{total_pages} pages successful")
        
        return full_text, total_pages, ocr_metadata
    
    async def _pdf_to_images(self, file_content: bytes) -> List[Image.Image]:
        """Convert PDF pages to images"""
        def convert():
            return convert_from_bytes(
                file_content,
                dpi=self.dpi,
                fmt='RGB',
                thread_count=2
            )
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, convert
        )
    
    async def _ocr_image(
        self, 
        image: Image.Image, 
        page_num: int, 
        use_vietocr: bool = True
    ) -> Tuple[str, float]:
        """
        OCR một trang image
        
        Returns:
            (extracted_text, confidence_score)
        """
        def process_image():
            try:
                # Preprocess image for better OCR
                processed_image = self._preprocess_image(image)
                
                # Try VietOCR first if available and requested
                if use_vietocr and self.vietocr_detector:
                    try:
                        text = self.vietocr_detector.predict(processed_image)
                        confidence = 0.85  # VietOCR doesn't provide confidence, assume good
                        return text, confidence
                    except Exception as e:
                        logger.warning(f"VietOCR failed for page {page_num}: {e}, falling back to Tesseract")
                
                # Fallback to Tesseract
                # Convert PIL to numpy array for Tesseract
                img_array = np.array(processed_image)
                
                # Get text with confidence
                data = pytesseract.image_to_data(
                    img_array, 
                    config=self.tesseract_config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Extract text and calculate average confidence
                texts = []
                confidences = []
                
                for i, conf in enumerate(data['conf']):
                    if int(conf) > 0:  # Valid confidence
                        text = data['text'][i].strip()
                        if text:
                            texts.append(text)
                            confidences.append(int(conf))
                
                extracted_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
                
                return extracted_text, avg_confidence
                
            except Exception as e:
                logger.error(f"OCR error for page {page_num}: {e}")
                return "", 0.0
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, process_image
        )
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy
        """
        try:
            # Convert to numpy array for OpenCV
            img_array = np.array(image)
            
            # Convert to grayscale if needed
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply filters to improve OCR
            # 1. Noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # 2. Increase contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(denoised)
            
            # 3. Threshold to binary
            _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(binary)
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}, using original")
            return image
    
    async def extract_text_from_image(
        self, 
        image_bytes: bytes, 
        use_vietocr: bool = True
    ) -> Tuple[str, float]:
        """
        Extract text from a single image
        """
        def process():
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Process with OCR
            return self._ocr_image(image, 1, use_vietocr)
        
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, process
        )
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported OCR languages"""
        try:
            langs = pytesseract.get_languages()
            supported = []
            
            if 'vie' in langs:
                supported.append('vietnamese')
            if 'eng' in langs:
                supported.append('english')
            
            if VIETOCR_AVAILABLE and self.vietocr_detector:
                supported.append('vietnamese_vietocr')
            
            return supported
            
        except Exception as e:
            logger.error(f"Failed to get supported languages: {e}")
            return ['english']  # Fallback
    
    async def validate_ocr_setup(self) -> Dict[str, Any]:
        """
        Validate OCR setup and return status
        """
        status = {
            "tesseract_available": False,
            "tesseract_languages": [],
            "vietocr_available": False,
            "opencv_available": False,
            "pdf2image_available": False
        }
        
        try:
            # Test Tesseract
            pytesseract.get_tesseract_version()
            status["tesseract_available"] = True
            status["tesseract_languages"] = pytesseract.get_languages()
        except Exception as e:
            logger.error(f"Tesseract not available: {e}")
        
        try:
            # Test VietOCR
            if VIETOCR_AVAILABLE and self.vietocr_detector:
                status["vietocr_available"] = True
        except Exception as e:
            logger.error(f"VietOCR not available: {e}")
        
        try:
            # Test OpenCV
            cv2.__version__
            status["opencv_available"] = True
        except Exception as e:
            logger.error(f"OpenCV not available: {e}")
        
        try:
            # Test pdf2image
            from pdf2image import convert_from_bytes
            status["pdf2image_available"] = True
        except Exception as e:
            logger.error(f"pdf2image not available: {e}")
        
        return status

# Global instance
ocr_service = OCRService()
