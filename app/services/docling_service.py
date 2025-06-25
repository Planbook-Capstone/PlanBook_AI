"""
Docling Service for advanced PDF processing and image extraction
"""

import logging
import asyncio
import base64
import io
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import PictureItem, TableItem, ImageRefMode

from app.services.supabase_service import supabase_service

logger = logging.getLogger(__name__)


class DoclingService:
    """Service for processing PDFs using Docling library"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self._setup_docling_converter()
    
    def _setup_docling_converter(self):
        """Setup Docling document converter with optimal settings"""
        try:
            # Configure pipeline options for image extraction
            pipeline_options = PdfPipelineOptions()
            pipeline_options.images_scale = 2.0  # Higher resolution for better quality
            pipeline_options.generate_page_images = True
            pipeline_options.generate_picture_images = True
            pipeline_options.generate_table_images = True
            
            # Create document converter
            self.doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            
            logger.info("Docling converter initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Docling converter: {e}")
            self.doc_converter = None
    
    async def extract_images_from_pdf(
        self, 
        pdf_content: bytes, 
        filename: str
    ) -> Dict[str, Any]:
        """
        Extract images from PDF using Docling
        
        Args:
            pdf_content: PDF file content as bytes
            filename: Original filename
            
        Returns:
            Dict containing extracted images and metadata
        """
        if not self.doc_converter:
            raise Exception("Docling converter not initialized")
        
        try:
            logger.info(f"Starting Docling image extraction for: {filename}")
            
            # Process PDF in thread executor
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._process_pdf_with_docling, pdf_content, filename
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Docling image extraction failed: {e}")
            raise
    
    def _process_pdf_with_docling(self, pdf_content: bytes, filename: str) -> Dict[str, Any]:
        """Process PDF with Docling in thread executor"""
        try:
            # Docling requires a file path or DocumentStream, not BytesIO
            # Create a temporary file
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                temp_file_path = temp_file.name

            try:
                # Convert document using file path
                conv_result = self.doc_converter.convert(temp_file_path)

                # Extract images and metadata
                extracted_data = {
                    "filename": filename,
                    "total_pages": len(conv_result.document.pages),
                    "images": [],
                    "tables": [],
                    "text_content": conv_result.document.export_to_markdown(),
                    "success": True
                }

                # Extract page images
                page_images = self._extract_page_images(conv_result)
                extracted_data["page_images"] = page_images

                # Extract figure/picture images
                picture_images = self._extract_picture_images(conv_result)
                extracted_data["images"].extend(picture_images)

                # Extract table images
                table_images = self._extract_table_images(conv_result)
                extracted_data["tables"].extend(table_images)

                logger.info(f"Docling extraction completed: {len(picture_images)} pictures, {len(table_images)} tables")

                return extracted_data

            finally:
                # Clean up temporary file
                import os
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file_path}: {e}")
            
        except Exception as e:
            logger.error(f"Error in Docling processing: {e}")
            raise
    
    def _extract_page_images(self, conv_result) -> List[Dict[str, Any]]:
        """Extract full page images"""
        page_images = []
        
        try:
            for page_no, page in conv_result.document.pages.items():
                if hasattr(page, 'image') and page.image:
                    # Convert PIL image to base64
                    img_buffer = io.BytesIO()
                    page.image.pil_image.save(img_buffer, format='PNG')
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    
                    page_images.append({
                        "page_number": page.page_no,
                        "image_data": img_base64,
                        "format": "png",
                        "type": "page_image",
                        "width": page.image.pil_image.width,
                        "height": page.image.pil_image.height
                    })
                    
        except Exception as e:
            logger.error(f"Error extracting page images: {e}")
        
        return page_images
    
    def _extract_picture_images(self, conv_result) -> List[Dict[str, Any]]:
        """Extract figure/picture images with captions"""
        pictures = []
        picture_counter = 0
        
        try:
            for element, _ in conv_result.document.iterate_items():
                if isinstance(element, PictureItem):
                    picture_counter += 1
                    
                    try:
                        # Get image from element
                        pil_image = element.get_image(conv_result.document)
                        
                        # Convert to base64
                        img_buffer = io.BytesIO()
                        pil_image.save(img_buffer, format='PNG')
                        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                        
                        # Extract caption/text
                        caption = getattr(element, 'caption', '') or getattr(element, 'text', '') or ''
                        
                        pictures.append({
                            "index": picture_counter,
                            "image_data": img_base64,
                            "format": "png",
                            "type": "figure",
                            "caption": caption.strip(),
                            "page": getattr(element, 'page', None),
                            "width": pil_image.width,
                            "height": pil_image.height,
                            "bbox": getattr(element, 'bbox', None)
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract picture {picture_counter}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error extracting picture images: {e}")
        
        return pictures
    
    def _extract_table_images(self, conv_result) -> List[Dict[str, Any]]:
        """Extract table images"""
        tables = []
        table_counter = 0
        
        try:
            for element, _ in conv_result.document.iterate_items():
                if isinstance(element, TableItem):
                    table_counter += 1
                    
                    try:
                        # Get image from element
                        pil_image = element.get_image(conv_result.document)
                        
                        # Convert to base64
                        img_buffer = io.BytesIO()
                        pil_image.save(img_buffer, format='PNG')
                        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                        
                        # Extract caption/text
                        caption = getattr(element, 'caption', '') or getattr(element, 'text', '') or ''
                        
                        tables.append({
                            "index": table_counter,
                            "image_data": img_base64,
                            "format": "png",
                            "type": "table",
                            "caption": caption.strip(),
                            "page": getattr(element, 'page', None),
                            "width": pil_image.width,
                            "height": pil_image.height,
                            "bbox": getattr(element, 'bbox', None)
                        })
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract table {table_counter}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error extracting table images: {e}")
        
        return tables
    
    async def upload_images_to_supabase(
        self,
        images: List[Dict[str, Any]],
        book_id: str
    ) -> List[Dict[str, Any]]:
        """Upload extracted images to Supabase storage"""
        if not supabase_service.is_available():
            logger.warning("Supabase service not available. Images will be returned with base64 data.")
            return self._prepare_images_without_upload(images, book_id)

        uploaded_images = []

        for img_data in images:
            try:
                # Create unique filename
                img_type = img_data.get("type", "image")
                img_index = img_data.get("index", 0)
                page = img_data.get("page", 1)

                # Generate unique filename with metadata
                base_filename = f"{book_id}_{img_type}_page{page}_{img_index}.png"
                unique_filename = supabase_service.generate_unique_filename(
                    base_filename,
                    prefix="docling"
                )

                # Upload to Supabase
                upload_result = await supabase_service.upload_base64_image(
                    base64_data=img_data["image_data"],
                    filename=unique_filename,
                    folder="textbook_images",
                    content_type="image/png"
                )

                if upload_result.get("success"):
                    # Add URL and metadata to image data
                    img_data_copy = img_data.copy()
                    img_data_copy["image_url"] = upload_result["url"]
                    img_data_copy["filename"] = unique_filename
                    img_data_copy["upload_status"] = "success"
                    img_data_copy["storage_path"] = upload_result["path"]
                    img_data_copy["file_size"] = upload_result["size"]

                    # Remove base64 data to save space (optional)
                    # del img_data_copy["image_data"]

                    uploaded_images.append(img_data_copy)
                    logger.info(f"✅ Uploaded image to Supabase: {unique_filename}")
                else:
                    logger.error(f"❌ Failed to upload image {unique_filename}: {upload_result.get('error')}")
                    # Keep original data with error status
                    img_data_copy = img_data.copy()
                    img_data_copy["filename"] = unique_filename
                    img_data_copy["upload_status"] = "failed"
                    img_data_copy["upload_error"] = upload_result.get("error")
                    uploaded_images.append(img_data_copy)

            except Exception as e:
                logger.error(f"Error uploading image: {e}")
                # Keep original data with error status
                img_data_copy = img_data.copy()
                img_data_copy["upload_status"] = "error"
                img_data_copy["upload_error"] = str(e)
                uploaded_images.append(img_data_copy)
                continue

        logger.info(f"📤 Supabase upload completed: {len([img for img in uploaded_images if img.get('upload_status') == 'success'])}/{len(images)} successful")
        return uploaded_images

    def _prepare_images_without_upload(
        self,
        images: List[Dict[str, Any]],
        book_id: str
    ) -> List[Dict[str, Any]]:
        """Prepare images metadata without actual upload (fallback)"""
        processed_images = []

        for img_data in images:
            try:
                # Create filename
                img_type = img_data.get("type", "image")
                img_index = img_data.get("index", 0)
                page = img_data.get("page", 1)
                filename = f"{book_id}_{img_type}_page{page}_{img_index}.png"

                # Add metadata without actual upload
                img_data_copy = img_data.copy()
                img_data_copy["filename"] = filename
                img_data_copy["upload_status"] = "pending_supabase_service"

                processed_images.append(img_data_copy)
                logger.debug(f"Prepared image for upload: {filename}")

            except Exception as e:
                logger.error(f"Error processing image: {e}")
                continue

        return processed_images


# Global service instance
docling_service = DoclingService()
