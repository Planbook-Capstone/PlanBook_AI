"""
Service để xử lý upload file PDF lên Supabase Storage
"""

import logging
import os
from typing import Dict, Any, Optional
from datetime import datetime
import uuid

from app.core.config import settings

logger = logging.getLogger(__name__)


class SupabaseStorageService:
    """Service để xử lý upload file PDF lên Supabase Storage"""
    
    def __init__(self):
        """Khởi tạo service với Supabase client"""
        self.supabase_client = None
        self.bucket_name = settings.SUPABASE_BUCKET_NAME
        self._service_initialized = False
    
    def _ensure_service_initialized(self):
        """Đảm bảo service được khởi tạo"""
        if not self._service_initialized:
            self._init_supabase_client()
            self._service_initialized = True
    
    def _init_supabase_client(self):
        """Khởi tạo Supabase client"""
        try:
            if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
                logger.warning("Supabase URL or KEY not configured")
                return
            
            from supabase import create_client
            
            self.supabase_client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_KEY
            )
            
            logger.info("✅ Supabase client initialized successfully")
            
        except ImportError:
            logger.error("❌ Supabase client not installed. Run: pip install supabase")
            self.supabase_client = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize Supabase client: {e}")
            self.supabase_client = None
    
    def is_available(self) -> bool:
        """Kiểm tra service có sẵn sàng không"""
        self._ensure_service_initialized()
        return self.supabase_client is not None
    
    async def upload_document_file(
        self,
        file_content: bytes,
        book_id: str,
        lesson_id: Optional[str] = None,
        original_filename: Optional[str] = None,
        file_type: str = "pdf"
    ) -> Dict[str, Any]:
        """
        Upload file document (PDF hoặc DOCX) lên Supabase Storage

        Args:
            file_content: Nội dung file
            book_id: ID của sách
            lesson_id: ID của bài học (optional)
            original_filename: Tên file gốc (optional)
            file_type: Loại file ("pdf" hoặc "docx")

        Returns:
            Dict chứa thông tin file đã upload và URL
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Supabase service not available"
            }
        
        try:
            # Xác định extension và content-type
            if file_type.lower() == "docx":
                extension = "docx"
                content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            else:  # default to PDF
                extension = "pdf"
                content_type = "application/pdf"

            # Tạo tên file unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_id = str(uuid.uuid4())[:8]

            if lesson_id:
                filename = f"{book_id}_{lesson_id}_{timestamp}_{file_id}.{extension}"
            else:
                filename = f"{book_id}_{timestamp}_{file_id}.{extension}"
            
            # Tạo đường dẫn file trong bucket theo cấu trúc thư mục
            file_path = f"{book_id}/{filename}"
            
            logger.info(f"Uploading {file_type.upper()} to Supabase: {file_path}")
            
            # Upload file lên Supabase Storage
            try:
                response = self.supabase_client.storage.from_(self.bucket_name).upload(
                    path=file_path,
                    file=file_content,
                    file_options={
                        "content-type": content_type,
                        "upsert": "true"  # Sửa từ True thành "true"
                    }
                )

                logger.info(f"Upload response: {response}")

            except Exception as upload_error:
                logger.error(f"Supabase upload exception: {upload_error}")
                return {
                    "success": False,
                    "error": f"Upload failed: {str(upload_error)}"
                }

            # Lấy public URL của file
            try:
                file_url = self.supabase_client.storage.from_(self.bucket_name).get_public_url(file_path)
                logger.info(f"Generated public URL: {file_url}")

            except Exception as url_error:
                logger.error(f"Failed to get public URL: {url_error}")
                # Fallback URL construction
                file_url = f"{settings.SUPABASE_URL}/storage/v1/object/public/{self.bucket_name}/{file_path}"
                logger.info(f"Using fallback URL: {file_url}")
            
            # Lưu thời gian upload
            upload_time = datetime.now()

            logger.info(f"Successfully uploaded {file_type.upper()} to Supabase: {file_url}")

            return {
                "success": True,
                "file_path": file_path,
                "file_url": file_url,
                "filename": filename,
                "book_id": book_id,
                "lesson_id": lesson_id,
                "bucket_name": self.bucket_name,
                "uploaded_at": upload_time.isoformat(),
                "upload_timestamp": upload_time  # Thêm timestamp object để sử dụng trong Qdrant
            }
            
        except Exception as e:
            logger.error(f"Error uploading document to Supabase: {e}")
            return {
                "success": False,
                "error": f"Upload error: {str(e)}"
            }

    async def upload_pdf_file(
        self,
        file_content: bytes,
        book_id: str,
        lesson_id: Optional[str] = None,
        original_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Backward compatibility method for PDF upload
        """
        return await self.upload_document_file(
            file_content=file_content,
            book_id=book_id,
            lesson_id=lesson_id,
            original_filename=original_filename,
            file_type="pdf"
        )

    async def upload_docx_file(
        self,
        file_content: bytes,
        book_id: str,
        lesson_id: Optional[str] = None,
        original_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload DOCX file lên Supabase Storage
        """
        return await self.upload_document_file(
            file_content=file_content,
            book_id=book_id,
            lesson_id=lesson_id,
            original_filename=original_filename,
            file_type="docx"
        )
    
    async def delete_file_by_url(self, file_url: str) -> Dict[str, Any]:
        """
        Xóa file từ Supabase Storage bằng URL

        Args:
            file_url: URL của file cần xóa

        Returns:
            Dict chứa kết quả xóa file
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Supabase service not available"
            }

        try:
            # Extract file path from URL
            # URL format: https://project.supabase.co/storage/v1/object/public/bucket/path/to/file
            if "/storage/v1/object/public/" in file_url:
                # Split URL to get path after bucket name
                url_parts = file_url.split("/storage/v1/object/public/")
                if len(url_parts) > 1:
                    # Remove bucket name from path
                    path_with_bucket = url_parts[1]
                    # Remove query parameters if any
                    path_with_bucket = path_with_bucket.split('?')[0]
                    # Remove bucket name (first part)
                    path_parts = path_with_bucket.split('/', 1)
                    if len(path_parts) > 1:
                        file_path = path_parts[1]
                    else:
                        return {
                            "success": False,
                            "error": f"Invalid URL format: cannot extract file path from {file_url}"
                        }
                else:
                    return {
                        "success": False,
                        "error": f"Invalid URL format: {file_url}"
                    }
            else:
                return {
                    "success": False,
                    "error": f"Invalid Supabase URL format: {file_url}"
                }

            return await self.delete_pdf_file(file_path)

        except Exception as e:
            logger.error(f"Error deleting file by URL: {e}")
            return {
                "success": False,
                "error": f"Delete error: {str(e)}"
            }

    async def delete_pdf_file(self, file_path: str) -> Dict[str, Any]:
        """
        Xóa file PDF từ Supabase Storage
        
        Args:
            file_path: Đường dẫn file trong bucket
            
        Returns:
            Dict chứa kết quả xóa file
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Supabase service not available"
            }
        
        try:
            response = self.supabase_client.storage.from_(self.bucket_name).remove([file_path])
            
            if hasattr(response, 'error') and response.error:
                logger.error(f"Supabase delete error: {response.error}")
                return {
                    "success": False,
                    "error": f"Delete failed: {response.error}"
                }
            
            logger.info(f"Successfully deleted PDF from Supabase: {file_path}")
            
            return {
                "success": True,
                "file_path": file_path,
                "deleted_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error deleting PDF from Supabase: {e}")
            return {
                "success": False,
                "error": f"Delete error: {str(e)}"
            }
    
    async def list_files_by_book_id(self, book_id: str) -> Dict[str, Any]:
        """
        Liệt kê tất cả file PDF của một book_id
        
        Args:
            book_id: ID của sách
            
        Returns:
            Dict chứa danh sách file
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Supabase service not available"
            }
        
        try:
            response = self.supabase_client.storage.from_(self.bucket_name).list(book_id)
            
            if hasattr(response, 'error') and response.error:
                logger.error(f"Supabase list error: {response.error}")
                return {
                    "success": False,
                    "error": f"List failed: {response.error}"
                }
            
            files = []
            for file_info in response:
                file_path = f"{book_id}/{file_info['name']}"
                public_url = self.supabase_client.storage.from_(self.bucket_name).get_public_url(file_path)
                
                files.append({
                    "name": file_info['name'],
                    "file_path": file_path,
                    "file_url": public_url,
                    "size": file_info.get('metadata', {}).get('size', 0),
                    "created_at": file_info.get('created_at'),
                    "updated_at": file_info.get('updated_at')
                })
            
            return {
                "success": True,
                "book_id": book_id,
                "files": files,
                "total_files": len(files)
            }
            
        except Exception as e:
            logger.error(f"Error listing files from Supabase: {e}")
            return {
                "success": False,
                "error": f"List error: {str(e)}"
            }


# Factory function - creates new instance each time
def get_supabase_storage_service() -> SupabaseStorageService:
    """Create new instance của SupabaseStorageService (thread-safe)"""
    return SupabaseStorageService()
