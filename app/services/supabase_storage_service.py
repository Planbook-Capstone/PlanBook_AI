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
    
    async def upload_pdf_file(
        self, 
        file_content: bytes, 
        book_id: str, 
        lesson_id: Optional[str] = None,
        original_filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload file PDF lên Supabase Storage
        
        Args:
            file_content: Nội dung file PDF
            book_id: ID của sách
            lesson_id: ID của bài học (optional)
            original_filename: Tên file gốc (optional)
            
        Returns:
            Dict chứa thông tin file đã upload và URL
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Supabase service not available"
            }
        
        try:
            # Tạo tên file unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_id = str(uuid.uuid4())[:8]
            
            if lesson_id:
                filename = f"{book_id}_{lesson_id}_{timestamp}_{file_id}.pdf"
            else:
                filename = f"{book_id}_{timestamp}_{file_id}.pdf"
            
            # Tạo đường dẫn file trong bucket theo cấu trúc thư mục
            file_path = f"{book_id}/{filename}"
            
            logger.info(f"Uploading PDF to Supabase: {file_path}")
            
            # Upload file lên Supabase Storage
            try:
                response = self.supabase_client.storage.from_(self.bucket_name).upload(
                    path=file_path,
                    file=file_content,
                    file_options={
                        "content-type": "application/pdf",
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

            logger.info(f"Successfully uploaded PDF to Supabase: {file_url}")

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
            logger.error(f"Error uploading PDF to Supabase: {e}")
            return {
                "success": False,
                "error": f"Upload error: {str(e)}"
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


# Singleton instance
_supabase_storage_service = None

def get_supabase_storage_service() -> SupabaseStorageService:
    """Lấy singleton instance của SupabaseStorageService"""
    global _supabase_storage_service
    if _supabase_storage_service is None:
        _supabase_storage_service = SupabaseStorageService()
    return _supabase_storage_service
