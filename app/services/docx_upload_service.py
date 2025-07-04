"""
Service để xử lý upload file DOCX offline và chuyển thành file DOCX online
"""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from fastapi import UploadFile

from app.services.google_drive_service import google_drive_service
from app.models.online_document_models import OnlineDocumentResponse, OnlineDocumentLinks

logger = logging.getLogger(__name__)


class DocxUploadService:
    """Service để xử lý upload file DOCX và chuyển thành online document"""
    
    def __init__(self):
        """Khởi tạo service với thư mục tạm"""
        self.temp_dir = Path(tempfile.gettempdir()) / "docx_uploads"
        self.temp_dir.mkdir(exist_ok=True)
    
    async def process_docx_upload_to_online(
        self, 
        uploaded_file: UploadFile,
        convert_to_google_docs: bool = True
    ) -> Dict[str, Any]:
        """
        Xử lý file DOCX upload và chuyển thành online document
        
        Args:
            uploaded_file: File DOCX được upload từ client
            convert_to_google_docs: Có convert thành Google Docs không (default: True)
            
        Returns:
            Dict chứa thông tin online document hoặc lỗi
        """
        temp_file_path = None
        
        try:
            # 1. Validate file type
            if not self._validate_docx_file(uploaded_file):
                return {
                    "success": False,
                    "error": "File không hợp lệ. Chỉ hỗ trợ file DOCX.",
                    "error_code": "INVALID_FILE_TYPE"
                }
            
            # 2. Validate file size (giới hạn 50MB)
            file_content = await uploaded_file.read()
            if not self._validate_file_size(file_content):
                return {
                    "success": False,
                    "error": "File quá lớn. Giới hạn tối đa 50MB.",
                    "error_code": "FILE_TOO_LARGE"
                }
            
            # 3. Lưu file tạm thời
            temp_file_path = await self._save_temp_file(file_content, uploaded_file.filename)
            if not temp_file_path:
                return {
                    "success": False,
                    "error": "Không thể lưu file tạm thời.",
                    "error_code": "TEMP_FILE_ERROR"
                }
            
            # 4. Upload lên Google Drive
            upload_result = await self._upload_to_google_drive(
                temp_file_path, 
                uploaded_file.filename,
                convert_to_google_docs
            )
            
            if not upload_result.get("success", False):
                return {
                    "success": False,
                    "error": f"Không thể upload lên Google Drive: {upload_result.get('error', 'Unknown error')}",
                    "error_code": "UPLOAD_FAILED"
                }
            
            # 5. Tạo response theo format chuẩn
            response_data = self._create_online_document_response(upload_result, uploaded_file.filename)
            
            logger.info(f"Successfully processed DOCX upload: {uploaded_file.filename}")
            return response_data
            
        except Exception as e:
            logger.error(f"Error processing DOCX upload: {e}")
            return {
                "success": False,
                "error": f"Lỗi xử lý file: {str(e)}",
                "error_code": "PROCESSING_ERROR"
            }
        
        finally:
            # 6. Cleanup: Xóa file tạm
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Deleted temporary file: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Could not delete temporary file: {e}")
    
    def _validate_docx_file(self, uploaded_file: UploadFile) -> bool:
        """
        Validate file type là DOCX
        
        Args:
            uploaded_file: File được upload
            
        Returns:
            bool: True nếu file hợp lệ
        """
        if not uploaded_file.filename:
            return False
        
        # Kiểm tra extension
        if not uploaded_file.filename.lower().endswith('.docx'):
            return False
        
        # Kiểm tra MIME type
        if uploaded_file.content_type and not uploaded_file.content_type in [
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/octet-stream'  # Fallback cho một số browser
        ]:
            return False
        
        return True
    
    def _validate_file_size(self, file_content: bytes) -> bool:
        """
        Validate kích thước file
        
        Args:
            file_content: Nội dung file
            
        Returns:
            bool: True nếu file size hợp lệ
        """
        max_size = 50 * 1024 * 1024  # 50MB
        return len(file_content) <= max_size and len(file_content) > 0
    
    async def _save_temp_file(self, file_content: bytes, filename: str) -> str:
        """
        Lưu file tạm thời
        
        Args:
            file_content: Nội dung file
            filename: Tên file gốc
            
        Returns:
            str: Đường dẫn file tạm hoặc None nếu lỗi
        """
        try:
            # Tạo tên file unique
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = self._sanitize_filename(filename)
            temp_filename = f"{timestamp}_{safe_filename}"
            temp_file_path = self.temp_dir / temp_filename
            
            # Lưu file
            with open(temp_file_path, 'wb') as f:
                f.write(file_content)
            
            logger.info(f"Saved temporary file: {temp_file_path}")
            return str(temp_file_path)
            
        except Exception as e:
            logger.error(f"Error saving temporary file: {e}")
            return None
    
    async def _upload_to_google_drive(
        self, 
        file_path: str, 
        original_filename: str,
        convert_to_google_docs: bool
    ) -> Dict[str, Any]:
        """
        Upload file lên Google Drive
        
        Args:
            file_path: Đường dẫn file local
            original_filename: Tên file gốc
            convert_to_google_docs: Có convert thành Google Docs không
            
        Returns:
            Dict: Kết quả upload
        """
        try:
            # Kiểm tra Google Drive service
            if not google_drive_service.is_available():
                return {
                    "success": False,
                    "error": "Google Drive service không khả dụng"
                }
            
            # Upload file
            upload_result = await google_drive_service.upload_docx_file(
                file_path=file_path,
                filename=original_filename,
                convert_to_google_docs=convert_to_google_docs
            )
            
            return upload_result
            
        except Exception as e:
            logger.error(f"Error uploading to Google Drive: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_online_document_response(
        self, 
        upload_result: Dict[str, Any], 
        original_filename: str
    ) -> Dict[str, Any]:
        """
        Tạo response theo format OnlineDocumentResponse
        
        Args:
            upload_result: Kết quả upload từ Google Drive
            original_filename: Tên file gốc
            
        Returns:
            Dict: Response data
        """
        try:
            links_data = upload_result.get("links", {})
            
            # Tạo OnlineDocumentLinks
            links = OnlineDocumentLinks(
                view=links_data.get("view", ""),
                edit=links_data.get("edit"),
                preview=links_data.get("preview"),
                download=links_data.get("download")
            )
            
            # Tạo OnlineDocumentResponse
            response = OnlineDocumentResponse(
                success=True,
                message="File DOCX đã được upload và chuyển thành online document thành công",
                file_id=upload_result.get("file_id"),
                filename=upload_result.get("filename", original_filename),
                mime_type=upload_result.get("mime_type"),
                links=links,
                primary_link=links_data.get("view", links_data.get("edit", "")),
                created_at=datetime.now().isoformat(),
                storage_provider="Google Drive",
                additional_info={
                    "original_filename": original_filename,
                    "web_view_link": upload_result.get("web_view_link"),
                    "web_content_link": upload_result.get("web_content_link")
                }
            )
            
            return response.model_dump()
            
        except Exception as e:
            logger.error(f"Error creating response: {e}")
            return {
                "success": False,
                "error": f"Lỗi tạo response: {str(e)}",
                "error_code": "RESPONSE_ERROR"
            }
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Làm sạch tên file để tránh lỗi
        
        Args:
            filename: Tên file gốc
            
        Returns:
            str: Tên file đã làm sạch
        """
        if not filename:
            return "document.docx"
        
        # Loại bỏ ký tự đặc biệt
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
        safe_filename = "".join(c for c in filename if c in safe_chars)
        
        # Đảm bảo có extension
        if not safe_filename.lower().endswith('.docx'):
            safe_filename = safe_filename.rsplit('.', 1)[0] + '.docx' if '.' in safe_filename else safe_filename + '.docx'
        
        # Giới hạn độ dài
        if len(safe_filename) > 100:
            name_part = safe_filename[:-5]  # Bỏ .docx
            safe_filename = name_part[:95] + '.docx'
        
        return safe_filename


# Singleton instance
docx_upload_service = DocxUploadService()
