"""
Service để upload file DOCX lên Google Drive và tạo link online
"""

import logging
import os
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

from app.core.config import settings

logger = logging.getLogger(__name__)


class GoogleDriveService:
    """Service để quản lý upload file lên Google Drive"""

    def __init__(self):
        self.service = None
        self.credentials = None
        self._initialize_service()

    def _initialize_service(self):
        """Khởi tạo Google Drive service"""
        try:
            # Kiểm tra credentials file
            credentials_path = getattr(settings, 'GOOGLE_DRIVE_CREDENTIALS_PATH', None)
            if not credentials_path or not os.path.exists(credentials_path):
                logger.warning("Google Drive credentials not found. Service will be disabled.")
                return

            # Tạo credentials từ service account
            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            self.credentials = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=SCOPES
            )

            # Tạo service
            self.service = build('drive', 'v3', credentials=self.credentials)
            logger.info("Google Drive service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Google Drive service: {e}")
            self.service = None

    def is_available(self) -> bool:
        """Kiểm tra service có sẵn sàng không"""
        return self.service is not None

    async def upload_docx_file(
        self, 
        file_path: str, 
        filename: str,
        convert_to_google_docs: bool = True
    ) -> Dict[str, Any]:
        """
        Upload file DOCX lên Google Drive

        Args:
            file_path: Đường dẫn file local
            filename: Tên file muốn lưu
            convert_to_google_docs: Có convert thành Google Docs không

        Returns:
            Dict chứa thông tin file đã upload
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Google Drive service not available"
            }

        try:
            # Chuẩn bị metadata
            file_metadata = {
                'name': filename,
            }

            # Thêm folder nếu có
            folder_id = getattr(settings, 'GOOGLE_DRIVE_FOLDER_ID', None)
            if folder_id:
                file_metadata['parents'] = [folder_id]

            # Chuẩn bị media upload
            if convert_to_google_docs:
                # Convert thành Google Docs
                media = MediaFileUpload(
                    file_path,
                    mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                )
                file_metadata['mimeType'] = 'application/vnd.google-apps.document'
            else:
                # Giữ nguyên format DOCX
                media = MediaFileUpload(
                    file_path,
                    mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                )

            # Upload file
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id,name,webViewLink,webContentLink,mimeType'
            ).execute()

            # Tạo permission để share
            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            self.service.permissions().create(
                fileId=file.get('id'),
                body=permission
            ).execute()

            # Tạo các link
            file_id = file.get('id')
            links = self._generate_links(file_id, file.get('mimeType'))

            logger.info(f"Successfully uploaded file to Google Drive: {file_id}")

            return {
                "success": True,
                "file_id": file_id,
                "filename": file.get('name'),
                "mime_type": file.get('mimeType'),
                "links": links,
                "web_view_link": file.get('webViewLink'),
                "web_content_link": file.get('webContentLink')
            }

        except HttpError as e:
            logger.error(f"Google Drive API error: {e}")
            return {
                "success": False,
                "error": f"Google Drive API error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"Error uploading to Google Drive: {e}")
            return {
                "success": False,
                "error": f"Upload error: {str(e)}"
            }

    def _generate_links(self, file_id: str, mime_type: str) -> Dict[str, str]:
        """Tạo các link cho file"""
        base_url = f"https://drive.google.com/file/d/{file_id}"
        
        if mime_type == 'application/vnd.google-apps.document':
            # Google Docs links
            return {
                "view": f"https://docs.google.com/document/d/{file_id}/view",
                "edit": f"https://docs.google.com/document/d/{file_id}/edit",
                "preview": f"https://docs.google.com/document/d/{file_id}/preview",
                "download": f"https://docs.google.com/document/d/{file_id}/export?format=docx"
            }
        else:
            # Regular file links
            return {
                "view": f"{base_url}/view",
                "download": f"{base_url}/view?usp=sharing",
                "preview": f"{base_url}/preview"
            }

    async def delete_file(self, file_id: str) -> bool:
        """Xóa file trên Google Drive"""
        if not self.is_available():
            return False

        try:
            self.service.files().delete(fileId=file_id).execute()
            logger.info(f"Deleted file from Google Drive: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file from Google Drive: {e}")
            return False

    async def cleanup_old_files(self, days_old: int = 7) -> int:
        """Xóa các file cũ trên Google Drive"""
        if not self.is_available():
            return 0

        try:
            from datetime import datetime, timedelta
            
            # Tính ngày cắt
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cutoff_str = cutoff_date.isoformat() + 'Z'

            # Tìm file cũ
            query = f"createdTime < '{cutoff_str}'"
            folder_id = getattr(settings, 'GOOGLE_DRIVE_FOLDER_ID', None)
            if folder_id:
                query += f" and '{folder_id}' in parents"

            results = self.service.files().list(
                q=query,
                fields="files(id,name,createdTime)"
            ).execute()

            files = results.get('files', [])
            deleted_count = 0

            for file in files:
                try:
                    await self.delete_file(file['id'])
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Error deleting old file {file['id']}: {e}")

            logger.info(f"Cleaned up {deleted_count} old files from Google Drive")
            return deleted_count

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0


# Singleton instance
google_drive_service = GoogleDriveService()
