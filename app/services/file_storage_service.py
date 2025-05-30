"""
File Storage Service sử dụng MongoDB GridFS
Thay thế local file storage bằng database storage cho scalability
"""
import gridfs
from motor.motor_asyncio import AsyncIOMotorGridFSBucket
import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from bson import ObjectId
import io

from app.database.connection import mongodb
from app.core.config import settings

logger = logging.getLogger(__name__)

class FileStorageService:
    """
    Service quản lý file storage với MongoDB GridFS
    """
    def __init__(self):
        self.bucket_name = "planbook_files"
        self._bucket = None
        self._sync_bucket = None
    
    async def get_gridfs_bucket(self):
        """Get GridFS bucket cho async operations"""
        if not self._bucket:
            if mongodb.database is None:
                await mongodb.connect()
            self._bucket = AsyncIOMotorGridFSBucket(
                mongodb.database, 
                bucket_name=self.bucket_name
            )
        return self._bucket
    
    def get_gridfs_bucket_sync(self):
        """Get GridFS bucket cho sync operations"""
        if not self._sync_bucket:
            if mongodb.sync_database is None:
                mongodb.connect_sync()
            self._sync_bucket = gridfs.GridFSBucket(
                mongodb.sync_database,
                bucket_name=self.bucket_name
            )
        return self._sync_bucket
    
    async def store_pdf_file(
        self, 
        file_content: bytes, 
        filename: str,
        metadata: Dict[str, Any]
    ) -> ObjectId:
        """Lưu PDF file vào GridFS với metadata"""
        try:
            bucket = await self.get_gridfs_bucket()
            
            # Enhanced metadata
            file_metadata = {
                **metadata,
                "file_type": "pdf",
                "content_type": "application/pdf",
                "file_size": len(file_content),
                "created_at": datetime.utcnow(),
                "original_filename": filename
            }
            
            file_id = await bucket.upload_from_stream(
                filename=filename,
                source=file_content,
                metadata=file_metadata
            )
            
            logger.info(f"Stored PDF to GridFS: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Error storing PDF to GridFS: {e}")
            raise
    
    async def store_docx_file(
        self,
        file_content: bytes,
        filename: str,
        metadata: Dict[str, Any]
    ) -> ObjectId:
        """Lưu DOCX file vào GridFS"""
        try:
            bucket = await self.get_gridfs_bucket()
            
            file_metadata = {
                **metadata,
                "file_type": "docx",
                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "file_size": len(file_content),
                "created_at": datetime.utcnow()
            }
            
            file_id = await bucket.upload_from_stream(
                filename=filename,
                source=file_content,
                metadata=file_metadata
            )
            
            logger.info(f"Stored DOCX to GridFS: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Error storing DOCX: {e}")
            raise
    
    async def get_file_by_id(self, file_id: ObjectId) -> Dict[str, Any]:
        """Lấy file từ GridFS bằng ID"""
        try:
            bucket = await self.get_gridfs_bucket()
            
            # Get file info
            file_info = await bucket.find({"_id": file_id}).to_list(length=1)
            if not file_info:
                raise FileNotFoundError(f"File not found: {file_id}")
            
            file_doc = file_info[0]
            
            # Download file content
            download_stream = await bucket.open_download_stream(file_id)
            file_content = await download_stream.read()
            
            return {
                "file_id": file_id,
                "filename": file_doc.get("filename", "unknown"),
                "content": file_content,
                "metadata": file_doc.get("metadata", {}),
                "upload_date": file_doc.get("uploadDate"),
                "length": file_doc.get("length", 0)
            }
        
        except Exception as e:
            logger.error(f"Error retrieving file from GridFS: {e}")
            raise
    
    async def list_files_by_type(
        self, 
        file_type: str, 
        limit: int = 50,
        **filters
    ) -> List[Dict[str, Any]]:
        """List files theo type và filters"""
        try:
            bucket = await self.get_gridfs_bucket()
            
            query = {"metadata.file_type": file_type}
            
            # Add additional filters
            for key, value in filters.items():
                query[f"metadata.{key}"] = value
            
            cursor = bucket.find(query).limit(limit).sort("uploadDate", -1)
            files = await cursor.to_list(length=limit)
            
            result = []
            for file in files:
                try:
                    # Ensure file object has all required attributes
                    if hasattr(file, '_id') and hasattr(file, 'filename'):
                        result.append({
                            "file_id": str(file._id),
                            "filename": file.filename,
                            "metadata": getattr(file, 'metadata', {}),
                            "upload_date": getattr(file, 'upload_date', None),
                            "file_size": getattr(file, 'length', 0)
                        })
                except Exception as item_error:
                    logger.warning(f"Error processing file item: {item_error}")
                    continue
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            # Return empty list instead of raising to prevent API crashes
            return []
    
    async def delete_file(self, file_id: ObjectId) -> bool:
        """Xóa file từ GridFS"""
        try:
            bucket = await self.get_gridfs_bucket()
            await bucket.delete(file_id)
            logger.info(f"Deleted file from GridFS: {file_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            return False
    
    async def get_file_stream_response(self, file_id: ObjectId):
        """Tạo streaming response cho download file"""
        from fastapi.responses import StreamingResponse
        
        try:
            bucket = await self.get_gridfs_bucket()
            
            # Get file info
            file_info = await bucket.find({"_id": file_id}).to_list(length=1)
            if not file_info:
                raise FileNotFoundError(f"File not found: {file_id}")
            
            file_doc = file_info[0]
            
            # Download file content directly
            download_stream = await bucket.open_download_stream(file_id)
            file_content = await download_stream.read()
            
            # Determine content type from metadata
            metadata = file_doc.get("metadata", {})
            content_type = metadata.get("content_type", "application/octet-stream")
            filename = file_doc.get("filename", "download")            # Clean filename for Content-Disposition header
            import urllib.parse
            safe_filename = urllib.parse.quote(filename.encode('utf-8'))
            
            return StreamingResponse(
                io.BytesIO(file_content),
                media_type=content_type,
                headers={
                    "Content-Disposition": f"attachment; filename*=UTF-8''{safe_filename}"
                }
            )
            
        except Exception as e:
            logger.error(f"Error creating file stream response: {e}")
            raise

# Global instance
file_storage = FileStorageService()
