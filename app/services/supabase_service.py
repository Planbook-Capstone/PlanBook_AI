"""
Supabase Service for file storage and management
"""

import logging
import asyncio
import base64
import io
from typing import Dict, List, Any, Optional
from pathlib import Path
import uuid
from datetime import datetime

from supabase import create_client, Client
from app.core.config import Settings

logger = logging.getLogger(__name__)
settings = Settings()


class SupabaseService:
    """Service for managing Supabase storage operations"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.bucket_name = settings.SUPABASE_BUCKET
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Supabase client"""
        try:
            if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
                logger.warning("Supabase credentials not configured. Service will be disabled.")
                return
            
            self.client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
            logger.info("Supabase client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Supabase service is available"""
        return self.client is not None
    
    async def upload_file(
        self,
        bucket_name: str,
        file_path: str,
        file_content: bytes,
        content_type: str = "application/octet-stream",
        upsert: bool = True
    ) -> Dict[str, Any]:
        """
        Upload file to Supabase storage

        Args:
            bucket_name: Name of the storage bucket
            file_path: Path where file will be stored in bucket
            file_content: File content as bytes
            content_type: MIME type of the file
            upsert: Whether to overwrite existing file

        Returns:
            Dict with upload result
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Supabase service not available"
            }

        try:
            # Upload file to storage
            file_options = {"content-type": content_type}
            if upsert:
                file_options["upsert"] = "true"  # Convert to string

            response = self.client.storage.from_(bucket_name).upload(
                path=file_path,
                file=file_content,
                file_options=file_options
            )

            # Check if upload was successful
            if hasattr(response, 'status_code'):
                success = response.status_code == 200
            else:
                # For newer versions, check if response contains error
                success = not hasattr(response, 'error') or response.error is None

            if success:
                # Get public URL
                public_url = self.client.storage.from_(bucket_name).get_public_url(file_path)

                return {
                    "success": True,
                    "url": public_url,
                    "path": file_path,
                    "bucket": bucket_name,
                    "size": len(file_content)
                }
            else:
                error_msg = getattr(response, 'text', str(response))
                logger.error(f"Upload failed: {error_msg}")
                return {
                    "success": False,
                    "error": f"Upload failed: {error_msg}"
                }

        except Exception as e:
            logger.error(f"Error uploading file to Supabase: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def upload_image(
        self,
        image_data: bytes,
        filename: str,
        folder: str = "textbook_images",
        content_type: str = "image/png"
    ) -> Dict[str, Any]:
        """
        Upload image to Supabase storage
        
        Args:
            image_data: Image content as bytes
            filename: Name of the image file
            folder: Folder path in bucket
            content_type: Image MIME type
            
        Returns:
            Dict with upload result
        """
        file_path = f"{folder}/{filename}"
        
        return await self.upload_file(
            bucket_name=self.bucket_name,
            file_path=file_path,
            file_content=image_data,
            content_type=content_type
        )
    
    async def upload_base64_image(
        self,
        base64_data: str,
        filename: str,
        folder: str = "textbook_images",
        content_type: str = "image/png"
    ) -> Dict[str, Any]:
        """
        Upload base64 encoded image to Supabase storage
        
        Args:
            base64_data: Base64 encoded image data
            filename: Name of the image file
            folder: Folder path in bucket
            content_type: Image MIME type
            
        Returns:
            Dict with upload result
        """
        try:
            # Decode base64 data
            image_bytes = base64.b64decode(base64_data)
            
            return await self.upload_image(
                image_data=image_bytes,
                filename=filename,
                folder=folder,
                content_type=content_type
            )
            
        except Exception as e:
            logger.error(f"Error uploading base64 image: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def delete_file(
        self,
        bucket_name: str,
        file_path: str
    ) -> Dict[str, Any]:
        """
        Delete file from Supabase storage
        
        Args:
            bucket_name: Name of the storage bucket
            file_path: Path of file to delete
            
        Returns:
            Dict with deletion result
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Supabase service not available"
            }
        
        try:
            response = self.client.storage.from_(bucket_name).remove([file_path])
            
            return {
                "success": True,
                "deleted_files": response
            }
            
        except Exception as e:
            logger.error(f"Error deleting file from Supabase: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_files(
        self,
        bucket_name: str,
        folder: str = "",
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        List files in Supabase storage bucket

        Args:
            bucket_name: Name of the storage bucket
            folder: Folder path to list
            limit: Maximum number of files to return (not used in current API)

        Returns:
            Dict with file list
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Supabase service not available"
            }

        try:
            # Note: Supabase storage list() doesn't support limit parameter
            response = self.client.storage.from_(bucket_name).list(path=folder)

            # Apply limit manually if needed
            if isinstance(response, list) and len(response) > limit:
                response = response[:limit]

            return {
                "success": True,
                "files": response
            }

        except Exception as e:
            logger.error(f"Error listing files from Supabase: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def generate_unique_filename(
        self,
        original_filename: str,
        prefix: str = "",
        suffix: str = ""
    ) -> str:
        """
        Generate unique filename with timestamp and UUID
        
        Args:
            original_filename: Original filename
            prefix: Prefix to add
            suffix: Suffix to add
            
        Returns:
            Unique filename
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        # Extract file extension
        file_path = Path(original_filename)
        name = file_path.stem
        ext = file_path.suffix
        
        # Build unique filename
        parts = []
        if prefix:
            parts.append(prefix)
        parts.extend([name, timestamp, unique_id])
        if suffix:
            parts.append(suffix)
        
        unique_name = "_".join(parts) + ext
        return unique_name


# Global service instance
supabase_service = SupabaseService()
