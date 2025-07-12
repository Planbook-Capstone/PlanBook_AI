"""
Service để lấy nội dung textbook từ Qdrant
"""
import logging
from typing import Dict, Any, Optional, List
from fastapi import HTTPException
from qdrant_client import models as qdrant_models

logger = logging.getLogger(__name__)


class TextbookRetrievalService:
    """Service để lấy nội dung textbook từ Qdrant"""
    
    def __init__(self):
        self.qdrant_service = None
    
    def _get_qdrant_service(self):
        """Lazy loading của qdrant service"""
        if not self.qdrant_service:
            from app.services.qdrant_service import get_qdrant_service
            self.qdrant_service = get_qdrant_service()
        return self.qdrant_service
    
    async def get_lesson_content(self, lesson_id: str) -> Dict[str, Any]:
        """
        Lấy nội dung lesson từ Qdrant
        
        Args:
            lesson_id: ID của lesson cần lấy
            
        Returns:
            Dict chứa lesson content và metadata
            
        Raises:
            HTTPException: Khi không tìm thấy lesson hoặc có lỗi
        """
        try:
            qdrant_service = self._get_qdrant_service()

            # Ensure Qdrant service is initialized
            qdrant_service._ensure_service_initialized()

            if not qdrant_service.qdrant_client:
                raise HTTPException(status_code=503, detail="Qdrant service not available")
            
            collections = qdrant_service.qdrant_client.get_collections().collections
            lesson_content = []
            found = False
            book_id = None
            collection_name = None
            
            for collection in collections:
                if collection.name.startswith("textbook_"):
                    try:
                        # Tìm tất cả chunks có lesson_id
                        search_result = qdrant_service.qdrant_client.scroll(
                            collection_name=collection.name,
                            scroll_filter=qdrant_models.Filter(
                                must=[
                                    qdrant_models.FieldCondition(
                                        key="lesson_id",
                                        match=qdrant_models.MatchValue(value=lesson_id),
                                    )
                                ]
                            ),
                            limit=100,
                            with_payload=True,
                        )
                        
                        if search_result[0]:
                            found = True
                            collection_name = collection.name
                            # Extract book_id từ collection name (textbook_abc123 -> abc123)
                            book_id = collection.name.replace("textbook_", "")

                            # Collect points with chunk_index for sorting
                            chunks_with_index = []
                            for point in search_result[0]:
                                payload = point.payload or {}
                                text_content = payload.get("text", "")
                                chunk_index = payload.get("chunk_index", 0)
                                if text_content and text_content.strip():
                                    chunks_with_index.append((chunk_index, text_content.strip()))

                            # Sort by chunk_index to maintain correct order
                            chunks_with_index.sort(key=lambda x: x[0])

                            # Extract sorted text content
                            for _, text_content in chunks_with_index:
                                lesson_content.append(text_content)
                            break
                            
                    except Exception as e:
                        logger.warning(f"Error searching in collection {collection.name}: {e}")
                        continue
            
            if not found:
                raise HTTPException(
                    status_code=404,
                    detail=f"Lesson with ID '{lesson_id}' not found"
                )
            
            # Ghép tất cả nội dung lại
            full_content = " ".join(lesson_content) if lesson_content else "No content available"
            
            return {
                "book_id": book_id,
                "lesson_id": lesson_id,
                "lesson_content": full_content,
                "collection_name": collection_name,
                "total_chunks": len(lesson_content),
                "content_length": len(full_content)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting lesson content for lesson_id {lesson_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    async def get_simple_lesson_content(self, lesson_id: str) -> Dict[str, Any]:
        """
        Lấy nội dung lesson đơn giản (chỉ lesson_content)
        
        Args:
            lesson_id: ID của lesson cần lấy
            
        Returns:
            Dict chỉ chứa lesson_content
        """
        result = await self.get_lesson_content(lesson_id)
        return {
            "lesson_content": result["lesson_content"]
        }
    
    async def get_lesson_with_metadata(self, lesson_id: str) -> Dict[str, Any]:
        """
        Lấy nội dung lesson kèm metadata đầy đủ
        
        Args:
            lesson_id: ID của lesson cần lấy
            
        Returns:
            Dict chứa lesson content và metadata đầy đủ
        """
        return await self.get_lesson_content(lesson_id)


# Factory function để tạo TextbookRetrievalService instance
def get_textbook_retrieval_service() -> TextbookRetrievalService:
    """
    Tạo TextbookRetrievalService instance mới

    Returns:
        TextbookRetrievalService: Fresh instance
    """
    return TextbookRetrievalService()
