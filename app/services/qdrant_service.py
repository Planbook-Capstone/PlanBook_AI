"""
Qdrant Service - Quản lý vector embeddings với Qdrant
"""

import logging
from typing import Dict, Any, List, Optional, Union, cast
import uuid
import datetime

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

from app.core.config import settings

logger = logging.getLogger(__name__)


class QdrantService:
    """Service quản lý vector embeddings với Qdrant"""

    def __init__(self):
        self.embedding_model = None
        self.qdrant_client = None
        self.vector_size: Optional[int] = None
        self._init_embedding_model()
        self._init_qdrant_client()

    def _init_embedding_model(self):
        """Khởi tạo mô hình embedding"""
        try:
            import torch
            import warnings

            model_name = settings.EMBEDDING_MODEL

            # Suppress the specific pin_memory warning when no CUDA is available
            if not torch.cuda.is_available():
                warnings.filterwarnings(
                    "ignore",
                    message=".*pin_memory.*no accelerator.*",
                    category=UserWarning,
                    module="torch.utils.data.dataloader",
                )

            # Initialize model with appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embedding_model = SentenceTransformer(model_name, device=device)
            self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(
                f"Embedding model initialized: {model_name} (dim={self.vector_size}, device={device})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
            self.vector_size = None

    def _init_qdrant_client(self):
        """Khởi tạo kết nối Qdrant"""
        try:
            self.qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
            )
            logger.info(f"Qdrant client initialized: {settings.QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            self.qdrant_client = None

    async def create_collection(self, collection_name: str) -> bool:
        """Tạo collection trong Qdrant"""
        if not self.qdrant_client or not self.embedding_model or not self.vector_size:
            logger.error(
                "Qdrant client, embedding model, or vector size not initialized"
            )
            return False

        try:
            # Kiểm tra xem collection đã tồn tại chưa
            collections = self.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]

            if collection_name in existing_names:
                # Xóa collection cũ nếu đã tồn tại
                self.qdrant_client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")

            # Tạo collection mới
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=self.vector_size, distance=qdrant_models.Distance.COSINE
                ),
            )

            logger.info(f"Created collection: {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False

    async def process_textbook(
        self,
        book_id: str,
        text_content: str,
        lesson_id: str,
        book_title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Xử lý sách giáo khoa đơn giản - chỉ lưu text content"""

        if (
            not self.qdrant_client
            or not self.embedding_model
            or self.vector_size is None
        ):
            return {
                "success": False,
                "error": "Qdrant client or embedding model not initialized",
            }

        try:
            # Tạo collection cho sách
            collection_name = f"textbook_{book_id}"
            collection_created = await self.create_collection(collection_name)

            if not collection_created:
                return {"success": False, "error": "Failed to create collection"}

            # Tạo chunks từ text content
            text_chunks = self._create_text_chunks_from_text(
                text_content,
                settings.MAX_CHUNK_SIZE,
                settings.CHUNK_OVERLAP,
            )

            # Chuẩn bị dữ liệu
            points = []
            import uuid

            # Tạo embeddings cho từng chunk
            for i, chunk_text in enumerate(text_chunks):
                chunk_id = str(uuid.uuid4())
                chunk_vector = self.embedding_model.encode(chunk_text).tolist()

                points.append(
                    qdrant_models.PointStruct(
                        id=chunk_id,
                        vector=chunk_vector,
                        payload={
                            "book_id": book_id,
                            "lesson_id": lesson_id,
                            "chunk_index": i,
                            "type": "lesson_content",
                            "text": chunk_text,
                        },
                    )
                )

            total_chunks = len(text_chunks)

            # Lưu vào Qdrant theo batch
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.qdrant_client.upsert(collection_name=collection_name, points=batch)
                logger.info(
                    f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} to Qdrant"
                )

            # Lưu metadata đơn giản
            zero_vector = [0.0] * self.vector_size

            metadata_point = qdrant_models.PointStruct(
                id=str(uuid.uuid4()),
                vector=zero_vector,
                payload={
                    "book_id": book_id,
                    "lesson_id": lesson_id,
                    "type": "metadata",
                    "total_chunks": total_chunks,
                    "processed_at": datetime.datetime.now().isoformat(),
                    "model": settings.EMBEDDING_MODEL,
                    "chunk_size": settings.MAX_CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP,
                    "book_title": book_title or "Unknown",
                },
            )

            self.qdrant_client.upsert(
                collection_name=collection_name, points=[metadata_point]
            )

            return {
                "success": True,
                "book_id": book_id,
                "lesson_id": lesson_id,
                "collection_name": collection_name,
                "total_chunks": total_chunks,
                "vector_dimension": self.vector_size,
            }

        except Exception as e:
            logger.error(f"Error processing textbook: {e}")
            return {"success": False, "error": str(e)}



    def _create_text_chunks_from_text(
        self, text: str, max_size: int, overlap: int
    ) -> List[str]:
        """Tạo chunks từ text content"""
        if not text or not text.strip():
            return []

        chunks = []
        text = text.strip()

        # Nếu text ngắn hơn max_size, trả về luôn
        if len(text) <= max_size:
            return [text]

        # Chia text thành các chunks với overlap
        start = 0
        while start < len(text):
            end = start + max_size

            # Nếu không phải chunk cuối, tìm vị trí ngắt tự nhiên
            if end < len(text):
                # Tìm dấu câu hoặc khoảng trắng gần nhất
                for i in range(end, start + max_size // 2, -1):
                    if text[i] in ".!?\n ":
                        end = i + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Di chuyển start với overlap
            start = max(start + max_size - overlap, end)

            # Tránh vòng lặp vô hạn
            if start >= len(text):
                break

        return chunks

    async def search_textbook(
        self, book_id: str, query: str, limit: int = 5
    ) -> Dict[str, Any]:
        """Tìm kiếm trong sách giáo khoa bằng vector similarity"""

        if not self.qdrant_client or not self.embedding_model:
            return {
                "success": False,
                "error": "Qdrant client or embedding model not initialized",
            }

        try:
            collection_name = f"textbook_{book_id}"

            # Kiểm tra xem collection có tồn tại không
            collections = self.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]

            if collection_name not in existing_names:
                return {
                    "success": False,
                    "error": f"Collection not found for book_id {book_id}. Please create embeddings first.",
                }

            # Tạo embedding cho query
            query_vector = self.embedding_model.encode(query).tolist()

            # Tìm kiếm trong Qdrant
            search_result = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                score_threshold=0.3,  # Giảm threshold để có nhiều kết quả hơn
            )

            # Format kết quả
            results = []
            for scored_point in search_result:
                # Sửa lỗi: Kiểm tra payload trước khi truy cập
                payload = scored_point.payload or {}

                # Bỏ qua metadata point bằng cách check payload
                if payload.get("type") == "metadata":
                    continue

                results.append(
                    {
                        "text": payload.get("text", ""),
                        "score": scored_point.score,
                        "chapter_title": payload.get("chapter_title", ""),
                        "lesson_title": payload.get("lesson_title", ""),
                        "lesson_id": payload.get("lesson_id", ""),
                        "chapter_id": payload.get("chapter_id", ""),
                        "type": payload.get("type", ""),
                    }
                )

            return {
                "success": True,
                "book_id": book_id,
                "query": query,
                "results": results,
            }

        except Exception as e:
            logger.error(f"Error searching textbook: {e}")
            return {"success": False, "error": str(e)}





    async def delete_textbook_by_id(self, book_id: str) -> Dict[str, Any]:
        """
        Xóa textbook bằng book_id (xóa collection trong Qdrant)
        
        Args:
            book_id: ID của textbook cần xóa
            
        Returns:
            Dict chứa kết quả xóa
        """
        try:
            if not self.qdrant_client:
                return {
                    "success": False,
                    "error": "Qdrant client not initialized"
                }

            collection_name = f"textbook_{book_id}"
            
            # Kiểm tra collection có tồn tại không
            collections = self.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]
            
            if collection_name not in existing_names:
                return {
                    "success": False,
                    "error": f"Textbook with ID '{book_id}' not found",
                    "book_id": book_id,
                    "collection_name": collection_name
                }
            
            # Lấy thông tin về collection trước khi xóa
            collection_info = self.qdrant_client.get_collection(collection_name)
            vector_count = getattr(collection_info, 'vectors_count', 0)
            
            # Xóa collection
            self.qdrant_client.delete_collection(collection_name)
            logger.info(f"Deleted textbook collection: {collection_name}")
            
            return {
                "success": True,
                "message": f"Textbook '{book_id}' deleted successfully",
                "book_id": book_id,
                "collection_name": collection_name,
                "deleted_vectors": vector_count
            }
            
        except Exception as e:
            logger.error(f"Error deleting textbook {book_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "book_id": book_id
            }

    async def delete_textbook_by_lesson_id(self, lesson_id: str) -> Dict[str, Any]:
        """
        Xóa textbook bằng lesson_id (tìm collection chứa lesson_id rồi xóa)
        
        Args:
            lesson_id: ID của lesson để tìm textbook cần xóa
            
        Returns:
            Dict chứa kết quả xóa
        """
        try:
            if not self.qdrant_client:
                return {
                    "success": False,
                    "error": "Qdrant client not initialized"
                }

            # Tìm collection chứa lesson_id
            collections = self.qdrant_client.get_collections().collections
            found_collection = None
            found_book_id = None

            for collection in collections:
                if collection.name.startswith("textbook_"):
                    try:
                        # Tìm kiếm lesson_id trong collection này
                        search_result = self.qdrant_client.scroll(
                            collection_name=collection.name,
                            scroll_filter=qdrant_models.Filter(
                                must=[
                                    qdrant_models.FieldCondition(
                                        key="lesson_id",
                                        match=qdrant_models.MatchValue(value=lesson_id),
                                    )
                                ]
                            ),
                            limit=1,
                            with_payload=True,
                        )

                        if search_result[0]:  # Tìm thấy lesson
                            found_collection = collection.name
                            found_book_id = collection.name.replace("textbook_", "")
                            break

                    except Exception as e:
                        logger.warning(f"Error searching in collection {collection.name}: {e}")
                        continue

            if not found_collection:
                return {
                    "success": False,
                    "error": f"No textbook found containing lesson_id: {lesson_id}",
                    "lesson_id": lesson_id
                }

            # Lấy thông tin về collection trước khi xóa
            collection_info = self.qdrant_client.get_collection(found_collection)
            vector_count = getattr(collection_info, 'vectors_count', 0)
            
            # Xóa collection
            self.qdrant_client.delete_collection(found_collection)
            logger.info(f"Deleted textbook collection: {found_collection} (found by lesson_id: {lesson_id})")
            
            return {
                "success": True,
                "message": f"Textbook containing lesson '{lesson_id}' deleted successfully",
                "lesson_id": lesson_id,
                "book_id": found_book_id,
                "collection_name": found_collection,
                "deleted_vectors": vector_count
            }
            
        except Exception as e:
            logger.error(f"Error deleting textbook by lesson_id {lesson_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "lesson_id": lesson_id
            }


# Singleton instance
qdrant_service = QdrantService()
