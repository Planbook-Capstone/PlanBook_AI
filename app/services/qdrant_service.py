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
            model_name = settings.EMBEDDING_MODEL
            self.embedding_model = SentenceTransformer(model_name)
            self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model initialized: {model_name} (dim={self.vector_size})")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None
            self.vector_size = None
    
    def _init_qdrant_client(self):
        """Khởi tạo kết nối Qdrant"""
        try:
            self.qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None
            )
            logger.info(f"Qdrant client initialized: {settings.QDRANT_URL}")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            self.qdrant_client = None
    
    async def create_collection(self, collection_name: str) -> bool:
        """Tạo collection trong Qdrant"""
        if not self.qdrant_client or not self.embedding_model or not self.vector_size:
            logger.error("Qdrant client, embedding model, or vector size not initialized")
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
                    size=self.vector_size,
                    distance=qdrant_models.Distance.COSINE
                )
            )
            
            logger.info(f"Created collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            return False
    
    async def process_textbook(
        self, 
        book_id: str,
        book_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Xử lý sách giáo khoa và lưu embeddings vào Qdrant"""
        
        if not self.qdrant_client or not self.embedding_model or self.vector_size is None:
            return {
                "success": False,
                "error": "Qdrant client or embedding model not initialized"
            }
        
        try:
            # Tạo collection cho sách
            collection_name = f"textbook_{book_id}"
            collection_created = await self.create_collection(collection_name)
            
            if not collection_created:
                return {
                    "success": False,
                    "error": "Failed to create collection"
                }
            
            # Chuẩn bị dữ liệu
            points = []
            total_chunks = 0
            
            # Duyệt qua cấu trúc sách
            for chapter in book_structure.get("chapters", []):
                chapter_id = chapter.get("chapter_id")
                chapter_title = chapter.get("chapter_title")
                
                for lesson in chapter.get("lessons", []):
                    lesson_id = lesson.get("lesson_id")
                    lesson_title = lesson.get("lesson_title")
                    
                    # Tạo chunk cho tiêu đề bài học
                    title_text = f"{chapter_title} - {lesson_title}"
                    title_id = f"title_{book_id}_{chapter_id}_{lesson_id}"
                    title_vector = self.embedding_model.encode(title_text).tolist()
                    
                    points.append(
                        qdrant_models.PointStruct(
                            id=title_id,
                            vector=title_vector,
                            payload={
                                "book_id": book_id,
                                "chapter_id": chapter_id,
                                "lesson_id": lesson_id,
                                "type": "title",
                                "text": title_text,
                                "chapter_title": chapter_title,
                                "lesson_title": lesson_title
                            }
                        )
                    )
                    total_chunks += 1
                    
                    # Xử lý nội dung bài học
                    content = lesson.get("content", [])
                    text_chunks = self._create_text_chunks(
                        content, 
                        settings.MAX_CHUNK_SIZE, 
                        settings.CHUNK_OVERLAP
                    )
                    
                    # Tạo embeddings cho từng chunk
                    for i, chunk_text in enumerate(text_chunks):
                        chunk_id = f"content_{book_id}_{chapter_id}_{lesson_id}_{i}"
                        chunk_vector = self.embedding_model.encode(chunk_text).tolist()
                        
                        points.append(
                            qdrant_models.PointStruct(
                                id=chunk_id,
                                vector=chunk_vector,
                                payload={
                                    "book_id": book_id,
                                    "chapter_id": chapter_id,
                                    "lesson_id": lesson_id,
                                    "chunk_index": i,
                                    "type": "content",
                                    "text": chunk_text,
                                    "chapter_title": chapter_title,
                                    "lesson_title": lesson_title
                                }
                            )
                        )
                        total_chunks += 1
            
            # Lưu vào Qdrant theo batch
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} to Qdrant")
            
            # Lưu metadata về quá trình xử lý
            # Sửa lỗi: vector=[0.0] * self.vector_size
            zero_vector = [0.0] * self.vector_size
            metadata_point = qdrant_models.PointStruct(
                id="metadata",
                vector=zero_vector,  # Vector rỗng cho metadata
                payload={
                    "book_id": book_id,
                    "total_chunks": total_chunks,
                    "processed_at": datetime.datetime.now().isoformat(),
                    "model": settings.EMBEDDING_MODEL,
                    "chunk_size": settings.MAX_CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP
                }
            )
            
            self.qdrant_client.upsert(
                collection_name=collection_name,
                points=[metadata_point]
            )
            
            return {
                "success": True,
                "book_id": book_id,
                "collection_name": collection_name,
                "total_chunks": total_chunks,
                "vector_dimension": self.vector_size
            }
            
        except Exception as e:
            logger.error(f"Error processing textbook: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _create_text_chunks(self, content: List[Dict[str, Any]], max_size: int, overlap: int) -> List[str]:
        """Tạo chunks từ nội dung bài học"""
        chunks = []
        current_chunk = ""
        
        for item in content:
            if item.get("type") == "text":
                text = item.get("text", "")
                
                # Nếu text ngắn, thêm vào chunk hiện tại
                if len(current_chunk) + len(text) <= max_size:
                    current_chunk += " " + text
                else:
                    # Lưu chunk hiện tại
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Bắt đầu chunk mới
                    current_chunk = text
            
            elif item.get("type") == "image" and item.get("description"):
                # Thêm mô tả hình ảnh
                desc = f"[Hình ảnh: {item.get('description')}]"
                
                if len(current_chunk) + len(desc) <= max_size:
                    current_chunk += " " + desc
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = desc
        
        # Thêm chunk cuối cùng
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def search_textbook(
        self,
        book_id: str,
        query: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """Tìm kiếm trong sách giáo khoa bằng vector similarity"""
        
        if not self.qdrant_client or not self.embedding_model:
            return {
                "success": False,
                "error": "Qdrant client or embedding model not initialized"
            }
        
        try:
            collection_name = f"textbook_{book_id}"
            
            # Kiểm tra xem collection có tồn tại không
            collections = self.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]
            
            if collection_name not in existing_names:
                return {
                    "success": False,
                    "error": f"Collection not found for book_id {book_id}. Please create embeddings first."
                }
            
            # Tạo embedding cho query
            query_vector = self.embedding_model.encode(query).tolist()
            
            # Tìm kiếm trong Qdrant
            search_result = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
                score_threshold=0.5  # Lọc kết quả có độ tương đồng > 0.5
            )
            
            # Format kết quả
            results = []
            for scored_point in search_result:
                # Bỏ qua metadata point
                if scored_point.id == "metadata":
                    continue
                
                # Sửa lỗi: Kiểm tra payload trước khi truy cập
                payload = scored_point.payload or {}
                    
                results.append({
                    "text": payload.get("text", ""),
                    "score": scored_point.score,
                    "chapter_title": payload.get("chapter_title", ""),
                    "lesson_title": payload.get("lesson_title", ""),
                    "lesson_id": payload.get("lesson_id", ""),
                    "chapter_id": payload.get("chapter_id", ""),
                    "type": payload.get("type", "")
                })
            
            return {
                "success": True,
                "book_id": book_id,
                "query": query,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error searching textbook: {e}")
            return {
                "success": False,
                "error": str(e)
            }

# Singleton instance
qdrant_service = QdrantService()
