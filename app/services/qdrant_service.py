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
        self, book_id: str, book_structure: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Xử lý sách giáo khoa và lưu embeddings vào Qdrant"""

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
            # Kiểm tra và xử lý book_structure
            if isinstance(book_structure, str):
                import json

                try:
                    book_structure = json.loads(book_structure)
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "error": f"Invalid JSON in book_structure: {e}",
                    }

            if not isinstance(book_structure, dict):
                return {
                    "success": False,
                    "error": f"book_structure must be dict, got {type(book_structure)}",
                }

            # Tạo collection cho sách
            collection_name = f"textbook_{book_id}"
            collection_created = await self.create_collection(collection_name)

            if not collection_created:
                return {"success": False, "error": "Failed to create collection"}

            # Chuẩn bị dữ liệu
            points = []
            total_chunks = 0

            # Duyệt qua cấu trúc sách
            for chapter in book_structure.get("chapters", []):
                chapter_id = chapter.get("chapter_id")
                chapter_title = chapter.get("chapter_title")
                chapter_start_page = chapter.get("start_page")
                chapter_end_page = chapter.get("end_page")

                for lesson in chapter.get("lessons", []):
                    lesson_id = lesson.get("lesson_id")
                    lesson_title = lesson.get("lesson_title")
                    lesson_start_page = lesson.get("start_page")
                    lesson_end_page = lesson.get("end_page")

                    # Lấy thông tin content đầy đủ
                    content = lesson.get("content", {})
                    lesson_images = (
                        content.get("images", []) if isinstance(content, dict) else []
                    )
                    lesson_pages = (
                        content.get("pages", []) if isinstance(content, dict) else []
                    )
                    lesson_total_pages = (
                        content.get("total_pages", 0)
                        if isinstance(content, dict)
                        else 0
                    )
                    lesson_has_images = (
                        content.get("has_images", False)
                        if isinstance(content, dict)
                        else False
                    )

                    # Tạo chunk cho tiêu đề bài học
                    title_text = f"{chapter_title} - {lesson_title}"
                    import uuid

                    title_id = str(uuid.uuid4())  # Sử dụng UUID thay vì string ID
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
                                "lesson_title": lesson_title,
                                "chapter_start_page": chapter_start_page,
                                "chapter_end_page": chapter_end_page,
                                "lesson_start_page": lesson_start_page,
                                "lesson_end_page": lesson_end_page,
                                "lesson_images": lesson_images,
                                "lesson_pages": lesson_pages,
                                "lesson_total_pages": lesson_total_pages,
                                "lesson_has_images": lesson_has_images,
                            },
                        )
                    )
                    total_chunks += 1

                    # Xử lý nội dung bài học
                    content = lesson.get("content", {})

                    # Xử lý content dựa trên cấu trúc thực tế
                    if isinstance(content, dict):
                        # Content là object với text, images, pages
                        lesson_text = content.get("text", "")
                        if lesson_text:
                            text_chunks = self._create_text_chunks_from_text(
                                lesson_text,
                                settings.MAX_CHUNK_SIZE,
                                settings.CHUNK_OVERLAP,
                            )
                        else:
                            text_chunks = []
                    elif isinstance(content, list):
                        # Content là array (legacy format)
                        text_chunks = self._create_text_chunks(
                            content, settings.MAX_CHUNK_SIZE, settings.CHUNK_OVERLAP
                        )
                    else:
                        # Content là string hoặc format khác
                        text_chunks = [str(content)] if content else []

                    # Tạo embeddings cho từng chunk
                    for i, chunk_text in enumerate(text_chunks):
                        chunk_id = str(uuid.uuid4())  # Sử dụng UUID thay vì string ID
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
                                    "lesson_title": lesson_title,
                                    "chapter_start_page": chapter_start_page,
                                    "chapter_end_page": chapter_end_page,
                                    "lesson_start_page": lesson_start_page,
                                    "lesson_end_page": lesson_end_page,
                                    "lesson_images": lesson_images,
                                    "lesson_pages": lesson_pages,
                                    "lesson_total_pages": lesson_total_pages,
                                    "lesson_has_images": lesson_has_images,
                                },
                            )
                        )
                        total_chunks += 1

            # Lưu vào Qdrant theo batch
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i : i + batch_size]
                self.qdrant_client.upsert(collection_name=collection_name, points=batch)
                logger.info(
                    f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} to Qdrant"
                )

            # Lưu metadata về quá trình xử lý VÀ toàn bộ book_structure gốc
            # Sửa lỗi: vector=[0.0] * self.vector_size
            zero_vector = [0.0] * self.vector_size

            # Lưu book_info từ book_structure
            book_info = book_structure.get("book_info", {})

            metadata_point = qdrant_models.PointStruct(
                id=str(uuid.uuid4()),  # Sử dụng UUID cho metadata
                vector=zero_vector,  # Vector rỗng cho metadata
                payload={
                    "book_id": book_id,
                    "type": "metadata",  # Thêm type để identify metadata point
                    "total_chunks": total_chunks,
                    "processed_at": datetime.datetime.now().isoformat(),
                    "model": settings.EMBEDDING_MODEL,
                    "chunk_size": settings.MAX_CHUNK_SIZE,
                    "chunk_overlap": settings.CHUNK_OVERLAP,
                    # Lưu toàn bộ book_structure gốc để có thể tái tạo format đầy đủ
                    "original_book_structure": book_structure,
                    "book_title": book_info.get("title", "Unknown"),
                    "book_subject": book_info.get("subject", "Unknown"),
                    "book_grade": book_info.get("grade", "Unknown"),
                    "book_total_pages": book_info.get("total_pages", 0),
                },
            )

            self.qdrant_client.upsert(
                collection_name=collection_name, points=[metadata_point]
            )

            return {
                "success": True,
                "book_id": book_id,
                "collection_name": collection_name,
                "total_chunks": total_chunks,
                "vector_dimension": self.vector_size,
            }

        except Exception as e:
            logger.error(f"Error processing textbook: {e}")
            return {"success": False, "error": str(e)}

    def _create_text_chunks(
        self, content: List[Dict[str, Any]], max_size: int, overlap: int
    ) -> List[str]:
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


# Singleton instance
qdrant_service = QdrantService()
