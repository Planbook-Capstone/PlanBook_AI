"""
Qdrant Service - Quản lý vector embeddings với Qdrant
"""

import logging
import threading
from typing import Dict, Any, List, Optional, Union, cast
import uuid
import datetime
import re

# Heavy imports sẽ được lazy load trong __init__ method

from app.core.config import settings
from app.services.semantic_analysis_service import get_semantic_analysis_service

logger = logging.getLogger(__name__)


class QdrantService:
    """
    Service quản lý vector embeddings với Qdrant
    """

    def __init__(self):
        """Initialize Qdrant service"""
        self.embedding_model = None
        self.qdrant_client = None
        self.vector_size: Optional[int] = None
        self._service_initialized = False
        self.semantic_analysis_service = get_semantic_analysis_service()

    def _ensure_service_initialized(self):
        """Ensure Qdrant service is initialized"""
        if not self._service_initialized:
            logger.info("🔄 QdrantService: First-time initialization triggered")
            self._init_embedding_model()
            self._init_qdrant_client()
            self._service_initialized = True
            logger.info("✅ QdrantService: Initialization completed")



    def _init_embedding_model(self):
        """Khởi tạo mô hình embedding"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            import warnings

            model_name = settings.EMBEDDING_MODEL
            logger.info(f"🔧 Initializing embedding model: {model_name}")

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
            logger.info(f"🔧 Using device: {device}")

            # Special handling for different models
            if "nvidia" in model_name.lower():
                logger.info(f"🔧 Loading nvidia model with trust_remote_code=True")
                self.embedding_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
            else:
                logger.info(f"🔧 Loading standard model")
                self.embedding_model = SentenceTransformer(model_name, device=device)

            self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(
                f"✅ Embedding model initialized successfully: {model_name} (dim={self.vector_size}, device={device})"
            )

            # Test encoding để đảm bảo model hoạt động
            test_text = "Test embedding"
            test_embedding = self.embedding_model.encode(test_text)
            logger.info(f"✅ Model test successful, embedding shape: {test_embedding.shape}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize embedding model: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            self.embedding_model = None
            self.vector_size = None

    def _init_qdrant_client(self):
        """Khởi tạo kết nối Qdrant"""
        try:
            from qdrant_client import QdrantClient
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
        self._ensure_service_initialized()
        if not self.qdrant_client or not self.embedding_model or not self.vector_size:
            logger.error(
                "Qdrant client, embedding model, or vector size not initialized"
            )
            return False

        try:
            from qdrant_client.http import models as qdrant_models
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
        text_content: Optional[str] = None,
        lesson_id: Optional[str] = None,
        book_title: Optional[str] = None,
        book_structure: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Xử lý sách giáo khoa và tạo embeddings

        Args:
            book_id: ID của sách
            text_content: Nội dung text đơn giản (cho simple processing)
            lesson_id: ID bài học (cho simple processing)
            book_title: Tiêu đề sách
            book_structure: Cấu trúc sách đầy đủ (cho full processing)
        """
        self._ensure_service_initialized()

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

            # Xử lý dữ liệu đầu vào
            if book_structure:
                # Xử lý book_structure đầy đủ
                return await self._process_book_structure(book_id, book_structure, collection_name)
            elif text_content:
                # Xử lý text_content đơn giản
                return await self._process_simple_text(book_id, text_content, lesson_id, book_title, collection_name)
            else:
                return {"success": False, "error": "Either book_structure or text_content must be provided"}



        except Exception as e:
            logger.error(f"Error processing textbook: {e}")
            return {"success": False, "error": str(e)}

    async def _process_simple_text(
        self, book_id: str, text_content: str, lesson_id: str, book_title: Optional[str], collection_name: str
    ) -> Dict[str, Any]:
        """Xử lý text content đơn giản"""
        from qdrant_client.http import models as qdrant_models

        # Tạo chunks từ text content
        text_chunks = self._create_text_chunks_from_text(
            text_content,
            settings.MAX_CHUNK_SIZE,
            settings.CHUNK_OVERLAP,
        )

        # Chuẩn bị dữ liệu
        points = []
        import uuid

        # Tạo embeddings cho từng chunk với semantic metadata
        for i, chunk_text in enumerate(text_chunks):
            chunk_id = str(uuid.uuid4())
            chunk_vector = self.embedding_model.encode(chunk_text).tolist()

            # Phân loại semantic cho chunk sử dụng LLM
            semantic_info = await self.semantic_analysis_service.analyze_content_semantic(chunk_text)

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
                        # Semantic metadata - Multi-label với confidence
                        "semantic_tags": semantic_info["semantic_tags"],
                        "key_concepts": semantic_info["key_concepts"],
                        "contains_examples": semantic_info["contains_examples"],
                        "contains_definitions": semantic_info["contains_definitions"],
                        "contains_formulas": semantic_info["contains_formulas"],
                        "estimated_difficulty": semantic_info["difficulty"],
                        "analysis_method": semantic_info["analysis_method"],
                        "word_count": len(chunk_text.split()),
                        "char_count": len(chunk_text),
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

        # Lưu metadata
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

    async def _process_book_structure(
        self, book_id: str, book_structure: Dict[str, Any], collection_name: str
    ) -> Dict[str, Any]:
        """Xử lý book structure đầy đủ"""
        from qdrant_client.http import models as qdrant_models

        points = []
        total_chunks = 0
        import uuid

        # Xử lý từng chapter và lesson
        for chapter in book_structure.get("chapters", []):
            chapter_id = chapter.get("id", "unknown")

            for lesson in chapter.get("lessons", []):
                lesson_id = lesson.get("id", "unknown")
                lesson_content = lesson.get("content", "")

                if not lesson_content.strip():
                    continue

                # Tạo chunks từ lesson content
                text_chunks = self._create_text_chunks_from_text(
                    lesson_content,
                    settings.MAX_CHUNK_SIZE,
                    settings.CHUNK_OVERLAP,
                )

                # Tạo embeddings cho từng chunk
                for i, chunk_text in enumerate(text_chunks):
                    chunk_id = str(uuid.uuid4())
                    chunk_vector = self.embedding_model.encode(chunk_text).tolist()

                    # Phân loại semantic cho chunk sử dụng LLM
                    semantic_info = await self.semantic_analysis_service.analyze_content_semantic(chunk_text)

                    points.append(
                        qdrant_models.PointStruct(
                            id=chunk_id,
                            vector=chunk_vector,
                            payload={
                                "book_id": book_id,
                                "chapter_id": chapter_id,
                                "lesson_id": lesson_id,
                                "chunk_index": i,
                                "type": "lesson_content",
                                "text": chunk_text,
                                # Semantic metadata - Multi-label với confidence
                                "semantic_tags": semantic_info["semantic_tags"],
                                "key_concepts": semantic_info["key_concepts"],
                                "contains_examples": semantic_info["contains_examples"],
                                "contains_definitions": semantic_info["contains_definitions"],
                                "contains_formulas": semantic_info["contains_formulas"],
                                "estimated_difficulty": semantic_info["difficulty"],
                                "analysis_method": semantic_info["analysis_method"],
                                "word_count": len(chunk_text.split()),
                                "char_count": len(chunk_text),
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

        # Lưu metadata
        zero_vector = [0.0] * self.vector_size
        metadata_point = qdrant_models.PointStruct(
            id=str(uuid.uuid4()),
            vector=zero_vector,
            payload={
                "book_id": book_id,
                "type": "metadata",
                "total_chunks": total_chunks,
                "processed_at": datetime.datetime.now().isoformat(),
                "model": settings.EMBEDDING_MODEL,
                "chunk_size": settings.MAX_CHUNK_SIZE,
                "chunk_overlap": settings.CHUNK_OVERLAP,
                "book_title": book_structure.get("title", "Unknown"),
                "total_chapters": len(book_structure.get("chapters", [])),
                "total_lessons": sum(len(ch.get("lessons", [])) for ch in book_structure.get("chapters", [])),
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



    def _create_text_chunks_from_text(
        self, text: str, max_size: int, overlap: int
    ) -> List[str]:
        """Tạo chunks từ text content với semantic awareness"""
        if not text or not text.strip():
            return []

        # Sử dụng semantic chunking nếu có thể
        try:
            return self._semantic_chunking(text, max_size, overlap)
        except Exception as e:
            logger.warning(f"Semantic chunking failed, fallback to simple chunking: {e}")
            return self._simple_chunking(text, max_size, overlap)

    def _semantic_chunking(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Chia chunks dựa trên ngữ nghĩa"""
        
        # 1. Phân tách theo cấu trúc văn bản
        semantic_chunks = []
        
        # Chia theo tiêu đề/đề mục (H1, H2, ##, etc.)
        header_pattern = r'(^#{1,6}\s+.+$|^[A-Z][^.!?]*:$|^\d+\.\s+[A-Z][^.!?]*$)'
        sections = re.split(header_pattern, text, flags=re.MULTILINE)
        
        current_section = ""
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
                
            # Kiểm tra nếu là header
            if re.match(header_pattern, section.strip(), re.MULTILINE):
                # Lưu section trước đó nếu có
                if current_section.strip():
                    semantic_chunks.extend(self._split_by_paragraphs(current_section, max_size))
                current_section = section + "\n"
            else:
                current_section += section
        
        # Xử lý section cuối
        if current_section.strip():
            semantic_chunks.extend(self._split_by_paragraphs(current_section, max_size))
        
        # Nếu không có structure rõ ràng, fallback về paragraph-based
        if not semantic_chunks:
            semantic_chunks = self._split_by_paragraphs(text, max_size)
        
        # Thêm overlap thông minh
        return self._add_semantic_overlap(semantic_chunks, overlap)

    def _split_by_paragraphs(self, text: str, max_size: int) -> List[str]:
        """Chia theo đoạn văn"""
        # Chia theo paragraph (2+ newlines)
        paragraphs = re.split(r'\n\s*\n', text.strip())
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # Nếu thêm paragraph này vào chunk hiện tại mà không vượt quá max_size
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            if len(test_chunk) <= max_size:
                current_chunk = test_chunk
            else:
                # Lưu chunk hiện tại
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Nếu paragraph quá dài, chia nhỏ hơn
                if len(para) > max_size:
                    chunks.extend(self._split_by_sentences(para, max_size))
                    current_chunk = ""
                else:
                    current_chunk = para
        
        # Thêm chunk cuối
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def _split_by_sentences(self, text: str, max_size: int) -> List[str]:
        """Chia theo câu khi paragraph quá dài"""
        
        # Chia theo câu (dấu .!?)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            if len(test_chunk) <= max_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Nếu câu quá dài, chia theo từ
                if len(sentence) > max_size:
                    chunks.extend(self._simple_chunking(sentence, max_size, 0))
                    current_chunk = ""
                else:
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks

    def _add_semantic_overlap(self, chunks: List[str], overlap: int) -> List[str]:
        """Thêm overlap thông minh dựa trên sentences"""
        if len(chunks) <= 1 or overlap <= 0:
            return chunks
        
        import re
        overlapped_chunks = []
        
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Lấy câu cuối của chunk trước
                prev_chunk = chunks[i-1]
                prev_sentences = re.split(r'(?<=[.!?])\s+', prev_chunk)
                
                # Lấy overlap_text từ chunk trước
                overlap_text = ""
                for j in range(len(prev_sentences)-1, -1, -1):
                    test_overlap = prev_sentences[j] + " " + overlap_text if overlap_text else prev_sentences[j]
                    if len(test_overlap) <= overlap:
                        overlap_text = test_overlap
                    else:
                        break
                
                # Kết hợp với chunk hiện tại
                if overlap_text:
                    overlapped_chunk = overlap_text + "\n\n" + chunk
                else:
                    overlapped_chunk = chunk
                
                overlapped_chunks.append(overlapped_chunk)
        
        return overlapped_chunks

    def _simple_chunking(self, text: str, max_size: int, overlap: int) -> List[str]:
        """Fallback chunking method (original logic)"""
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
        self, book_id: str, query: str, limit: int = 5, semantic_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Tìm kiếm trong sách giáo khoa bằng vector similarity"""
        from qdrant_client.http import models as qdrant_models
        self._ensure_service_initialized()

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

            # Chuẩn bị filters cho Qdrant
            qdrant_filter = None
            if semantic_filters:
                filter_conditions = []

                # Filter by semantic tags
                if "semantic_tags" in semantic_filters:
                    tag_types = semantic_filters["semantic_tags"]
                    if isinstance(tag_types, str):
                        tag_types = [tag_types]

                    # Tạo filter cho semantic_tags array
                    for tag_type in tag_types:
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key="semantic_tags",
                                match=qdrant_models.MatchAny(any=[tag_type])
                            )
                        )

                # Filter by difficulty
                if "difficulty" in semantic_filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="estimated_difficulty",
                            match=qdrant_models.MatchValue(value=semantic_filters["difficulty"])
                        )
                    )

                # Filter by content features
                if "has_examples" in semantic_filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="contains_examples",
                            match=qdrant_models.MatchValue(value=semantic_filters["has_examples"])
                        )
                    )

                if "has_formulas" in semantic_filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="contains_formulas",
                            match=qdrant_models.MatchValue(value=semantic_filters["has_formulas"])
                        )
                    )

                # Filter by lesson_id
                if "lesson_id" in semantic_filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="lesson_id",
                            match=qdrant_models.MatchValue(value=semantic_filters["lesson_id"])
                        )
                    )

                if filter_conditions:
                    qdrant_filter = qdrant_models.Filter(
                        must=filter_conditions
                    )

            # Tìm kiếm trong Qdrant
            search_result = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=qdrant_filter,
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

                # Chuẩn bị semantic metadata
                semantic_tags = payload.get("semantic_tags", [])

                # Backward compatibility: convert old semantic_type to new format
                if not semantic_tags and "semantic_type" in payload:
                    semantic_tags = [{"type": payload["semantic_type"], "confidence": 0.8}]

                results.append(
                    {
                        "text": payload.get("text", ""),
                        "score": scored_point.score,
                        "chapter_title": payload.get("chapter_title", ""),
                        "lesson_title": payload.get("lesson_title", ""),
                        "lesson_id": payload.get("lesson_id", ""),
                        "chapter_id": payload.get("chapter_id", ""),
                        "type": payload.get("type", ""),
                        "semantic_tags": semantic_tags,
                        "key_concepts": payload.get("key_concepts", []),
                        "estimated_difficulty": payload.get("estimated_difficulty", "basic"),
                        "contains_examples": payload.get("contains_examples", False),
                        "contains_definitions": payload.get("contains_definitions", False),
                        "contains_formulas": payload.get("contains_formulas", False),
                        "analysis_method": payload.get("analysis_method", "unknown")
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

    async def global_search(
        self, query: str, limit: int = 5, semantic_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Tìm kiếm toàn bộ trong tất cả collections"""
        from qdrant_client.http import models as qdrant_models

        if not self.qdrant_client or not self.embedding_model:
            return {
                "success": False,
                "error": "Qdrant client or embedding model not initialized",
            }

        try:
            # Lấy tất cả collections
            collections = self.qdrant_client.get_collections().collections
            textbook_collections = [c.name for c in collections if c.name.startswith("textbook_")]

            if not textbook_collections:
                return {
                    "success": True,
                    "query": query,
                    "results": [],
                    "message": "No textbook collections found"
                }

            # Tạo embedding cho query
            query_vector = self.embedding_model.encode(query).tolist()

            # Chuẩn bị filters cho Qdrant (tái sử dụng logic từ search_textbook)
            qdrant_filter = None
            if semantic_filters:
                filter_conditions = []

                # Filter by semantic tags
                if "semantic_tags" in semantic_filters:
                    tag_types = semantic_filters["semantic_tags"]
                    if isinstance(tag_types, str):
                        tag_types = [tag_types]

                    for tag_type in tag_types:
                        filter_conditions.append(
                            qdrant_models.FieldCondition(
                                key="semantic_tags",
                                match=qdrant_models.MatchAny(any=[tag_type])
                            )
                        )

                # Filter by difficulty
                if "difficulty" in semantic_filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="estimated_difficulty",
                            match=qdrant_models.MatchValue(value=semantic_filters["difficulty"])
                        )
                    )

                # Filter by content features
                if "has_examples" in semantic_filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="contains_examples",
                            match=qdrant_models.MatchValue(value=semantic_filters["has_examples"])
                        )
                    )

                if "has_formulas" in semantic_filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="contains_formulas",
                            match=qdrant_models.MatchValue(value=semantic_filters["has_formulas"])
                        )
                    )

                # Filter by lesson_id
                if "lesson_id" in semantic_filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="lesson_id",
                            match=qdrant_models.MatchValue(value=semantic_filters["lesson_id"])
                        )
                    )

                if filter_conditions:
                    qdrant_filter = qdrant_models.Filter(
                        must=filter_conditions
                    )

            # Tìm kiếm trong tất cả collections và gộp kết quả
            all_results = []

            for collection_name in textbook_collections:
                try:
                    search_result = self.qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        query_filter=qdrant_filter,
                        limit=limit,
                        with_payload=True,
                        score_threshold=0.3,
                    )

                    # Extract book_id from collection name
                    book_id = collection_name.replace("textbook_", "")

                    # Format kết quả
                    for scored_point in search_result:
                        payload = scored_point.payload or {}

                        # Bỏ qua metadata point
                        if payload.get("type") == "metadata":
                            continue

                        # Chuẩn bị semantic metadata
                        semantic_tags = payload.get("semantic_tags", [])

                        # Backward compatibility
                        if not semantic_tags and "semantic_type" in payload:
                            semantic_tags = [{"type": payload["semantic_type"], "confidence": 0.8}]

                        all_results.append({
                            "text": payload.get("text", ""),
                            "score": scored_point.score,
                            "book_id": book_id,
                            "chapter_title": payload.get("chapter_title", ""),
                            "lesson_title": payload.get("lesson_title", ""),
                            "lesson_id": payload.get("lesson_id", ""),
                            "chapter_id": payload.get("chapter_id", ""),
                            "type": payload.get("type", ""),
                            "semantic_tags": semantic_tags,
                            "key_concepts": payload.get("key_concepts", []),
                            "estimated_difficulty": payload.get("estimated_difficulty", "basic"),
                            "contains_examples": payload.get("contains_examples", False),
                            "contains_definitions": payload.get("contains_definitions", False),
                            "contains_formulas": payload.get("contains_formulas", False),
                            "analysis_method": payload.get("analysis_method", "unknown")
                        })

                except Exception as e:
                    logger.warning(f"Error searching collection {collection_name}: {e}")
                    continue

            # Sắp xếp theo score và giới hạn kết quả
            all_results.sort(key=lambda x: x["score"], reverse=True)
            final_results = all_results[:limit]

            return {
                "success": True,
                "query": query,
                "results": final_results,
                "collections_searched": len(textbook_collections),
                "total_results_found": len(all_results)
            }

        except Exception as e:
            logger.error(f"Error in global search: {e}")
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
            from qdrant_client.http import models as qdrant_models
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



    def get_semantic_suggestions(self, book_id: str) -> Dict[str, Any]:
        """Lấy thống kê semantic để gợi ý filter options"""
        
        if not self.qdrant_client:
            return {"success": False, "error": "Qdrant client not initialized"}
        
        try:
            collection_name = f"textbook_{book_id}"
            
            # Get all points to analyze
            points = self.qdrant_client.scroll(
                collection_name=collection_name,
                limit=1000,  # Adjust based on your data size
                with_payload=True
            )[0]  # scroll returns (points, next_page_token)
            
            # Analyze semantic distribution
            semantic_stats = {
                "semantic_types": {},
                "difficulty_levels": {},
                "content_features": {
                    "has_examples": 0,
                    "has_definitions": 0,
                    "has_formulas": 0,
                },
                "concepts": {},
                "total_chunks": 0
            }
            
            for point in points:
                payload = point.payload or {}
                
                # Skip metadata points
                if payload.get("type") == "metadata":
                    continue
                    
                semantic_stats["total_chunks"] += 1
                
                # Count semantic tags (multi-label)
                semantic_tags = payload.get("semantic_tags", [])
                for tag_info in semantic_tags:
                    if isinstance(tag_info, dict) and "type" in tag_info:
                        tag_type = tag_info["type"]
                        semantic_stats["semantic_types"][tag_type] = (
                            semantic_stats["semantic_types"].get(tag_type, 0) + 1
                        )
                
                # Count difficulty levels
                difficulty = payload.get("estimated_difficulty", "basic")
                semantic_stats["difficulty_levels"][difficulty] = (
                    semantic_stats["difficulty_levels"].get(difficulty, 0) + 1
                )
                
                # Count content features
                if payload.get("contains_examples"):
                    semantic_stats["content_features"]["has_examples"] += 1
                if payload.get("contains_definitions"):
                    semantic_stats["content_features"]["has_definitions"] += 1
                if payload.get("contains_formulas"):
                    semantic_stats["content_features"]["has_formulas"] += 1
                
                # Count key concepts
                key_concepts = payload.get("key_concepts", [])
                for concept in key_concepts:
                    semantic_stats["concepts"][concept] = (
                        semantic_stats["concepts"].get(concept, 0) + 1
                    )
            
            # Sort and limit concepts
            sorted_concepts = sorted(
                semantic_stats["concepts"].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:20]
            
            semantic_stats["top_concepts"] = dict(sorted_concepts)
            del semantic_stats["concepts"]  # Remove full list
            
            return {
                "success": True,
                "book_id": book_id,
                "statistics": semantic_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting semantic suggestions: {e}")
            return {"success": False, "error": str(e)}


# Factory function để tạo QdrantService instance
def get_qdrant_service() -> QdrantService:
    """
    Tạo QdrantService instance mới

    Returns:
        QdrantService: Fresh instance
    """
    return QdrantService()
