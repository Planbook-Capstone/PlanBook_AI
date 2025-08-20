"""
Qdrant Service - Quản lý vector embeddings với Qdrant
"""

import logging
from typing import Dict, Any, Optional
import datetime

# Heavy imports sẽ được lazy load trong __init__ method

from app.core.config import settings
from app.services.semantic_analysis_service import get_semantic_analysis_service
from app.services.smart_chunking_service import get_smart_chunking_service

logger = logging.getLogger(__name__)


class QdrantService:
    """
    Service quản lý vector embeddings với Qdrant - Individual Collections per Book
    """

    def __init__(self):
        """Initialize Qdrant service"""
        self.embedding_model = None
        self.qdrant_client = None
        self.vector_size: Optional[int] = None
        self._service_initialized = False
        self.semantic_analysis_service = get_semantic_analysis_service()
        self.smart_chunking_service = get_smart_chunking_service()

    def is_available(self) -> bool:
        """Check if Qdrant service is available"""
        try:
            self._ensure_service_initialized()
            return (
                self._service_initialized and
                self.qdrant_client is not None and
                self.embedding_model is not None
            )
        except Exception as e:
            logger.error(f"❌ QdrantService availability check failed: {e}")
            return False

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

            logger.info(f"🔄 Initializing Qdrant client...")
            logger.info(f"   - URL: {settings.QDRANT_URL}")
            logger.info(f"   - API Key: {'***' if settings.QDRANT_API_KEY else 'None'}")

            self.qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
            )

            # Test connection
            collections = self.qdrant_client.get_collections()
            logger.info(f"✅ Qdrant client initialized successfully: {settings.QDRANT_URL}")
            logger.info(f"   - Found {len(collections.collections)} existing collections")

        except Exception as e:
            logger.error(f"❌ Failed to initialize Qdrant client: {e}")
            logger.error(f"   - URL: {settings.QDRANT_URL}")
            logger.error(f"   - Make sure Qdrant server is running")
            self.qdrant_client = None



    def _ensure_collection_exists(self, collection_name: str) -> bool:
        """
        Đảm bảo collection tồn tại - tự động tạo nếu chưa có

        Args:
            collection_name: Tên collection cần đảm bảo tồn tại

        Returns:
            bool: True nếu collection tồn tại hoặc được tạo thành công, False nếu có lỗi
        """
        if not self.qdrant_client or not self.vector_size:
            logger.error("Cannot create collection: Qdrant client or vector size not initialized")
            return False

        try:
            from qdrant_client.http import models as qdrant_models

            # Kiểm tra xem collection đã tồn tại chưa
            collections = self.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]

            if collection_name not in existing_names:
                logger.info(f"Collection '{collection_name}' not found. Creating new collection...")

                # Tạo collection mới
                self.qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=self.vector_size, distance=qdrant_models.Distance.COSINE
                    ),
                )
                logger.info(f"✅ Successfully created new collection: {collection_name}")

                # Tạo payload index cho các trường quan trọng
                logger.info(f"Creating indexes for collection: {collection_name}")
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="book_id",
                    field_schema="keyword",
                )
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="lesson_id",
                    field_schema="keyword",
                )
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="type",
                    field_schema="keyword",
                )
                self.qdrant_client.create_payload_index(
                    collection_name=collection_name,
                    field_name="content_type",
                    field_schema="keyword",
                )
                logger.info(f"✅ Successfully created indexes for collection: {collection_name}")
            else:
                logger.info(f"✅ Collection '{collection_name}' already exists - ready to add content")

            return True

        except Exception as e:
            logger.error(f"❌ Error ensuring collection '{collection_name}' exists: {e}")
            return False

    async def process_textbook(
        self,
        book_id: str,
        content: Optional[Any] = None,  # Có thể là str hoặc Dict
        lesson_id: Optional[str] = None,
        content_type: str = "textbook",  # "textbook" | "guide"
        file_url: Optional[str] = None,  # URL của file PDF trên Supabase
        uploaded_at: Optional[str] = None,  # Thời gian upload file
        # Backward compatibility parameters
        text_content: Optional[str] = None,
        book_content: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Xử lý sách giáo khoa/guide và tạo embeddings vào collection riêng

        Args:
            book_id: ID của sách hoặc guide
            content: Nội dung cần xử lý (str cho guide, Dict hoặc str cho textbook)
            lesson_id: ID bài học
            content_type: Loại nội dung ("textbook" hoặc "guide")
            text_content: [DEPRECATED] Sử dụng content thay thế
            book_content: [DEPRECATED] Sử dụng content thay thế
        """
        self._ensure_service_initialized()

        # Debug logging
        logger.info(f"🔍 process_textbook called with:")
        logger.info(f"   - book_id: {book_id}")
        logger.info(f"   - lesson_id: {lesson_id} (type: {type(lesson_id)})")
        logger.info(f"   - content_type: {content_type}")
        logger.info(f"   - has_content: {content is not None}")
        logger.info(f"   - content_type_obj: {type(content)}")
        logger.info(f"   - has_text_content (deprecated): {text_content is not None}")
        logger.info(f"   - has_book_content (deprecated): {book_content is not None}")

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
            # Xác định collection name dựa trên content_type và book_id
            if content_type == "guide":
                collection_name = f"guide_{book_id}"
            else:  # textbook
                collection_name = f"textbook_{book_id}"

            # Đảm bảo collection tồn tại - tự động tạo nếu chưa có
            if not self._ensure_collection_exists(collection_name):
                logger.error(f"❌ Failed to create or access collection '{collection_name}'. Check Qdrant connection and permissions.")
                return {"success": False, "error": f"Failed to create or access collection '{collection_name}'. Check Qdrant connection and logs for details."}

            # Xử lý dữ liệu đầu vào - thống nhất cho cả guide và textbook
            content_text = None

            # Ưu tiên sử dụng parameter content mới
            if content is not None:
                if isinstance(content, dict):
                    content_text = str(content)  # Convert dict to string
                else:
                    content_text = str(content)  # Convert to string
            # Backward compatibility với các parameter cũ
            elif content_type == "guide" and text_content:
                content_text = text_content
            elif content_type == "textbook" and book_content:
                if isinstance(book_content, dict):
                    content_text = str(book_content)  # Convert dict to string
                else:
                    content_text = str(book_content)  # Convert to string

            if not content_text:
                return {"success": False, "error": "Missing required content. Please provide 'content' parameter."}

            # Xử lý content thống nhất
            return await self._process_content_to_collection(
                book_id=book_id,
                content=content_text,
                content_type=content_type,
                lesson_id=lesson_id,
                collection_name=collection_name,
                file_url=file_url,
                uploaded_at=uploaded_at
            )
        except Exception as e:
            logger.error(f"Error processing textbook: {e}")
            return {"success": False, "error": str(e)}

    async def _process_content_to_collection(
        self, book_id: str, content: str, content_type: str, collection_name: str,
        lesson_id: Optional[str] = None, file_url: Optional[str] = None, uploaded_at: Optional[str] = None
    ) -> Dict[str, Any]:
        """Xử lý nội dung text vào collection cụ thể - hàm thống nhất cho cả textbook và guide"""
        from qdrant_client.http import models as qdrant_models

        # Debug logging
        logger.info(f"🔍 Processing content to collection: {collection_name}")
        logger.info(f"   - book_id: {book_id}")
        logger.info(f"   - lesson_id: {lesson_id} (type: {type(lesson_id)})")
        logger.info(f"   - content_type: {content_type}")

        # Tạo chunks từ content sử dụng smart chunking service
        chunk_infos = self.smart_chunking_service.chunk_textbook_content(
            content,
            max_tokens=settings.MAX_CHUNK_SIZE
        )

        # Chuẩn bị dữ liệu
        points = []
        import uuid

        # Tạo embeddings cho từng chunk với semantic metadata từ smart chunking
        for i, chunk_info in enumerate(chunk_infos):
            try:
                chunk_id = str(uuid.uuid4())
                chunk_text = getattr(chunk_info, 'text', '')
                if not chunk_text:
                    logger.warning(f"Empty chunk text at index {i}, skipping...")
                    continue

                chunk_vector = self.embedding_model.encode(chunk_text).tolist()

                # Sử dụng semantic info từ smart chunking service
                semantic_info = {
                    'chunk_type': chunk_info.chunk_type,
                    'semantic_tag': chunk_info.semantic_tag,
                    'concepts': chunk_info.concepts,
                    'token_count': chunk_info.token_count,
                    'is_semantic_complete': chunk_info.is_semantic_complete
                }

                # Debug logging để kiểm tra cấu trúc dữ liệu
                if i == 0:
                    logger.info(f"🔍 ChunkInfo attributes: {dir(chunk_info)}")
                    logger.info(f"🔍 semantic_info keys: {list(semantic_info.keys())}")
                    logger.info(f"🔍 semantic_tag value: {semantic_info.get('semantic_tag', 'NOT_FOUND')}")

                # Xác định content_type_detail dựa trên content_type
                if content_type == "guide":
                    content_type_detail = "guide_content"
                else:
                    content_type_detail = "lesson_content"

                # Đảm bảo lesson_id có giá trị hợp lệ
                safe_lesson_id = lesson_id if lesson_id is not None else ""

                # Debug logging cho chunk đầu tiên
                if i == 0:
                    logger.info(f"🔍 Creating point with lesson_id: {lesson_id} -> safe_lesson_id: {safe_lesson_id} (type: {type(safe_lesson_id)})")

                points.append(
                    qdrant_models.PointStruct(
                        id=chunk_id,
                        vector=chunk_vector,
                        payload={
                            "book_id": book_id,
                            "lesson_id": safe_lesson_id,
                            "chunk_index": i,
                            "type": "content",
                            "content_type": content_type_detail,
                            "text": chunk_text,
                            # Smart chunking metadata - defensive access
                            "chunk_type": semantic_info.get("chunk_type", "content"),
                            "semantic_tag": semantic_info.get("semantic_tag", "theory"),
                            "concepts": semantic_info.get("concepts", []),
                            "token_count": semantic_info.get("token_count", 0),
                            "is_semantic_complete": semantic_info.get("is_semantic_complete", False),
                            "parent_title": getattr(chunk_info, 'parent_title', ''),
                            "overlap_context": getattr(chunk_info, 'overlap_context', ''),
                            # Legacy compatibility - defensive access
                            "semantic_tags": [semantic_info.get("semantic_tag", "theory")],
                            "key_concepts": semantic_info.get("concepts", []),
                            "contains_examples": semantic_info.get("chunk_type", "content") == "example",
                            "contains_definitions": semantic_info.get("chunk_type", "content") == "definition",
                            "contains_formulas": "formula" in chunk_text.lower(),
                            "estimated_difficulty": "basic",
                            "analysis_method": "smart_chunking",
                            "word_count": len(chunk_text.split()),
                            "char_count": len(chunk_text),
                            "processed_at": datetime.datetime.now().isoformat(),
                        },
                    )
                )
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
                logger.error(f"ChunkInfo type: {type(chunk_info)}")
                logger.error(f"ChunkInfo attributes: {dir(chunk_info) if hasattr(chunk_info, '__dict__') else 'No attributes'}")
                continue

        total_chunks = len(chunk_infos)

        # Lưu vào Qdrant theo batch vào collection cụ thể
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.qdrant_client.upsert(collection_name=collection_name, points=batch)
            logger.info(
                f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} to collection {collection_name}"
            )

        # Lưu metadata vào collection
        zero_vector = [0.0] * self.vector_size
        safe_lesson_id = lesson_id if lesson_id is not None else ""
        metadata_payload = {
            "book_id": book_id,
            "lesson_id": safe_lesson_id,
            "type": "metadata",
            "content_type": content_type,
            "total_chunks": total_chunks,
            "processed_at": datetime.datetime.now().isoformat(),
            "model": settings.EMBEDDING_MODEL,
            "chunk_size": settings.MAX_CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
        }

        # Debug logging cho file metadata
        logger.info(f"🔍 Adding file metadata to Qdrant:")
        logger.info(f"   - file_url: {file_url} (type: {type(file_url)})")
        logger.info(f"   - uploaded_at: {uploaded_at} (type: {type(uploaded_at)})")

        # Thêm fileUrl và uploaded_at nếu có
        if file_url:
            metadata_payload["file_url"] = file_url
            logger.info(f"✅ Added file_url to metadata: {file_url}")
        else:
            logger.warning("⚠️  file_url is None or empty, not adding to metadata")

        if uploaded_at:
            metadata_payload["uploaded_at"] = uploaded_at
            logger.info(f"✅ Added uploaded_at to metadata: {uploaded_at}")
        else:
            logger.warning("⚠️  uploaded_at is None or empty, not adding to metadata")

        metadata_point = qdrant_models.PointStruct(
            id=str(uuid.uuid4()),
            vector=zero_vector,
            payload=metadata_payload,
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







    async def search_textbook(
        self, book_id: str, query: str, limit: int = 5, semantic_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Tìm kiếm trong sách giáo khoa bằng vector similarity trong collection riêng theo book_id"""
        from qdrant_client.http import models as qdrant_models
        self._ensure_service_initialized()

        if not self.qdrant_client or not self.embedding_model:
            return {
                "success": False,
                "error": "Qdrant client or embedding model not initialized",
            }

        try:
            # Sử dụng collection riêng cho textbook
            collection_name = f"textbook_{book_id}"

            # Kiểm tra xem collection có tồn tại không
            collections = self.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]

            if collection_name not in existing_names:
                return {
                    "success": False,
                    "error": f"Textbook collection {collection_name} not found. Please create embeddings first.",
                }

            # Tạo embedding cho query
            query_vector = self.embedding_model.encode(query).tolist()

            # Chuẩn bị filters cho Qdrant - bắt buộc filter theo book_id và type="content"
            filter_conditions = [
                qdrant_models.FieldCondition(
                    key="book_id",
                    match=qdrant_models.MatchValue(value=book_id)
                ),
                qdrant_models.FieldCondition(
                    key="type",
                    match=qdrant_models.MatchValue(value="content")
                )
            ]

            # Thêm semantic filters nếu có
            if semantic_filters:
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

                # Filter by lesson_id (nếu muốn tìm trong lesson cụ thể)
                if "lesson_id" in semantic_filters:
                    filter_conditions.append(
                        qdrant_models.FieldCondition(
                            key="lesson_id",
                            match=qdrant_models.MatchValue(value=semantic_filters["lesson_id"])
                        )
                    )

            # Tạo filter cuối cùng
            qdrant_filter = qdrant_models.Filter(must=filter_conditions)

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

    async def get_lessons_by_type(
        self,
        content_type: str = "textbook",
        book_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Lấy bài học từ Qdrant theo loại content và book_id - Optimized version
        Chỉ lấy những thông tin cần thiết để tăng tốc độ response

        Args:
            content_type: Loại content ("textbook" hoặc "guide")
            book_id: ID của sách (optional, để filter theo book cụ thể)

        Returns:
            Dict chứa danh sách bài học với thông tin cơ bản
        """
        from qdrant_client.http import models as qdrant_models

        self._ensure_service_initialized()

        if not self.qdrant_client:
            return {
                "success": False,
                "error": "Qdrant client not initialized"
            }

        try:
            # Lấy danh sách tất cả collections
            collections_response = self.qdrant_client.get_collections()
            all_collections = [col.name for col in collections_response.collections]

            # Lọc collections theo content_type
            if content_type == "textbook":
                target_collections = [col for col in all_collections if col.startswith("textbook_")]
            elif content_type == "guide":
                target_collections = [col for col in all_collections if col.startswith("guide_")]
            else:
                target_collections = [col for col in all_collections if col.startswith("textbook_") or col.startswith("guide_")]

            # Nếu có book_id, lọc thêm theo book_id
            if book_id:
                target_collections = [col for col in target_collections if col.endswith(f"_{book_id}")]

            logger.info(f"Found {len(target_collections)} {content_type} collections" + (f" for book_id={book_id}" if book_id else ""))

            lessons = []

            for collection_name in target_collections:
                try:
                    # Tối ưu: Lấy tất cả metadata points nhưng không lấy vectors để tăng tốc
                    search_result = self.qdrant_client.scroll(
                        collection_name=collection_name,
                        scroll_filter=qdrant_models.Filter(
                            must=[
                                qdrant_models.FieldCondition(
                                    key="type",
                                    match=qdrant_models.MatchValue(value="metadata")
                                )
                            ]
                        ),
                        limit=100,  # Tăng limit để lấy nhiều lessons trong 1 collection
                        with_payload=True,
                        with_vectors=False  # Không cần vectors để tăng tốc
                    )

                    # Xử lý tất cả metadata points trong collection
                    for point in search_result[0]:  # search_result[0] chứa danh sách points
                        payload = point.payload

                        # Chỉ lấy những field cần thiết
                        lesson_info = {
                            "book_id": payload.get("book_id", ""),
                            "lesson_id": payload.get("lesson_id", ""),
                            "file_url": payload.get("file_url", ""),
                            "uploaded_at": payload.get("uploaded_at", payload.get("processed_at", "")),
                            "processed_at": payload.get("processed_at", ""),
                            "content_type": payload.get("content_type", content_type),
                            "collection_name": collection_name,
                            "total_chunks": payload.get("total_chunks", 0)
                        }

                        # Chỉ thêm vào danh sách nếu có book_id và lesson_id
                        if lesson_info["book_id"] and lesson_info["lesson_id"]:
                            lessons.append(lesson_info)

                except Exception as e:
                    logger.warning(f"Error processing collection {collection_name}: {e}")
                    continue

            # Sắp xếp theo uploaded_at (mới nhất trước), fallback processed_at
            lessons.sort(key=lambda x: x.get("uploaded_at", x.get("processed_at", "")), reverse=True)

            logger.info(f"Retrieved {len(lessons)} {content_type} lessons from Qdrant" + (f" for book_id={book_id}" if book_id else ""))

            return {
                "success": True,
                "lessons": lessons,
                "total_lessons": len(lessons),
                "collections_processed": len(target_collections),
                "content_type": content_type,
                "book_id": book_id
            }

        except Exception as e:
            logger.error(f"Error getting {content_type} lessons: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def get_all_lessons(self) -> Dict[str, Any]:
        """
        Backward compatibility method - lấy tất cả textbook lessons
        """
        return await self.get_lessons_by_type(content_type="textbook")

    async def get_all_guides(self) -> Dict[str, Any]:
        """
        Lấy tất cả guide lessons
        """
        return await self.get_lessons_by_type(content_type="guide")

    async def get_file_urls_for_deletion(self, book_id: str, lesson_id: Optional[str] = None) -> list:
        """
        Lấy danh sách file URLs cần xóa từ Supabase trước khi xóa khỏi Qdrant

        Args:
            book_id: ID của book
            lesson_id: ID của lesson (optional)

        Returns:
            List các file URLs cần xóa
        """
        from qdrant_client.http import models as qdrant_models

        self._ensure_service_initialized()

        if not self.qdrant_client:
            return []

        file_urls = []

        try:
            # Lấy danh sách collections
            collections_response = self.qdrant_client.get_collections()
            all_collections = [col.name for col in collections_response.collections]

            # Tìm collections liên quan đến book_id
            target_collections = []
            for col in all_collections:
                if col.endswith(f"_{book_id}"):
                    target_collections.append(col)

            # Tạo filter
            filter_conditions = [
                qdrant_models.FieldCondition(
                    key="type",
                    match=qdrant_models.MatchValue(value="metadata")
                ),
                qdrant_models.FieldCondition(
                    key="book_id",
                    match=qdrant_models.MatchValue(value=book_id)
                )
            ]

            # Nếu có lesson_id, thêm filter
            if lesson_id:
                filter_conditions.append(
                    qdrant_models.FieldCondition(
                        key="lesson_id",
                        match=qdrant_models.MatchValue(value=lesson_id)
                    )
                )

            search_filter = qdrant_models.Filter(must=filter_conditions)

            # Tìm kiếm trong các collections
            for collection_name in target_collections:
                try:
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=collection_name,
                        scroll_filter=search_filter,
                        limit=1000,
                        with_payload=True,
                        with_vectors=False
                    )

                    points = scroll_result[0]
                    for point in points:
                        file_url = point.payload.get("file_url")
                        if file_url and file_url not in file_urls:
                            file_urls.append(file_url)

                except Exception as e:
                    logger.warning(f"Error searching collection {collection_name}: {e}")
                    continue

            logger.info(f"Found {len(file_urls)} file URLs for deletion: book_id={book_id}, lesson_id={lesson_id}")
            return file_urls

        except Exception as e:
            logger.error(f"Error getting file URLs for deletion: {e}")
            return []

    async def update_lesson_id_in_book(
        self, book_id: str, old_lesson_id: str, new_lesson_id: str
    ) -> Dict[str, Any]:
        """
        Update tất cả lessonID cũ thành lessonID mới trong một bookID

        Args:
            book_id: ID của book
            old_lesson_id: lessonID cũ cần thay đổi
            new_lesson_id: lessonID mới

        Returns:
            Dict chứa kết quả update
        """
        from qdrant_client.http import models as qdrant_models

        self._ensure_service_initialized()

        if not self.qdrant_client:
            return {
                "success": False,
                "error": "Qdrant client not initialized"
            }

        try:
            # Tìm collection cho book_id (textbook hoặc guide)
            collections = self.qdrant_client.get_collections()
            target_collection = None

            for collection in collections.collections:
                collection_name = collection.name
                if collection_name == f"textbook_{book_id}" or collection_name == f"guide_{book_id}":
                    target_collection = collection_name
                    break

            if not target_collection:
                return {
                    "success": False,
                    "error": f"No collection found for book_id '{book_id}'"
                }

            logger.info(f"Updating lesson_id from '{old_lesson_id}' to '{new_lesson_id}' in collection '{target_collection}'")

            # Tìm tất cả points có old_lesson_id
            filter_condition = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="book_id",
                        match=qdrant_models.MatchValue(value=book_id)
                    ),
                    qdrant_models.FieldCondition(
                        key="lesson_id",
                        match=qdrant_models.MatchValue(value=old_lesson_id)
                    )
                ]
            )

            # Scroll để lấy tất cả points cần update
            scroll_result = self.qdrant_client.scroll(
                collection_name=target_collection,
                scroll_filter=filter_condition,
                limit=10000,  # Lấy nhiều points
                with_payload=True,
                with_vectors=False
            )

            points_to_update = scroll_result[0]

            if not points_to_update:
                return {
                    "success": False,
                    "error": f"No points found with lesson_id '{old_lesson_id}' in book '{book_id}'"
                }

            logger.info(f"Found {len(points_to_update)} points to update")

            # Update từng point
            updated_count = 0
            for point in points_to_update:
                try:
                    # Update payload với lesson_id mới
                    self.qdrant_client.set_payload(
                        collection_name=target_collection,
                        payload={"lesson_id": new_lesson_id},
                        points=[point.id]
                    )
                    updated_count += 1
                except Exception as e:
                    logger.error(f"Error updating point {point.id}: {e}")
                    continue

            logger.info(f"Successfully updated {updated_count} points")

            return {
                "success": True,
                "book_id": book_id,
                "old_lesson_id": old_lesson_id,
                "new_lesson_id": new_lesson_id,
                "collection_name": target_collection,
                "points_updated": updated_count,
                "message": f"Updated {updated_count} points from lesson_id '{old_lesson_id}' to '{new_lesson_id}' in book '{book_id}'"
            }

        except Exception as e:
            logger.error(f"Error updating lesson_id in book {book_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to update lesson_id: {str(e)}"
            }

    async def update_book_id_in_qdrant(
        self, old_book_id: str, new_book_id: str
    ) -> Dict[str, Any]:
        """
        Update bookID cũ thành bookID mới trong Qdrant (rename collection và update metadata)

        Args:
            old_book_id: bookID cũ
            new_book_id: bookID mới

        Returns:
            Dict chứa kết quả update
        """
        from qdrant_client.http import models as qdrant_models

        self._ensure_service_initialized()

        if not self.qdrant_client:
            return {
                "success": False,
                "error": "Qdrant client not initialized"
            }

        try:
            # Tìm collections cho old_book_id
            collections = self.qdrant_client.get_collections()
            old_collections = []

            for collection in collections.collections:
                collection_name = collection.name
                if collection_name == f"textbook_{old_book_id}" or collection_name == f"guide_{old_book_id}":
                    old_collections.append(collection_name)

            if not old_collections:
                return {
                    "success": False,
                    "error": f"No collections found for book_id '{old_book_id}'"
                }

            logger.info(f"Found {len(old_collections)} collections to update for book_id '{old_book_id}'")

            results = []

            for old_collection_name in old_collections:
                try:
                    # Xác định loại collection và tạo tên mới
                    if old_collection_name.startswith("textbook_"):
                        new_collection_name = f"textbook_{new_book_id}"
                        content_type = "textbook"
                    else:  # guide_
                        new_collection_name = f"guide_{new_book_id}"
                        content_type = "guide"

                    logger.info(f"Processing collection: {old_collection_name} -> {new_collection_name}")

                    # Kiểm tra xem collection mới đã tồn tại chưa
                    existing_collections = [c.name for c in self.qdrant_client.get_collections().collections]
                    if new_collection_name in existing_collections:
                        return {
                            "success": False,
                            "error": f"Collection '{new_collection_name}' already exists. Cannot update book_id to '{new_book_id}'"
                        }

                    # Tạo collection mới
                    if not self._ensure_collection_exists(new_collection_name):
                        return {
                            "success": False,
                            "error": f"Failed to create new collection '{new_collection_name}'"
                        }

                    # Lấy tất cả points từ collection cũ
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=old_collection_name,
                        limit=10000,  # Lấy nhiều points
                        with_payload=True,
                        with_vectors=True
                    )

                    points_to_copy = scroll_result[0]
                    logger.info(f"Found {len(points_to_copy)} points to copy from {old_collection_name}")

                    if points_to_copy:
                        # Tạo points mới với book_id được update
                        new_points = []
                        for point in points_to_copy:
                            # Update payload với book_id mới
                            updated_payload = point.payload.copy()
                            updated_payload["book_id"] = new_book_id

                            new_points.append(
                                qdrant_models.PointStruct(
                                    id=point.id,
                                    vector=point.vector,
                                    payload=updated_payload
                                )
                            )

                        # Upsert points vào collection mới theo batch
                        batch_size = 100
                        for i in range(0, len(new_points), batch_size):
                            batch = new_points[i:i + batch_size]
                            self.qdrant_client.upsert(
                                collection_name=new_collection_name,
                                points=batch
                            )
                            logger.info(f"Copied batch {i//batch_size + 1}/{(len(new_points)-1)//batch_size + 1}")

                    # Xóa collection cũ
                    self.qdrant_client.delete_collection(old_collection_name)
                    logger.info(f"Deleted old collection: {old_collection_name}")

                    results.append({
                        "old_collection": old_collection_name,
                        "new_collection": new_collection_name,
                        "content_type": content_type,
                        "points_copied": len(points_to_copy),
                        "success": True
                    })

                except Exception as e:
                    logger.error(f"Error processing collection {old_collection_name}: {e}")
                    results.append({
                        "old_collection": old_collection_name,
                        "success": False,
                        "error": str(e)
                    })

            # Kiểm tra kết quả tổng thể
            successful_updates = [r for r in results if r.get("success")]
            failed_updates = [r for r in results if not r.get("success")]

            if failed_updates:
                return {
                    "success": False,
                    "old_book_id": old_book_id,
                    "new_book_id": new_book_id,
                    "results": results,
                    "error": f"Some collections failed to update: {len(failed_updates)} failed, {len(successful_updates)} succeeded"
                }

            return {
                "success": True,
                "old_book_id": old_book_id,
                "new_book_id": new_book_id,
                "collections_updated": len(successful_updates),
                "results": results,
                "message": f"Successfully updated book_id from '{old_book_id}' to '{new_book_id}' across {len(successful_updates)} collections"
            }

        except Exception as e:
            logger.error(f"Error updating book_id from {old_book_id} to {new_book_id}: {e}")
            return {
                "success": False,
                "error": f"Failed to update book_id: {str(e)}"
            }

    async def global_search(
        self, query: str, limit: int = 10, book_id: Optional[str] = None, lesson_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Tìm kiếm toàn cục trong tất cả sách giáo khoa với filter tùy chọn"""
        from qdrant_client.http import models as qdrant_models

        if not self.qdrant_client or not self.embedding_model:
            return {
                "success": False,
                "error": "Qdrant client or embedding model not initialized",
            }

        try:
            # Lấy danh sách tất cả collections
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]

            # Lọc các collection textbook và guide
            textbook_collections = [name for name in collection_names if name.startswith("textbook_")]
            guide_collections = [name for name in collection_names if name.startswith("guide_")]

            if book_id:
                # Nếu có book_id, chỉ tìm trong collection của book đó
                target_collection = f"textbook_{book_id}"
                if target_collection not in collection_names:
                    return {
                        "success": True,
                        "query": query,
                        "results": [],
                        "message": f"Collection {target_collection} not found"
                    }
                search_collections = [target_collection]
            else:
                # Nếu không có book_id, tìm trong tất cả collections
                search_collections = textbook_collections + guide_collections

            if not search_collections:
                return {
                    "success": True,
                    "query": query,
                    "results": [],
                    "message": "No textbook or guide collections found"
                }

            # Tạo embedding cho query
            query_vector = self.embedding_model.encode(query).tolist()

            # Chuẩn bị filters cho Qdrant - chỉ tìm content, không tìm metadata
            filter_conditions = [
                qdrant_models.FieldCondition(
                    key="type",
                    match=qdrant_models.MatchValue(value="content")
                )
            ]

            # Thêm filter cho lesson_id nếu có
            if lesson_id:
                filter_conditions.append(
                    qdrant_models.FieldCondition(
                        key="lesson_id",
                        match=qdrant_models.MatchValue(value=lesson_id)
                    )
                )



            # Tạo filter cuối cùng
            qdrant_filter = qdrant_models.Filter(must=filter_conditions)

            # Tìm kiếm trong tất cả collections
            all_results = []

            for collection_name in search_collections:
                try:
                    search_result = self.qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=query_vector,
                        query_filter=qdrant_filter,
                        limit=limit,
                        with_payload=True,
                        score_threshold=0.3,
                    )

                    # Thêm collection_name vào mỗi result
                    for result in search_result:
                        result.payload["source_collection"] = collection_name
                        all_results.append(result)

                except Exception as e:
                    logger.warning(f"Error searching in collection {collection_name}: {e}")
                    continue

            # Sắp xếp kết quả theo score
            all_results.sort(key=lambda x: x.score, reverse=True)

            # Lấy top results
            search_result = all_results[:limit]

            # Format kết quả
            results = []
            for scored_point in search_result:
                payload = scored_point.payload or {}

                # Bỏ qua metadata point (đã được filter ở trên nhưng double check)
                if payload.get("type") == "metadata":
                    continue

                # Chuẩn bị semantic metadata
                semantic_tags = payload.get("semantic_tags", [])

                # Backward compatibility
                if not semantic_tags and "semantic_type" in payload:
                    semantic_tags = [{"type": payload["semantic_type"], "confidence": 0.8}]

                results.append({
                    "text": payload.get("text", ""),
                    "score": scored_point.score,
                    "book_id": payload.get("book_id", ""),
                    "lesson_id": payload.get("lesson_id", ""),
                    "type": payload.get("type", ""),
                    "content_type": payload.get("content_type", ""),
                    "semantic_tags": semantic_tags,
                    "key_concepts": payload.get("key_concepts", []),
                    "estimated_difficulty": payload.get("estimated_difficulty", "basic"),
                    "contains_examples": payload.get("contains_examples", False),
                    "contains_definitions": payload.get("contains_definitions", False),
                    "contains_formulas": payload.get("contains_formulas", False),
                    "analysis_method": payload.get("analysis_method", "unknown")
                })

            # Sắp xếp theo score (kết quả đã được giới hạn bởi limit trong search)
            results.sort(key=lambda x: x["score"], reverse=True)

            return {
                "success": True,
                "query": query,
                "results": results,
                "collections_searched": search_collections,
                "total_results_found": len(results)
            }

        except Exception as e:
            logger.error(f"Error in global search: {e}")
            return {"success": False, "error": str(e)}

    async def delete_textbook_by_book_id(self, book_id: str) -> Dict[str, Any]:
        """
        Xóa textbook bằng book_id (xóa collection riêng của book_id)

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

            # Xác định collection name theo pattern textbook_bookId
            collection_name = f"textbook_{book_id}"

            # Kiểm tra collection có tồn tại không
            collections = self.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]

            if collection_name not in existing_names:
                return {
                    "success": False,
                    "error": f"Textbook collection {collection_name} not found",
                    "book_id": book_id
                }

            # Xóa toàn bộ collection
            self.qdrant_client.delete_collection(collection_name=collection_name)

            logger.info(f"Deleted collection: {collection_name} for book_id: {book_id}")

            return {
                "success": True,
                "message": f"Textbook '{book_id}' deleted successfully (collection removed)",
                "book_id": book_id,
                "collection_name": collection_name,
                "operation": "collection_deleted"
            }

        except Exception as e:
            logger.error(f"Error deleting textbook {book_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "book_id": book_id
            }

    async def delete_book_clean(self, book_id: str) -> Dict[str, Any]:
        """
        Xóa toàn bộ book (collection) - Clean version với exception handling

        Args:
            book_id: ID của book cần xóa

        Returns:
            Dict chứa thông tin xóa thành công

        Raises:
            ValueError: Nếu book_id không hợp lệ
            RuntimeError: Nếu Qdrant client chưa khởi tạo
            FileNotFoundError: Nếu collection không tồn tại
            Exception: Các lỗi khác từ Qdrant
        """
        self._ensure_service_initialized()

        if not book_id or not book_id.strip():
            raise ValueError("book_id cannot be empty")

        if not self.qdrant_client:
            raise RuntimeError("Qdrant client not initialized")

        # Tìm tất cả collections liên quan đến book_id (cả textbook và guide)
        collections = self.qdrant_client.get_collections().collections
        existing_names = [c.name for c in collections]

        target_collections = []
        for name in existing_names:
            if name.endswith(f"_{book_id}"):
                target_collections.append(name)

        if not target_collections:
            raise FileNotFoundError(f"Book '{book_id}' not found (no collections found for book_id)")

        # Xóa tất cả collections liên quan
        deleted_collections = []
        for collection_name in target_collections:
            self.qdrant_client.delete_collection(collection_name=collection_name)
            deleted_collections.append(collection_name)
            logger.info(f"✅ Deleted collection: {collection_name}")

        logger.info(f"✅ Deleted book '{book_id}' ({len(deleted_collections)} collections)")

        return {
            "book_id": book_id,
            "deleted_collections": deleted_collections,
            "operation": "book_deleted",
            "message": f"Book '{book_id}' and all its lessons deleted successfully from {len(deleted_collections)} collections"
        }

    async def delete_lesson_clean(self, lesson_id: str) -> Dict[str, Any]:
        """
        Xóa lesson cụ thể - Clean version với exception handling

        Args:
            lesson_id: ID của lesson cần xóa

        Returns:
            Dict chứa thông tin xóa thành công

        Raises:
            ValueError: Nếu lesson_id không hợp lệ
            RuntimeError: Nếu Qdrant client chưa khởi tạo
            FileNotFoundError: Nếu lesson không tồn tại
            Exception: Các lỗi khác từ Qdrant
        """
        from qdrant_client.http import models as qdrant_models

        self._ensure_service_initialized()

        if not lesson_id or not lesson_id.strip():
            raise ValueError("lesson_id cannot be empty")

        if not self.qdrant_client:
            raise RuntimeError("Qdrant client not initialized")

        # Tìm lesson trong các collections
        collections = self.qdrant_client.get_collections().collections
        textbook_collections = [c.name for c in collections if c.name.startswith("textbook_")]

        if not textbook_collections:
            raise FileNotFoundError("No textbook collections found")

        # Tạo filter để tìm lesson
        lesson_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="lesson_id",
                    match=qdrant_models.MatchValue(value=lesson_id)
                ),
                qdrant_models.FieldCondition(
                    key="type",
                    match=qdrant_models.MatchValue(value="content")
                )
            ]
        )

        # Tìm lesson trong từng collection
        found_collection = None
        found_book_id = None

        for collection_name in textbook_collections:
            try:
                scroll_result = self.qdrant_client.scroll(
                    collection_name=collection_name,
                    scroll_filter=lesson_filter,
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )

                points = scroll_result[0]
                if points:
                    found_collection = collection_name
                    found_book_id = points[0].payload.get("book_id", collection_name.replace("textbook_", ""))
                    break

            except Exception as e:
                logger.warning(f"Error checking collection {collection_name}: {e}")
                continue

        if not found_collection:
            raise FileNotFoundError(f"Lesson '{lesson_id}' not found in any textbook collection")

        # Xóa lesson
        delete_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="lesson_id",
                    match=qdrant_models.MatchValue(value=lesson_id)
                )
            ]
        )

        self.qdrant_client.delete(
            collection_name=found_collection,
            points_selector=qdrant_models.FilterSelector(filter=delete_filter)
        )

        logger.info(f"✅ Deleted lesson '{lesson_id}' from book '{found_book_id}' (collection: {found_collection})")

        return {
            "lesson_id": lesson_id,
            "book_id": found_book_id,
            "collection_name": found_collection,
            "operation": "lesson_deleted",
            "message": f"Lesson '{lesson_id}' deleted successfully from book '{found_book_id}'"
        }

    async def delete_lesson_in_book_clean(self, book_id: str, lesson_id: str) -> Dict[str, Any]:
        """
        Xóa lesson cụ thể trong book cụ thể - Clean version với exception handling

        Args:
            book_id: ID của book chứa lesson
            lesson_id: ID của lesson cần xóa

        Returns:
            Dict chứa thông tin xóa thành công

        Raises:
            ValueError: Nếu book_id hoặc lesson_id không hợp lệ
            RuntimeError: Nếu Qdrant client chưa khởi tạo
            FileNotFoundError: Nếu book hoặc lesson không tồn tại
            Exception: Các lỗi khác từ Qdrant
        """
        from qdrant_client.http import models as qdrant_models

        self._ensure_service_initialized()

        if not book_id or not book_id.strip():
            raise ValueError("book_id cannot be empty")

        if not lesson_id or not lesson_id.strip():
            raise ValueError("lesson_id cannot be empty")

        if not self.qdrant_client:
            raise RuntimeError("Qdrant client not initialized")

        collection_name = f"textbook_{book_id}"

        # Kiểm tra collection có tồn tại không
        collections = self.qdrant_client.get_collections().collections
        existing_names = [c.name for c in collections]

        if collection_name not in existing_names:
            raise FileNotFoundError(f"Book '{book_id}' not found (collection '{collection_name}' does not exist)")

        # Kiểm tra lesson có tồn tại trong collection không
        lesson_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="lesson_id",
                    match=qdrant_models.MatchValue(value=lesson_id)
                ),
                qdrant_models.FieldCondition(
                    key="type",
                    match=qdrant_models.MatchValue(value="content")
                )
            ]
        )

        try:
            scroll_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=lesson_filter,
                limit=1,
                with_payload=True,
                with_vectors=False
            )

            points = scroll_result[0]
            if not points:
                raise FileNotFoundError(f"Lesson '{lesson_id}' not found in book '{book_id}'")

        except Exception as e:
            if "not found" in str(e).lower():
                raise FileNotFoundError(f"Lesson '{lesson_id}' not found in book '{book_id}'")
            raise

        # Xóa lesson
        delete_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="lesson_id",
                    match=qdrant_models.MatchValue(value=lesson_id)
                )
            ]
        )

        self.qdrant_client.delete(
            collection_name=collection_name,
            points_selector=qdrant_models.FilterSelector(filter=delete_filter)
        )

        logger.info(f"✅ Deleted lesson '{lesson_id}' from book '{book_id}' (collection: {collection_name})")

        return {
            "lesson_id": lesson_id,
            "book_id": book_id,
            "collection_name": collection_name,
            "operation": "lesson_deleted_in_book",
            "message": f"Lesson '{lesson_id}' deleted successfully from book '{book_id}'"
        }

    async def check_lesson_id_exists(self, lesson_id: str) -> Dict[str, Any]:
        """
        Kiểm tra lesson_id đã tồn tại trong các textbook collections chưa

        Args:
            lesson_id: ID của lesson cần kiểm tra

        Returns:
            Dict chứa thông tin về lesson_id existence
        """
        self._ensure_service_initialized()

        if not self.qdrant_client:
            return {
                "success": False,
                "exists": False,
                "error": "Qdrant client not initialized"
            }

        try:
            from qdrant_client.http import models as qdrant_models

            # Lấy tất cả collections
            collections = self.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]

            # Tìm trong tất cả textbook collections
            textbook_collections = [name for name in existing_names if name.startswith("textbook_")]

            if not textbook_collections:
                return {
                    "success": True,
                    "exists": False,
                    "message": "No textbook collections found - lesson_id is available"
                }

            # Tạo filter để tìm lesson_id
            lesson_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="lesson_id",
                        match=qdrant_models.MatchValue(value=lesson_id)
                    ),
                    qdrant_models.FieldCondition(
                        key="type",
                        match=qdrant_models.MatchValue(value="content")
                    )
                ]
            )

            # Tìm trong từng collection
            for collection_name in textbook_collections:
                try:
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=collection_name,
                        scroll_filter=lesson_filter,
                        limit=1,
                        with_payload=True,
                        with_vectors=False
                    )

                    points = scroll_result[0]

                    if points:
                        # Lesson_id đã tồn tại
                        existing_point = points[0]
                        existing_book_id = existing_point.payload.get("book_id", "unknown")

                        return {
                            "success": True,
                            "exists": True,
                            "lesson_id": lesson_id,
                            "existing_book_id": existing_book_id,
                            "collection_name": collection_name,
                            "message": f"Lesson ID '{lesson_id}' already exists in book '{existing_book_id}'"
                        }
                except Exception as e:
                    logger.warning(f"Error checking collection {collection_name}: {e}")
                    continue

            # Lesson_id chưa tồn tại trong bất kỳ collection nào
            return {
                "success": True,
                "exists": False,
                "lesson_id": lesson_id,
                "message": f"Lesson ID '{lesson_id}' is available"
            }

        except Exception as e:
            logger.error(f"Error checking lesson_id existence {lesson_id}: {e}")
            return {
                "success": False,
                "exists": False,
                "error": f"Error checking lesson_id: {str(e)}",
                "lesson_id": lesson_id
            }

    async def delete_textbook_by_lesson_id(self, lesson_id: str) -> Dict[str, Any]:
        """
        Xóa lesson bằng lesson_id (xóa tất cả points có lesson_id từ collection riêng)

        Args:
            lesson_id: ID của lesson cần xóa

        Returns:
            Dict chứa kết quả xóa
        """
        from qdrant_client.http import models as qdrant_models

        try:
            if not self.qdrant_client:
                return {
                    "success": False,
                    "error": "Qdrant client not initialized"
                }

            # Tìm lesson trong các textbook collections
            collections = self.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]
            textbook_collections = [name for name in existing_names if name.startswith("textbook_")]

            if not textbook_collections:
                return {
                    "success": False,
                    "error": "No textbook collections found",
                    "lesson_id": lesson_id
                }

            # Tạo filter để tìm lesson_id
            delete_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="lesson_id",
                        match=qdrant_models.MatchValue(value=lesson_id)
                    )
                ]
            )

            # Tìm lesson trong từng collection
            found_collection = None
            found_book_id = None

            for collection_name in textbook_collections:
                try:
                    scroll_result = self.qdrant_client.scroll(
                        collection_name=collection_name,
                        scroll_filter=delete_filter,
                        limit=1,
                        with_payload=True,
                        with_vectors=False
                    )

                    if scroll_result[0]:  # Tìm thấy lesson
                        found_collection = collection_name
                        found_book_id = scroll_result[0][0].payload.get("book_id", "unknown")
                        break
                except Exception as e:
                    logger.warning(f"Error checking collection {collection_name}: {e}")
                    continue

            if not found_collection:
                return {
                    "success": False,
                    "error": f"No lesson found with lesson_id: {lesson_id}",
                    "lesson_id": lesson_id
                }

            # Xóa tất cả points có lesson_id này từ collection tìm thấy
            delete_result = self.qdrant_client.delete(
                collection_name=found_collection,
                points_selector=qdrant_models.FilterSelector(
                    filter=delete_filter
                )
            )

            logger.info(f"Deleted all points for lesson_id: {lesson_id} from collection: {found_collection}")

            return {
                "success": True,
                "message": f"Lesson '{lesson_id}' deleted successfully from collection {found_collection}",
                "lesson_id": lesson_id,
                "book_id": found_book_id,
                "collection_name": found_collection,
                "operation_info": delete_result.operation_id if hasattr(delete_result, 'operation_id') else "completed"
            }

        except Exception as e:
            logger.error(f"Error deleting textbook by lesson_id {lesson_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "lesson_id": lesson_id
            }

    async def get_textbook_info_by_book_id(self, book_id: str) -> Dict[str, Any]:
        """
        Lấy thông tin chi tiết về textbook theo book_id từ collection riêng

        Args:
            book_id: ID của textbook cần lấy thông tin

        Returns:
            Dict chứa thông tin chi tiết về textbook
        """
        self._ensure_service_initialized()

        if not self.qdrant_client:
            return {
                "success": False,
                "error": "Qdrant client not initialized"
            }

        try:
            from qdrant_client.http import models as qdrant_models

            # Kiểm tra cả textbook và guide collections
            textbook_collection = f"textbook_{book_id}"
            guide_collection = f"guide_{book_id}"

            collections = self.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]

            # Ưu tiên textbook collection, fallback sang guide collection
            if textbook_collection in existing_names:
                collection_name = textbook_collection
                content_type = "textbook"
            elif guide_collection in existing_names:
                collection_name = guide_collection
                content_type = "guide"
            else:
                return {
                    "success": False,
                    "error": f"No collection found for book_id '{book_id}'. Checked: {textbook_collection}, {guide_collection}",
                    "book_id": book_id
                }

            # Tìm metadata point của textbook
            metadata_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="book_id",
                        match=qdrant_models.MatchValue(value=book_id)
                    ),
                    qdrant_models.FieldCondition(
                        key="type",
                        match=qdrant_models.MatchValue(value="metadata")
                    )
                ]
            )

            metadata_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=metadata_filter,
                limit=1,
                with_payload=True,
                with_vectors=False
            )

            if not metadata_result[0]:
                return {
                    "success": False,
                    "error": f"Textbook with book_id '{book_id}' not found",
                    "book_id": book_id
                }

            # Lấy metadata
            metadata_point = metadata_result[0][0]
            metadata_payload = metadata_point.payload or {}

            # Tìm content points để lấy thống kê
            content_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="book_id",
                        match=qdrant_models.MatchValue(value=book_id)
                    ),
                    qdrant_models.FieldCondition(
                        key="type",
                        match=qdrant_models.MatchValue(value="content")
                    )
                ]
            )

            content_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=content_filter,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )

            # Tính toán thống kê
            total_chunks = len(content_result[0])
            unique_lessons = set()

            for point in content_result[0]:
                payload = point.payload or {}
                lesson_id = payload.get("lesson_id")
                if lesson_id:
                    unique_lessons.add(lesson_id)

            return {
                "success": True,
                "book_id": book_id,
                "content_type": content_type,  # Thêm thông tin loại content
                "book_info": {
                    "book_id": book_id,
                    "content_type": content_type,
                    "total_chunks": metadata_payload.get("total_chunks", total_chunks),
                    "total_lessons": len(unique_lessons),
                    "processed_at": metadata_payload.get("processed_at"),
                },
                "metadata": metadata_payload,  # Trả về toàn bộ metadata bao gồm file_url và uploaded_at
                "statistics": {
                    "total_chunks": total_chunks,
                    "total_lessons": len(unique_lessons),
                    "unique_lesson_ids": list(unique_lessons)
                },
                "collection_name": collection_name
            }

        except Exception as e:
            logger.error(f"Error getting textbook info for book_id {book_id}: {e}")
            return {
                "success": False,
                "error": f"Error getting textbook info: {str(e)}",
                "book_id": book_id
            }

    async def get_textbook_lessons_by_book_id(self, book_id: str) -> Dict[str, Any]:
        """
        Lấy danh sách lessons theo book_id từ collection riêng

        Args:
            book_id: ID của textbook

        Returns:
            Dict chứa danh sách lessons
        """
        self._ensure_service_initialized()

        if not self.qdrant_client:
            return {
                "success": False,
                "error": "Qdrant client not initialized"
            }

        try:
            from qdrant_client.http import models as qdrant_models

            # Sử dụng collection riêng cho textbook
            collection_name = f"textbook_{book_id}"

            # Kiểm tra collection có tồn tại không
            collections = self.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]

            if collection_name not in existing_names:
                return {
                    "success": False,
                    "error": f"Textbook collection {collection_name} not found",
                    "book_id": book_id
                }

            # Tìm tất cả content points của book_id
            content_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="book_id",
                        match=qdrant_models.MatchValue(value=book_id)
                    ),
                    qdrant_models.FieldCondition(
                        key="type",
                        match=qdrant_models.MatchValue(value="content")
                    )
                ]
            )

            content_result = self.qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=content_filter,
                limit=1000,
                with_payload=True,
                with_vectors=False
            )

            if not content_result[0]:
                return {
                    "success": False,
                    "error": f"No lessons found for book_id '{book_id}'",
                    "book_id": book_id
                }

            # Tổng hợp thông tin lessons
            lessons_map = {}

            for point in content_result[0]:
                payload = point.payload or {}
                lesson_id = payload.get("lesson_id")

                if lesson_id and lesson_id not in lessons_map:
                    lessons_map[lesson_id] = {
                        "lesson_id": lesson_id,
                        "book_id": book_id,
                        "chunk_count": 0,
                        "has_examples": False,
                        "has_definitions": False,
                        "has_formulas": False
                    }

                if lesson_id:
                    lessons_map[lesson_id]["chunk_count"] += 1

                    # Cập nhật content features
                    if payload.get("contains_examples"):
                        lessons_map[lesson_id]["has_examples"] = True
                    if payload.get("contains_definitions"):
                        lessons_map[lesson_id]["has_definitions"] = True
                    if payload.get("contains_formulas"):
                        lessons_map[lesson_id]["has_formulas"] = True

            lessons_list = list(lessons_map.values())
            lessons_list.sort(key=lambda x: x["lesson_id"])  # Sort by lesson_id

            return {
                "success": True,
                "book_id": book_id,
                "total_lessons": len(lessons_list),
                "lessons": lessons_list,
                "collection_name": collection_name
            }

        except Exception as e:
            logger.error(f"Error getting textbook lessons for book_id {book_id}: {e}")
            return {
                "success": False,
                "error": f"Error getting textbook lessons: {str(e)}",
                "book_id": book_id
            }

    async def get_lesson_info_by_lesson_id(self, lesson_id: str, book_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy thông tin chi tiết về lesson theo lesson_id từ các textbook collections

        Tối ưu hóa: Nếu có book_id thì tìm trực tiếp trong collection đó,
        chỉ trả về metadata (đặc biệt là file_url) mà không cần chunks để tiết kiệm thời gian

        Args:
            lesson_id: ID của lesson
            book_id: Optional - ID của book để tìm trực tiếp trong collection cụ thể

        Returns:
            Dict chứa thông tin metadata của lesson (không bao gồm chunks)
        """
        self._ensure_service_initialized()

        if not self.qdrant_client:
            return {
                "success": False,
                "error": "Qdrant client not initialized"
            }

        try:
            from qdrant_client.http import models as qdrant_models

            # Xác định collections cần tìm
            if book_id:
                # Nếu có book_id, tìm trực tiếp trong collection cụ thể
                target_collections = [f"textbook_{book_id}"]
                logger.info(f"Searching for lesson_id '{lesson_id}' in specific collection: textbook_{book_id}")
            else:
                # Nếu không có book_id, tìm trong tất cả textbook collections
                collections = self.qdrant_client.get_collections().collections
                existing_names = [c.name for c in collections]
                target_collections = [name for name in existing_names if name.startswith("textbook_")]
                logger.info(f"Searching for lesson_id '{lesson_id}' in all textbook collections: {target_collections}")

            if not target_collections:
                return {
                    "success": False,
                    "error": "No textbook collections found" + (f" for book_id '{book_id}'" if book_id else ""),
                    "lesson_id": lesson_id,
                    "book_id": book_id
                }

            # Tìm lesson trong collections (chỉ lấy 1 point để có metadata)
            lesson_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="lesson_id",
                        match=qdrant_models.MatchValue(value=lesson_id)
                    ),
                    qdrant_models.FieldCondition(
                        key="type",
                        match=qdrant_models.MatchValue(value="content")
                    )
                ]
            )

            found_collection = None
            lesson_point = None

            for collection_name in target_collections:
                try:
                    result = self.qdrant_client.scroll(
                        collection_name=collection_name,
                        scroll_filter=lesson_filter,
                        limit=1,  # Chỉ lấy 1 point để có metadata
                        with_payload=True,
                        with_vectors=False  # Không cần vectors
                    )

                    if result[0]:  # Tìm thấy lesson
                        found_collection = collection_name
                        lesson_point = result[0][0]  # Lấy point đầu tiên
                        break
                except Exception as e:
                    logger.warning(f"Error checking collection {collection_name}: {e}")
                    continue

            if not lesson_point:
                return {
                    "success": False,
                    "error": f"Lesson with lesson_id '{lesson_id}' not found" + (f" in book '{book_id}'" if book_id else ""),
                    "lesson_id": lesson_id,
                    "book_id": book_id
                }

            # Lấy metadata từ point đầu tiên (không cần lấy tất cả chunks)
            payload = lesson_point.payload or {}

            # Đếm tổng số chunks trong lesson (nếu cần)
            count_filter = qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="lesson_id",
                        match=qdrant_models.MatchValue(value=lesson_id)
                    ),
                    qdrant_models.FieldCondition(
                        key="type",
                        match=qdrant_models.MatchValue(value="content")
                    )
                ]
            )

            try:
                count_result = self.qdrant_client.count(
                    collection_name=found_collection,
                    count_filter=count_filter
                )
                total_chunks = count_result.count
            except Exception as e:
                logger.warning(f"Error counting chunks: {e}")
                total_chunks = 1

            # Tạo lesson info với metadata cần thiết
            lesson_data = {
                "lessonId": lesson_id,
                "bookId": payload.get("book_id", "Unknown"),
                "fileUrl": payload.get("file_url", ""),
                "uploaded_at": payload.get("uploaded_at", ""),
                "processed_at": payload.get("processed_at", ""),
                "content_type": payload.get("content_type", "textbook"),
                "total_chunks": total_chunks,
                "collection_name": found_collection,
                # Thêm một số metadata hữu ích khác
                "lesson_title": payload.get("lesson_title", ""),
                "chapter_title": payload.get("chapter_title", ""),
                "page_range": payload.get("page_range", ""),
                "word_count_total": payload.get("word_count_total", 0),
                "estimated_difficulty": payload.get("estimated_difficulty", "basic")
            }

            return {
                "success": True,
                "data": lesson_data,
                "message": f"Retrieved lesson '{lesson_id}' metadata successfully" + (f" from book '{book_id}'" if book_id else "")
            }

        except Exception as e:
            logger.error(f"Error getting lesson info for lesson_id {lesson_id}: {e}")
            return {
                "success": False,
                "error": f"Error getting lesson info: {str(e)}",
                "lesson_id": lesson_id,
                "book_id": book_id
            }

    async def get_all_textbooks(self) -> Dict[str, Any]:
        """
        Lấy danh sách tất cả textbooks từ individual collections

        Returns:
            Dict chứa danh sách textbooks với metadata
        """
        from qdrant_client.http import models as qdrant_models

        try:
            if not self.qdrant_client:
                logger.error("❌ Qdrant client not initialized in get_all_textbooks")
                return {
                    "success": False,
                    "error": "Qdrant client not initialized"
                }

            # Lấy tất cả collections
            collections = self.qdrant_client.get_collections().collections
            existing_names = [c.name for c in collections]

            logger.info(f"🔍 Found {len(existing_names)} collections: {existing_names}")

            # Tìm các collections textbook và guide
            textbook_collections = [name for name in existing_names if name.startswith(('textbook_', 'guide_'))]

            if not textbook_collections:
                return {
                    "success": True,
                    "textbooks": [],
                    "message": "No textbooks found. No textbook or guide collections exist."
                }

            # Tìm metadata từ tất cả collections
            textbooks = []

            for collection_name in textbook_collections:
                try:
                    logger.info(f"🔍 Checking collection: {collection_name}")

                    # Tìm metadata points trong collection này
                    metadata_filter = qdrant_models.Filter(
                        must=[
                            qdrant_models.FieldCondition(
                                key="type",
                                match=qdrant_models.MatchValue(value="metadata")
                            )
                        ]
                    )

                    scroll_result = self.qdrant_client.scroll(
                        collection_name=collection_name,
                        scroll_filter=metadata_filter,
                        limit=100,
                        with_payload=True,
                        with_vectors=False
                    )

                    if scroll_result[0]:
                        for point in scroll_result[0]:
                            payload = point.payload or {}

                            # Lấy thông tin từ metadata point
                            book_id = payload.get("book_id", "unknown")
                            content_type = payload.get("content_type", "unknown")

                            textbook_data = {
                                "book_id": book_id,
                                "content_type": content_type,
                                "collection_name": collection_name,
                                "total_chunks": payload.get("total_chunks", 0),
                                "processed_at": payload.get("processed_at"),
                                "model": payload.get("model", "unknown"),
                                "chunk_size": payload.get("chunk_size", 0),
                                "chunk_overlap": payload.get("chunk_overlap", 0),
                            }
                            textbooks.append(textbook_data)

                except Exception as e:
                    logger.warning(f"Error processing collection {collection_name}: {e}")
                    continue

            return {
                "success": True,
                "textbooks": textbooks,
                "total_textbooks": len(textbooks),
                "collections_checked": textbook_collections
            }

        except Exception as e:
            logger.error(f"Error getting all textbooks: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Factory function để tạo QdrantService instance
def get_qdrant_service() -> QdrantService:
    """
    Tạo QdrantService instance mới

    Returns:
        QdrantService: Fresh instance
    """
    return QdrantService()
